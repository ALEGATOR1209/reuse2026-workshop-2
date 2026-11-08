"""NeuroMechFly body (flygym 2.x) driven by "actions" (raise a leg, spread the wings, ...).

The fly is tethered (its thorax is held in place, like a fly on a ball in the lab), so
moving legs can't tip it over. Legs are position-controlled by MuJoCo actuators; the
wings and antennae have no muscles in the model, so we set their angles directly each step.

Every user gets their own body (one shared MuJoCo model, one state per user), stepped
together on a few threads. Nothing is rendered here: the browser gets the fly's meshes once
(`model_payload`) and then a stream of its body's poses (`pose_stream`), and draws the fly
in 3D itself, so every viewer can rotate their own view.
"""

import asyncio
import base64
import gzip
import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation

# Each action is a set of joint-angle offsets from the neutral pose (radians), found by
# trying joints one at a time and rendering the result. Index 0 is "rest".
# "{s}" is filled with l / r for the two sides.
FRONT = {
    "c_thorax-{s}f_coxa-pitch": -0.9,               # swing the leg forward/up
    "c_thorax-{s}f_coxa-roll": +1.6,                # ... and out to the side
    "{s}f_coxa-{s}f_trochanterfemur-pitch": -0.9,   # straighten the femur
    "{s}f_trochanterfemur-{s}f_tibia-pitch": -0.4,  # and the tibia
}
MIDDLE = {"c_thorax-{s}m_coxa-roll": +1.2, "{s}m_coxa-{s}m_trochanterfemur-pitch": -0.8}
HIND = {"c_thorax-{s}h_coxa-roll": -1.2, "{s}h_coxa-{s}h_trochanterfemur-pitch": -1.0}

# Each action's index is a class label in the saved recordings: only ever append new actions.
ACTIONS = [  # (name, legs: {side: offsets}, wings, antennae)
    ("rest", {}, None, None),
    ("right front leg up", {"r": FRONT}, None, None),
    ("left front leg up", {"l": FRONT}, None, None),
    ("both front legs up", {"l": FRONT, "r": FRONT}, None, None),
    ("wings out", {}, "out", None),
    ("flap wings", {}, "flap", None),
    ("middle legs out", {"l": MIDDLE, "r": MIDDLE}, None, None),
    ("hind legs out", {"l": HIND, "r": HIND}, None, None),
    ("antennae up", {}, None, "up"),
    ("wiggle antennae", {}, None, "wiggle"),
]
WING_OUT_ROLL = -1.0     # wing roll sweeps the wing out sideways
FLAP_HZ = 3.0            # slowed down ~70x from a real fly so the eye can follow it
# Antennae: the whole antenna turns where the pedicel meets the head (funiculus and arista
# follow). Yaw tilts it outward so the arista points up; negative pitch swings it forward.
ANT_JOINTS = [f"c_head-{s}_pedicel-{ax}" for s in "lr" for ax in ("pitch", "yaw")]
ANT_UP = (-0.5, 0.5)     # (pitch, yaw) for "antennae up": forward and out, aristae pointing up
# "wiggle antennae": each antenna swings between up-and-forward (pitch -0.6, yaw 0.45) and
# back (pitch +0.3, yaw -0.05), left and right in antiphase: (pitch, yaw) = centre + amplitude * sin
ANT_WIGGLE = ((-0.15, 0.2), (0.45, 0.25))
WIGGLE_HZ = 2.0          # slowed down like the wing flap
FLOOR_Z = 1.40           # where the feet are in the neutral pose (mm); the floor is drawn by the browser
TIMESTEP = 5e-4          # physics step (s); 5x flygym's default, same final poses, 5x cheaper
IDLE_FORGET_S = 900      # drop a body nobody has watched for 15 minutes


class Body:
    """One user's fly: its own MuJoCo state, commands and latest pose."""

    def __init__(self, data, n_actions):
        self.data = data
        self.command = np.zeros(n_actions)   # wanted level of each action, 0..1
        self.level = np.zeros(n_actions)     # current (smoothed) level
        self.t = 0.0                          # body time (s), drives the wing flap
        self.viewers = 0
        self.last_seen = time.time()
        self.settled = False
        self.pose, self.frame_id = None, 0


class BodyPool:
    def __init__(self, fps=25, threads=6):
        self.fps, self.threads = fps, threads
        self._bodies, self._lock = {}, threading.Lock()
        self._build_lock = threading.Lock()
        self._built = False
        self._thread = None
        self._model_json = None

    # ---- the shared model ----
    def _build(self):
        with self._build_lock:
            if self._built:
                return
            from flygym import Simulation
            from flygym.anatomy import ActuatedDOFPreset, AxisOrder, JointPreset, Skeleton
            from flygym.compose import KinematicPosePreset, NeuroMechFly
            from flygym.compose.fly.base_fly import ActuatorType
            from flygym.compose.world.tethered_world import TetheredWorld
            from flygym.utils.math import Rotation3D

            fly = NeuroMechFly()
            skel = Skeleton(joint_preset=JointPreset.ALL_BIOLOGICAL, axis_order=AxisOrder.ROLL_PITCH_YAW)
            pose = KinematicPosePreset.NEUTRAL
            fly.add_joints(skel, neutral_pose=pose)
            fly.add_actuators(skel.get_actuated_dofs_from_preset(ActuatedDOFPreset.LEGS_ACTIVE_ONLY),
                              actuator_type="position", neutral_input=pose, kp=50)
            fly.colorize()
            world = TetheredWorld()
            world.add_fly(fly, [0, 0, 1.5], Rotation3D(format="quat", values=[1, 0, 0, 0]))
            sim = Simulation(world)
            m = sim.mj_model
            m.opt.timestep = TIMESTEP

            # Leg targets: neutral pose + offsets per action, in actuator (= ctrl) order.
            order = [d.name for d in fly.get_actuated_jointdofs_order(ActuatorType.POSITION)]
            act_ids = [mj.mj_name2id(m, mj.mjtObj.mjOBJ_ACTUATOR, f"{fly.name}/{o}-position") for o in order]
            assert act_ids == list(range(m.nu)), "unexpected actuator order"
            angles = pose.get_pose_by_axis_order(AxisOrder.ROLL_PITCH_YAW).joint_angles_lookup_rad
            self.neutral = np.array([angles.get(n, 0.0) for n in order])
            self.leg_dirs = np.zeros((len(ACTIONS), len(order)))
            for a, (_, legs, _, _) in enumerate(ACTIONS):
                for side, offsets in legs.items():
                    for name, off in offsets.items():
                        self.leg_dirs[a, order.index(name.format(s=side))] = off

            # Wings have no muscles in the model, only springs back to rest. We set their angles
            # ourselves, so remove the springs and gravity on them so they stay where we put them.
            wings = [f"c_thorax-{s}_wing-{ax}" for s in "lr" for ax in ("roll", "yaw")]
            jids = [mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, f"{fly.name}/{w}") for w in wings]
            self.wing_q = np.array([m.jnt_qposadr[j] for j in jids])   # l roll, l yaw, r roll, r yaw
            self.wing_v = np.array([m.jnt_dofadr[j] for j in jids])
            for j in jids:
                m.jnt_stiffness[j] = 0
                m.body_gravcomp[m.jnt_bodyid[j]] = 1
            # Same for the antennae (pedicel joints; the funiculus and arista keep their springs).
            jids = [mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, f"{fly.name}/{a}") for a in ANT_JOINTS]
            self.ant_q = np.array([m.jnt_qposadr[j] for j in jids])    # l pitch, l yaw, r pitch, r yaw
            self.ant_v = np.array([m.jnt_dofadr[j] for j in jids])
            for j in jids:
                m.jnt_stiffness[j] = 0
            for s in "lr":
                for part in ("pedicel", "funiculus", "arista"):
                    m.body_gravcomp[mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, f"{fly.name}/{s}_{part}")] = 1
            mode_of = lambda i, mode: [a for a, act in enumerate(ACTIONS) if act[i] == mode][0]
            self.i_out, self.i_flap = mode_of(2, "out"), mode_of(2, "flap")
            self.i_ant_up, self.i_wiggle = mode_of(3, "up"), mode_of(3, "wiggle")
            self.periodic = [self.i_flap, self.i_wiggle]   # actions that keep moving while on

            self.m = m
            self.template = mj.MjData(m)              # neutral pose, settled
            mj.mj_copyData(self.template, m, sim.mj_data)
            self.template.ctrl[:] = self.neutral
            mj.mj_step(m, self.template, nstep=2000)
            self.steps = int(round(1 / self.fps / TIMESTEP))
            self.geoms = np.flatnonzero(m.geom_type == mj.mjtGeom.mjGEOM_MESH)
            self._built = True

    @staticmethod
    def _color(m, g):
        """A body part's colour: its material colour times the average colour of its texture, if any."""
        mat = m.geom_matid[g]
        if mat < 0:
            return m.geom_rgba[g].copy()
        rgba = m.mat_rgba[mat].copy()
        tex = m.mat_texid[mat][mj.mjtTextureRole.mjTEXROLE_RGB]
        if tex >= 0:
            a, n, c = m.tex_adr[tex], m.tex_width[tex] * m.tex_height[tex], m.tex_nchannel[tex]
            rgba[:3] *= m.tex_data[a:a + n * c].reshape(n, c)[:, :3].mean(0) / 255
        return rgba

    def model_payload(self):
        """Gzipped JSON: every mesh of the fly (vertices, triangles) and which body part uses which mesh."""
        if self._model_json is None:
            self._build()
            m = self.m
            meshes, verts, faces, geoms, seen = [], [], [], [], {}
            for g in self.geoms:
                mid = int(m.geom_dataid[g])
                if mid not in seen:
                    va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
                    fa, fn = m.mesh_faceadr[mid], m.mesh_facenum[mid]
                    seen[mid] = len(meshes)
                    meshes.append({"v0": int(sum(len(v) for v in verts) // 3), "nv": int(vn),   # in vertices
                                   "f0": int(sum(len(f) for f in faces) // 3), "nf": int(fn)})  # in triangles
                    verts.append(m.mesh_vert[va:va + vn].astype(np.float32).ravel())
                    faces.append(m.mesh_face[fa:fa + fn].astype(np.uint16).ravel())
                name = mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, g).split("/")[-1]
                geoms.append({"name": name, "mesh": seen[mid], "rgba": np.round(self._color(m, g), 3).tolist()})
            b64 = lambda a: base64.b64encode(np.concatenate(a).tobytes()).decode()
            payload = {"meshes": meshes, "geoms": geoms, "verts": b64(verts), "faces": b64(faces),
                       "floor_z": FLOOR_Z, "actions": [a[0] for a in ACTIONS], "fps": self.fps}
            self._model_json = gzip.compress(json.dumps(payload).encode(), 6)
        return self._model_json

    # ---- bodies ----
    def body(self, sid):
        self._build()
        with self._lock:
            b = self._bodies.get(sid)
            if b is None:
                d = mj.MjData(self.m)
                mj.mj_copyData(d, self.m, self.template)
                b = self._bodies[sid] = Body(d, len(ACTIONS))
                self._advance(b, force=True)
            b.last_seen = time.time()
            return b

    def set_levels(self, sid, levels):
        b = self.body(sid)
        levels = np.clip(np.asarray(levels, float)[:len(ACTIONS)], 0, 1)
        b.command = np.r_[levels, np.zeros(len(ACTIONS) - len(levels))]
        b.settled = False

    def stats(self):
        with self._lock:
            return {"bodies": len(self._bodies), "watched": sum(b.viewers > 0 for b in self._bodies.values())}

    def _wing_angles(self, b):
        out, flap = b.level[self.i_out], b.level[self.i_flap]
        roll = WING_OUT_ROLL * max(out, 0.7 * flap)
        yaw = flap * (0.45 + 0.55 * math.sin(2 * math.pi * FLAP_HZ * b.t))  # yaw lifts the wings up
        return np.array([roll, yaw, roll, yaw])

    def _antenna_angles(self, b):
        up, wig = b.level[self.i_ant_up], b.level[self.i_wiggle]
        sin = math.sin(2 * math.pi * WIGGLE_HZ * b.t)
        (cp, cy), (ap, ay) = ANT_WIGGLE
        out = []
        for side in (1, -1):   # left, right (antiphase)
            out += [up * ANT_UP[0] + wig * (cp - side * ap * sin), up * ANT_UP[1] + wig * (cy + side * ay * sin)]
        return np.array(out)

    def _advance(self, b, force=False):
        """Move one body forward by one frame (1/fps s). Bodies standing still are not simulated."""
        m, d = self.m, b.data
        # ease toward the command (~0.25 s time constant) so the limbs move smoothly
        b.level += (b.command - b.level) * min(1.0, 4.0 / self.fps)
        b.t += 1 / self.fps
        moving = b.level[self.periodic].max() > 1e-3            # flapping wings or wiggling antennae
        if not force and b.settled and not moving:
            return
        d.ctrl[:] = self.neutral + b.level @ self.leg_dirs
        wings, ant = self._wing_angles(b), self._antenna_angles(b)
        d.qpos[self.wing_q], d.qvel[self.wing_v] = wings, 0
        d.qpos[self.ant_q], d.qvel[self.ant_v] = ant, 0
        mj.mj_step(m, d, nstep=self.steps)                       # releases the GIL: bodies run in parallel
        d.qpos[self.wing_q], d.qvel[self.wing_v] = wings, 0
        d.qpos[self.ant_q], d.qvel[self.ant_v] = ant, 0
        mj.mj_kinematics(m, d)
        b.settled = np.abs(b.command - b.level).max() < 1e-3 and np.abs(d.qvel).max() < 1e-3
        quat = Rotation.from_matrix(d.geom_xmat[self.geoms].reshape(-1, 3, 3)).as_quat()   # x y z w
        pose = np.hstack([d.geom_xpos[self.geoms], quat[:, [3, 0, 1, 2]]]).astype(np.float32)  # x y z, qw qx qy qz
        b.pose = json.dumps({"p": base64.b64encode(pose.tobytes()).decode(), "l": np.round(b.level, 3).tolist()})
        b.frame_id += 1

    def _loop(self):
        self._build()
        pool = ThreadPoolExecutor(self.threads, thread_name_prefix="flybody")
        while True:
            t0 = time.time()
            with self._lock:
                watched = [b for b in self._bodies.values() if b.viewers > 0]
                for sid in [s for s, b in self._bodies.items() if b.viewers == 0 and t0 - b.last_seen > IDLE_FORGET_S]:
                    del self._bodies[sid]
            if watched:
                list(pool.map(self._advance, watched))
            # keep a steady frame rate: aim for the next tick, not "now + 1/fps"
            self._next = max(getattr(self, "_next", t0), t0 - 1 / self.fps) + 1 / self.fps
            time.sleep(max(0.0, self._next - time.time()) if watched else 0.1)

    def start(self):
        with self._build_lock:
            if self._thread is None:
                self._thread = threading.Thread(target=self._loop, daemon=True, name="flybodies")
                self._thread.start()

    async def pose_stream(self, sid):
        """Server-sent events: this user's body pose, once per physics frame. Only watched bodies move."""
        self.start()
        b = self.body(sid)
        b.viewers += 1
        last = -1
        try:
            while True:
                if b.frame_id != last and b.pose:
                    last = b.frame_id
                    yield f"data: {b.pose}\n\n"
                b.last_seen = time.time()
                await asyncio.sleep(0.5 / self.fps)
        finally:
            b.viewers -= 1
