"""Export the fly's poses so a browser can animate it without a physics engine.

    uv run main.py poses        # -> flymnist/static/fly-poses.json (used by flymnist/static/flypose.js)

The server (body.py) moves the fly with MuJoCo physics. For a static copy of the app (no Python
server) we only need what students actually see, and that turns out to be simple:

- every joint of the fly is a hinge, and the thorax is held in place, so the pose of every body
  part follows from the joint angles by forward kinematics (the kinematic tree is exported);
- leg joints are position-controlled (kp=50) against joint springs (k=10), so where a leg settles
  is a linear function of the command: rest angle + 5/6 of the offset. We settle each action in
  the real physics (level 1) and export the joint-angle change it causes. Several actions mix by
  adding their changes times their levels, exactly as body.py adds their leg offsets;
- wings and antennae are set directly by formulas in body.py; the browser evaluates the same
  formulas (the constants are exported, read from body.py).

The legs follow the command within a few ms, much faster than a frame, so no extra lag is modelled.
"""

import json
import math
from pathlib import Path

import mujoco as mj
import numpy as np

from . import body as B

OUT = Path(__file__).parent / "static" / "fly-poses.json"


def _name(m, kind, i):
    return mj.mj_id2name(m, kind, i).split("/")[-1]


def _settle(pool, b, levels, max_frames=300):
    """Hold `levels` (no easing) until the body stops moving; return its joint angles."""
    b.command[:] = levels
    b.level[:] = levels
    for _ in range(max_frames):
        pool._advance(b, force=True)
        if np.abs(b.data.qvel).max() < 1e-6:
            break
    return b.data.qpos.copy()


WING = {"flap_roll": 0.7, "yaw_mid": 0.45, "yaw_amp": 0.55}   # the numbers inside body.py's _wing_angles


def wing_angles(out, flap, t):
    roll = B.WING_OUT_ROLL * max(out, WING["flap_roll"] * flap)
    yaw = flap * (WING["yaw_mid"] + WING["yaw_amp"] * math.sin(2 * math.pi * B.FLAP_HZ * t))
    return [roll, yaw, roll, yaw]


def antenna_angles(up, wig, t):
    s = math.sin(2 * math.pi * B.WIGGLE_HZ * t)
    (cp, cy), (ap, ay) = B.ANT_WIGGLE
    return [v for side in (1, -1) for v in (up * B.ANT_UP[0] + wig * (cp - side * ap * s),
                                            up * B.ANT_UP[1] + wig * (cy + side * ay * s))]


def _check_formulas(pool):
    """The exported wing/antenna formulas must match body.py's (they are what the browser evaluates)."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        b = B.Body(None, len(B.ACTIONS))
        b.level, b.t = rng.random(len(B.ACTIONS)), rng.random() * 10
        lv = b.level
        assert np.allclose(pool._wing_angles(b), wing_angles(lv[pool.i_out], lv[pool.i_flap], b.t)), "wing formula changed"
        assert np.allclose(pool._antenna_angles(b), antenna_angles(lv[pool.i_ant_up], lv[pool.i_wiggle], b.t)), \
            "antenna formula changed"


def build(pool=None):
    pool = pool or B.BodyPool()
    pool._build()
    m = pool.m
    assert m.nq == m.njnt and (m.jnt_type == mj.mjtJoint.mjJNT_HINGE).all(), "expected hinge joints only"
    assert not m.jnt_limited.any() and np.abs(m.jnt_pos).max() == 0 and np.abs(m.qpos0).max() == 0

    # Every body a drawn geom hangs from, up to the (fixed) thorax, in MuJoCo order (parents first).
    need = set()
    for g in pool.geoms:
        bid = int(m.geom_bodyid[g])
        while bid > 0:
            need.add(bid)
            bid = int(m.body_parentid[bid])
    order = sorted(need)
    root = order[0]
    assert all(int(m.body_parentid[b]) in need or b == root for b in order) and m.body_mocapid[root] >= 0
    bidx = {b: i for i, b in enumerate(order)}

    # Joints (all hinges) of those bodies, in MuJoCo order = the order forward kinematics applies them.
    joints = [j for b in order for j in range(m.body_jntadr[b], m.body_jntadr[b] + m.body_jntnum[b])]
    jidx = {j: i for i, j in enumerate(joints)}
    jq = np.array([m.jnt_qposadr[j] for j in joints])
    by_name = {_name(m, mj.mjtObj.mjOBJ_JOINT, j): i for i, j in enumerate(joints)}

    # Joints the browser sets itself (body.py sets these qpos directly every frame).
    wing_j = [by_name[f"c_thorax-{s}_wing-{ax}"] for s in "lr" for ax in ("roll", "yaw")]
    ant_j = [by_name[a] for a in B.ANT_JOINTS]
    assert [jq[i] for i in wing_j] == list(pool.wing_q) and [jq[i] for i in ant_j] == list(pool.ant_q)
    driven = set(wing_j) | set(ant_j)

    n = len(B.ACTIONS)
    b = pool.body("__poses_export__")
    try:
        q_rest = _settle(pool, b, np.zeros(n))[jq]
        root_pos, root_quat = b.data.xpos[root].copy(), b.data.xquat[root].copy()
        actions = []
        for a, (name, *_rest) in enumerate(B.ACTIONS):
            deltas = []
            if a not in pool.periodic and a > 0:   # periodic actions only move driven joints
                lv = np.zeros(n)
                lv[a] = 1
                dq = _settle(pool, b, lv)[jq] - q_rest
                deltas = [[i, round(float(dq[i]), 5)] for i in range(len(joints))
                          if i not in driven and abs(dq[i]) > 2e-4]
            actions.append({"name": name, "dq": deltas})
        _settle(pool, b, np.zeros(n))
    finally:
        with pool._lock:
            pool._bodies.pop("__poses_export__", None)

    _check_formulas(pool)
    r = lambda x, k: [round(float(v), k) for v in x]
    ease = min(1.0, 4.0 / pool.fps)   # body.py: level += (command - level) * min(1, 4/fps), once per frame
    wq, aq = q_rest[wing_j], q_rest[ant_j]
    assert np.abs(wq).max() < 1e-9 and np.abs(aq).max() < 1e-9, "driven joints rest at 0"
    (cp, cy), (ap, ay) = B.ANT_WIGGLE
    return {
        "about": "NeuroMechFly pose data for flypose.js, made by `uv run main.py poses` (flymnist/poses.py)",
        "fps": pool.fps,
        # the same easing as a rate per second: level += (command - level) * (1 - exp(-ease_rate * dt))
        "ease_rate": round(-pool.fps * math.log(1 - ease), 6) if ease < 1 else 1e6,
        "rest_action": 0,
        "actions": actions,
        # bodies: [parent index, pos, quat (in the parent's frame; -1 = the world, for the thorax held in
        # place), first joint, number of joints]
        "bodies": [[bidx.get(int(m.body_parentid[bb]), -1),
                    r(root_pos if bb == root else m.body_pos[bb], 6), r(root_quat if bb == root else m.body_quat[bb], 7),
                    jidx[m.body_jntadr[bb]] if m.body_jntnum[bb] else 0, int(m.body_jntnum[bb])]
                   for bb in order],
        "body_names": [_name(m, mj.mjtObj.mjOBJ_BODY, bb) for bb in order],
        # joints: [axis x, y, z] (hinges through the body origin) and the settled rest angle
        "joint_axes": [r(m.jnt_axis[j], 0) for j in joints],
        "joint_names": [_name(m, mj.mjtObj.mjOBJ_JOINT, j) for j in joints],
        "rest": r(q_rest, 5),
        # geoms, in the order of /api/fly/model and of the pose stream: [body index, pos, quat]
        "geoms": [[bidx[int(m.geom_bodyid[g])], r(m.geom_pos[g], 6), r(m.geom_quat[g], 7)] for g in pool.geoms],
        "geom_names": [_name(m, mj.mjtObj.mjOBJ_GEOM, g) for g in pool.geoms],
        # body.py _wing_angles: roll = out_roll * max(out, flap_roll * flap);
        #                       yaw = flap * (yaw_mid + yaw_amp * sin(2 pi hz t)); joints l roll, l yaw, r roll, r yaw
        "wings": {"joints": wing_j, "out": pool.i_out, "flap": pool.i_flap, "out_roll": B.WING_OUT_ROLL,
                  **WING, "hz": B.FLAP_HZ},
        # body.py _antenna_angles: for side = +1 (left), -1 (right):
        #   pitch = up * up_pitch + wiggle * (centre_pitch - side * amp_pitch * sin(2 pi hz t))
        #   yaw   = up * up_yaw   + wiggle * (centre_yaw   + side * amp_yaw   * sin(2 pi hz t))
        "antennae": {"joints": ant_j, "up": pool.i_ant_up, "wiggle": pool.i_wiggle, "up_angles": list(B.ANT_UP),
                     "centre": [cp, cy], "amp": [ap, ay], "hz": B.WIGGLE_HZ},
        "periodic": list(pool.periodic),
    }


def export(out=OUT):
    data = build()
    text = json.dumps(data, separators=(",", ":"))
    Path(out).write_text(text)
    print(f"wrote {out} ({len(text) / 1024:.1f} KB, {len(data['geoms'])} geoms, {len(data['rest'])} joints)")
    return data
