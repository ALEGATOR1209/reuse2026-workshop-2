// Animate the NeuroMechFly body without a physics engine ("pose blending").
//
//   import { loadPoser } from "./flypose.js";
//   const poser = await loadPoser("fly-poses.json");
//   poser.setLevels(levels);                 // 10 numbers 0..1, in the order of the fly's actions
//   const { pose, levels, moving } = poser.update(performance.now() / 1000);
//
// `pose` has the layout of one pose frame of /api/fly/stream: per geom (in /api/fly/model order)
// x y z qw qx qy qz in the world frame. `moving` is false when nothing changed since the last
// update, so the caller can skip redrawing.
//
// How: fly-poses.json (made by `uv run main.py poses`, see flymnist/poses.py) holds the fly's
// kinematic tree (every joint is a hinge, the thorax is fixed), its rest joint angles, and how much
// each action turns each leg joint once the physics settles. Joint angles = rest + sum of
// level * change, wings and antennae from the same formulas as flymnist/body.py, then forward
// kinematics like MuJoCo's. Levels ease toward the command like the server's (~0.23 s time constant).

export async function loadPoser(url = "fly-poses.json") {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status} ${res.statusText}`);
  return createPoser(await res.json());
}

// q = a * b (quaternions as [w, x, y, z] at offsets)
function qmul(o, oi, a, ai, b, bi) {
  const aw = a[ai], ax = a[ai + 1], ay = a[ai + 2], az = a[ai + 3];
  const bw = b[bi], bx = b[bi + 1], by = b[bi + 2], bz = b[bi + 3];
  o[oi] = aw * bw - ax * bx - ay * by - az * bz;
  o[oi + 1] = aw * bx + ax * bw + ay * bz - az * by;
  o[oi + 2] = aw * by - ax * bz + ay * bw + az * bx;
  o[oi + 3] = aw * bz + ax * by - ay * bx + az * bw;
}
// o = p + rotate(q, v)
function qrotAdd(o, oi, p, pi, q, qi, v, vi) {
  const w = q[qi], x = q[qi + 1], y = q[qi + 2], z = q[qi + 3];
  const vx = v[vi], vy = v[vi + 1], vz = v[vi + 2];
  const tx = 2 * (y * vz - z * vy), ty = 2 * (z * vx - x * vz), tz = 2 * (x * vy - y * vx);   // t = 2 q.xyz × v
  o[oi] = p[pi] + vx + w * tx + (y * tz - z * ty);
  o[oi + 1] = p[pi + 1] + vy + w * ty + (z * tx - x * tz);
  o[oi + 2] = p[pi + 2] + vz + w * tz + (x * ty - y * tx);
}

export function createPoser(d) {
  const nA = d.actions.length, nB = d.bodies.length, nJ = d.rest.length, nG = d.geoms.length;
  const TAU = 2 * Math.PI;
  const flat = (rows, k) => Float64Array.from(rows.flatMap((r) => r[k]));
  const parent = Int32Array.from(d.bodies, (b) => b[0]);
  const bpos = flat(d.bodies, 1), bquat = flat(d.bodies, 2);
  const jadr = Int32Array.from(d.bodies, (b) => b[3]), jnum = Int32Array.from(d.bodies, (b) => b[4]);
  const axes = Float64Array.from(d.joint_axes.flat());
  const rest = Float64Array.from(d.rest);
  const gbody = Int32Array.from(d.geoms, (g) => g[0]), gpos = flat(d.geoms, 1), gquat = flat(d.geoms, 2);
  const world = { pos: new Float64Array(3), quat: Float64Array.of(1, 0, 0, 0) };
  const W = d.wings, A = d.antennae, periodic = d.periodic;

  const command = new Float64Array(nA), level = new Float64Array(nA);
  const q = new Float64Array(nJ);
  const xpos = new Float64Array(nB * 3), xquat = new Float64Array(nB * 4), jq = new Float64Array(4);
  const pose = new Float32Array(nG * 7), levelsOut = new Float32Array(nA);
  let lastT = null, first = true;

  function jointAngles(t) {
    q.set(rest);
    for (let a = 0; a < nA; a++) {
      const l = level[a];
      if (!l) continue;
      for (const [j, dq] of d.actions[a].dq) q[j] += l * dq;
    }
    // wings (flymnist/body.py _wing_angles): l roll, l yaw, r roll, r yaw
    const out = level[W.out], flap = level[W.flap];
    const roll = W.out_roll * Math.max(out, W.flap_roll * flap);
    const yaw = flap * (W.yaw_mid + W.yaw_amp * Math.sin(TAU * W.hz * t));
    [roll, yaw, roll, yaw].forEach((v, i) => (q[W.joints[i]] = v));
    // antennae (_antenna_angles): l pitch, l yaw, r pitch, r yaw; left and right in antiphase
    const up = level[A.up], wig = level[A.wiggle], s = Math.sin(TAU * A.hz * t);
    [1, -1].forEach((side, k) => {
      q[A.joints[2 * k]] = up * A.up_angles[0] + wig * (A.centre[0] - side * A.amp[0] * s);
      q[A.joints[2 * k + 1]] = up * A.up_angles[1] + wig * (A.centre[1] + side * A.amp[1] * s);
    });
  }

  function kinematics() {   // like MuJoCo's mj_kinematics: body = parent * (pos, quat), then each hinge in order
    for (let b = 0; b < nB; b++) {
      const p = parent[b], P = p < 0 ? world.pos : xpos, Q = p < 0 ? world.quat : xquat;
      const pp = p < 0 ? 0 : p * 3, pq = p < 0 ? 0 : p * 4;
      qrotAdd(xpos, b * 3, P, pp, Q, pq, bpos, b * 3);
      qmul(xquat, b * 4, Q, pq, bquat, b * 4);
      for (let j = jadr[b], e = j + jnum[b]; j < e; j++) {
        const h = q[j] / 2, sh = Math.sin(h);
        jq[0] = Math.cos(h); jq[1] = axes[j * 3] * sh; jq[2] = axes[j * 3 + 1] * sh; jq[3] = axes[j * 3 + 2] * sh;
        qmul(xquat, b * 4, xquat, b * 4, jq, 0);   // hinges pass through the body origin, so xpos stays
      }
      const k = b * 4, n = Math.hypot(xquat[k], xquat[k + 1], xquat[k + 2], xquat[k + 3]);
      for (let i = 0; i < 4; i++) xquat[k + i] /= n;
    }
    const gq = [0, 0, 0, 0], gp = [0, 0, 0];
    for (let g = 0; g < nG; g++) {
      const b = gbody[g];
      qrotAdd(gp, 0, xpos, b * 3, xquat, b * 4, gpos, g * 3);
      qmul(gq, 0, xquat, b * 4, gquat, g * 4);
      pose.set(gp, g * 7); pose.set(gq, g * 7 + 3);
    }
  }

  return {
    actions: d.actions.map((a) => a.name),
    geomNames: d.geom_names,

    setLevels(levels) {
      for (let a = 0; a < nA; a++) {
        const v = +(levels?.[a] ?? 0);
        command[a] = Number.isFinite(v) ? Math.min(1, Math.max(0, v)) : 0;
      }
      command[d.rest_action] = 0;   // like /api/fly/command: "rest" is the absence of every other action
    },

    update(tSeconds) {
      const dt = lastT === null ? 0 : Math.max(0, tSeconds - lastT);
      lastT = tSeconds;
      // ease toward the command like body.py (exactly its per-frame step at its frame rate)
      const f = 1 - Math.exp(-d.ease_rate * dt);
      let changed = first, settled = true;
      for (let a = 0; a < nA; a++) {
        let l = level[a] + (command[a] - level[a]) * f;
        if (Math.abs(command[a] - l) < 1e-3) l = command[a];   // close enough: stop easing
        else settled = false;
        if (l !== level[a]) changed = true;
        level[a] = l;
      }
      const periodicOn = periodic.some((a) => level[a] > 1e-3);
      const moving = changed || periodicOn;
      if (moving) { jointAngles(tSeconds); kinematics(); }
      first = false;
      levelsOut.set(level);
      return { pose, levels: levelsOut, moving, settled: settled && !periodicOn };
    },
  };
}
