"""Compare the browser's pose blending (flymnist/static/flypose.js, run in node) with the server's physics.

    uv run python tests/compare_flypose.py [--json flymnist/static/fly-poses.json]

Both get the same action levels at the same frame times (the server's 25 fps). For every scenario it
prints the error of every body part's position (mm; the fly is ~2.5 mm long) and orientation (degrees),
over all frames and in the final settled pose.
"""

import argparse
import base64
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flymnist.body import ACTIONS, BodyPool  # noqa: E402

NODE = r"""
import { createPoser } from %s;
import { readFileSync, writeFileSync } from "node:fs";
const [jsonPath, scenPath, outPath] = process.argv.slice(1);
const scen = JSON.parse(readFileSync(scenPath, "utf8")), d = JSON.parse(readFileSync(jsonPath, "utf8"));
const out = [];
for (const s of scen) {
  const p = createPoser(d);
  p.update(s.t0);
  for (let k = 0; k < s.t.length; k++) {
    p.setLevels(s.cmd[k]);
    const { pose } = p.update(s.t[k]);
    out.push(...pose);
  }
}
writeFileSync(outPath, Buffer.from(new Float32Array(out).buffer));
"""


def scenarios(fps):
    n = len(ACTIONS)
    sec = lambda s: int(round(s * fps))

    def hold(parts):   # [(levels dict, seconds)]
        cmd = []
        for lv, s in parts:
            v = [0.0] * n
            for a, x in lv.items():
                v[a] = x
            cmd += [v] * sec(s)
        return cmd

    out = [(f"{a}: {ACTIONS[a][0]}", hold([({}, 0.4), ({a: 1}, 2.0), ({}, 1.2)])) for a in range(1, n)]
    out += [
        ("combo: right front + both front (same joints)", hold([({}, 0.2), ({1: 1, 3: 1}, 2.0), ({}, 1.2)])),
        ("combo: both front + wings out + antennae up", hold([({}, 0.2), ({3: 1, 4: 1, 8: 1}, 2.0), ({}, 1.2)])),
        ("combo: flap + wiggle + middle legs", hold([({}, 0.2), ({5: 1, 6: 1, 9: 1}, 2.0), ({}, 1.2)])),
        ("partial: middle .5, hind .7, wings out .3, ant up .6",
         hold([({}, 0.2), ({6: .5, 7: .7, 4: .3, 8: .6}, 2.0), ({}, 1.2)])),
    ]
    rng = np.random.default_rng(1)   # like live decoding: a new set of levels every ~0.3 s
    parts = [({int(a): float(rng.uniform(0.3, 1)) for a in rng.choice(np.arange(1, n), rng.integers(1, 4), replace=False)},
              0.32) for _ in range(20)] + [({}, 1.2)]
    out.append(("live-like random levels", hold(parts)))
    return out


def server_run(pool, sid, cmds):
    b = pool.body(sid)   # created and advanced once, like on the first /api/fly/stream
    t0, ts, poses = b.t, [], []
    for c in cmds:
        if not np.array_equal(c, b.command):
            pool.set_levels(sid, c)
        pool._advance(b)
        ts.append(b.t)
        poses.append(np.frombuffer(base64.b64decode(json.loads(b.pose)["p"]), np.float32).reshape(-1, 7))
    return t0, ts, np.array(poses)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(ROOT / "flymnist/static/fly-poses.json"))
    ap.add_argument("--tmp", default=None, help="folder for the scratch files (default: a temp folder)")
    a = ap.parse_args()
    import tempfile
    tmp = Path(a.tmp or tempfile.mkdtemp())
    tmp.mkdir(parents=True, exist_ok=True)

    pool = BodyPool()
    scen = scenarios(pool.fps)
    runs = []
    for i, (name, cmds) in enumerate(scen):
        t0, ts, poses = server_run(pool, f"cmp{i}", cmds)
        runs.append({"name": name, "t0": t0, "t": ts, "cmd": cmds, "server": poses})
    (tmp / "scen.json").write_text(json.dumps([{k: r[k] for k in ("t0", "t", "cmd")} for r in runs]))
    js = json.dumps((ROOT / "flymnist/static/flypose.js").as_uri())
    subprocess.run(["node", "--input-type=module", "-e", NODE % js, str(a.json), str(tmp / "scen.json"),
                    str(tmp / "poser.bin")], check=True)
    poser = np.fromfile(tmp / "poser.bin", np.float32).reshape(-1, pool.geoms.size, 7)

    print(f"{'scenario':55s} {'pos err mm: max':>15s} {'mean':>8s} {'settled max':>11s} {'rot deg: max':>12s} {'mean':>6s}")
    k, worst = 0, 0.0
    for r in runs:
        s, p = r["server"], poser[k:k + len(r["t"])]
        k += len(r["t"])
        dp = np.linalg.norm(s[..., :3] - p[..., :3], axis=-1)
        dot = np.clip(np.abs((s[..., 3:] * p[..., 3:]).sum(-1)), 0, 1)
        dr = np.degrees(2 * np.arccos(dot))
        rest_tail = next(i for i, c in enumerate(reversed(r["cmd"])) if any(c))   # frames back at rest at the end
        on_end = len(r["cmd"]) - rest_tail - 1                                       # last frame of the last hold
        settled = max(dp[on_end].max(), dp[-1].max())   # end of the "on" hold, and back at rest
        worst = max(worst, dp.max())
        print(f"{r['name']:55s} {dp.max():15.4f} {dp.mean():8.5f} {settled:11.5f} {dr.max():12.3f} {dr.mean():6.3f}")
    print(f"worst position error: {worst:.4f} mm")


if __name__ == "__main__":
    main()
