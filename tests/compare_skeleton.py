"""Check that the browser's stick figure (flymnist/static/skeleton.js, run in node) equals flymnist/skeleton.py.

    uv run python tests/compare_skeleton.py

Renders the same random upper-body poses (with random hands; plus edge cases: hidden hands, a zero-length segment,
a hidden shoulder, a zero-size hand) in both and prints the largest pixel difference. Needs node >= 18.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flymnist import skeleton  # noqa: E402

NODE = r"""
import { render } from %s;
import { readFileSync, writeFileSync } from "node:fs";
const L = JSON.parse(readFileSync(process.argv[2], "utf8"));
const out = new Float32Array(L.length * 36 * 48);
L.forEach(([lm, hands], k) => out.set(render(lm, hands), k * 1728));
writeFileSync(process.argv[3], Buffer.from(out.buffer));
"""


def poses(n=300, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        lm = np.zeros((33, 3))
        cx, cy, w = rng.uniform(100, 540), rng.uniform(100, 380), rng.uniform(40, 250)
        lm[:, 0], lm[:, 1] = cx + rng.normal(0, w, 33), cy + rng.normal(0, w, 33)
        lm[11, :2], lm[12, :2] = (cx + w / 2, cy), (cx - w / 2, cy + rng.normal(0, 5))
        lm[:, 2] = rng.uniform(0.2, 1.0, 33)
        lm[[0, 11, 12], 2] = 0.9
        hands = {}
        for side, wr in (("left", 15), ("right", 16)):          # random hands near the wrists (some missing)
            if rng.random() < 0.8:
                hands[side] = (lm[wr, :2] + rng.normal(0, w * 0.15, (21, 2))).tolist()
            else:
                hands[side] = None
        out.append((lm, hands))
    a = out[0][0].copy(); a[15:23, 2] = 0.1; out.append((a, None))           # hands hidden, no hand landmarks
    a = out[1][0].copy(); a[13] = a[15]; out.append((a, out[1][1]))         # zero-length forearm
    a = out[2][0].copy(); a[11, 2] = 0.1; out.append((a, out[2][1]))        # a shoulder hidden -> blank
    h = dict(out[3][1]); h["left"] = [h["left"][0]] * 21 if h["left"] else None; out.append((out[3][0], h))  # palm length 0
    return out


def main():
    L = poses()
    py = np.stack([skeleton.render(lm, hands) for lm, hands in L])
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "lm.json").write_text(json.dumps([[lm.tolist(), hands] for lm, hands in L]))
        (d / "run.mjs").write_text(NODE % json.dumps(str(ROOT / "flymnist/static/skeleton.js")))
        subprocess.run(["node", str(d / "run.mjs"), str(d / "lm.json"), str(d / "out.bin")], check=True)
        js = np.fromfile(d / "out.bin", np.float32).reshape(py.shape)
    diff = np.abs(py - js)
    print(f"{len(L)} poses: max |python - js| = {diff.max():.2e}, identical images: {(diff.max((1, 2)) == 0).sum()}")
    sys.exit(0 if diff.max() < 1e-6 else 1)


if __name__ == "__main__":
    main()
