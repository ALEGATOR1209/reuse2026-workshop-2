"""MediaPipe Pose landmarks for every photo in the downloaded HaGRIDv2 shards (step 'pose' of pretrain.py).

Standalone (MediaPipe needs Python <= 3.12, the project uses 3.13); pretrain.py runs it as

    uv run --no-project --python 3.11 --with mediapipe==0.10.14 --with pyarrow python -I flymnist/pretrain_pose.py data/datasets/hagrid

Writes pose-<shard>.npz next to each <split>-<shard>.parquet: label (n,), size (n, 2) = (w, h) and
landmarks (n, 33, 4) = (x px, y px, z, visibility), NaN where no person was found.
"""

import io
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np


def run_shard(path):
    import mediapipe as mp
    import pyarrow.parquet as pq
    from PIL import Image

    out = path.with_name("pose-" + path.stem.split("-")[-1] + ".npz")      # train-00272.parquet -> pose-00272.npz
    if out.exists():
        return out.name, "exists"
    t = pq.read_table(path)
    labels = t.column("label").to_numpy()
    images = t.column("image").to_pylist()
    lm = np.full((len(images), 33, 4), np.nan, np.float32)
    size = np.zeros((len(images), 2), np.int32)
    with mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1) as pose:
        for i, rec in enumerate(images):
            im = Image.open(io.BytesIO(rec["bytes"])).convert("RGB")
            size[i] = im.size
            r = pose.process(np.asarray(im))
            if r.pose_landmarks:
                lm[i] = [(p.x * im.size[0], p.y * im.size[1], p.z, p.visibility) for p in r.pose_landmarks.landmark]
    np.savez_compressed(out.with_suffix(".tmp.npz"), label=labels, size=size, landmarks=lm)
    out.with_suffix(".tmp.npz").replace(out)
    return out.name, f"{len(images)} photos, {np.isfinite(lm[:, 0, 0]).sum()} with a person"


if __name__ == "__main__":
    shards = sorted(Path(sys.argv[1]).glob("*-*.parquet"))
    with Pool(int(sys.argv[2]) if len(sys.argv) > 2 else 8) as pool:
        for name, msg in pool.imap_unordered(run_shard, shards):
            print(f"  {name}: {msg}", flush=True)
