"""MediaPipe Hand Landmarker (21 points per hand) for the downloaded HaGRIDv2 photos (after pretrain_pose.py).

Standalone (MediaPipe needs Python <= 3.12); pretrain.py runs it as

    uv run --no-project --python 3.11 --with mediapipe==0.10.14 --with pyarrow python -I flymnist/pretrain_hands.py \
        data/datasets/hagrid data/pretrain/models/hand_landmarker.task

Same model file as the browser (tracker.js). Per photo: the whole photo first; if fewer than 2 hands are
found and there is a person, again on a square crop around both pose wrists (upscaled), which finds small
hands of people far from the camera. Writes hands-<shard>.npz next to pose-<shard>.npz:
hands (n, 2, 21, 2) pixel x, y (NaN = no hand), score (n, 2) handedness score, crop (n,) bool.
Hands are NOT assigned to left / right here: skeleton.assign_hands does that (nearest pose wrist).
"""

import io
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

MODEL = None


def _found(res, x0, y0, sx, sy):
    out = []
    for lms, hd in zip(res.hand_landmarks, res.handedness):
        out.append((np.array([[x0 + p.x * sx, y0 + p.y * sy] for p in lms], np.float32), hd[0].score))
    return out


def run_shard(path):
    import mediapipe as mp
    import pyarrow.parquet as pq
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision
    from PIL import Image

    shard = path.stem.split("-")[-1]
    out = path.with_name(f"hands-{shard}.npz")
    if out.exists():
        return out.name, "exists"
    pose = np.load(path.with_name(f"pose-{shard}.npz"))["landmarks"]
    images = pq.read_table(path, columns=["image"]).column("image").to_pylist()
    hands = np.full((len(images), 2, 21, 2), np.nan, np.float32)
    score = np.zeros((len(images), 2), np.float32)
    crop = np.zeros(len(images), bool)
    opts = vision.HandLandmarkerOptions(base_options=mpt.BaseOptions(model_asset_path=MODEL), num_hands=2,
                                        running_mode=vision.RunningMode.IMAGE,
                                        min_hand_detection_confidence=0.3, min_hand_presence_confidence=0.3)
    with vision.HandLandmarker.create_from_options(opts) as det:
        for i, rec in enumerate(images):
            im = Image.open(io.BytesIO(rec["bytes"])).convert("RGB")
            w, h = im.size
            found = _found(det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(im))), 0, 0, w, h)
            lm = pose[i]
            if len(found) < 2 and np.isfinite(lm[0, 0]):
                sw = float(np.hypot(*(lm[11, :2] - lm[12, :2])))
                wr = lm[[15, 16], :2]
                c = wr.mean(0)
                half = max(float(np.abs(wr[0] - wr[1]).max()) / 2 + 0.9 * sw, 0.6 * sw, 16.0)
                box = (int(c[0] - half), int(c[1] - half), int(c[0] + half), int(c[1] + half))
                sub = im.crop(box).resize((384, 384), Image.BILINEAR)
                s = (box[2] - box[0]) / 384
                again = _found(det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(sub))), box[0], box[1],
                               (box[2] - box[0]), (box[3] - box[1]))
                if len(again) > len(found):
                    found, crop[i] = again, True
            for k, (pts, sc) in enumerate(found[:2]):
                hands[i, k], score[i, k] = pts, sc
    np.savez_compressed(out.with_suffix(".tmp.npz"), hands=hands, score=score, crop=crop)
    out.with_suffix(".tmp.npz").replace(out)
    n = np.isfinite(hands[:, :, 0, 0]).sum(1)
    return out.name, f"{len(images)} photos: 2 hands {np.sum(n == 2)}, 1 hand {np.sum(n == 1)}, none {np.sum(n == 0)}, via crop {crop.sum()}"


def _init(model):
    global MODEL
    MODEL = model


if __name__ == "__main__":
    shards = sorted(p for p in Path(sys.argv[1]).glob("*-*.parquet"))
    with Pool(int(sys.argv[3]) if len(sys.argv) > 3 else 10, initializer=_init, initargs=(sys.argv[2],)) as pool:
        for name, msg in pool.imap_unordered(run_shard, shards):
            print(f"  {name}: {msg}", flush=True)
