"""A pre-trained gesture model from HaGRIDv2 photos (alternative class model; not used by the app).

    uv run main.py pretrain                 # all steps below; each one resumes / skips if already done
    uv run main.py pretrain --step download # ~1.7 GB of HaGRIDv2 parquet shards -> data/datasets/hagrid/
    uv run main.py pretrain --step pose     # MediaPipe Pose on every photo (own uv env, Python 3.11)
    uv run main.py pretrain --step build    # crop + label + augment -> 36x48 eye images
    uv run main.py pretrain --step simulate # every image through the brain (~1 h on CPU, checkpointed)
    uv run main.py pretrain --step train    # decoders + evaluation on the teacher's recordings

Gestures -> fly actions (indices into camera.CLASSES): the person's RIGHT hand up -> right front
leg up, LEFT hand up -> left front leg up, both hands up -> both front legs up, hand heart -> flap
wings, anything else (hands down, other gestures) -> rest. Labels come from the person's own
anatomy (MediaPipe Pose); the brain sees the unmirrored camera frame, so the person's right hand is
on the image's LEFT. "Both hands up" is mostly synthesised by mirroring one-hand-up photos.

Output: data/pretrained_gestures.joblib (same format as data/camera_session.joblib) and
data/pretrained_gestures.report.json. It CONTAINS DERIVED HaGRIDv2 IMAGES (CC BY-SA 4.0 style
license with research / non-commercial terms, see LICENSE below): never publish it (e.g. on the
static site). The teacher's recordings (data/camera_session.joblib) are only read, for evaluation.
"""

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

from .data import DATA_DIR

HAGRID_DIR = DATA_DIR / "datasets" / "hagrid"
WORK_DIR = DATA_DIR / "pretrain"
OUT = DATA_DIR / "pretrained_gestures.joblib"
REPORT = DATA_DIR / "pretrained_gestures.report.json"

SOURCE = "HaGRIDv2 (Kapitanov et al., arXiv:2412.01508), 512 px copy: huggingface.co/datasets/testdummyvt/hagRIDv2_512px"
LICENSE = ("HaGRID license (github.com/hukenovs/hagrid/blob/master/license/en_us.pdf): Creative Commons "
           "Attribution-ShareAlike 4.0; the authors describe it for research / non-commercial use. Derived images "
           "must keep attribution + share-alike and must not be published with the app.")
_URL = ("https://huggingface.co/datasets/testdummyvt/hagRIDv2_512px/resolve/"
        "6b846043ac83cbee666cd27492509e9317bb20ee/data/train/train-{:05d}-of-01503.parquet")

# HaGRIDv2 label ids (the parquet's class_label names) and the train shards we use. The train split
# is sorted by label, 547 photos per shard; shards are spread over each label's range (different people).
HAGRID_LABELS = {4: "grabbing", 6: "hand_heart", 7: "hand_heart2", 8: "holy", 9: "like", 13: "no_gesture",
                 15: "one", 16: "palm", 17: "peace", 21: "stop", 23: "take_picture", 30: "timeout", 33: "xsign"}
SHARDS = {
    "hand_heart": [272, 283, 294, 304], "hand_heart2": [313, 324, 336, 347],
    "no_gesture": [603, 604, 605, 606],              # the whole (small) no_gesture class
    "holy": [365, 390], "timeout": [1345], "xsign": [1470], "take_picture": [1020],  # two hands at the chest
    "palm": [700, 713, 726], "stop": [925, 945], "one": [670], "peace": [755], "like": [430],  # often a hand up
}


# HaGRID's test split has other people than its train split (split by user_id): for cross-person tests.
HAGRID_TEST_DIR = DATA_DIR / "datasets" / "hagrid_test"
_URL_TEST = _URL.replace("data/train/train-{:05d}-of-01503", "data/test/test-{:05d}-of-00301")
SHARDS_TEST = {"hand_heart": [57], "hand_heart2": [69], "holy": [75], "no_gesture": [118, 119], "palm": [138, 144],
               "stop": [186], "one": [132], "timeout": [270], "xsign": [297], "like": [87], "peace": [150]}


def download(split="train"):
    folder, url, shards = ((HAGRID_DIR, _URL, SHARDS) if split == "train" else (HAGRID_TEST_DIR, _URL_TEST, SHARDS_TEST))
    folder.mkdir(parents=True, exist_ok=True)
    shards = sorted({s for v in shards.values() for s in v})
    for k, s in enumerate(shards):
        path = folder / f"{split}-{s:05d}.parquet"
        if path.exists():
            continue
        t = time.time()
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(url.format(s), tmp)
        tmp.replace(path)
        print(f"  {split} shard {k + 1}/{len(shards)}: {path.name} {path.stat().st_size / 1e6:.0f} MB, {time.time() - t:.0f}s", flush=True)
    (folder / "SOURCE.txt").write_text(f"{SOURCE}\n\n{LICENSE}\n")


def pose(folder=HAGRID_DIR):
    """MediaPipe Pose in its own uv environment (MediaPipe has no Python 3.13 wheels)."""
    script = Path(__file__).with_name("pretrain_pose.py")
    subprocess.run(["uv", "run", "--no-project", "--python", "3.11", "--with", "mediapipe==0.10.14", "--with", "pyarrow",
                    "python", "-I", str(script), str(folder)], check=True)


# ---------------------------------------------------------------- labels from pose
REST, RIGHT, LEFT, BOTH, HEART = 0, 1, 2, 3, 5          # indices into camera.CLASSES / body.ACTIONS
GESTURES = [REST, RIGHT, LEFT, BOTH, HEART]
_HEART_IDS = {6, 7}
_CHEST_TWO_HANDS = {8, 23, 30, 33}                       # holy, take_picture, timeout, xsign: hard negatives for heart
NOSE, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 0, 11, 12, 13, 14, 15, 16   # MediaPipe Pose (L/R = the person's own side)


def load_poses(folder=HAGRID_DIR):
    """All photos with their HaGRID label and pose: dict of arrays (one row per photo)."""
    cols = {k: [] for k in ("shard", "row", "hagrid", "size", "lm")}
    for f in sorted(folder.glob("pose-*.npz")):
        z = np.load(f)
        n = len(z["label"])
        cols["shard"].append(np.full(n, int(f.stem.split("-")[1])))
        cols["row"].append(np.arange(n))
        cols["hagrid"].append(z["label"])
        cols["size"].append(z["size"])
        cols["lm"].append(z["landmarks"])
    return {k: np.concatenate(v) for k, v in cols.items()}


def hand_states(lm, margin=0.15):
    """Per photo: usable (frontal, big enough), left/right hand state 'U' up / 'D' down / '?' unclear.

    Up: wrist visible and above its own shoulder by margin x shoulder width. Down: wrist visible and
    below the shoulder, or wrist hidden with the elbow below the shoulder (hands out of frame)."""
    vis = lambda i: np.nan_to_num(lm[:, i, 3])
    x = lambda i: lm[:, i, 0]
    y = lambda i: lm[:, i, 1]
    sw = x(L_SH) - x(R_SH)               # facing the camera: the person's left shoulder is on the image right
    ok = (vis(L_SH) > .5) & (vis(R_SH) > .5) & (vis(NOSE) > .5) & np.nan_to_num(sw > 30)
    states = []
    for wr, el, sh in [(L_WR, L_EL, L_SH), (R_WR, R_EL, R_SH)]:
        up = (vis(wr) > .5) & (y(wr) < y(sh) - margin * sw)
        down = ((vis(wr) > .5) & (y(wr) > y(sh))) | ((vis(wr) <= .5) & (vis(el) > .3) & (y(el) > y(sh)))
        states.append(np.where(up, "U", np.where(down, "D", "?")))
    return ok, states[0], states[1], sw


def plan(per_class=2000, seed=0, folder=HAGRID_DIR):
    """Which photo becomes which training image: (photo index, class, flip, mirror) for every image.

    mirror: 0 none, +1 copy the image-right half onto the left, -1 the other way (around the shoulder midpoint)."""
    P = load_poses(folder)
    ok, ls, rs, sw = hand_states(P["lm"])
    hag = P["hagrid"]
    heart = np.isin(hag, list(_HEART_IDS))
    rng = np.random.default_rng(seed)
    left_up, right_up = ok & (ls == "U") & (rs == "D") & ~heart, ok & (rs == "U") & (ls == "D") & ~heart
    down = ok & (ls == "D") & (rs == "D")
    one_up = np.flatnonzero(left_up | right_up)
    items = []                                          # (photo, class, flip, mirror)
    pick = lambda pool, n: rng.permutation(pool)[:n] if len(pool) >= n else rng.choice(pool, n)

    # One hand up. The brain sees the unmirrored frame: the person's right hand is on the image left.
    # A horizontal flip turns a right-hand-up photo into a left-hand-up one.
    for cls, want_left in [(RIGHT, False), (LEFT, True)]:
        for i in pick(one_up, per_class):
            items.append((i, cls, bool(left_up[i]) != want_left, 0))
    # Both hands up: mirror the raised-hand half of a one-hand-up photo (plus the few real ones).
    both = np.flatnonzero(ok & (ls == "U") & (rs == "U") & ~heart)
    for i in both:
        items.append((i, BOTH, bool(rng.integers(2)), 0))
    for i in pick(one_up, per_class - len(both)):
        items.append((i, BOTH, bool(rng.integers(2)), 1 if left_up[i] else -1))   # left hand is on the image right
    # Hand heart (hands at the chest), some made symmetric so symmetry alone doesn't mean "both up".
    for i in pick(np.flatnonzero(down & heart), per_class):
        items.append((i, HEART, bool(rng.integers(2)), int(rng.choice([-1, 1])) if rng.random() < .15 else 0))
    # Rest: hands down, incl. two-hands-at-the-chest gestures (hard negatives for the heart) and no_gesture.
    chest = np.flatnonzero(down & np.isin(hag, list(_CHEST_TWO_HANDS)))
    other = np.flatnonzero(down & ~heart & ~np.isin(hag, list(_CHEST_TWO_HANDS)))
    n_chest = int(per_class * .4)
    for i in np.concatenate([pick(chest, n_chest), pick(other, per_class - n_chest)]):
        items.append((i, REST, bool(rng.integers(2)), int(rng.choice([-1, 1])) if rng.random() < .15 else 0))
    it = np.array(items, dtype=[("photo", int), ("cls", int), ("flip", bool), ("mirror", int)])
    return P, it


# ---------------------------------------------------------------- framing + the fly's eye view
# A framing draws, per image: crop width = k x shoulder width, shoulder line at fraction s of the crop
# height, horizontal offset ox (x crop width) of the shoulder midpoint, and a horizontal squash asp.
FRAMINGS = {
    # (a) like the teacher's laptop webcam: head and shoulders fill the frame, shoulders near the bottom.
    # Picked by a pixel-level pass (train on HaGRID, test on the teacher's frames) over k x s.
    "webcam": {"k": (2.45, 2.75), "s": (0.85, 0.95), "ox": (-0.06, 0.06), "asp": (0.95, 1.05)},
    # tight pose crop (needs live pose estimation in the browser)
    "pose": {"k": (3.0, 3.0), "s": (0.55, 0.55), "ox": (0.0, 0.0), "asp": (1.0, 1.0)},
}
CANVAS = (160, 120)       # the browser draws the camera frame into a 160x120 canvas, then preprocess_frame


def _render_shard(args):
    import io as _io

    import pyarrow.parquet as pq
    from PIL import Image

    from .camera import preprocess_frame

    shard, rows, lms, specs, out_raw = args
    table = pq.read_table(HAGRID_DIR / f"train-{shard:05d}.parquet", columns=["image"]).column("image")
    eye, raw = [], []
    for r, lm, (k, s, ox, asp, flip, mirror, seam) in zip(rows, lms, specs):
        g = np.asarray(Image.open(_io.BytesIO(table[int(r)]["bytes"].as_py())).convert("L"))
        sw = lm[L_SH, 0] - lm[R_SH, 0]
        cx, sy = (lm[L_SH, 0] + lm[R_SH, 0]) / 2, (lm[L_SH, 1] + lm[R_SH, 1]) / 2
        W = k * sw
        H = 0.75 * W
        W = W * asp
        x0, y0 = cx + ox * W - W / 2, sy - s * H
        pad = int(max(W, H)) + 2
        gp = np.pad(g, pad, mode="edge")              # outside the photo: repeat its edge, not black
        if mirror:
            c = int(round(cx + seam * sw)) + pad
            xs = np.arange(gp.shape[1])
            src = np.clip(2 * c - xs, 0, gp.shape[1] - 1)
            keep_right = mirror > 0
            gp = np.where((xs < c if keep_right else xs > c)[None, :], gp[:, src], gp)
        box = (x0 + pad, y0 + pad, x0 + pad + W, y0 + pad + H)
        im = Image.fromarray(gp).resize(CANVAS, Image.BILINEAR, box=box)
        if flip:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        buf = _io.BytesIO()
        im.save(buf, "PNG")
        eye.append(preprocess_frame(buf.getvalue()))
        if out_raw:
            raw.append(np.asarray(im))
    return np.stack(eye), (np.stack(raw) if out_raw else None)


def render(P, items, framing, seed=0, raw=False, workers=12):
    """Images as the fly's eye gets them, (n, 36, 48), in the order of items."""
    from multiprocessing import get_context

    rng = np.random.default_rng(seed + 1)
    f = FRAMINGS[framing] if isinstance(framing, str) else framing
    u = lambda key: rng.uniform(*f[key], len(items))
    specs = np.stack([u("k"), u("s"), u("ox"), u("asp"), items["flip"], items["mirror"],
                      rng.uniform(-0.06, 0.06, len(items))], 1)
    photo = items["photo"]
    shards = P["shard"][photo]
    jobs, where = [], []
    for sh in np.unique(shards):
        idx = np.flatnonzero(shards == sh)
        jobs.append((int(sh), P["row"][photo[idx]], P["lm"][photo[idx]], specs[idx], raw))
        where.append(idx)
    eye = np.zeros((len(items), 36, 48), np.float32)
    raws = np.zeros((len(items), CANVAS[1], CANVAS[0]), np.uint8) if raw else None
    with get_context("spawn").Pool(workers) as pool:
        for idx, (e, r) in zip(where, pool.imap(_render_shard, jobs)):
            eye[idx] = e
            if raw:
                raws[idx] = r
    return eye, raws, specs


# ---------------------------------------------------------------- the brain (same model as the class model)
CHUNK = 32                 # frames per brain batch / checkpoint file; chunk i always uses seed SEED + i
SEED = 0
_W = {}


def _brain_init(threads):
    import torch

    from .brain import FlyBrain
    from .data import load_annotations, load_connectome
    from .retina import build_retina

    torch.set_num_threads(threads)
    pre, post, weight, root_ids = load_connectome()
    input_idx, uv = build_retina(load_annotations(root_ids), "lamina")
    _W["brain"], _W["uv"], _W["n"] = FlyBrain(pre, post, weight, len(root_ids), input_idx), uv, len(root_ids)


def _brain_chunk(args):
    """Simulate one chunk (camera timing: 50 ms per frame at up to 500 Hz, lamina input) and save it."""
    import scipy.sparse as sp
    import torch

    from .camera import DURATION_MS
    from .retina import images_to_rates

    i, images, out, *rest = args
    rate = rest[0] if rest else 500.0
    gen = torch.Generator().manual_seed(SEED + i)
    t = time.time()
    c = _W["brain"].run(images_to_rates(images, _W["uv"], rate), DURATION_MS, generator=gen)
    m = sp.csr_matrix(c.clamp(max=255).numpy().astype(np.uint8))
    if out is not None:
        tmp = Path(str(out) + ".tmp.npz")
        sp.save_npz(tmp, m)
        tmp.replace(out)
    return i, time.time() - t, m


def simulate(images, counts_dir, procs=5, threads=2, log=lambda *a: print(*a, flush=True), rate=500.0):
    """Spike counts (sparse uint8, n x N) for images (n, 36, 48), checkpointed per chunk in counts_dir."""
    from multiprocessing import get_context

    import scipy.sparse as sp

    counts_dir = Path(counts_dir)
    counts_dir.mkdir(parents=True, exist_ok=True)
    n_chunks = (len(images) + CHUNK - 1) // CHUNK
    path = lambda i: counts_dir / f"chunk-{i:05d}.npz"
    todo = [i for i in range(n_chunks) if not path(i).exists()]
    log(f"brain: {len(images)} frames in {n_chunks} chunks, {len(todo)} to simulate on {procs} processes x {threads} threads")
    t0, done = time.time(), 0
    if todo:
        with get_context("spawn").Pool(procs, initializer=_brain_init, initargs=(threads,)) as pool:
            jobs = [(i, images[i * CHUNK:(i + 1) * CHUNK], path(i), rate) for i in todo]
            for i, dt, _ in pool.imap_unordered(_brain_chunk, jobs):
                done += 1
                el = time.time() - t0
                log(f"  chunk {i}: {dt:.0f}s; {done}/{len(todo)} done, {el / 60:.1f} min, ~{el / done * (len(todo) - done) / 60:.0f} min left")
    return sp.vstack([sp.load_npz(path(i)) for i in range(n_chunks)]).tocsr()


def benchmark(images, configs=((1, 14), (2, 7), (3, 4), (4, 3), (5, 2), (7, 2)), chunks_per_proc=1):
    """Frames per second of the brain for (processes, threads) setups (no checkpoints written)."""
    from multiprocessing import get_context

    res = {}
    for procs, threads in configs:
        with get_context("spawn").Pool(procs, initializer=_brain_init, initargs=(threads,)) as pool:
            pool.map(_brain_chunk, [(0, images[:2], None)] * procs)        # warm-up (loading the connectome)
            jobs = [(k, images[k * CHUNK:(k + 1) * CHUNK], None) for k in range(procs * chunks_per_proc)]
            t = time.time()
            pool.map(_brain_chunk, jobs, chunksize=1)
            el = time.time() - t
        res[f"{procs}x{threads}"] = len(jobs) * CHUNK / el
        print(f"  {procs} processes x {threads} threads: {res[f'{procs}x{threads}']:.2f} frames/s", flush=True)
    return res


# ---------------------------------------------------------------- steps
def build(framing="webcam", per_class=2000, seed=0):
    """Training images for one framing -> WORK_DIR/<framing>/images.npz (+ contact sheet of samples)."""
    out = WORK_DIR / framing / "images.npz"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    P, it = plan(per_class, seed)
    eye, _, specs = render(P, it, framing, seed)
    ph = it["photo"]
    np.savez_compressed(out, images=eye, labels=it["cls"], flip=it["flip"], mirror=it["mirror"], specs=specs,
                        shard=P["shard"][ph], row=P["row"][ph], hagrid=P["hagrid"][ph])
    print(f"  {framing}: {len(eye)} images, per class {np.bincount(it['cls'], minlength=10)[GESTURES].tolist()}", flush=True)
    return out


def contact_sheet(framing, path, per_class=10, seed=3):
    """Sample training images per class: the camera picture (160x120 canvas) and what the fly's eye gets."""
    from PIL import Image, ImageDraw

    from .camera import CLASSES

    P, it = plan()
    rng = np.random.default_rng(seed)
    sel = np.concatenate([rng.choice(np.flatnonzero(it["cls"] == c), per_class, replace=False) for c in GESTURES])
    eye, raw, _ = render(P, it[sel], framing, seed, raw=True)
    tw, th, lab = 120, 90, 150
    sheet = Image.new("L", (lab + per_class * (2 * tw + 8), len(GESTURES) * (th + 6)), 255)
    d = ImageDraw.Draw(sheet)
    for r, c in enumerate(GESTURES):
        d.text((4, r * (th + 6) + th // 2), CLASSES[c], fill=0)
        for j in range(per_class):
            i = r * per_class + j
            x, y = lab + j * (2 * tw + 8), r * (th + 6)
            sheet.paste(Image.fromarray(raw[i]).resize((tw, th)), (x, y))
            sheet.paste(Image.fromarray((eye[i] * 255).astype(np.uint8)).resize((tw, th), Image.NEAREST), (x + tw, y))
    sheet.save(path)


def brain_masks():
    """The neuron groups the server's decoders read (same as server.create_app)."""
    from .data import load_annotations, load_connectome
    from .retina import build_retina

    _, _, _, root_ids = load_connectome()
    ann = load_annotations(root_ids)
    input_idx, _ = build_retina(ann, "lamina")
    not_input = np.ones(len(root_ids), bool)
    not_input[input_idx] = False
    return {"descending": (ann.super_class == "descending").to_numpy() & not_input,
            "central brain": (ann.super_class == "central").to_numpy() & not_input,
            "whole brain": not_input}


def teacher_data():
    """The teacher's class recordings (read only): images, labels, counts, recording session per frame.

    Sessions = the three places / times the teacher recorded (each recorded every action once, in
    order, starting with rest); the short last block of 'right front leg up' is a fourth session."""
    import joblib
    import scipy.sparse as sp

    s = joblib.load(DATA_DIR / "camera_session.joblib")
    y = np.asarray(s["labels"], int)
    starts = np.r_[0, np.flatnonzero(np.diff(y)) + 1]
    ends = np.r_[starts[1:], len(y)]
    session, g, seen = np.zeros(len(y), int), -1, set()
    for a, b in zip(starts, ends):            # a new session starts with "rest" or with an action already recorded
        if y[a] == REST or y[a] in seen or g < 0:
            g, seen = g + 1, set()
        seen.add(y[a])
        session[a:b] = g
    return np.stack(s["images"]).astype(np.float32), y, sp.csr_matrix(s["counts"]), session


def simulate_step(framings=("webcam", "pose"), procs=5, threads=2):
    """Brain spike counts for every built framing, checkpointed under WORK_DIR/<framing>/counts/."""
    import scipy.sparse as sp

    for fr in framings:
        out = WORK_DIR / fr / "counts.npz"
        if out.exists():
            continue
        images = np.load(WORK_DIR / fr / "images.npz")["images"]
        t = time.time()
        counts = simulate(images, WORK_DIR / fr / "counts", procs, threads)
        sp.save_npz(out.with_suffix(".tmp.npz"), counts)
        out.with_suffix(".tmp.npz").replace(out)
        print(f"  {fr}: {len(images)} frames in {(time.time() - t) / 60:.1f} min "
              f"({len(images) / (time.time() - t):.1f} frames/s, {procs} processes x {threads} threads)", flush=True)


# ---------------------------------------------------------------- evaluation + the final model
FEATURES = ["descending", "central brain", "whole brain", "pixels"]


def _features(name, images, counts, cols):
    from .readout import spike_features

    return images.reshape(len(images), -1) if name == "pixels" else spike_features(counts, cols)


def _fit_eval(train, test, masks, smooth=5):
    """Fit every decoder on train (images, labels, counts) the way camera.train_decoders does, test on test."""
    import scipy.sparse as sp
    from sklearn.metrics import confusion_matrix

    from .camera import _make_model

    (Xi, y, C), (Ti, ty, TC, runs) = train, test
    active = np.asarray(sp.csr_matrix(C).sum(0)).ravel() > 0
    out = {}
    for name in FEATURES:
        cols = None if name == "pixels" else np.flatnonzero(masks[name] & active)
        m = _make_model().fit(_features(name, Xi, C, cols), y)
        proba = m.predict_proba(_features(name, Ti, TC, cols))
        pred = m.classes_[proba.argmax(1)]
        # smoothed: mean probability over the last `smooth` frames of the same recording (as a live demo could)
        sm = np.empty_like(proba)
        for r in np.unique(runs):
            idx = np.flatnonzero(runs == r)
            cs = np.cumsum(proba[idx], 0)
            lo = np.maximum(np.arange(len(idx)) - smooth + 1, 0)
            sm[idx] = (cs - np.where(lo[:, None] > 0, cs[lo - 1], 0)) / (np.arange(len(idx)) - lo + 1)[:, None]
        spred = m.classes_[sm.argmax(1)]
        g = np.isin(ty, GESTURES)
        out[name] = {
            "accuracy": float((pred[g] == ty[g]).mean()),
            "accuracy_smoothed": float((spred[g] == ty[g]).mean()),
            "per_class": {int(c): float((pred[ty == c] == c).mean()) for c in GESTURES if (ty == c).any()},
            "confusion": confusion_matrix(ty[g], pred[g], labels=GESTURES).tolist(),
            "other_gesture_as_rest": float((pred[~g] == REST).mean()) if (~g).any() else None,
        }
    return out


def _runs(y):
    return np.cumsum(np.r_[0, np.diff(y) != 0])


def evaluate(framing, masks, teacher, log=lambda *a: print(*a, flush=True)):
    """(a)/(b): train on HaGRID only, test on the teacher's frames; in-domain: grouped CV on HaGRID.
    (c): HaGRID + the teacher's other sessions -> each held-out session, vs. the teacher's data alone."""
    import scipy.sparse as sp
    from sklearn.model_selection import GroupKFold

    from .camera import _make_model

    z = np.load(WORK_DIR / framing / "images.npz")
    X, y, groups = z["images"], z["labels"], z["shard"]
    C = sp.load_npz(WORK_DIR / framing / "counts.npz")
    TX, ty, TC, sess = teacher
    sess = np.minimum(sess, 2)                     # the short 4th block joins the 3rd session for testing
    keep = np.isin(ty, GESTURES + [9])             # 9 = wiggle antennae: "anything else", should be rest
    TX, ty, TC, sess = TX[keep], ty[keep], TC[keep], sess[keep]
    res = {"teacher_frames": _fit_eval((X, y, C), (TX, ty, TC, _runs(ty)), masks)}
    log(f"  [{framing}] HaGRID only -> teacher's frames: " +
        ", ".join(f"{k} {v['accuracy']:.3f}/{v['accuracy_smoothed']:.3f}" for k, v in res["teacher_frames"].items()))

    active = np.asarray(C.sum(0)).ravel() > 0
    cv = {}
    for name in ["whole brain", "pixels"]:
        cols = None if name == "pixels" else np.flatnonzero(masks[name] & active)
        F = _features(name, X, C, cols)
        cv[name] = float(np.mean([(_make_model().fit(F[tr], y[tr]).predict(F[te]) == y[te]).mean()
                                  for tr, te in GroupKFold(4).split(F, y, groups)]))
    res["hagrid_grouped_cv"] = cv
    log(f"  [{framing}] HaGRID grouped CV (photos of other people): {cv}")

    if framing == "webcam":
        mixed, alone = {}, {}
        for s in np.unique(sess):
            tr, te = sess != s, sess == s
            test = (TX[te], ty[te], TC[te], _runs(ty[te]))
            trm = tr & np.isin(ty, GESTURES)
            mixed[int(s)] = _fit_eval((np.concatenate([X, TX[trm]]), np.concatenate([y, ty[trm]]),
                                       sp.vstack([C, TC[trm]]).tocsr()), test, masks)
            alone[int(s)] = _fit_eval((TX[trm], ty[trm], TC[trm]), test, masks)
        avg = lambda d, k: {n: float(np.mean([d[s][n][k] for s in d])) for n in FEATURES}
        res["c_hagrid_plus_teacher_loso"] = {"per_session": mixed, "mean": avg(mixed, "accuracy"),
                                             "mean_smoothed": avg(mixed, "accuracy_smoothed")}
        res["teacher_only_loso"] = {"per_session": alone, "mean": avg(alone, "accuracy"),
                                    "mean_smoothed": avg(alone, "accuracy_smoothed")}
        log(f"  [c] HaGRID + teacher, held-out session: {res['c_hagrid_plus_teacher_loso']['mean']}")
        log(f"      teacher only, held-out session:     {res['teacher_only_loso']['mean']}")
    return res


def framing_search(per_class=1200, log=lambda *a: print(*a, flush=True)):
    """Quick pixel-level pass to choose the framing: decoders on pixels, HaGRID -> the teacher's frames."""
    from .camera import _make_model

    TX, ty, _, _ = teacher_data()
    g = np.isin(ty, GESTURES)
    TX, ty = TX[g].reshape(g.sum(), -1), ty[g]
    P, it = plan(per_class)
    res = {}
    for k in [1.8, 2.2, 2.6, 3.0, 3.5]:
        for s in [0.6, 0.75, 0.9]:
            fr = {"k": (k - .15, k + .15), "s": (s - .05, s + .05), "ox": (-.06, .06), "asp": (.95, 1.05)}
            X = render(P, it, fr)[0].reshape(len(it), -1)
            res[f"k={k} s={s}"] = acc = float((_make_model().fit(X, it["cls"]).predict(TX) == ty).mean())
            log(f"  crop width {k} x shoulders, shoulders at {s:.0%} height: teacher's frames {acc:.3f}")
    return res


def finish(framing="webcam", include_teacher=False, results=None):
    """Train the decoders exactly like the server (camera.train_decoders) and save the class-model file."""
    import joblib
    import scipy.sparse as sp

    from .camera import CLASSES, export_decoder, train_decoders

    z = np.load(WORK_DIR / framing / "images.npz")
    images, labels = list(z["images"]), z["labels"].tolist()
    counts = sp.load_npz(WORK_DIR / framing / "counts.npz").astype(np.uint8)
    n_hagrid = len(images)
    if include_teacher:
        TX, ty, TC, _ = teacher_data()
        g = np.isin(ty, GESTURES)
        images += list(TX[g]); labels += ty[g].tolist(); counts = sp.vstack([counts, TC[g]])
    masks = brain_masks()
    # like the browser's recordings: no counts for the eye-input neurons (decoders never read them)
    counts = sp.csr_matrix(counts.multiply(masks["whole brain"][None, :].astype(np.uint8)), dtype=np.uint8)
    counts.eliminate_zeros()
    models, metrics = train_decoders(images, labels, counts, masks)
    for v in models.values():
        export_decoder(v)                            # what the browser gets: make sure it exports
    meta = {"contains_third_party_images": "HaGRIDv2", "do_not_publish": True, "source": SOURCE, "license": LICENSE,
            "framing": FRAMINGS[framing] | {"name": framing}, "hagrid_frames": n_hagrid,
            "teacher_frames": len(images) - n_hagrid, "classes": {c: CLASSES[c] for c in GESTURES},
            "built": time.strftime("%Y-%m-%d %H:%M"), "brain": "lamina input, 50 ms, 500 Hz, torch FlyBrain, seed 0 + chunk"}
    tmp = OUT.with_suffix(".tmp")
    joblib.dump({"images": images, "labels": labels, "counts": counts, "models": models, "metrics": metrics,
                 "version": 1, "trained_on": None, "include_class": True, "meta": meta}, tmp, compress=3)
    tmp.replace(OUT)
    report = {"warning": "Contains images derived from HaGRIDv2 (third-party, share-alike, research/non-commercial). "
                         "Never publish data/pretrained_gestures.joblib (e.g. on the static site).",
              **meta, "frames_per_class": {CLASSES[c]: int(np.sum(np.asarray(labels) == c)) for c in GESTURES},
              "train_decoders_cv": metrics, **(results or {})}
    REPORT.write_text(json.dumps(report, indent=1))
    print(f"saved {OUT} ({OUT.stat().st_size / 1e6:.0f} MB) and {REPORT}", flush=True)


def skeleton_test(per_class=1200):
    """What-if (not used by the app): the fly sees a drawn MediaPipe skeleton (pose crop) instead of the
    camera image. Pixel decoder, cross-person grouped CV on HaGRID. Note L/R/both labels come from the same
    pose, so only heart vs rest is a fair test here."""
    import io as _io

    from PIL import Image, ImageDraw
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import GroupKFold, cross_val_predict

    from .camera import _make_model, preprocess_frame

    edges = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (15, 17), (15, 19), (15, 21), (16, 18), (16, 20),
             (16, 22), (11, 23), (12, 24)]
    P, it = plan(per_class)
    X = []
    for ph, _, flip, mirror in it:
        lm = P["lm"][ph].copy()
        sw, cx, sy = lm[L_SH, 0] - lm[R_SH, 0], (lm[L_SH, 0] + lm[R_SH, 0]) / 2, (lm[L_SH, 1] + lm[R_SH, 1]) / 2
        if mirror:
            for a in ([11, 13, 15, 17, 19, 21, 23] if mirror > 0 else [12, 14, 16, 18, 20, 22, 24]):
                b = a + 1 if a % 2 else a - 1
                lm[b] = lm[a]
                lm[b, 0] = 2 * cx - lm[a, 0]
        W = 3 * sw
        f = lambda i: ((lm[i, 0] - cx) / W * 160 + 80, (lm[i, 1] - sy) / (.75 * W) * 120 + .55 * 120)
        im = Image.new("L", CANVAS, 0)
        d = ImageDraw.Draw(im)
        for a, b in edges:
            d.line([f(a), f(b)], fill=255, width=4)
        hx, hy = f(NOSE)
        r = .3 * sw / W * 160
        d.ellipse([hx - r, hy - r, hx + r, hy + r], outline=255, width=4)
        if flip:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        buf = _io.BytesIO()
        im.save(buf, "PNG")
        X.append(preprocess_frame(buf.getvalue()).ravel())
    y = it["cls"]
    pred = cross_val_predict(_make_model(), np.array(X), y, groups=P["shard"][it["photo"]], cv=GroupKFold(4))
    return {"hagrid_grouped_cv": float((pred == y).mean()),
            "per_class": {int(c): float((pred[y == c] == c).mean()) for c in GESTURES},
            "confusion": confusion_matrix(y, pred, labels=GESTURES).tolist()}


def _cached(name, fn):
    path = WORK_DIR / f"{name}.json"
    if not path.exists():
        path.write_text(json.dumps(fn(), indent=1))
    return json.loads(path.read_text())


def run(step="all", include_teacher=False):
    """include_teacher: also train on the teacher's recordings, option (c). Off by default: on held-out
    sessions it was no better than (a) or than the teacher's data alone, and the teacher's "flap wings"
    recordings are not a hand heart."""
    steps = ["download", "pose", "build", "simulate", "train"] if step == "all" else [step]
    if "download" in steps:
        download()
        download("test")
    if "pose" in steps:
        pose()
        pose(HAGRID_TEST_DIR)
    if "build" in steps:
        for fr in FRAMINGS:
            build(fr)
        contact_sheet("webcam", WORK_DIR / "webcam" / "samples.png")
    if "simulate" in steps:
        bench = WORK_DIR / "benchmark.json"
        if not bench.exists():   # pick processes x threads by speed (torch scales poorly across threads)
            images = np.load(WORK_DIR / "webcam" / "images.npz")["images"]
            bench.write_text(json.dumps(benchmark(images, configs=[(1, os.cpu_count()), (5, 2), (os.cpu_count() // 2, 1)])))
        best = max(json.loads(bench.read_text()).items(), key=lambda kv: kv[1])[0]
        simulate_step(tuple(FRAMINGS), *map(int, best.split("x")))
    if "train" in steps:
        masks, teacher = brain_masks(), teacher_data()
        results = {
            "evaluation": {f"({'a' if fr == 'webcam' else 'b'}) {fr}": _cached(f"eval_{fr}", lambda: evaluate(fr, masks, teacher))
                           for fr in FRAMINGS},
            "framing_search_pixels_teacher_frames": _cached("framing_search", framing_search),
            "what_if_skeleton_input": _cached("skeleton_test", skeleton_test),
            "brain_speed_frames_per_s": json.loads((WORK_DIR / "benchmark.json").read_text()),
            "final_model": "(c) HaGRID webcam framing + teacher's recordings" if include_teacher else "(a) HaGRID webcam framing only",
        }
        finish("webcam", include_teacher, results)
