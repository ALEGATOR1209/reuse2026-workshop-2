"""Pre-trained gesture model on stick figures (flymnist/skeleton.py, spec 2: body + magnified hands).

    uv run main.py pretrain --step download     # HaGRIDv2 shards (train + test split), then:
    uv run main.py pretrain --step pose         # MediaPipe Pose on every photo, then:
    uv run main.py pretrain --step skeleton     # hand landmarks (first time), experiments, brain, model (~1.5 h)

Landmarks: MediaPipe Pose (body) + Hand Landmarker (21 points per hand) of HaGRIDv2 photos; each hand is given to
the person's left / right by the nearest pose wrist (skeleton.assign_hands, same rule as the browser).

Train (HaGRID train-split people), 2,000 frames per gesture:
  right / left hand up: one-hand-up photos (a flip turns one into the other),
  both hands up: mostly synthetic (one arm + hand mirrored onto the other side),
  hand heart: both HaGRID heart classes, half of them moved up to face height (wrists + hands raised,
    elbows half as much), 15% mirror-symmetrised,
  rest: hands down (no_gesture and one-hand gestures at the chest), 100 blank frames (nobody in view) and,
    40% of it, the two-hands-at-the-chest gestures (holy, timeout, xsign, take_picture): the drawn hands
    tell them from the heart (REST_CHEST; with / without compared in the report),
  every frame: small rotation + landmark noise.
Test: HaGRID test-split people (HaGRID splits by user_id), real landmarks, no augmentation; extra test
groups: two-hands-at-the-chest gestures (should be rest), real face-height hearts, synthetic face-height
hearts, synthetic both-up. Few-shot: a new gesture recorded by ONE person (30 jittered frames of one photo)
added to the pretrained data, tested on other people. Every image goes through the brain (lamina, 50 ms,
skeleton.RATE_HZ).

Output: data/pretrained_skeleton.joblib (class-model format + "landmarks" to re-render; only stick figures,
counts and landmarks, no photos) and data/pretrained_skeleton.report.json. Derived from HaGRIDv2 landmarks:
if the decoders are ever published, credit HaGRIDv2 (CC BY-SA 4.0). The spec-1 model is kept as
data/pretrained_skeleton_v1.joblib.
"""

import json
import time

import numpy as np
import scipy.sparse as sp

from . import pretrain as pt
from . import skeleton as sk
from .data import DATA_DIR

WORK = pt.WORK_DIR / "skeleton2"
OUT = DATA_DIR / "pretrained_skeleton.joblib"
REPORT = DATA_DIR / "pretrained_skeleton.report.json"
HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
HAND_MODEL = pt.WORK_DIR / "models" / "hand_landmarker.task"
REST_CHEST = True                          # put holy / timeout / xsign / take_picture into rest?
_LEFT = [11, 13, 15, 17, 19, 21]           # the person's left arm + hand points (MediaPipe Pose); right = index + 1
LIKE, PEACE, XSIGN = 9, 17, 33             # HaGRID ids of the few-shot gestures
NEW = 9                                    # class index the few-shot gesture is recorded as ("wiggle antennae" slot)


def hands_step():
    """Hand Landmarker on all downloaded photos (own uv environment, like pretrain.pose)."""
    import subprocess
    import urllib.request
    from pathlib import Path

    if not HAND_MODEL.exists():
        HAND_MODEL.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL)
    script = Path(__file__).with_name("pretrain_hands.py")
    for folder in (pt.HAGRID_DIR, pt.HAGRID_TEST_DIR):
        subprocess.run(["uv", "run", "--no-project", "--python", "3.11", "--with", "mediapipe==0.10.14", "--with", "pyarrow",
                        "python", "-I", str(script), str(folder), str(HAND_MODEL)], check=True)


# ---------------------------------------------------------------- landmark samples: pose (33, 3) + hands (2, 21, 2)
def load_hands(folder):
    """Detected hands per photo, (n, 2, 21, 2), in the order of pretrain.load_poses(folder); NaN = none."""
    return np.concatenate([np.load(folder / f"hands-{f.stem.split('-')[1]}.npz")["hands"] for f in sorted(folder.glob("pose-*.npz"))])


def samples(P, HD, idx):
    """Photos -> pose (k, 33, 3) [x, y, visibility] and hands (k, 2, 21, 2) [the person's left, right]."""
    pose = P["lm"][idx][:, :, [0, 1, 3]].astype(np.float64)
    hands = np.full((len(idx), 2, 21, 2), np.nan)
    for k, i in enumerate(idx):
        found = [h for h in HD[i] if np.isfinite(h[0, 0])]
        a = sk.assign_hands(pose[k], found)
        for j, side in enumerate(("left", "right")):
            if a[side] is not None:
                hands[k, j] = a[side]
    return pose, hands


def _flip(pose, hands):
    """Mirror image: x around the shoulder midpoint; the hand on the image left is now the person's right."""
    pose, hands = pose.copy(), hands[::-1].copy()
    m = pose[11, 0] + pose[12, 0]
    pose[:, 0] = m - pose[:, 0]
    hands[..., 0] = m - hands[..., 0]
    return pose, hands


def _mirror_arm(pose, hands, keep_left):
    """Copy one arm (and its hand) onto the other side, mirrored around the shoulder midpoint."""
    pose, hands = pose.copy(), hands.copy()
    m = pose[11, 0] + pose[12, 0]
    for a in _LEFT:
        src, dst = (a, a + 1) if keep_left else (a + 1, a)
        pose[dst] = pose[src]
        pose[dst, 0] = m - pose[src, 0]
    src, dst = (0, 1) if keep_left else (1, 0)
    hands[dst] = hands[src]
    hands[dst, :, 0] = m - hands[src, :, 0]
    return pose, hands


def _raise(pose, hands, by):
    """Hands up by `by` shoulder widths (wrists and hand points; elbows half as much): heart at face height."""
    pose, hands = pose.copy(), hands.copy()
    dy = by * np.hypot(*(pose[11, :2] - pose[12, :2]))
    pose[15:23, 1] -= dy
    pose[13:15, 1] -= dy / 2
    hands[..., 1] -= dy
    return pose, hands


def _augment(pose, hands, rng, rot_deg=10, noise=0.03, hand_noise=0.008):
    pose, hands = pose.copy(), hands.copy()
    c = (pose[11, :2] + pose[12, :2]) / 2
    w = np.hypot(*(pose[11, :2] - pose[12, :2]))
    t = np.deg2rad(rng.uniform(-rot_deg, rot_deg))
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    pose[:, :2] = (pose[:, :2] - c) @ R.T + c + rng.normal(0, noise * w, (33, 2))
    hands = (hands - c) @ R.T + c + rng.normal(0, hand_noise * w, hands.shape)
    return pose, hands


def render_all(pose, hands, layout=None):
    hd = lambda h: None if not np.isfinite(h[0, 0]) else h
    return np.stack([sk.render(p, {"left": hd(h[0]), "right": hd(h[1])}, layout) for p, h in zip(pose, hands)])


def _states(P):
    ok, ls, rs, _ = pt.hand_states(P["lm"])
    hag = P["hagrid"]
    return ok, ls, rs, np.isin(hag, list(pt._HEART_IDS)), np.isin(hag, list(pt._CHEST_TWO_HANDS))


def build_train(per_class=2000, blanks=100, seed=0, rest_chest=REST_CHEST, exclude=()):
    """Training samples (augmented) from train-split photos -> pose, hands, labels, HaGRID label of the source photo."""
    P, HD = pt.load_poses(), load_hands(pt.HAGRID_DIR)
    ok, ls, rs, heart, chest = _states(P)
    ok &= ~np.isin(P["hagrid"], list(exclude))
    rng = np.random.default_rng(seed)
    pick = lambda pool, n: rng.permutation(pool)[:n] if len(pool) >= n else rng.choice(pool, n)
    left_up, right_up = ok & (ls == "U") & (rs == "D") & ~heart, ok & (rs == "U") & (ls == "D") & ~heart
    down = ok & (ls == "D") & (rs == "D")
    one_up = np.flatnonzero(left_up | right_up)
    items = []                                           # (photo, class, flip, mirror, raise by)
    for cls, want_left in [(pt.RIGHT, False), (pt.LEFT, True)]:
        items += [(i, cls, bool(left_up[i]) != want_left, 0, 0.0) for i in pick(one_up, per_class)]
    both = np.flatnonzero(ok & (ls == "U") & (rs == "U") & ~heart)
    items += [(i, pt.BOTH, bool(rng.integers(2)), 0, 0.0) for i in both]
    items += [(i, pt.BOTH, bool(rng.integers(2)), 1 if left_up[i] else -1, 0.0) for i in pick(one_up, per_class - len(both))]
    hearts = np.flatnonzero(ok & heart & ~((ls == "U") ^ (rs == "U")))     # chest (and a few real face-height) hearts
    for i in pick(hearts, per_class):
        sym = int(rng.choice([-1, 1])) if rng.random() < .15 else 0
        up = float(rng.uniform(0.4, 1.4)) if (ls[i] == "D" and rng.random() < 0.5) else 0.0
        items.append((i, pt.HEART, bool(rng.integers(2)), sym, up))
    other = np.flatnonzero(down & ~heart & ~chest)
    n_chest = int(per_class * .4) if rest_chest else 0
    pool = np.concatenate([pick(np.flatnonzero(down & chest), n_chest), pick(other, per_class - n_chest)]) if n_chest else pick(other, per_class)
    for i in pool:
        items.append((i, pt.REST, bool(rng.integers(2)), int(rng.choice([-1, 1])) if rng.random() < .15 else 0, 0.0))
    ph = np.array([t[0] for t in items])
    pose, hands = samples(P, HD, ph)
    for k, (_, _, flip, mirror, up) in enumerate(items):
        p, h = pose[k], hands[k]
        if mirror:
            p, h = _mirror_arm(p, h, keep_left=mirror > 0)       # +1: keep the image-right side = the person's left
        if up:
            p, h = _raise(p, h, up)
        if flip:
            p, h = _flip(p, h)
        pose[k], hands[k] = _augment(p, h, rng)
    labels = np.array([t[1] for t in items] + [pt.REST] * blanks)
    pose = np.concatenate([pose, np.zeros((blanks, 33, 3))])        # nobody in view: blank picture = rest
    hands = np.concatenate([hands, np.full((blanks, 2, 21, 2), np.nan)])
    src = np.concatenate([P["hagrid"][ph], np.full(blanks, -1)])
    return pose, hands, labels, src


TEST_GROUPS = ["rest", "rest: two hands at chest", "right up", "left up", "both up (real)", "heart at chest",
               "heart at face (real)", "heart at face (synthetic)", "both up (synthetic)"]
_TEST_LABEL = [pt.REST, pt.REST, pt.RIGHT, pt.LEFT, pt.BOTH, pt.HEART, pt.HEART, pt.HEART, pt.BOTH]


def build_test(max_per_group=600, seed=0):
    """Test samples from test-split people (real landmarks, no augmentation) -> pose, hands, labels, group index,
    HaGRID label of the source photo."""
    P, HD = pt.load_poses(pt.HAGRID_TEST_DIR), load_hands(pt.HAGRID_TEST_DIR)
    ok, ls, rs, heart, chest = _states(P)
    ok &= ~np.isin(P["hagrid"], [LIKE, PEACE])           # those two are kept for the few-shot test
    down = ok & (ls == "D") & (rs == "D")
    rng = np.random.default_rng(seed)
    one_l, one_r = ok & (ls == "U") & (rs == "D") & ~heart, ok & (rs == "U") & (ls == "D") & ~heart
    masks = [down & ~heart & ~chest, down & chest, one_r, one_l, ok & (ls == "U") & (rs == "U") & ~heart,
             down & heart, ok & heart & (ls == "U") & (rs == "U"), down & heart, one_l | one_r]
    poses, handss, groups, srcs = [], [], [], []
    for g, m in enumerate(masks):
        idx = rng.permutation(np.flatnonzero(m))[:max_per_group]
        p, h = samples(P, HD, idx)
        if TEST_GROUPS[g] == "heart at face (synthetic)":
            ups = rng.uniform(0.5, 1.2, len(idx))
            p, h = map(np.stack, zip(*[_raise(a, b, u) for a, b, u in zip(p, h, ups)]))
        if TEST_GROUPS[g] == "both up (synthetic)":
            p, h = map(np.stack, zip(*[_mirror_arm(a, b, keep_left=bool(one_l[i])) for a, b, i in zip(p, h, idx)]))
        poses.append(p); handss.append(h); groups += [g] * len(idx); srcs.append(P["hagrid"][idx])
    groups = np.array(groups)
    return np.concatenate(poses), np.concatenate(handss), np.array(_TEST_LABEL)[groups], groups, np.concatenate(srcs)


def score(pred, ty, groups):
    """Accuracy per test group, heart/rest summary, confusion on real poses (synthetic groups excluded)."""
    from sklearn.metrics import confusion_matrix

    real = ~np.isin(groups, [TEST_GROUPS.index("heart at face (synthetic)"), TEST_GROUPS.index("both up (synthetic)")])
    per = {TEST_GROUPS[g]: float((pred[groups == g] == ty[groups == g]).mean()) for g in np.unique(groups)}
    rest_g = np.isin(groups, [0, 1])
    return {"accuracy_real": float((pred[real] == ty[real]).mean()), "per_group": per,
            "rest_called_heart": {TEST_GROUPS[g]: float((pred[groups == g] == pt.HEART).mean()) for g in (0, 1)},
            "heart_called_rest": float((pred[groups == 5] == pt.REST).mean()),
            "n_per_group": {TEST_GROUPS[g]: int((groups == g).sum()) for g in np.unique(groups)},
            "confusion_real": confusion_matrix(ty[real], pred[real], labels=pt.GESTURES).tolist(),
            "rest_accuracy_all": float((pred[rest_g] == pt.REST).mean())}


# ---------------------------------------------------------------- experiments
def layout_experiment(log=lambda *a: print(*a, flush=True)):
    """Pixel decoders: v1 (no hands) vs hands in place vs hand insets, rest with / without the chest gestures."""
    from .camera import _make_model

    tp, th, ty, tg, _ = build_test()
    res = {}
    for rest_chest in (True, False):
        pose, hands, y, _ = build_train(rest_chest=rest_chest)
        for layout in ("none", "inplace", "inset"):
            X = render_all(pose, hands, layout).reshape(len(y), -1)
            TX = render_all(tp, th, layout).reshape(len(ty), -1)
            s = score(_make_model().fit(X, y).predict(TX), ty, tg)
            res[f"{layout}, rest {'with' if rest_chest else 'without'} chest gestures"] = s
            log(f"  {layout:8s} rest_chest={rest_chest!s:5s}: real {s['accuracy_real']:.3f} | " +
                " | ".join(f"{k} {v:.2f}" for k, v in s["per_group"].items()))
    return res


def brain_layout_experiment(rest_chest=REST_CHEST, procs=5, threads=2, every=5, log=lambda *a: print(*a, flush=True)):
    """The same through the brain on a subset (every 5th train / 2nd test frame): in place vs insets."""
    tp, th, ty, tg, _ = build_test()
    pose, hands, y, _ = build_train(rest_chest=rest_chest)
    itr, ite = np.arange(0, len(y), every), np.arange(0, len(ty), 2)
    masks = pt.brain_masks()
    keep = masks["whole brain"][None, :].astype(np.uint8)
    res = {}
    for layout in ("inplace", "inset"):
        X, TX = render_all(pose[itr], hands[itr], layout), render_all(tp[ite], th[ite], layout)
        C = pt.simulate(X, WORK / f"exp_{layout}_train", procs, threads, rate=sk.RATE_HZ)
        TC = pt.simulate(TX, WORK / f"exp_{layout}_test", procs, threads, rate=sk.RATE_HZ)
        C, TC = [sp.csr_matrix(M.multiply(keep), dtype=np.uint8) for M in (C, TC)]
        res[layout] = {k: score(p, ty[ite], tg[ite]) for k, p in _predict_all((X, y[itr], C), (TX, TC), masks).items()}
        res[layout]["drive"] = drive(C, masks, X)
        log(f"  [{layout}] " + ", ".join(f"{k} {v['accuracy_real']:.3f} (heart chest {v['per_group'].get('heart at chest', 0):.2f}, "
                                          f"rest->heart {v['rest_called_heart']['rest']:.2f})"
                                          for k, v in res[layout].items() if k != "drive"))
    return res


def _predict_all(train, test, masks, decoders=None):
    from .camera import _make_model
    from .readout import spike_features

    (X, y, C), (TX, TC) = train, test
    active = np.asarray(C.sum(0)).ravel() > 0
    out = {}
    for name in decoders or pt.FEATURES:
        if name == "pixels":
            out[name] = _make_model().fit(X.reshape(len(X), -1), y).predict(TX.reshape(len(TX), -1))
        else:
            cols = np.flatnonzero(masks[name] & active)
            out[name] = _make_model().fit(spike_features(C, cols), y).predict(spike_features(TC, cols))
    return out


def drive(C, masks, X):
    """How strongly the pictures drive the brain: mean spikes / active neurons per frame and group."""
    out = {"image_mean": float(X.mean())}
    for name in ["descending", "central brain", "whole brain"]:
        M = C[:, np.flatnonzero(masks[name])]
        out[name] = {"spikes_per_frame": float(M.sum() / M.shape[0]), "active_per_frame": float(M.getnnz() / M.shape[0]),
                     "ever_active": int((np.asarray(M.sum(0)).ravel() > 0).sum()), "group_size": int(M.shape[1])}
    return out


def few_shot(train, src, test, masks, procs=5, threads=2, people=3, frames=30, seed=1, repeat=1, decoders=None,
             log=lambda *a: print(*a, flush=True)):
    """A new gesture recorded by ONE person (30 jittered frames of one train-split photo) + the pretrained data
    without any photo of that gesture -> test the new gesture on other people (test split), and the old classes.
    repeat: the recorded frames count `repeat` times in training (like a class weight; 67 ~ balanced with
    the 2,000 pretrained frames per class). decoders: only these (default all)."""
    (X, y, C), (TX, ty, TC, tg, tsrc) = train, test
    P, HD = pt.load_poses(), load_hands(pt.HAGRID_DIR)
    TP, THD = pt.load_poses(pt.HAGRID_TEST_DIR), load_hands(pt.HAGRID_TEST_DIR)
    okT = _states(TP)[0]
    ok = _states(P)[0]
    rng = np.random.default_rng(seed)
    keep = masks["whole brain"][None, :].astype(np.uint8)
    res = {"repeat": repeat}
    if repeat == 1 and decoders is None:
        base = _predict_all((X, y, C), (TX, TC), masks)
        res["old_classes_before"] = {k: score(p, ty, tg)["accuracy_real"] for k, p in base.items()}
    for gid, name in [(PEACE, "peace"), (LIKE, "like (thumb up)"), (XSIGN, "arms crossed (xsign)")]:
        has_hand = np.isfinite(HD[:, :, 0, 0]).any(1)
        cand = np.flatnonzero(ok & (P["hagrid"] == gid) & has_hand)
        test_idx = np.flatnonzero(okT & (TP["hagrid"] == gid))[:500]
        tp, th = samples(TP, THD, test_idx)
        NX = render_all(tp, th)
        NC = sp.csr_matrix(pt.simulate(NX, WORK / f"fewshot_{gid}_test", procs, threads, rate=sk.RATE_HZ).multiply(keep), dtype=np.uint8)
        drop = src != gid                                   # pretrained data without that gesture
        out = {"people": [], "test_frames": int(len(test_idx))}
        for k, i in enumerate(rng.choice(cand, people, replace=False)):
            p0, h0 = samples(P, HD, np.array([i]))
            pose, hands = map(np.stack, zip(*[_augment(p0[0], h0[0], rng, rot_deg=4, noise=0.015, hand_noise=0.004)
                                              for _ in range(frames)]))
            RX = render_all(pose, hands)
            RC = sp.csr_matrix(pt.simulate(RX, WORK / f"fewshot_{gid}_rec{k}", procs, threads, rate=sk.RATE_HZ).multiply(keep),
                               dtype=np.uint8)
            tr = (np.concatenate([X[drop]] + [RX] * repeat), np.concatenate([y[drop], np.full(frames * repeat, NEW)]),
                  sp.vstack([C[drop]] + [RC] * repeat).tocsr())
            preds = _predict_all(tr, (np.concatenate([TX, NX]), sp.vstack([TC, NC]).tocsr()), masks, decoders)
            r = {}
            for dec, p in preds.items():
                po, pn = p[:len(TX)], p[len(TX):]
                m = tsrc != gid                             # old-class test frames (xsign photos are in the chest group)
                r[dec] = {"new_gesture_recall": float((pn == NEW).mean()),
                          "new_gesture_predicted_as": {int(c): int((pn == c).sum()) for c in np.unique(pn)},
                          "old_classes_accuracy": score(po[m], ty[m], tg[m])["accuracy_real"],
                          "old_classes_false_new": float((po[m] == NEW).mean())}
            out["people"].append({"photo": int(i), **r})
            log(f"  few-shot {name}, person {k + 1}: " + ", ".join(
                f"{d} new {v['new_gesture_recall']:.2f} / old {v['old_classes_accuracy']:.3f}" for d, v in r.items()))
        out["mean"] = {d: {m: float(np.mean([pp[d][m] for pp in out["people"]]))
                           for m in ("new_gesture_recall", "old_classes_accuracy", "old_classes_false_new")} for d in (decoders or pt.FEATURES)}
        res[name] = out
    return res


# ---------------------------------------------------------------- the model
def run(procs=5, threads=2):
    import joblib

    from .camera import CLASSES, train_decoders

    log = lambda *a: print(*a, flush=True)
    WORK.mkdir(parents=True, exist_ok=True)
    if any(not f.with_name(f.name.replace("pose-", "hands-")).exists()
           for d in (pt.HAGRID_DIR, pt.HAGRID_TEST_DIR) for f in d.glob("pose-*.npz")):
        hands_step()                                            # Hand Landmarker on the photos (once, ~4 min)
    cache = lambda name, fn: (json.loads((WORK / f"{name}.json").read_text()) if (WORK / f"{name}.json").exists()
                              else (lambda r: ((WORK / f"{name}.json").write_text(json.dumps(r, indent=1)), r)[1])(fn()))
    results = {"layout_pixels": cache("layout_pixels", layout_experiment),
               "layout_brain_subset": cache("layout_brain", lambda: brain_layout_experiment(procs=procs, threads=threads))}
    if not (WORK / "train.npz").exists():
        pose, hands, y, src = build_train()
        np.savez_compressed(WORK / "train.npz", pose=pose, hands=hands, labels=y, src=src, images=render_all(pose, hands))
    if not (WORK / "test.npz").exists():
        tp, th, ty, tg, tsrc = build_test()
        np.savez_compressed(WORK / "test.npz", pose=tp, hands=th, labels=ty, groups=tg, src=tsrc, images=render_all(tp, th))
    tr, te = np.load(WORK / "train.npz"), np.load(WORK / "test.npz")
    X, y, src, TX, ty, tg, tsrc = tr["images"], tr["labels"], tr["src"], te["images"], te["labels"], te["groups"], te["src"]
    log(f"stick figures v{sk.SPEC_VERSION}: train {np.bincount(y, minlength=10)[pt.GESTURES].tolist()}, "
        f"test {dict(zip(TEST_GROUPS, np.bincount(tg).tolist()))}")
    t = time.time()
    C = pt.simulate(X, WORK / "counts_train", procs, threads, rate=sk.RATE_HZ)
    TC = pt.simulate(TX, WORK / "counts_test", procs, threads, rate=sk.RATE_HZ)
    speed = (len(X) + len(TX)) / (time.time() - t)
    masks = pt.brain_masks()
    keep = masks["whole brain"][None, :].astype(np.uint8)      # like the browser: no eye-input counts
    C, TC = [sp.csr_matrix(M.multiply(keep), dtype=np.uint8) for M in (C, TC)]
    preds = _predict_all((X, y, C), (TX, TC), masks)
    results["cross_person"] = {k: score(p, ty, tg) for k, p in preds.items()}
    for k, v in results["cross_person"].items():
        log(f"  {k:13s} real {v['accuracy_real']:.3f} | " + " | ".join(f"{g} {a:.2f}" for g, a in v["per_group"].items()))
    results["brain_drive"] = drive(C, masks, X)
    results["few_shot"] = cache("few_shot", lambda: few_shot((X, y, C), src, (TX, ty, TC, tg, tsrc), masks, procs, threads))
    v1 = DATA_DIR / "pretrained_skeleton_v1.report.json"
    if v1.exists():
        results["v1_cross_person"] = {k: v["accuracy_real_poses"] for k, v in json.loads(v1.read_text())["cross_person"].items()}
    models, metrics = train_decoders(list(X), y.tolist(), C, masks)
    meta = {"input": f"stick figure + magnified hands (flymnist/skeleton.py spec {sk.SPEC_VERSION}) instead of the camera picture",
            "spec_version": sk.SPEC_VERSION, "rate_hz": sk.RATE_HZ,
            "training_data": "stick figures rendered from MediaPipe Pose + Hand landmarks of HaGRIDv2 photos (no photos stored); "
                             "credit HaGRIDv2 (CC BY-SA 4.0) if the decoders are ever published",
            "source": pt.SOURCE, "license": pt.LICENSE, "classes": {c: CLASSES[c] for c in pt.GESTURES},
            "brain": f"lamina input, 50 ms, {sk.RATE_HZ:.0f} Hz for a white pixel, torch FlyBrain, seed 0 + chunk",
            "rest_includes_two_hands_at_chest": REST_CHEST, "built": time.strftime("%Y-%m-%d %H:%M")}
    landmarks = {"pose": tr["pose"].astype(np.float32), "hands": tr["hands"].astype(np.float32),
                 "note": "pose (n, 33, 3) [x, y, visibility] px; hands (n, 2, 21, 2) [the person's left, right] px, NaN = none; "
                         "render with skeleton.render(pose, {'left': hands[0], 'right': hands[1]})"}
    joblib.dump({"images": list(X), "labels": y.tolist(), "counts": C, "models": models, "metrics": metrics, "version": 1,
                 "trained_on": None, "include_class": True, "meta": meta, "landmarks": landmarks}, OUT.with_suffix(".tmp"), compress=3)
    OUT.with_suffix(".tmp").replace(OUT)
    REPORT.write_text(json.dumps({**meta, "frames_per_class": {CLASSES[c]: int((y == c).sum()) for c in pt.GESTURES},
                                  "train_decoders_cv": metrics, **results, "test_people": "HaGRIDv2 test split (other people)",
                                  "brain_frames_per_s": round(speed, 1), "brain_setup": f"{procs} processes x {threads} threads"},
                                 indent=1))
    log(f"saved {OUT} and {REPORT}")
    return results
