"""Stick figures: the camera tab's second input ("Camera picture | Stick figure").

Instead of the camera picture, the fly sees a stick figure of the person's upper-body pose
(flymnist/skeleton.py, drawn in the browser by static/skeleton.js from MediaPipe landmarks).
Unlike camera pictures, stick figures look the same in any room and for anyone, so decoders
trained on other people work out of the box:

- data/pretrained_skeleton.joblib (`uv run main.py pretrain --step skeleton`): stick figures of
  HaGRIDv2 people (no photos), their spike counts and decoders for 5 gestures. Until anyone trains,
  everyone uses its decoders.
- Train = a fixed subset of those pre-trained frames (SERVER_PER_CLASS per gesture, so a Train takes
  seconds, not minutes) + the class's stick-figure recordings (the teacher's) + the visitor's own.
  Own frames weigh more (up to MAX_WEIGHT pre-trained frames each), so ~30 frames of a new gesture
  are enough. Decoders read the neurons active in >= MIN_ACTIVE of the pre-trained (or of the own) frames.
- Recordings are StickSessions, separate from the camera-picture sessions: the stick figure, label
  and spike counts of each frame plus its pose landmarks and the stick-figure spec version, so the
  frames can be drawn again (and re-simulated, in the browser) when the spec changes.
- The static site ships a smaller subset (static_export) for training in the browser.

Nothing here hard-codes the spec: the input rate and spec version come from the pre-trained
model's meta and from flymnist/skeleton.py.
"""

import base64
import gzip
import json
import re
import threading

import numpy as np
import scipy.sparse as sp

from . import skeleton as sk
from .camera import IMG_SHAPE, export_decoder, train_decoders
from .data import DATA_DIR
from .sessions import Session

PRETRAINED = DATA_DIR / "pretrained_skeleton.joblib"
SERVER_PER_CLASS = 600      # pre-trained frames per gesture in a Train on the server (~10 s for all decoders)
STATIC_PER_CLASS = 200      # ... shipped with the static site, for training in the browser
STATIC_TOP = 2000           # whole-brain neurons shipped with the static site (largest pre-trained weights)
MIN_ACTIVE = 0.02           # decoders read neurons that fire in >= 2% of the pre-trained (or of the own) frames
MAX_WEIGHT = 10.0           # one own frame counts as up to this many pre-trained frames
SEED = 0
LICENSE = ("Derived from HaGRIDv2 (Kapitanov et al., arXiv:2412.01508), CC BY-SA 4.0; research / non-commercial "
           "use per its authors. Stick figures and spike counts only: no photos.")


def spec_version():
    """Version of the stick-figure drawing this app uses (flymnist/skeleton.py; skeleton.js is identical)."""
    return int(getattr(sk, "SPEC_VERSION", 1))


def _meta_spec(meta):
    if "spec_version" in meta:
        return int(meta["spec_version"])
    m = re.search(r"spec v(?:ersion )?(\d+)", str(meta.get("input", "")))
    return int(m.group(1)) if m else 1


def _b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def own_weights(labels, per_class):
    """Weight of each own frame: a gesture's own frames together weigh about as much as one
    gesture of the pre-trained data (per_class frames), each frame between 1 and MAX_WEIGHT."""
    y = np.asarray(labels, int)
    if not per_class or not len(y):
        return np.ones(len(y))
    n = np.bincount(y, minlength=10)[y]
    return np.clip(per_class / n, 1.0, MAX_WEIGHT)


# ---------------------------------------------------------------- landmarks
def parse_landmarks(item):
    """{pose: 33 x [x, y, vis] | null, hands: {left, right: 21 x [x, y] | null}, size?: [w, h]}
    (tracker.js, pixels of the unmirrored frame) -> compact numpy dict, or None."""
    if not item:
        return None
    pose = item.get("pose")
    hands = item.get("hands") or {}
    out = {"pose": None if pose is None else np.asarray(pose, np.float32).reshape(-1, 3)}
    for side in ("left", "right"):
        h = hands.get(side)
        out[side] = None if h is None else np.asarray(h, np.float32).reshape(-1, 2)
    if item.get("size"):
        out["size"] = [int(v) for v in item["size"][:2]]
    if out["pose"] is not None and len(out["pose"]) != 33:
        raise ValueError("pose needs 33 landmarks")
    return out


def landmarks_json(lm):
    if lm is None:
        return None
    r = lambda a: None if a is None else np.round(np.asarray(a, float), 2).tolist()
    return {"pose": r(lm.get("pose")), "hands": {"left": r(lm.get("left")), "right": r(lm.get("right"))},
            **({"size": lm["size"]} if lm.get("size") else {})}


# ---------------------------------------------------------------- recordings
class StickSession(Session):
    """A Session of stick-figure recordings: also the landmarks and spec version of every frame,
    and the spec version its decoders were trained for."""

    def __init__(self, path, n_neurons):
        self.landmarks, self.specs, self.model_spec = [], [], None
        super().__init__(path, n_neurons)

    def _load_extra(self, saved):
        self.landmarks = list(saved.get("landmarks", [None] * len(self.images)))
        self.specs = [int(v) for v in saved.get("specs", [1] * len(self.images))]
        self.model_spec = saved.get("model_spec")

    def _state(self):
        return {**super()._state(), "kind": "skeleton", "landmarks": self.landmarks, "specs": self.specs,
                "model_spec": self.model_spec}

    def add(self, images, labels, counts, landmarks=None, spec=None):
        with self.lock:
            self.landmarks += list(landmarks) if landmarks is not None else [None] * len(images)
            self.specs += [int(spec if spec is not None else spec_version())] * len(images)
            super().add(images, labels, counts)

    def set_models(self, models, metrics, trained_on=None, include_class=True, spec=None):
        with self.lock:
            self.model_spec = spec
            super().set_models(models, metrics, trained_on=trained_on, include_class=include_class)

    def clear(self):
        with self.lock:
            self.landmarks, self.specs, self.model_spec = [], [], None
            super().clear()

    def stale(self, spec):
        """Frames drawn with another spec version (redrawn from their landmarks in the browser)."""
        return [i for i, s in enumerate(self.specs) if s != spec]

    def current(self, spec):
        return [i for i, s in enumerate(self.specs) if s == spec]

    def label_counts_spec(self, n_classes, spec):
        y = [self.labels[i] for i in self.current(spec)]
        return np.bincount(np.asarray(y, int), minlength=n_classes).tolist()

    def replace(self, rows, images, counts, spec):
        """Frames redrawn with the current spec: new images and spike counts for those rows."""
        with self.lock:
            rows = [int(i) for i in rows]
            if any(not 0 <= i < len(self.images) for i in rows):
                raise ValueError("no such frame")
            C = self.counts.tolil()
            new = sp.csr_matrix(counts, dtype=np.uint8)
            for k, i in enumerate(rows):
                self.images[i] = images[k]
                C[i] = new[k]
                self.specs[i] = int(spec)
            self.counts = C.tocsr()
            self.version += 1
            self.save()


# ---------------------------------------------------------------- the pre-trained model
class Pretrained:
    """data/pretrained_skeleton.joblib, loaded once (in the background at server start): its decoders,
    and the subset of its frames that Train mixes in. Never written."""

    def __init__(self, masks, path=PRETRAINED):
        self.path, self.masks = path, masks
        self._lock = threading.Lock()
        self.ready = threading.Event()
        self.error = None
        self.models, self.metrics, self.meta = {}, {}, {}
        self.images = self.labels = self.counts = None
        self.per_class, self.version, self.spec, self.active = 0, "none", None, None
        self.frames_per_class = {}
        self._export = None

    @property
    def available(self):
        return self.path.exists()

    def load(self):
        with self._lock:
            if self.ready.is_set():
                return self
            try:
                if self.path.exists():
                    self._load()
            except Exception as e:  # noqa: BLE001  (the app works without it: own recordings only)
                self.error = f"{type(e).__name__}: {e}"
                print(f"stick figures: could not load {self.path.name}: {self.error}", flush=True)
            finally:
                self.ready.set()
        return self

    def _load(self):
        import joblib

        s = joblib.load(self.path)
        y = np.asarray(s["labels"], int)
        rng = np.random.default_rng(SEED)
        per = min(SERVER_PER_CLASS, int(np.bincount(y)[np.unique(y)].min()))
        sel = np.sort(np.concatenate([rng.choice(np.flatnonzero(y == c), per, replace=False) for c in np.unique(y)]))
        self.labels = y[sel]
        self.images = np.stack([s["images"][i] for i in sel]).astype(np.float32).reshape(len(sel), *IMG_SHAPE)
        self.counts = sp.csr_matrix(s["counts"])[sel].astype(np.uint8)
        self.per_class = per
        frac = np.asarray((self.counts > 0).sum(0)).ravel() / len(sel)
        self.active = frac >= MIN_ACTIVE
        self.models, self.metrics, self.meta = s["models"], s["metrics"], dict(s.get("meta", {}))
        self.frames_per_class = {int(c): int((y == c).sum()) for c in np.unique(y)}
        self.spec = _meta_spec(self.meta)
        self.version = f"{self.meta.get('built', '')}-{len(y)}".replace(" ", "_")
        print(f"stick figures: pre-trained model {self.path.name} ({len(y)} frames, spec v{self.spec}); "
              f"Train mixes in {per} per gesture", flush=True)

    @property
    def rate_hz(self):
        return float(self.meta.get("rate_hz", sk.RATE_HZ))

    def info(self):
        if not self.ready.is_set():
            return {"available": self.available, "loading": True}
        if not self.models:
            return {"available": False, "loading": False, "error": self.error}
        return {"available": True, "loading": False, "spec_version": self.spec,
                "classes": sorted(self.frames_per_class), "frames_per_class": self.frames_per_class,
                "frames_in_train": self.per_class, "metrics": self.metrics, "built": self.meta.get("built")}

    def export(self):
        """The pre-trained decoders as plain weights (gzipped JSON, cached)."""
        if self._export is None:
            self._export = gzip.compress(json.dumps({
                "version": f"pretrained:{self.version}", "source": "pretrained",
                "decoders": {k: export_decoder(v) for k, v in self.models.items()}}).encode(), 6)
        return self._export


# ---------------------------------------------------------------- training
def train(pre, parts, masks, progress=None, spec=None):
    """Decoders from the pre-trained subset + the current-spec frames of `parts` (StickSessions:
    the class's and/or the visitor's). Returns (models, metrics, n_frames)."""
    spec = spec_version() if spec is None else spec
    have_pre = pre is not None and pre.labels is not None
    images, labels, counts, weights, groups = [], [], [], [], []
    if have_pre:
        images.append(pre.images); labels.append(pre.labels); counts.append(pre.counts)
        weights.append(np.ones(len(pre.labels))); groups.append(pre.labels)
    extra_n = 0
    for k, p in enumerate(parts):
        with p.lock:
            idx = p.current(spec)
            if not idx:
                continue
            im = np.stack([p.images[i] for i in idx]).astype(np.float32).reshape(len(idx), *IMG_SHAPE)
            y = np.asarray([p.labels[i] for i in idx], int)
            c = p.counts[idx]
        images.append(im); labels.append(y); counts.append(c)
        weights.append(own_weights(y, pre.per_class if have_pre else 0)); groups.append(1000 * (k + 1) + y)
        extra_n += len(idx)
    if not labels:
        raise ValueError("Record at least 5 frames of at least two actions first.")
    y = np.concatenate(labels)
    C = sp.vstack(counts).tocsr()
    active = pre.active.copy() if have_pre else np.zeros(C.shape[1], bool)
    if extra_n:
        E = C[len(pre.labels):] if have_pre else C
        active |= np.asarray((E > 0).sum(0)).ravel() >= max(1.0, MIN_ACTIVE * E.shape[0])
    cols = {name: np.flatnonzero(m & active) for name, m in masks.items()}
    models, metrics = train_decoders(np.concatenate(images), y, C, masks, progress,
                                     sample_weight=np.concatenate(weights), groups=np.concatenate(groups), cols=cols)
    for m in metrics.values():
        m["frames"] = {"pretrained": int(len(pre.labels)) if have_pre else 0, "recorded": int(extra_n)}
    return models, metrics, len(y)


# ---------------------------------------------------------------- static site
def static_export(path=PRETRAINED, per_class=STATIC_PER_CLASS, top=STATIC_TOP):
    """A compact subset of the pre-trained frames for training in the browser (static site):
    per_class frames per gesture, spike counts of the descending and central-brain neurons the
    pre-trained decoders read plus the `top` whole-brain neurons with the largest decoder weights,
    as a sparse matrix; and the stick figures (for the pixels baseline). No photos."""
    import joblib

    s = joblib.load(path)
    y = np.asarray(s["labels"], int)
    rng = np.random.default_rng(SEED + 1)
    per = min(per_class, int(np.bincount(y)[np.unique(y)].min()))
    sel = np.sort(np.concatenate([rng.choice(np.flatnonzero(y == c), per, replace=False) for c in np.unique(y)]))
    wb = s["models"]["whole brain"]
    lr = wb["model"][-1]
    imp = np.abs(lr.coef_).max(0)                 # weights on standardized spike counts
    cols = set(np.asarray(wb["cols"])[np.argsort(-imp)[:top]].tolist())
    for k in ("descending", "central brain"):
        if k in s["models"] and s["models"][k]["cols"] is not None:
            cols |= set(np.asarray(s["models"][k]["cols"]).tolist())
    cols = np.asarray(sorted(cols), np.int32)
    C = sp.csr_matrix(s["counts"])[sel][:, cols].tocsr()
    C.sort_indices()
    assert len(cols) < 65536
    imgs = np.round(np.stack([s["images"][i] for i in sel]).reshape(len(sel), -1) * 255).astype(np.uint8)
    meta = dict(s.get("meta", {}))
    return {"license": LICENSE, "spec_version": _meta_spec(meta), "rate_hz": float(meta.get("rate_hz", sk.RATE_HZ)),
            "n": int(len(sel)), "per_class": per, "image_shape": list(IMG_SHAPE), "labels": y[sel].tolist(),
            "cols": _b64(cols), "indptr": _b64(C.indptr.astype(np.int32)), "idx": _b64(C.indices.astype(np.uint16)),
            "val": _b64(np.minimum(C.data, 255).astype(np.uint8)), "images": _b64(imgs),
            "min_active": MIN_ACTIVE, "max_weight": MAX_WEIGHT}
