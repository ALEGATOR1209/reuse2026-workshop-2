"""Per-user camera sessions, plus the class model everyone starts from.

The class model is the teacher's session (whoever uses the app on this computer). A
student's session holds only their own recordings; until they train, they use the class
decoders. When they train, their decoders learn from the class recordings plus their own.

Each recording is stored as its preprocessed image, its label and the spike counts the
user's own brain (in their browser) produced for it. Sessions are saved to disk, so a
server restart or a page reload keeps everyone's work.

The camera tab has two inputs, kept apart so their recordings never mix: camera pictures
(Session; data/camera_session.joblib and data/sessions/<id>.joblib) and stick figures of the
person's pose (StickSession, flymnist/stick.py; data/skeleton_session.joblib and
data/sessions/skeleton/<id>.joblib), which also keep the pose landmarks of every frame.
"""

import re
import threading
import time
from pathlib import Path

import joblib
import numpy as np
import scipy.sparse as sp

from .data import DATA_DIR

CLASS_FILE = DATA_DIR / "camera_session.joblib"
SESSIONS_DIR = DATA_DIR / "sessions"
SKELETON_CLASS_FILE = DATA_DIR / "skeleton_session.joblib"
SKELETON_SUBDIR = "skeleton"            # students' stick-figure sessions: SESSIONS_DIR / SKELETON_SUBDIR
_SID = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


class Session:
    def __init__(self, path, n_neurons):
        self.path, self.n = Path(path), n_neurons
        self.lock = threading.RLock()     # re-entrant: StickSession wraps these methods
        self.images, self.labels = [], []
        self.counts = sp.csr_matrix((0, n_neurons), dtype=np.uint8)
        self.models, self.metrics = {}, {}
        self.version = 0              # bumps whenever recordings or decoders change
        self.trained_on = None        # for students: class version their decoders include
        self.include_class = True
        if self.path.exists():
            s = joblib.load(self.path)
            self.images, self.labels = list(s["images"]), list(s["labels"])
            self.counts = s.get("counts", self.counts)
            self.models, self.metrics = s.get("models", {}), s.get("metrics", {})
            self.version = s.get("version", 1)
            self.trained_on = s.get("trained_on")
            self.include_class = s.get("include_class", True)
            self._load_extra(s)

    def _load_extra(self, saved):
        """Subclasses: read their own fields from the saved dict."""

    @property
    def missing_counts(self):
        """Recordings without spike counts (made before brains ran in the browser)."""
        return len(self.images) - self.counts.shape[0]

    def add(self, images, labels, counts):
        with self.lock:
            self.images += list(images)
            self.labels += [int(x) for x in labels]
            self.counts = sp.vstack([self.counts, sp.csr_matrix(counts, dtype=np.uint8)]).tocsr()
            self.version += 1
            self.save()

    def set_models(self, models, metrics, trained_on=None, include_class=True):
        with self.lock:
            self.models, self.metrics = models, metrics
            self.trained_on, self.include_class = trained_on, include_class
            self.version += 1
            self.save()

    def clear(self):
        with self.lock:
            self.images, self.labels, self.models, self.metrics = [], [], {}, {}
            self.counts = sp.csr_matrix((0, self.n), dtype=np.uint8)
            self.trained_on = None
            self.version += 1
            self.save()

    def label_counts(self, n_classes):
        return np.bincount(np.asarray(self.labels, int), minlength=n_classes).tolist()

    def _state(self):
        return {"images": self.images, "labels": self.labels, "counts": self.counts, "models": self.models,
                "metrics": self.metrics, "version": self.version, "trained_on": self.trained_on,
                "include_class": self.include_class}

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        joblib.dump(self._state(), tmp)
        tmp.replace(self.path)


class SessionStore:
    """The class session plus one session per student. kind: "camera" (camera pictures) or
    "skeleton" (stick figures, flymnist/stick.py); the two kinds live in separate files."""

    def __init__(self, n_neurons, kind="camera", session_cls=Session, last_seen=None):
        self.n, self.kind, self.session_cls = n_neurons, kind, session_cls
        self.cls = session_cls(CLASS_FILE if kind == "camera" else SKELETON_CLASS_FILE, n_neurons)
        self._users, self._lock = {}, threading.Lock()
        self.last_seen = {} if last_seen is None else last_seen     # may be shared between the two kinds

    def user_path(self, sid):
        return (SESSIONS_DIR if self.kind == "camera" else SESSIONS_DIR / SKELETON_SUBDIR) / f"{sid}.joblib"

    def get(self, sid, teacher=False):
        if teacher:
            return self.cls
        if not _SID.match(sid or ""):
            raise ValueError("bad session id")
        with self._lock:
            self.last_seen[sid] = time.time()
            if sid not in self._users:
                self._users[sid] = self.session_cls(self.user_path(sid), self.n)
            return self._users[sid]

    def active_users(self, within_s=600):
        now = time.time()
        return sum(1 for t in self.last_seen.values() if now - t < within_s)
