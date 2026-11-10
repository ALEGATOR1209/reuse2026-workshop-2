"""Checks for the stick-figure input's storage and training (flymnist/stick.py), without the server or a browser.

    uv run python tests/test_stick.py

Uses small random data in a temporary folder; nothing in data/ is read or written.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flymnist import sessions, stick  # noqa: E402

N = 500                                   # neurons
rng = np.random.default_rng(0)


def frames(label, k, shift):
    """k frames of one 'gesture': its own image blob and its own group of active neurons."""
    imgs = [np.zeros((36, 48), np.float32) for _ in range(k)]
    for im in imgs:
        im[5 + shift:15 + shift, 10:20] = 1
    C = np.zeros((k, N), np.uint8)
    C[:, 50 * shift:50 * shift + 40] = rng.integers(0, 4, (k, 40))
    lm = {"pose": np.zeros((33, 3), np.float32), "left": None, "right": np.ones((21, 2), np.float32)}
    return imgs, [label] * k, sp.csr_matrix(C), [lm] * k


class FakePre:                            # the parts of stick.Pretrained that train() uses
    def __init__(self, per):
        I, y, C = [], [], []
        for lab, sh in [(0, 0), (1, 2), (2, 4)]:
            a, b, c, _ = frames(lab, per, sh)
            I += a; y += b; C.append(c)
        self.images, self.labels, self.counts = np.stack(I), np.asarray(y), sp.vstack(C).tocsr()
        self.per_class = per
        self.active = np.asarray((self.counts > 0).sum(0)).ravel() / len(y) >= stick.MIN_ACTIVE


def main():
    tmp = Path(tempfile.mkdtemp())
    masks = {"descending": np.arange(N) < 100, "central brain": (np.arange(N) >= 100) & (np.arange(N) < 300),
             "whole brain": np.ones(N, bool)}

    # weights: a gesture's own frames weigh about one pre-trained gesture, 1..MAX_WEIGHT each
    w = stick.own_weights([8] * 30 + [0] * 600, 600)
    assert np.allclose(w[:30], stick.MAX_WEIGHT) and np.allclose(w[30:], 1.0), w[:3]

    # a StickSession keeps landmarks + spec per frame, survives a reload, and is separate from camera sessions
    s = stick.StickSession(tmp / "skeleton" / "a.joblib", N)
    s.add(*frames(8, 30, 6)[:3], frames(8, 30, 6)[3], spec=1)
    s.add(*frames(0, 10, 0)[:3], None, spec=2)
    s2 = stick.StickSession(tmp / "skeleton" / "a.joblib", N)
    assert len(s2.images) == 40 and s2.specs == [1] * 30 + [2] * 10 and s2.landmarks[0]["right"].shape == (21, 2)
    assert s2.stale(2) == list(range(30)) and s2.label_counts_spec(10, 2)[0] == 10
    a, _, c, _ = frames(8, 30, 6)
    s2.replace(list(range(30)), a, c, 2)
    assert s2.stale(2) == [] and stick.StickSession(tmp / "skeleton" / "a.joblib", N).specs == [2] * 40

    # the store puts stick sessions in their own folder
    sessions.SESSIONS_DIR = tmp / "sessions"
    store = sessions.SessionStore.__new__(sessions.SessionStore)
    store.kind = "skeleton"
    assert store.user_path("abcdefgh12") == tmp / "sessions" / "skeleton" / "abcdefgh12.joblib"

    # training: pre-trained subset + own frames of the current spec; a new gesture (label 8) is learned
    pre = FakePre(100)
    models, metrics, n = stick.train(pre, [s2], masks, spec=2)
    assert n == 300 + 40, n
    assert metrics["whole brain"]["frames"] == {"pretrained": 300, "recorded": 40}
    m = models["whole brain"]
    X = np.log1p(c[:, m["cols"]].toarray().astype(np.float32))
    assert (m["model"].predict(X) == 8).all()
    assert set(models) == {"descending", "central brain", "whole brain", "pixels"}
    print("stick tests passed:", {k: round(v["cv_accuracy"], 3) for k, v in metrics.items()})


if __name__ == "__main__":
    main()
