"""Webcam -> fly eyes -> fly brain -> "motor decoder" -> what the fly's body does.

You record a few seconds of a gesture of your own choice for each fly action you want
(any gesture works, as long as it looks different from the others). Every frame is shown to the fly brain (same eye mapping as MNIST; the brain runs in each
user's browser), and a logistic regression learns to tell the gestures apart from the
brain's spikes. The default decoder reads only the descending neurons, the ~1,300 neurons
that carry commands from the brain to the body. The connectome itself is not trained.

This module: frame preprocessing, decoder training (on the server) and decoder export
(weights the browser applies to its own spikes).
"""

import base64
import io

import numpy as np
import scipy.sparse as sp
from PIL import Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .body import ACTIONS
from .readout import spike_features

CLASSES = [a[0] for a in ACTIONS]   # one class per fly action; you choose any gesture for it
IMG_SHAPE = (36, 48)        # what the eye gets: rows, cols (4:3 like a webcam)
DURATION_MS = 50.0          # brain time per camera frame (shorter than MNIST, for responsiveness)
FOLDS = 5


def preprocess_frame(image_bytes):
    """Webcam frame -> local-contrast image in [0, 1], shape IMG_SHAPE.

    Lamina cells respond to contrast (centre minus surround), not absolute brightness,
    so we do the same: grey, downsample, subtract a blurred copy, keep the magnitude.
    """
    g = Image.open(io.BytesIO(image_bytes)).convert("L").resize(IMG_SHAPE[::-1], Image.BOX)
    g = np.asarray(g, np.float32) / 255.0
    c = np.abs(g - ndimage.gaussian_filter(g, 2.0))
    return np.clip(c / (np.percentile(c, 99) + 1e-3), 0, 1).astype(np.float32)


def _make_model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000))


def check_trainable(labels):
    y = np.asarray(labels, int)
    if len(set(y.tolist())) < 2 or np.bincount(y).max() < 5:
        raise ValueError("Record at least 5 frames of at least two actions first.")


def train_decoders(images, labels, counts, masks, progress=None, task="Training the motor decoder"):
    """Fit one decoder per neuron group (plus a pixels-only baseline) and cross-validate it.

    images: (F, 36, 48) float; labels: (F,) action index; counts: sparse (F, N) spike counts;
    masks: {name: bool mask over neurons}. Returns (models, metrics).
    """
    y = np.asarray(labels, int)
    check_trainable(y)
    counts = sp.csr_matrix(counts)
    images = np.asarray(images, np.float32).reshape(len(y), -1)
    names = list(masks) + ["pixels"]
    if progress:
        progress.start(task, "Fitting decoders (5-fold cross-validation)", len(names) * (FOLDS + 1))
    active = np.asarray(counts.sum(0)).ravel() > 0
    cols = {name: np.flatnonzero(m & active) for name, m in masks.items()}

    # Honest cross-validation: consecutive webcam frames are nearly identical, so each fold
    # holds out a contiguous block of every action, not random frames.
    folds = np.zeros(len(y), int)
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        folds[idx] = np.minimum(np.arange(len(idx)) * FOLDS // len(idx), FOLDS - 1)

    models, metrics = {}, {}
    for name in names:
        if name != "pixels" and len(cols[name]) == 0:
            metrics[name] = {"cv_accuracy": None, "neurons": 0}
            if progress:
                progress.advance(FOLDS + 1)
            continue
        X = images if name == "pixels" else spike_features(counts, cols[name])
        correct = []
        for k in range(FOLDS):
            tr, te = folds != k, folds == k
            if len(np.unique(y[tr])) >= 2 and te.sum() > 0:
                correct.append((_make_model().fit(X[tr], y[tr]).predict(X[te]) == y[te]).mean())
            if progress:
                progress.advance()
        models[name] = {"model": _make_model().fit(X, y), "cols": None if name == "pixels" else cols[name]}
        metrics[name] = {"cv_accuracy": float(np.mean(correct)) if correct else None, "neurons": int(X.shape[1])}
        if progress:
            progress.advance()
    return models, metrics


def _b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def export_decoder(entry):
    """A trained decoder as plain numbers the browser can apply:
    p = softmax(coef @ ((log1p(counts[cols]) - mean) / scale) + intercept), or a sigmoid for 2 classes."""
    pipe = entry["model"]
    scaler, lr = pipe[0], pipe[-1]
    return {
        "cols": None if entry["cols"] is None else _b64(np.asarray(entry["cols"], np.int32)),
        "n_features": int(len(scaler.mean_)),
        "mean": _b64(scaler.mean_.astype(np.float32)),
        "scale": _b64(scaler.scale_.astype(np.float32)),
        "coef": _b64(lr.coef_.astype(np.float32)),
        "intercept": lr.intercept_.astype(float).tolist(),
        "classes": lr.classes_.astype(int).tolist(),
    }
