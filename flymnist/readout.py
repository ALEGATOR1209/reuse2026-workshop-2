"""Train linear readouts on the fly brain's spike counts.

The connectome is never trained. For each brain region we fit a logistic
regression from that region's spike counts to the digit label. The input
neurons we stimulated are excluded from every region (except "eye input"), so
a readout can only succeed if the information travelled through synapses.
"""

import json

import joblib
import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import load_annotations, load_connectome, load_mnist
from .simulate import load_features

REGIONS = {
    # name: annotation super_class(es); None = whole brain
    "optic lobe": ["optic"],
    "visual projection (optic lobe -> central brain)": ["visual_projection"],
    "central brain": ["central"],
    "descending (brain -> body)": ["descending"],
    "whole brain": None,
}


def _fit_eval(Xtr, ytr, Xte, yte, Cs=(0.001, 0.01, 0.1)):
    """Pick C (L2 strength) by validation loss, refit on all training data.

    Returns a metrics dict (accuracy, cross-entropy loss on train/val/test) and the model.
    """
    a, b, ya, yb = train_test_split(Xtr, ytr, test_size=0.2, random_state=0, stratify=ytr)
    make = lambda C, it: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=it))
    val_loss = {C: log_loss(yb, make(C, 500).fit(a, ya).predict_proba(b), labels=range(10)) for C in Cs}
    best = min(val_loss, key=val_loss.get)
    model = make(best, 1000).fit(Xtr, ytr)
    loss = lambda X, y: log_loss(y, model.predict_proba(X), labels=range(10))
    m = {"accuracy": model.score(Xte, yte), "train_acc": model.score(Xtr, ytr),
         "train_loss": loss(Xtr, ytr), "val_loss": val_loss[best], "test_loss": loss(Xte, yte),
         "C": best, "val_loss_per_C": val_loss}
    return m, model


def spike_features(counts, cols):
    """Readout input: log(1 + spike count) of the chosen neurons. counts: sparse or dense (n, N)."""
    X = counts[:, cols]
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    return np.log1p(X.astype(np.float32))


def readouts_path(feature_file):
    return feature_file.with_suffix(".readouts.joblib")


def run(feature_file, out_json=None):
    Xtr, ytr, Xte, yte, input_idx = load_features(feature_file)
    _, _, _, root_ids = load_connectome()
    super_class = load_annotations(root_ids).super_class.fillna("unknown").to_numpy()
    n_train, n_test = len(ytr), len(yte)

    is_input = np.zeros(len(root_ids), bool)
    is_input[input_idx] = True
    active = np.asarray((Xtr > 0).sum(0)).ravel() > 0  # neurons that spiked for >=1 training image

    rows, models = [], {}
    xtr_px, _, xte_px, _ = load_mnist()
    m, model = _fit_eval(xtr_px[:n_train].reshape(n_train, -1), ytr, xte_px[:n_test].reshape(n_test, -1), yte)
    rows.append({"features": "BASELINE: raw pixels (no brain)", "neurons": 784, "active": 784, **m})
    models["pixels"] = {"name": rows[-1]["features"], "cols": None, "model": model, "accuracy": m["accuracy"]}

    def add(name, mask):
        cols = np.flatnonzero(mask & active)
        if len(cols) == 0:
            rows.append({"features": name, "neurons": int(mask.sum()), "active": 0, "accuracy": 0.1,
                         "train_acc": 0.1, "train_loss": np.log(10), "val_loss": np.log(10), "test_loss": np.log(10), "C": None})
            return
        m, model = _fit_eval(spike_features(Xtr, cols), ytr, spike_features(Xte, cols), yte)
        rows.append({"features": name, "neurons": int(mask.sum()), "active": len(cols), **m})
        models[name] = {"name": name, "cols": cols, "model": model, "accuracy": m["accuracy"]}
        print(f"  {name:50s} acc {m['accuracy']:.3f}  test loss {m['test_loss']:.3f}", flush=True)

    add("eye input (stimulated neurons, noisy pixels)", is_input)
    for name, classes in REGIONS.items():
        region = np.ones(len(root_ids), bool) if classes is None else np.isin(super_class, classes)
        add(name, region & ~is_input)

    print(f"\nTrained on {n_train} digits, tested on {n_test} digits ({feature_file.name})")
    print("loss = cross-entropy (chance level = ln 10 = 2.303)")
    print(f"{'readout from':50s} {'active':>7s} {'train acc':>9s} {'test acc':>8s} {'train loss':>10s} {'val loss':>8s} {'test loss':>9s} {'C':>6s}")
    for r in rows:
        print(f"{r['features']:50s} {r['active']:7d} {r['train_acc']:9.3f} {r['accuracy']:8.3f} "
              f"{r['train_loss']:10.3f} {r['val_loss']:8.3f} {r['test_loss']:9.3f} {str(r['C']):>6s}")
    joblib.dump({"feature_file": feature_file.name, "n_train": n_train, "readouts": models}, readouts_path(feature_file))
    print("saved", readouts_path(feature_file))
    if out_json:
        out_json.write_text(json.dumps({"feature_file": feature_file.name, "n_train": n_train, "n_test": n_test, "results": rows}, indent=2), encoding="utf-8")
        print("saved", out_json)
    return rows


# ---------------------------------------------------------------------------
# Per-hemisphere readouts (left-only / right-only / both), saved to separate files
# so the standard readouts used by the web app are never overwritten.
# ---------------------------------------------------------------------------
HEMI_REGIONS = {"whole brain": None, "optic lobe": ["optic"], "central brain": ["central"],
                "descending (brain -> body)": ["descending"]}


def hemispheres_path(feature_file):
    return feature_file.with_suffix(".hemispheres.joblib")


def run_hemispheres(feature_file):
    """Train one readout per (region, side) on the given simulation; inputs excluded as usual."""
    Xtr, ytr, Xte, yte, input_idx = load_features(feature_file)
    _, _, _, root_ids = load_connectome()
    ann = load_annotations(root_ids)
    super_class = ann.super_class.fillna("unknown").to_numpy()
    side = ann.side.to_numpy()
    is_input = np.zeros(len(root_ids), bool)
    is_input[input_idx] = True
    active = np.asarray((Xtr > 0).sum(0)).ravel() > 0

    rows, models = [], {}
    for region, classes in HEMI_REGIONS.items():
        in_region = np.ones(len(root_ids), bool) if classes is None else np.isin(super_class, classes)
        for s in ["left", "right", "both"]:
            mask = in_region & ~is_input & ((side == s) if s != "both" else True)
            cols = np.flatnonzero(mask & active)
            if len(cols) == 0:
                rows.append({"region": region, "side": s, "active": 0, "accuracy": 0.1, "test_loss": float(np.log(10))})
                continue
            m, model = _fit_eval(spike_features(Xtr, cols), ytr, spike_features(Xte, cols), yte)
            rows.append({"region": region, "side": s, "active": len(cols), **m})
            models[f"{region} | {s}"] = {"region": region, "side": s, "cols": cols, "model": model, "accuracy": m["accuracy"]}
            print(f"  {region:28s} {s:5s} active {len(cols):6d}  test acc {m['accuracy']:.3f}  test loss {m['test_loss']:.3f}", flush=True)

    out = hemispheres_path(feature_file)
    joblib.dump({"feature_file": feature_file.name, "n_train": len(ytr), "readouts": models}, out)
    out.with_suffix(".json").write_text(json.dumps({"feature_file": feature_file.name, "n_train": len(ytr),
                                                    "n_test": len(yte), "results": rows}, indent=2, default=float), encoding="utf-8")
    print("saved", out, "and", out.with_suffix(".json"))
    return rows
