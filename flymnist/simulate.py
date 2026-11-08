"""Show MNIST digits to the fly brain and record spike counts of every neuron."""

import time

import numpy as np
import scipy.sparse as sp
import torch

from .brain import FlyBrain
from .data import DATA_DIR, load_annotations, load_connectome, load_mnist
from .retina import build_retina, images_to_rates


def feature_path(layer, rate_hz, duration_ms, shuffle, split=False):
    tag = ("_shuffled" if shuffle else "") + ("_split" if split else "")
    return DATA_DIR / f"spikes_{layer}_{int(rate_hz)}Hz_{int(duration_ms)}ms{tag}.npz"


def shuffle_connectome(pre, post, weight, seed=0):
    """Control: keep every neuron's outputs (count, sign, strength) but send them to random targets."""
    rng = np.random.default_rng(seed)
    return pre, rng.permutation(post), weight


def _simulate(brain, images, uv, rate_hz, duration_ms, batch, gen, label):
    chunks, t0 = [], time.time()
    for i in range(0, len(images), batch):
        rates = images_to_rates(images[i:i + batch], uv, rate_hz)
        counts = brain.run(rates, duration_ms, generator=gen)
        chunks.append(sp.csr_matrix(counts.clamp(max=255).numpy().astype(np.uint8)))
        done = min(i + batch, len(images))
        el = time.time() - t0
        print(f"  {label}: {done}/{len(images)} images, {el:.0f}s elapsed, ~{el / done * (len(images) - done):.0f}s left", flush=True)
    return sp.vstack(chunks).tocsr()


def run(n_train, n_test, layer="lamina", rate_hz=500.0, duration_ms=100.0, batch=32, shuffle=False, seed=0, split=False):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)  # same seed -> same input spikes as the intact run
    pre, post, weight, root_ids = load_connectome()
    if shuffle:
        pre, post, weight = shuffle_connectome(pre, post, weight, seed)
    ann = load_annotations(root_ids)
    input_idx, uv = build_retina(ann, layer)
    brain = FlyBrain(pre, post, weight, len(root_ids), input_idx)
    if split:
        from .lesion import cross_hemisphere_mask
        print(f"split brain: removed {brain.set_lesion(cut_synapses=cross_hemisphere_mask(ann, pre, post)):,} left<->right connections")
    print(f"brain: {len(root_ids)} neurons, {len(pre)} connections; {len(input_idx)} input neurons ({layer}); "
          f"{duration_ms:.0f} ms per image at up to {rate_hz:.0f} Hz{'; SHUFFLED wiring' if shuffle else ''}")

    xtr, ytr, xte, yte = load_mnist()
    X_train = _simulate(brain, xtr[:n_train], uv, rate_hz, duration_ms, batch, gen, "train")
    X_test = _simulate(brain, xte[:n_test], uv, rate_hz, duration_ms, batch, gen, "test")

    out = feature_path(layer, rate_hz, duration_ms, shuffle, split)
    np.savez_compressed(
        out,
        Xtr_data=X_train.data, Xtr_indices=X_train.indices, Xtr_indptr=X_train.indptr, Xtr_shape=X_train.shape,
        Xte_data=X_test.data, Xte_indices=X_test.indices, Xte_indptr=X_test.indptr, Xte_shape=X_test.shape,
        y_train=ytr[:n_train], y_test=yte[:n_test], input_idx=input_idx,
        n_train=n_train, n_test=n_test,
    )
    print("saved", out)
    return out


def load_features(path):
    z = np.load(path)
    m = lambda k: sp.csr_matrix((z[f"{k}_data"], z[f"{k}_indices"], z[f"{k}_indptr"]), shape=tuple(z[f"{k}_shape"]))
    return m("Xtr"), z["y_train"], m("Xte"), z["y_test"], z["input_idx"]
