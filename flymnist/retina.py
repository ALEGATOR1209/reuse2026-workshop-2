"""Project MNIST images onto the fly's eye.

In FlyWire coordinates every visual cell type of one eye (photoreceptors,
lamina cells, ...) lies on a nearly flat sheet, and all those sheets are
parallel (within ~4 degrees). For each eye we take one 2D frame (PCA of the
R7/R8 photoreceptor positions), project every input neuron into it, stretch
each cell type to the unit square and give each neuron the pixel intensity at
its position. Both eyes see the same image.

Pixel intensity sets the neuron's Poisson firing rate, the same mechanism the
fly-brain model uses for optogenetic activation.
"""

import numpy as np
import torch
import torch.nn.functional as F

INPUT_LAYERS = {
    # photoreceptors only: most faithful, but the signal fades in the optic lobe
    "photoreceptors": ["R1-6", "R7", "R8"],
    # photoreceptors + lamina monopolar cells (the lamina's output to the medulla)
    "lamina": ["R1-6", "R7", "R8", "L1", "L2", "L3", "L4", "L5"],
}


def _xyz(ann, rows):
    return ann.loc[rows, ["pos_x", "pos_y", "pos_z"]].to_numpy(float)


def build_retina(ann, layer="photoreceptors"):
    """Return (input_idx, uv): simulation indices of input neurons and their image coords in [0,1]^2."""
    types = INPUT_LAYERS[layer]
    idx_all, uv_all = [], []
    for side in ["left", "right"]:
        side_mask = (ann.side == side).to_numpy()
        ref = _xyz(ann, np.flatnonzero(ann.cell_type.isin(["R7", "R8"]).to_numpy() & side_mask))
        _, _, vt = np.linalg.svd(ref - ref.mean(0), full_matrices=False)
        basis = vt[:2] * np.sign(vt[:2, np.abs(vt[:2]).argmax(1)].diagonal())[:, None]
        for t in types:
            rows = np.flatnonzero((ann.cell_type == t).to_numpy() & side_mask)
            uv = (_xyz(ann, rows) - ref.mean(0)) @ basis.T
            lo, hi = np.percentile(uv, 1, 0), np.percentile(uv, 99, 0)
            idx_all.append(rows)
            uv_all.append(np.clip((uv - lo) / (hi - lo), 0, 1))
    return np.concatenate(idx_all), np.concatenate(uv_all).astype(np.float32)


def images_to_rates(images, uv, max_rate_hz):
    """images (B, 28, 28) in [0, 1] -> firing rates (B, n_inputs) in Hz."""
    img = torch.as_tensor(images).unsqueeze(1)                      # (B,1,H,W)
    grid = torch.as_tensor(uv * 2 - 1).view(1, 1, -1, 2)            # grid_sample wants [-1, 1]
    grid = grid.expand(img.shape[0], -1, -1, -1)
    vals = F.grid_sample(img, grid, mode="bilinear", align_corners=True)  # (B,1,1,n)
    return vals.view(img.shape[0], -1) * max_rate_hz
