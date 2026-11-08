"""Pack everything a browser needs to run its own copy of the fly brain into one file.

Layout: uint32 header length, a JSON header (padded to 8 bytes), then the arrays the
header lists (name, dtype, offset, length), each 8-byte aligned. Connections are grouped
by presynaptic neuron (CSR, sorted by target within a row) and the targets are stored as
differences from the previous target in the row, which compresses ~25% better.
The file is gzipped once and served as-is; browsers unzip it themselves.
"""

import gzip
import hashlib
import json

import numpy as np

from .brain import PARAMS
from .data import DATA_DIR

WEB_DIR = DATA_DIR / "web"
FORMAT_VERSION = 1


def _pack(arrays, meta):
    sections, blobs, offset = [], [], 0
    for name, a in arrays.items():
        a = np.ascontiguousarray(a)
        sections.append({"name": name, "dtype": a.dtype.name, "offset": offset, "length": int(a.size)})
        b = a.tobytes()
        b += b"\0" * (-len(b) % 8)
        blobs.append(b)
        offset += len(b)
    header = json.dumps({**meta, "sections": sections}).encode()
    header += b" " * (-(len(header) + 4) % 8)
    return np.uint32(len(header)).tobytes() + header + b"".join(blobs)


def brain_file(pre, post, weight, n, input_idx, uv, side, region, cell_type, tag):
    """Return (path, version) of the gzipped brain file, building it on first use (~30 s)."""
    path = WEB_DIR / f"brain_{tag}.bin.gz"
    if path.exists():
        return path, hashlib.sha1(path.read_bytes()[:1 << 20]).hexdigest()[:12]
    WEB_DIR.mkdir(parents=True, exist_ok=True)
    order = np.lexsort((post, pre))
    ps, po, w = pre[order], post[order].astype(np.int64), weight[order]
    indptr = np.r_[0, np.bincount(ps, minlength=n).cumsum()].astype(np.int32)
    delta = po.copy()
    delta[1:] -= po[:-1]
    starts = indptr[:-1][np.diff(indptr) > 0]
    delta[starts] = po[starts]                        # first target of each row is absolute

    names, type_idx = np.unique(np.asarray(cell_type, dtype=str), return_inverse=True)
    side_code = np.select([np.asarray(side) == "left", np.asarray(side) == "right"], [1, 2], 0).astype(np.uint8)
    arrays = {
        "indptr": indptr,
        "post_delta": delta.astype(np.int32),
        "weight": w.astype(np.int16),                  # signed synapse counts (-2405..1897)
        "input_idx": np.asarray(input_idx, np.int32),
        "uv": np.asarray(uv, np.float32).ravel(),      # image position of each input neuron, [0,1]^2
        "side": side_code,                             # 0 other, 1 left, 2 right
        "region": np.asarray(region, np.uint8),
        "cell_type": type_idx.astype(np.uint16),
    }
    meta = {"format": FORMAT_VERSION, "n": int(n), "synapses": int(len(pre)), "params": PARAMS, "dt": 0.1,
            "cell_type_names": names.tolist()}
    data = gzip.compress(_pack(arrays, meta), 6)
    path.write_bytes(data)
    return path, hashlib.sha1(data[:1 << 20]).hexdigest()[:12]
