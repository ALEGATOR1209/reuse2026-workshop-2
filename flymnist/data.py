"""Download and load the FlyWire connectome, its cell annotations, and MNIST."""

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Pinned to a fixed commit of eonsystemspbc/fly-brain so results are reproducible.
_FLY_BRAIN = "https://raw.githubusercontent.com/eonsystemspbc/fly-brain/a3db62f9436074e485c0278290c2164ed6150808/data"
FILES = {
    "2025_Completeness_783.csv": f"{_FLY_BRAIN}/2025_Completeness_783.csv",
    "2025_Connectivity_783.parquet": f"{_FLY_BRAIN}/2025_Connectivity_783.parquet",
    # Cell types / super classes for every FlyWire neuron (Schlegel et al. 2024).
    "neuron_annotations.tsv": "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/supplemental_files/Supplemental_file1_neuron_annotations.tsv",
}


def download():
    DATA_DIR.mkdir(exist_ok=True)
    for name, url in FILES.items():
        path = DATA_DIR / name
        if path.exists():
            print(f"ok      {name}")
            continue
        print(f"fetch   {name} ...")
        urllib.request.urlretrieve(url, path)
    load_mnist()  # triggers torchvision download
    print("data ready in", DATA_DIR)


def load_connectome():
    """Return (pre, post, weight, root_ids).

    weight = synapse count x sign (+1 excitatory / -1 inhibitory), exactly the
    'Excitatory x Connectivity' column used by the fly-brain model.
    Neuron i in the simulation is root_ids[i].
    """
    root_ids = pd.read_csv(DATA_DIR / "2025_Completeness_783.csv", index_col=0).index.to_numpy()
    con = pd.read_parquet(
        DATA_DIR / "2025_Connectivity_783.parquet",
        columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"],
    )
    return (
        con["Presynaptic_Index"].to_numpy(np.int64),
        con["Postsynaptic_Index"].to_numpy(np.int64),
        con["Excitatory x Connectivity"].to_numpy(np.float32),
        root_ids,
    )


def load_annotations(root_ids):
    """Annotation table aligned to simulation order (one row per neuron)."""
    ann = pd.read_csv(
        DATA_DIR / "neuron_annotations.tsv", sep="\t", low_memory=False,
        usecols=["root_id", "pos_x", "pos_y", "pos_z", "super_class", "cell_class", "cell_type", "side"],
    ).drop_duplicates("root_id").set_index("root_id")
    return ann.reindex(root_ids).reset_index()


def load_mnist():
    """Return (x_train, y_train, x_test, y_test); images are float32 in [0, 1], shape (n, 28, 28)."""
    from torchvision.datasets import MNIST

    root = DATA_DIR / "mnist"
    tr = MNIST(root, train=True, download=True)
    te = MNIST(root, train=False, download=True)
    f = lambda d: d.data.numpy().astype(np.float32) / 255.0
    return f(tr), tr.targets.numpy(), f(te), te.targets.numpy()
