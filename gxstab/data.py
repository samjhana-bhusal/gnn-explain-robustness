"""
Dataset registry.

Every dataset is normalised to a single PyG ``Data`` object with an undirected,
coalesced ``edge_index``, a boolean ``train/val/test`` split, and integer labels.
Heterophilous graphs ship with 10 pre-defined splits; we take split 0 by default
and expose the rest for seed sweeps.

Deliberately excluded from the default path: Elliptic Bitcoin (153 MB download,
~1 GB on disk). It is available behind ``include_elliptic=True`` with an explicit
opt-in, per the plan's day-11 go/no-go.
"""

import os

import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, HeterophilousGraphDataset
from torch_geometric.utils import to_undirected, remove_self_loops, coalesce

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

# name -> (loader-family, pyg-name)
CITATION = {
    "cora": "Cora",
    "citeseer": "CiteSeer",
    "pubmed": "PubMed",
}
HETEROPHILOUS = {
    "roman-empire": "Roman-empire",
    "amazon-ratings": "Amazon-ratings",
    "minesweeper": "Minesweeper",
    "tolokers": "Tolokers",
    "questions": "Questions",
}

# Rough homophily class for reporting; measured exactly by ``edge_homophily``.
HOMOPHILY_HINT = {
    "cora": "high", "citeseer": "high", "pubmed": "high",
    "roman-empire": "low", "amazon-ratings": "low", "minesweeper": "low",
    "tolokers": "low", "questions": "low",
}


def _canonicalize(data):
    """Undirected, no self-loops, coalesced. Idempotent."""
    ei = data.edge_index
    ei, _ = remove_self_loops(ei)
    ei = to_undirected(ei, num_nodes=data.num_nodes)
    ei = coalesce(ei, num_nodes=data.num_nodes)
    data.edge_index = ei
    return data


def _pick_split(data, split_idx: int):
    """Heterophilous datasets carry [N, 10] mask matrices; select one column."""
    for attr in ("train_mask", "val_mask", "test_mask"):
        m = getattr(data, attr)
        if m.dim() == 2:
            setattr(data, attr, m[:, split_idx].clone())
    return data


def edge_homophily(data) -> float:
    """Fraction of edges whose endpoints share a label (Zhu et al. 2020)."""
    u, v = data.edge_index
    return float((data.y[u] == data.y[v]).float().mean())


def load_dataset(name: str, split_idx: int = 0, root: str = DATA_ROOT):
    """
    Returns (data, meta) where meta has num_features, num_classes, homophily.

    ``split_idx`` selects one of the 10 heterophilous splits (ignored for
    citation graphs, which have a single canonical public split).
    """
    name = name.lower()
    if name in CITATION:
        ds = Planetoid(root=root, name=CITATION[name], transform=T.NormalizeFeatures())
        data = _canonicalize(ds[0])
    elif name in HETEROPHILOUS:
        ds = HeterophilousGraphDataset(root=root, name=HETEROPHILOUS[name])
        data = _canonicalize(ds[0])
        data = _pick_split(data, split_idx)
    else:
        raise KeyError(
            f"Unknown dataset {name!r}. Known: "
            f"{sorted(CITATION) + sorted(HETEROPHILOUS)}"
        )

    meta = {
        "name": name,
        "num_features": ds.num_features,
        "num_classes": ds.num_classes,
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.edge_index.size(1)),
        "homophily": edge_homophily(data),
        "homophily_hint": HOMOPHILY_HINT.get(name, "unknown"),
        "split_idx": split_idx,
    }
    return data, meta


def available_datasets(include_elliptic: bool = False):
    names = sorted(CITATION) + sorted(HETEROPHILOUS)
    if include_elliptic:
        names.append("elliptic")
    return names
