"""Shared harness: model caching, target sampling, node-level records."""

from __future__ import annotations

import os

import numpy as np
import torch

from gxstab.data import load_dataset
from gxstab.models import GNN, train_model
from gxstab.utils import set_seed, get_device

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "results")
CKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "ckpts")


def get_trained_model(dataset, backbone, seed, *, hidden=64, epochs=200,
                      device="cpu", split_idx=0, force=False):
    """Train (or load cached) a backbone on a dataset+seed. Returns (model, data, meta, acc)."""
    os.makedirs(CKPT_DIR, exist_ok=True)
    tag = f"{dataset}_{backbone}_s{seed}_sp{split_idx}"
    path = os.path.join(CKPT_DIR, tag + ".pt")

    set_seed(seed)
    data, meta = load_dataset(dataset, split_idx=split_idx)
    dev = get_device(device)
    data = data.to(dev)
    model = GNN(meta["num_features"], hidden, meta["num_classes"], backbone, K=2).to(dev)

    if os.path.exists(path) and not force:
        blob = torch.load(path, map_location=dev)
        model.load_state_dict(blob["state_dict"])
        acc = blob["acc"]
    else:
        acc = train_model(model, data, epochs=epochs)
        torch.save({"state_dict": model.state_dict(), "acc": acc}, path)
    model.eval()
    return model, data, meta, acc


def sample_targets(model, data, meta, *, n=60, conf_min=0.6, deg_max=None,
                   stratify_degree=True, seed=0):
    """
    Confident, correctly-classified test nodes as explanation targets. Optionally
    capped by degree and stratified across the degree range so results are not
    dominated by one regime.
    """
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        probs = torch.exp(logits)
        pred = logits.argmax(-1)
    conf = probs.max(-1).values
    deg = torch.zeros(meta["num_nodes"], device=data.x.device)
    idx0 = data.edge_index[0]
    deg.scatter_add_(0, idx0, torch.ones_like(idx0, dtype=deg.dtype))

    base = (pred == data.y) & data.test_mask
    if deg_max is not None:
        base = base & (deg <= deg_max)
    mask = base & (conf > conf_min)
    cand = torch.where(mask)[0].cpu().numpy()

    # Fallback: some backbones (e.g. linear SGC) rarely exceed an absolute
    # confidence bar. Take the most-confident correct test nodes instead so the
    # target set is comparable across backbones.
    if len(cand) < n:
        pool = torch.where(base)[0]
        if len(pool) > 0:
            order = pool[torch.argsort(conf[pool], descending=True)]
            cand = order[:max(n, len(cand))].cpu().numpy()
    if len(cand) == 0:
        return []

    if stratify_degree and len(cand) > n:
        d = deg.cpu().numpy()[cand]
        order = np.argsort(d)
        picks = order[np.linspace(0, len(order) - 1, n).astype(int)]
        cand = cand[np.unique(picks)]
    else:
        rng = np.random.default_rng(seed)
        cand = rng.choice(cand, size=min(n, len(cand)), replace=False)
    return [int(v) for v in cand]


def node_confidence(model, data, v):
    with torch.no_grad():
        p = torch.exp(model(data.x, data.edge_index)[v])
    c = int(p.argmax())
    return c, float(p[c])
