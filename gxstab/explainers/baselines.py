"""
Parameter-free edge-importance baselines.

If a learned explainer cannot beat these on faithfulness *and* stability, the
learning is not buying anything -- so every results table reports them. All are
restricted to the target's attack surface (Lemma 1).

  random   -- uniform noise (the chance floor).
  degree   -- inverse endpoint degree; low-degree edges carry sharper signal.
  cosine   -- feature cosine similarity of the endpoints (homophily proxy).
  homorank -- same-label endpoints ranked above cross-label (label-aware proxy).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from gxstab.surface import surface_ball, in_surface


def _surface_mask(edge_index, node_idx, K, num_nodes):
    ball = surface_ball(node_idx, K, edge_index, num_nodes)
    u = edge_index[0].tolist()
    w = edge_index[1].tolist()
    return np.array([in_surface(u[i], w[i], ball) for i in range(len(u))], dtype=bool), u, w


def random_edge_scores(model, x, edge_index, node_idx, *, K=2, num_classes=None):
    m, _, _ = _surface_mask(edge_index, node_idx, K, x.size(0))
    rng = np.random.default_rng()
    s = rng.random(edge_index.size(1))
    s[~m] = 0.0
    return s


def degree_edge_scores(model, x, edge_index, node_idx, *, K=2, num_classes=None):
    N = x.size(0)
    deg = np.zeros(N)
    for a in edge_index[0].tolist():
        deg[a] += 1
    m, u, w = _surface_mask(edge_index, node_idx, K, N)
    s = np.zeros(edge_index.size(1))
    for i in range(len(u)):
        if m[i]:
            d = max(deg[u[i]], 1) * max(deg[w[i]], 1)
            s[i] = 1.0 / np.sqrt(d)
    return s


def cosine_edge_scores(model, x, edge_index, node_idx, *, K=2, num_classes=None):
    m, u, w = _surface_mask(edge_index, node_idx, K, x.size(0))
    xn = F.normalize(x, dim=1)
    s = np.zeros(edge_index.size(1))
    idx = np.where(m)[0]
    if len(idx):
        uu = torch.tensor([u[i] for i in idx])
        vv = torch.tensor([w[i] for i in idx])
        cos = (xn[uu] * xn[vv]).sum(dim=1).detach().cpu().numpy()
        s[idx] = np.clip(cos, 0.0, None)
    return s


def homorank_edge_scores(model, x, edge_index, node_idx, *, K=2, num_classes=None):
    """Uses the model's *predicted* labels (no ground-truth leakage)."""
    with torch.no_grad():
        pred = model(x, edge_index).argmax(dim=-1).cpu().numpy()
    m, u, w = _surface_mask(edge_index, node_idx, K, x.size(0))
    s = np.zeros(edge_index.size(1))
    for i in range(len(u)):
        if m[i]:
            s[i] = 1.0 if pred[u[i]] == pred[w[i]] else 0.25
    return s
