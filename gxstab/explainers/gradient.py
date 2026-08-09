"""
Cheap, certifiable edge explainers.

These are the base explainers we smooth: each needs only a handful of forward/
backward passes, so wrapping them in n=200-1000 Monte-Carlo smoothing samples per
node stays tractable on a laptop.

  grad       -- |d logit_c / d (edge gate)| via a differentiable edge weight.
  ig         -- integrated gradients of the edge gate from 0 to 1.
  occlusion  -- prediction drop when each surface edge is individually removed.
"""

from __future__ import annotations

import numpy as np
import torch

from gxstab.surface import surface_ball, in_surface


def _edge_gate_forward(model, x, edge_index, gate):
    """
    Run the model with a per-edge multiplicative gate. Works for any PyG conv
    that accepts ``edge_weight``; for others we fall back to gating messages via
    a masked edge_index at gate in {0,1}. Here we use edge_weight where possible.
    """
    try:
        out = model(x, edge_index, edge_weight=gate)
    except TypeError:
        out = model(x, edge_index)
    return out


def _supports_edge_weight(model):
    return model.model_type in ("GCN", "SGC", "APPNP")


def grad_edge_scores(model, x, edge_index, node_idx, *, K=2, num_classes=None):
    model.eval()
    E = edge_index.size(1)
    num_nodes = x.size(0)

    if _supports_edge_weight(model):
        gate = torch.ones(E, device=x.device, requires_grad=True)
        out = _edge_gate_forward(model, x, edge_index, gate)
        c = int(out[node_idx].argmax())
        out[node_idx, c].backward()
        scores = gate.grad.detach().abs().cpu().numpy()
    else:
        # Gradient wrt inputs is ill-defined per edge for GAT/SAGE; use occlusion.
        return occlusion_edge_scores(
            model, x, edge_index, node_idx, K=K, num_classes=num_classes
        )
    return _restrict(scores, edge_index, node_idx, K, num_nodes)


def integrated_gradient_edge_scores(
    model, x, edge_index, node_idx, *, K=2, num_classes=None, steps=20
):
    model.eval()
    E = edge_index.size(1)
    num_nodes = x.size(0)
    if not _supports_edge_weight(model):
        return occlusion_edge_scores(
            model, x, edge_index, node_idx, K=K, num_classes=num_classes
        )

    total = torch.zeros(E, device=x.device)
    # baseline gate = 0 (all edges off), target gate = 1.
    with torch.no_grad():
        c = int(model(x, edge_index).argmax(dim=-1)[node_idx])
    for s in range(1, steps + 1):
        alpha = s / steps
        gate = torch.full((E,), alpha, device=x.device, requires_grad=True)
        out = _edge_gate_forward(model, x, edge_index, gate)
        model.zero_grad(set_to_none=True)
        out[node_idx, c].backward()
        total += gate.grad.detach()
    scores = (total / steps).abs().cpu().numpy()  # gate diff (1-0)=1 factor
    return _restrict(scores, edge_index, node_idx, K, num_nodes)


def occlusion_edge_scores(model, x, edge_index, node_idx, *, K=2, num_classes=None):
    model.eval()
    num_nodes = x.size(0)
    ball = surface_ball(node_idx, K, edge_index, num_nodes)
    u = edge_index[0].tolist()
    w = edge_index[1].tolist()

    with torch.no_grad():
        base = model(x, edge_index)
        c = int(base[node_idx].argmax())
        base_p = float(torch.exp(base[node_idx, c]))

    # Group directed edges into undirected surface pairs; occlude both directions.
    pair_to_dirs = {}
    for i in range(len(u)):
        if not in_surface(u[i], w[i], ball):
            continue
        e = (min(u[i], w[i]), max(u[i], w[i]))
        pair_to_dirs.setdefault(e, []).append(i)

    scores = np.zeros(edge_index.size(1))
    keep_all = np.ones(edge_index.size(1), dtype=bool)
    for e, dirs in pair_to_dirs.items():
        mask = keep_all.copy()
        for d in dirs:
            mask[d] = False
        with torch.no_grad():
            out = model(x, edge_index[:, mask])
            p = float(torch.exp(out[node_idx, c]))
        drop = max(base_p - p, 0.0)
        for d in dirs:
            scores[d] = drop
    return scores


def _restrict(scores, edge_index, node_idx, K, num_nodes):
    ball = surface_ball(node_idx, K, edge_index, num_nodes)
    u = edge_index[0].tolist()
    w = edge_index[1].tolist()
    out = np.asarray(scores, dtype=float).copy()
    for i in range(len(u)):
        if not in_surface(u[i], w[i], ball):
            out[i] = 0.0
    return out
