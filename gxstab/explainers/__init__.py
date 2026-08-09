"""
Explainers behind one interface.

Every explainer exposes ``edge_scores(model, x, edge_index, node_idx, *, K,
num_classes) -> np.ndarray`` returning one non-negative importance score per
directed edge in ``edge_index`` (higher = more important for the target's
prediction). ``top_k_edges`` reduces a score vector to an undirected top-k set
restricted to the target's attack surface, which is the object we smooth and
certify.

Scores outside the target's K-hop attack surface are forced to zero: by Lemma 1
they cannot influence the prediction, so any nonzero score there is an artifact.
"""

from __future__ import annotations

import numpy as np

from gxstab.surface import surface_ball, in_surface

from .gradient import grad_edge_scores, integrated_gradient_edge_scores, occlusion_edge_scores
from .baselines import random_edge_scores, degree_edge_scores, cosine_edge_scores, homorank_edge_scores
from .legacy import gnnexplainer_edge_scores, subgraph_mcts_edge_scores

REGISTRY = {
    # cheap / certifiable
    "grad": grad_edge_scores,
    "ig": integrated_gradient_edge_scores,
    "occlusion": occlusion_edge_scores,
    # parameter-free baselines
    "random": random_edge_scores,
    "degree": degree_edge_scores,
    "cosine": cosine_edge_scores,
    "homorank": homorank_edge_scores,
    # expensive / fragility-comparison (migrated from prototype)
    "gnnexplainer": gnnexplainer_edge_scores,
    "mcts": subgraph_mcts_edge_scores,
}

CERTIFIABLE = ("grad", "ig", "occlusion", "degree", "cosine", "homorank")


def get_explainer(name: str):
    if name not in REGISTRY:
        raise KeyError(f"unknown explainer {name!r}. Known: {sorted(REGISTRY)}")
    return REGISTRY[name]


def mask_to_surface(scores, edge_index, node_idx, K, num_nodes):
    """Zero out scores for edges off the target's attack surface (Lemma 1)."""
    ball = surface_ball(node_idx, K, edge_index, num_nodes)
    u = edge_index[0].tolist()
    w = edge_index[1].tolist()
    out = np.asarray(scores, dtype=float).copy()
    for i in range(len(u)):
        if not in_surface(u[i], w[i], ball):
            out[i] = 0.0
    return out


def top_k_edges(scores, edge_index, node_idx, k, K, num_nodes, target_bias=True):
    """
    Reduce a directed score vector to <=k undirected (min,max) edges on the
    target's surface, highest score first. Ties broken toward target-incident
    edges when ``target_bias`` so dense binary masks still center on the target.
    """
    scores = mask_to_surface(scores, edge_index, node_idx, K, num_nodes)
    u = edge_index[0].tolist()
    w = edge_index[1].tolist()

    best = {}
    for i in range(len(u)):
        s = float(scores[i])
        if s <= 0.0:
            continue
        a, b = u[i], w[i]
        e = (min(a, b), max(a, b))
        incident = 1 if (target_bias and (a == node_idx or b == node_idx)) else 0
        key = (s, incident)
        if e not in best or key > best[e]:
            best[e] = key

    ordered = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
    return [e for e, _ in ordered[:k]]
