"""
Novel extensions for the GNN-XAI robustness benchmark.

This module implements the two research contributions described in the report:

  1. Homophily Attribution Index (HAI) -- a metric quantifying what fraction of an
     explanation's attribution weight lands on homophilous (same-label) edges, and
     how that fraction shifts under adversarial topological perturbation.

  2. XAI-Guided Topological Defense -- a sparsifier that uses explainer edge
     attribution scores to prune injected adversarial edges from a node's
     neighborhood, restoring the model's prediction.

Both are kept deliberately explainer-agnostic where possible so they can be reused
across the three paradigms benchmarked in evaluate.py.
"""

import numpy as np
import torch

from explainers import get_local_neighborhood, run_gnn_explainer


# --------------------------------------------------------------------------- #
# Extension 1: Homophily Attribution Index (HAI)
# --------------------------------------------------------------------------- #
def homophily_attribution_index(edges, y, weights=None):
    """
    Fraction of an explanation's attribution weight that connects same-label nodes.

    Args:
        edges:   list of (u, v) global node-id tuples (the explanatory edges).
        y:       node-label tensor for the whole graph.
        weights: optional per-edge attribution weights aligned with ``edges``.
                 When ``None`` (e.g. binary top-k masks) every edge is weighted
                 uniformly, so HAI reduces to the same-label edge fraction.

    Returns:
        HAI in [0, 1], or ``None`` when there are no explanatory edges (an
        undefined index we do not want to silently report as 0).
    """
    if not edges:
        return None

    labels = y.cpu().numpy() if torch.is_tensor(y) else np.asarray(y)

    total = 0.0
    same = 0.0
    for i, (u, v) in enumerate(edges):
        w = 1.0 if weights is None else float(weights[i])
        total += w
        if labels[u] == labels[v]:
            same += w

    return (same / total) if total > 0 else None


def hai_shift(hai_clean, hai_perturbed):
    """
    Signed change in homophily focus, HAI(perturbed) - HAI(clean).

    Negative values mean the explanation was pulled toward heterophilous
    (cross-class) edges by the attack -- i.e. the explainer was distracted.
    Returns ``None`` if either operand is undefined.
    """
    if hai_clean is None or hai_perturbed is None:
        return None
    return hai_perturbed - hai_clean


# --------------------------------------------------------------------------- #
# Extension 2: XAI-Guided Topological Defense
# --------------------------------------------------------------------------- #
def _target_class_prob(model, x, edge_index, node_idx, target_class):
    with torch.no_grad():
        out = model(x, edge_index)
        prob = torch.exp(out[node_idx])[target_class].item()
        pred = out[node_idx].argmax().item()
    return prob, pred


def xai_guided_defense(
    model,
    x,
    edge_index,
    node_idx,
    target_class,
    num_classes,
    keep_percentile=50.0,
):
    """
    Defend a target node by pruning low-attribution edges in its neighborhood.

    The attacker has injected edges into ``node_idx``'s 2-hop neighborhood
    (``edge_index`` is the *perturbed* graph). We run GNNExplainer on the
    perturbed graph to score every edge, then keep only the higher-scoring edges
    inside the neighborhood while leaving the rest of the graph untouched. The
    intuition: genuine, prediction-supporting edges receive high attribution,
    whereas randomly injected adversarial edges do not, so thresholding the
    attribution mask acts as a targeted denoiser.

    Args:
        edge_index:      the perturbed (attacked) edge index.
        keep_percentile: keep neighborhood edges whose attribution is at or above
                         this percentile of neighborhood-edge scores (50 -> keep
                         the top half). Edges outside the neighborhood are always
                         kept so global context is preserved.

    Returns:
        dict with the attacked and defended target-class probabilities and
        predictions, plus how many neighborhood edges were pruned.
    """
    model.eval()

    # Baseline: prediction on the attacked graph before any defense.
    attacked_prob, attacked_pred = _target_class_prob(
        model, x, edge_index, node_idx, target_class
    )

    # Score every edge of the perturbed graph.
    mask = run_gnn_explainer(model, x, edge_index, node_idx, num_classes)

    # Identify which edges fall inside the target's 2-hop neighborhood.
    subset, _, _, _, _ = get_local_neighborhood(
        node_idx, 2, edge_index, x, torch.zeros(x.size(0), dtype=torch.long)
    )
    neighborhood = set(subset.cpu().numpy().tolist())

    u_edges = edge_index[0].cpu().numpy()
    v_edges = edge_index[1].cpu().numpy()

    local_edge_ids = [
        i for i in range(edge_index.size(1))
        if u_edges[i] in neighborhood and v_edges[i] in neighborhood
    ]

    # Degenerate case: nothing local to prune, defense is a no-op.
    if not local_edge_ids:
        return {
            "attacked_prob": attacked_prob,
            "attacked_pred": attacked_pred,
            "defended_prob": attacked_prob,
            "defended_pred": attacked_pred,
            "pruned_edges": 0,
            "neighborhood_edges": 0,
        }

    local_scores = mask[local_edge_ids]
    tau = np.percentile(local_scores, keep_percentile)

    keep_ids = []
    pruned = 0
    for i in range(edge_index.size(1)):
        in_neighborhood = u_edges[i] in neighborhood and v_edges[i] in neighborhood
        if in_neighborhood:
            if mask[i] >= tau:
                keep_ids.append(i)
            else:
                pruned += 1
        else:
            # Preserve everything outside the attacked neighborhood.
            keep_ids.append(i)

    defended_edge_index = edge_index[:, keep_ids]
    defended_prob, defended_pred = _target_class_prob(
        model, x, defended_edge_index, node_idx, target_class
    )

    return {
        "attacked_prob": attacked_prob,
        "attacked_pred": attacked_pred,
        "defended_prob": defended_prob,
        "defended_pred": defended_pred,
        "pruned_edges": int(pruned),
        "neighborhood_edges": len(local_edge_ids),
    }
