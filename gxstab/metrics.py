"""
Evaluation metrics.

Stability (how much an explanation moves under perturbation):
  * topk_jaccard          -- overlap of top-k undirected edge sets.
  * rank_correlation      -- Kendall-tau / Spearman on full surface score vectors
                             (captures reordering that Jaccard misses).

Faithfulness (does the explanation capture the model's logic):
  * fidelity              -- standard Fid-minus / Fid-plus (hard edge removal).
  * robust_fidelity       -- randomized/marginalized variant that avoids the
                             out-of-distribution artifact of hard removal
                             (Zheng et al., "Towards Robust Fidelity", 2023).

Well-posedness:
  * prediction_conditioned -- gate that keeps only targets whose argmax is
                             unchanged and whose confidence moved < delta, so
                             "the explanation changed" is not just tracking a
                             changed decision. This makes the fragility claim
                             well-defined (fixes the prototype's conflation).

Secondary analysis:
  * homophily_attribution_index / hai_shift -- migrated from the prototype.
"""

from __future__ import annotations

import numpy as np
import torch

from gxstab.surface import surface_ball, in_surface


# --------------------------------------------------------------------------- #
# Stability
# --------------------------------------------------------------------------- #
def topk_jaccard(edges_a, edges_b) -> float:
    a, b = set(map(_canon, edges_a)), set(map(_canon, edges_b))
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _canon(e):
    u, v = e
    return (min(u, v), max(u, v))


def rank_correlation(scores_a, scores_b, edge_index, node_idx, K, num_nodes):
    """
    Kendall-tau and Spearman between two directed score vectors, computed on the
    undirected surface edges only (off-surface entries are structurally zero and
    would inflate agreement). Returns (kendall, spearman); NaN if < 2 edges.
    """
    from scipy.stats import kendalltau, spearmanr

    ball = surface_ball(node_idx, K, edge_index, num_nodes)
    u = edge_index[0].tolist()
    w = edge_index[1].tolist()
    per_pair_a, per_pair_b = {}, {}
    for i in range(len(u)):
        if not in_surface(u[i], w[i], ball):
            continue
        e = (min(u[i], w[i]), max(u[i], w[i]))
        per_pair_a[e] = max(per_pair_a.get(e, 0.0), float(scores_a[i]))
        per_pair_b[e] = max(per_pair_b.get(e, 0.0), float(scores_b[i]))

    keys = sorted(set(per_pair_a) | set(per_pair_b))
    if len(keys) < 2:
        return float("nan"), float("nan")
    va = [per_pair_a.get(k, 0.0) for k in keys]
    vb = [per_pair_b.get(k, 0.0) for k in keys]
    kt = kendalltau(va, vb).correlation
    sp = spearmanr(va, vb).correlation
    return (float(kt) if kt == kt else 0.0), (float(sp) if sp == sp else 0.0)


# --------------------------------------------------------------------------- #
# Faithfulness
# --------------------------------------------------------------------------- #
def _target_prob(model, x, edge_index, node_idx, c):
    with torch.no_grad():
        out = model(x, edge_index)
    return float(torch.exp(out[node_idx, c]))


def _remove_pairs(edge_index, pairs):
    drop = set()
    for a, b in pairs:
        drop.add((a, b))
        drop.add((b, a))
    keep = [
        i for i in range(edge_index.size(1))
        if (int(edge_index[0, i]), int(edge_index[1, i])) not in drop
    ]
    return edge_index[:, keep]


def fidelity(model, x, edge_index, node_idx, explanation_edges, node_pred_class=None):
    """
    Standard hard fidelity on the target's surface.

      Fid- = p(c|G) - p(c|G \\ E_exp)     [necessity; higher is better]
      Fid+ = p(c|G) - p(c|G_exp)          [sufficiency; lower is better]

    G_exp keeps only explanation edges *within the surface* and all edges
    outside the surface (which by Lemma 1 cannot affect the target anyway).
    """
    with torch.no_grad():
        c = node_pred_class if node_pred_class is not None else int(model(x, edge_index)[node_idx].argmax())
    p_orig = _target_prob(model, x, edge_index, node_idx, c)

    exp = set(map(_canon, explanation_edges))
    p_minus = _target_prob(model, x, _remove_pairs(edge_index, exp), node_idx, c)

    # Fid+: within surface keep only explanation edges.
    K = getattr(model, "K", 2)
    ball = surface_ball(node_idx, K, edge_index, x.size(0))
    keep = []
    for i in range(edge_index.size(1)):
        a, b = int(edge_index[0, i]), int(edge_index[1, i])
        if in_surface(a, b, ball):
            if (min(a, b), max(a, b)) in exp:
                keep.append(i)
        else:
            keep.append(i)
    p_plus = _target_prob(model, x, edge_index[:, keep], node_idx, c)

    return {"fid_minus": p_orig - p_minus, "fid_plus": p_orig - p_plus,
            "p_orig": p_orig, "p_minus": p_minus, "p_plus": p_plus}


def robust_fidelity(
    model, x, edge_index, node_idx, explanation_edges, *,
    node_pred_class=None, alpha=0.5, samples=20, seed=0,
):
    """
    Randomized fidelity that marginalizes over partial removals to avoid the
    distribution shift of deleting a whole subgraph at once (Zheng et al. 2023).

      robust Fid- : each explanation edge removed independently w.p. alpha,
                    averaged over ``samples`` draws.
      robust Fid+ : each *non-explanation surface* edge removed w.p. alpha,
                    explanation edges always kept.
    """
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        c = node_pred_class if node_pred_class is not None else int(model(x, edge_index)[node_idx].argmax())
    p_orig = _target_prob(model, x, edge_index, node_idx, c)

    exp = set(map(_canon, explanation_edges))
    K = getattr(model, "K", 2)
    ball = surface_ball(node_idx, K, edge_index, x.size(0))
    surface_pairs = set()
    for i in range(edge_index.size(1)):
        a, b = int(edge_index[0, i]), int(edge_index[1, i])
        if a != b and in_surface(a, b, ball):
            surface_pairs.add((min(a, b), max(a, b)))
    non_exp = surface_pairs - exp

    minus_drops, plus_drops = [], []
    for _ in range(samples):
        rm_minus = {e for e in exp if rng.random() < alpha}
        minus_drops.append(p_orig - _target_prob(model, x, _remove_pairs(edge_index, rm_minus), node_idx, c))

        rm_plus = {e for e in non_exp if rng.random() < alpha}
        plus_drops.append(p_orig - _target_prob(model, x, _remove_pairs(edge_index, rm_plus), node_idx, c))

    return {
        "robust_fid_minus": float(np.mean(minus_drops)),
        "robust_fid_plus": float(np.mean(plus_drops)),
        "alpha": alpha, "samples": samples,
    }


# --------------------------------------------------------------------------- #
# Well-posedness gate
# --------------------------------------------------------------------------- #
def prediction_conditioned(model, x, clean_ei, pert_ei, node_idx, delta=0.05):
    """
    True iff the target keeps its argmax label AND |p_clean - p_pert| < delta on
    that class. Only such (node, perturbation) pairs enter the fragility stats,
    so a moved explanation cannot be an artifact of a moved decision.
    """
    with torch.no_grad():
        clean = torch.exp(model(x, clean_ei)[node_idx])
        pert = torch.exp(model(x, pert_ei)[node_idx])
    c = int(clean.argmax())
    same = int(pert.argmax()) == c
    stable = abs(float(clean[c]) - float(pert[c])) < delta
    return bool(same and stable), {"pred_class": c, "p_clean": float(clean[c]), "p_pert": float(pert[c])}


# --------------------------------------------------------------------------- #
# Secondary: homophily attribution (migrated from prototype extensions.py)
# --------------------------------------------------------------------------- #
def homophily_attribution_index(edges, y, weights=None):
    if not edges:
        return None
    labels = y.cpu().numpy() if torch.is_tensor(y) else np.asarray(y)
    total = same = 0.0
    for i, (u, v) in enumerate(edges):
        wt = 1.0 if weights is None else float(weights[i])
        total += wt
        if labels[u] == labels[v]:
            same += wt
    return (same / total) if total > 0 else None


def hai_shift(hai_clean, hai_perturbed):
    if hai_clean is None or hai_perturbed is None:
        return None
    return hai_perturbed - hai_clean
