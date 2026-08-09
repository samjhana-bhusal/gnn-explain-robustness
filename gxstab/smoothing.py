"""
Randomized-smoothing of a base edge explainer over the attack surface.

The smoothing coordinates are the target's surface entries (Lemma 1): each
*present* surface edge is independently deleted with probability ``p_d``, and
each *candidate absent* surface non-edge is independently inserted with
probability ``p_i``. For each sampled graph we run a base explainer and record
which undirected surface edges land in its top-k. The smoothed explanation is
the set of edges with the highest inclusion probability

    pi_e = P[ e in TopK(phi(G_tilde)) ].

The certificate (certify.py) turns Monte-Carlo estimates of pi (with
Clopper-Pearson confidence bounds) into a provable radius of top-k invariance.

Insertion coordinates are a *bounded candidate pool* -- by default the non-edges
with both endpoints in the target's ball -- so the coordinate count stays
O(|ball|^2) instead of O(|ball| * N). The certificate statement is then precise:
robust to any r_d deletions of surface edges and r_i insertions from the
declared candidate pool.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from gxstab.surface import surface_ball, surface_edges
from gxstab.explainers import top_k_edges


@dataclass
class Surface:
    """The smoothing coordinate system for one target node."""
    node_idx: int
    K: int
    present: list          # surface edges (min,max) that may be deleted
    absent: list           # candidate non-edges (min,max) that may be inserted

    @property
    def n_present(self):
        return len(self.present)

    @property
    def n_absent(self):
        return len(self.absent)


def build_surface(v, edge_index, num_nodes, K=2, insert_cap=None):
    """
    Construct the smoothing surface for ``v``: present = surface edges;
    absent = non-edges with both endpoints in the ball (bounded candidate pool).
    """
    present = surface_edges(v, K, edge_index, num_nodes)
    ball = sorted(surface_ball(v, K, edge_index, num_nodes))
    present_set = set(present)
    absent = []
    for i in range(len(ball)):
        for j in range(i + 1, len(ball)):
            e = (ball[i], ball[j])
            if e not in present_set:
                absent.append(e)
                if insert_cap is not None and len(absent) >= insert_cap:
                    break
        if insert_cap is not None and len(absent) >= insert_cap:
            break
    return Surface(node_idx=v, K=K, present=present, absent=absent)


@dataclass
class SmoothConfig:
    p_d: float = 0.0          # per-present-edge deletion prob (0 in insertion mode)
    p_i: float = 0.3          # per-candidate insertion flip prob
    n_samples: int = 200
    k: int = 5
    seed: int = 0
    mode: str = "insert"      # "insert": never delete genuine edges (paper default)


def _base_edges_without_surface(edge_index, surface: Surface):
    """Directed edges of G with all present-surface edges removed (we re-add the
    sampled subset each draw). Returns (base_ei, present_dir_lookup)."""
    drop = set()
    for a, b in surface.present:
        drop.add((a, b))
        drop.add((b, a))
    keep = [
        i for i in range(edge_index.size(1))
        if (int(edge_index[0, i]), int(edge_index[1, i])) not in drop
    ]
    return edge_index[:, keep]


def _assemble(base_ei, active_pairs, device):
    if not active_pairs:
        return base_ei
    rows, cols = [], []
    for a, b in active_pairs:
        rows += [a, b]
        cols += [b, a]
    extra = torch.tensor([rows, cols], dtype=torch.long, device=device)
    return torch.cat([base_ei, extra], dim=1)


def sample_graph(base_ei, surface: Surface, present_mask, absent_mask, device):
    """Assemble a smoothing draw: present edges kept where mask True, plus
    inserted absent edges where mask True."""
    active = [surface.present[i] for i in range(surface.n_present) if present_mask[i]]
    active += [surface.absent[j] for j in range(surface.n_absent) if absent_mask[j]]
    return _assemble(base_ei, active, device)


def inclusion_probabilities(model, x, edge_index, surface: Surface, explainer_fn,
                            cfg: SmoothConfig, num_classes=None):
    """
    Monte-Carlo estimate of pi_e for every present surface edge.

    Returns a dict with per-edge counts (# draws where e in top-k) and n_samples,
    from which certify.py derives Clopper-Pearson bounds. Only *present* surface
    edges are certifiable explanation members (an inserted edge is adversarial by
    construction); inserted edges appearing in top-k are ignored for pi.
    """
    device = x.device
    rng = np.random.default_rng(cfg.seed)
    base_ei = _base_edges_without_surface(edge_index, surface)
    v, K = surface.node_idx, surface.K
    N = x.size(0)

    counts = {e: 0 for e in surface.present}
    for _ in range(cfg.n_samples):
        # insertion mode: genuine edges always kept; only candidates are flipped.
        pm = (rng.random(surface.n_present) >= cfg.p_d) if cfg.mode != "insert" \
            else np.ones(surface.n_present, dtype=bool)
        am = rng.random(surface.n_absent) < cfg.p_i        # insert candidate w.p. p_i
        g = sample_graph(base_ei, surface, pm, am, device)
        scores = explainer_fn(model, x, g, v, K=K, num_classes=num_classes)
        tk = top_k_edges(scores, g, v, cfg.k, K, N)
        tk = set(tk)
        for e in surface.present:
            if e in tk:
                counts[e] += 1

    return {"counts": counts, "n_samples": cfg.n_samples,
            "pi_hat": {e: counts[e] / cfg.n_samples for e in surface.present}}


def inclusion_probabilities_local(model, ctx, explainer_fn, cfg: SmoothConfig, num_classes=None):
    """
    Same as ``inclusion_probabilities`` but runs every smoothing draw on the
    target's relabeled (K+1)-hop subgraph (subgraph.LocalContext), which is far
    smaller than the full graph. Returns inclusion data keyed by GLOBAL edges.
    """
    device = ctx.x_local.device
    rng = np.random.default_rng(cfg.seed)
    v, K = ctx.v_local, ctx.K
    Nl = ctx.x_local.size(0)

    local_surface = Surface(node_idx=v, K=K, present=ctx.present_local, absent=ctx.absent_local)
    base_ei = _base_edges_without_surface(ctx.edge_index_local, local_surface)
    local_to_global = {le: ge for le, ge in zip(ctx.present_local, ctx.present_global)}

    counts = {ge: 0 for ge in ctx.present_global}
    for _ in range(cfg.n_samples):
        pm = np.ones(local_surface.n_present, dtype=bool) if cfg.mode == "insert" \
            else (rng.random(local_surface.n_present) >= cfg.p_d)
        am = rng.random(local_surface.n_absent) < cfg.p_i
        g = sample_graph(base_ei, local_surface, pm, am, device)
        scores = explainer_fn(model, ctx.x_local, g, v, K=K, num_classes=num_classes)
        tk = set(top_k_edges(scores, g, v, cfg.k, K, Nl))
        for le in ctx.present_local:
            if le in tk:
                counts[local_to_global[le]] += 1

    return {"counts": counts, "n_samples": cfg.n_samples,
            "pi_hat": {ge: counts[ge] / cfg.n_samples for ge in ctx.present_global}}


def exact_inclusion_probabilities(model, x, edge_index, surface: Surface,
                                  explainer_fn, cfg: SmoothConfig, num_classes=None,
                                  max_coords=16):
    """
    Exact pi by enumerating all 2^(n_present+n_absent) smoothing outcomes.
    Only for tiny surfaces (test/validation use). Returns exact pi_e per present
    edge under the given (p_d, p_i).
    """
    device = x.device
    v, K = surface.node_idx, surface.K
    N = x.size(0)
    nP, nA = surface.n_present, surface.n_absent
    if nP + nA > max_coords:
        raise ValueError(f"exact pi needs <= {max_coords} coords, got {nP + nA}")

    base_ei = _base_edges_without_surface(edge_index, surface)
    pi = {e: 0.0 for e in surface.present}
    insert_mode = cfg.mode == "insert"
    for code in range(1 << (nP + nA)):
        pm = [(code >> i) & 1 for i in range(nP)]           # 1 = keep present
        am = [(code >> (nP + j)) & 1 for j in range(nA)]    # 1 = insert absent
        prob = 1.0
        for i in range(nP):
            if insert_mode:
                prob *= 1.0 if pm[i] else 0.0               # genuine edges always kept
            else:
                prob *= (1 - cfg.p_d) if pm[i] else cfg.p_d
        if prob == 0.0:
            continue
        for j in range(nA):
            prob *= cfg.p_i if am[j] else (1 - cfg.p_i)
        g = sample_graph(base_ei, surface, pm, am, device)
        scores = explainer_fn(model, x, g, v, K=K, num_classes=num_classes)
        tk = set(top_k_edges(scores, g, v, cfg.k, K, N))
        for e in surface.present:
            if e in tk:
                pi[e] += prob
    return pi


def smoothed_topk(pi_hat: dict, k: int):
    """Top-k present edges by inclusion probability (ties broken by edge id)."""
    ranked = sorted(pi_hat.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]), reverse=True)
    return [e for e, _ in ranked[:k]]
