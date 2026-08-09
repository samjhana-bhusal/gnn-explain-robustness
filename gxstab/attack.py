"""
Prediction-preserving injection attack on the smoothed explainer (GXAttack-lite).

Greedy white-box adversary: add candidate edges (from the receptive-field pool)
one at a time, each chosen to most reduce the smoothed top-1 edge's inclusion
probability, subject to keeping the target's argmax label fixed and its
confidence within delta. The empirical attack radius is the number of insertions
needed to change the smoothed top-k.

Two uses:
  * Tightness: compare empirical attack radius to the certified radius. The
    certificate is sound iff certified <= empirical on every node; the gap
    measures how conservative the certificate is.
  * A real, honest attack baseline (à la GXAttack) against which the smoothed
    explainer is the defense.

All smoothing runs on the (K+1)-hop subgraph for speed.
"""

from __future__ import annotations

import numpy as np
import torch

from gxstab.explainers import REGISTRY
from gxstab.subgraph import build_local_context
from gxstab.smoothing import (Surface, inclusion_probabilities_local, smoothed_topk,
                              SmoothConfig)


def _local_surface(ctx):
    return Surface(node_idx=ctx.v_local, K=ctx.K,
                   present=ctx.present_local, absent=ctx.absent_local)


def _pred_ok(model, ctx, injected_local, base_c, base_p, delta):
    """Argmax preserved and |dp| < delta after injecting ``injected_local`` edges."""
    ei = ctx.edge_index_local
    if injected_local:
        rows, cols = [], []
        for a, b in injected_local:
            rows += [a, b]; cols += [b, a]
        extra = torch.tensor([rows, cols], dtype=torch.long, device=ei.device)
        ei = torch.cat([ei, extra], dim=1)
    with torch.no_grad():
        p = torch.exp(model(ctx.x_local, ei)[ctx.v_local])
    return int(p.argmax()) == base_c and abs(float(p[base_c]) - base_p) < delta


def _smoothed_topk_with_injection(model, ctx, injected_local, explainer_fn, cfg, num_classes):
    """Smoothed top-k when the adversary has already inserted ``injected_local``.
    The injected edges become part of the (fixed) base graph; smoothing still
    flips the remaining candidate pool."""
    base_absent = [e for e in ctx.absent_local if e not in set(injected_local)]
    aug_present = list(ctx.present_local)  # genuine edges unchanged
    surf = Surface(node_idx=ctx.v_local, K=ctx.K, present=aug_present, absent=base_absent)
    # Fold injected edges into the base graph by adding them to edge_index_local.
    ei = ctx.edge_index_local
    if injected_local:
        rows, cols = [], []
        for a, b in injected_local:
            rows += [a, b]; cols += [b, a]
        ei = torch.cat([ei, torch.tensor([rows, cols], dtype=torch.long, device=ei.device)], dim=1)
    ctx2 = _shallow_ctx(ctx, ei)
    pid = inclusion_probabilities_local(model, ctx2, explainer_fn, cfg, num_classes=num_classes)
    return smoothed_topk(pid["pi_hat"], cfg.k), pid


def _shallow_ctx(ctx, new_ei):
    import copy
    c = copy.copy(ctx)
    c.edge_index_local = new_ei
    return c


def empirical_attack_radius(model, data, meta, v, *, explainer="grad", p_i=0.3,
                            k=1, n_samples=120, insert_cap=30, max_r=6, delta=0.05,
                            seed=0):
    """Greedy injection radius that changes the smoothed top-k (or max_r+1 if the
    attack fails within budget)."""
    ctx = build_local_context(model, data.x, data.edge_index, v, meta["num_nodes"], K=2, insert_cap=insert_cap)
    if len(ctx.present_local) < k + 1 or len(ctx.absent_local) < 1:
        return None
    cfg = SmoothConfig(p_d=0.0, p_i=p_i, n_samples=n_samples, k=k, seed=seed, mode="insert")
    fn = REGISTRY[explainer]
    nc = meta["num_classes"]

    with torch.no_grad():
        p0 = torch.exp(model(ctx.x_local, ctx.edge_index_local)[ctx.v_local])
    base_c, base_p = int(p0.argmax()), float(p0.max())

    clean_tk, _ = _smoothed_topk_with_injection(model, ctx, [], fn, cfg, nc)
    clean_tk = set(clean_tk)

    injected = []
    pool = list(ctx.absent_local)
    for r in range(1, max_r + 1):
        best, best_score = None, 1e9
        for cand in pool:
            if cand in injected:
                continue
            trial = injected + [cand]
            if not _pred_ok(model, ctx, trial, base_c, base_p, delta):
                continue
            tk, pid = _smoothed_topk_with_injection(model, ctx, trial, fn, cfg, nc)
            if set(tk) != clean_tk:
                return r  # attack succeeded at radius r
            # score = inclusion prob of the current top-1 (lower = closer to flip)
            top_pi = max(pid["pi_hat"].values()) if pid["pi_hat"] else 0.0
            if top_pi < best_score:
                best_score, best = top_pi, cand
        if best is None:
            break
        injected.append(best)
    return max_r + 1  # not broken within budget
