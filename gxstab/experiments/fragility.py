"""
Study 1: empirical explanation fragility under prediction-preserving local attacks.

For each (dataset, backbone, explainer, seed, target, attack, radius) we:
  1. explain the clean graph,
  2. apply a radius-r surface attack (perturb.py),
  3. record whether the sample is prediction-conditioned (argmax unchanged,
     |dp| < delta) so we only count decision-stable perturbations,
  4. re-explain and record top-k Jaccard vs. clean.

This is the corrected version of the prototype's headline experiment: local
budget, symmetric flips, prediction conditioning. It answers "how much does the
explanation move when the decision does not?" -- well-posed this time. One row
per (config, target, attack, radius).
"""

from __future__ import annotations

import os
import numpy as np
import torch

import torch

from gxstab.explainers import REGISTRY, top_k_edges
from gxstab.perturb import perturb
from gxstab.subgraph import build_local_context
from gxstab import metrics as M
from gxstab.experiments.common import get_trained_model, sample_targets, RESULTS_DIR


def _explain_local(model, ctx, ename, k, nc):
    """Top-k on the target's subgraph, mapped back to global edge ids. Exact by
    Lemma 1 (the (K+1)-hop subgraph preserves the target's logits)."""
    s = REGISTRY[ename](model, ctx.x_local, ctx.edge_index_local, ctx.v_local, K=ctx.K, num_classes=nc)
    tk_local = top_k_edges(s, ctx.edge_index_local, ctx.v_local, k, ctx.K, ctx.x_local.size(0))
    l2g = ctx.subset.tolist()
    return {(min(l2g[a], l2g[b]), max(l2g[a], l2g[b])) for a, b in tk_local}


def run_fragility(datasets, backbones, explainers, seeds, *,
                  radii=(1, 2, 3, 4), attacks=("random", "degree", "homophily"),
                  k=5, n_targets=40, conf_min=0.6, delta=0.05, use_local=True,
                  out="fragility.parquet"):
    # use_local=True explains on the (K+1)-hop subgraph (fast, exact) -- valid for
    # grad/occlusion/baselines. GNNExplainer/MCTS self-extract a subgraph, so they
    # must run on the full graph (use_local=False); use only on small datasets.
    import pandas as pd

    rows = []
    for dataset in datasets:
        for backbone in backbones:
            for seed in seeds:
                model, data, meta, acc = get_trained_model(dataset, backbone, seed)
                N = meta["num_nodes"]
                nc = meta["num_classes"]
                targets = sample_targets(model, data, meta, n=n_targets, conf_min=conf_min, seed=seed)
                rng = np.random.default_rng(seed)

                for v in targets:
                    ctx = build_local_context(model, data.x, data.edge_index, v, N, K=2)
                    with torch.no_grad():
                        p_clean_vec = torch.exp(model(ctx.x_local, ctx.edge_index_local)[ctx.v_local])
                    c = int(p_clean_vec.argmax())
                    p_clean = float(p_clean_vec[c])
                    if use_local:
                        clean_tk = {e: _explain_local(model, ctx, e, k, nc) for e in explainers}
                    else:
                        clean_tk = {e: set(top_k_edges(REGISTRY[e](model, data.x, data.edge_index, v, K=2, num_classes=nc),
                                                       data.edge_index, v, k, 2, N)) for e in explainers}

                    for attack in attacks:
                        for r in radii:
                            pei, _ = perturb(v, data.edge_index, N, radius=r, K=2,
                                             y=data.y, method=attack, rng=rng)
                            pctx = build_local_context(model, data.x, pei, v, N, K=2)
                            with torch.no_grad():
                                p_pert_vec = torch.exp(model(pctx.x_local, pctx.edge_index_local)[pctx.v_local])
                            ok = (int(p_pert_vec.argmax()) == c) and (abs(p_clean - float(p_pert_vec[c])) < delta)
                            for ename in explainers:
                                if use_local:
                                    tk2 = _explain_local(model, pctx, ename, k, nc)
                                else:
                                    tk2 = set(top_k_edges(REGISTRY[ename](model, data.x, pei, v, K=2, num_classes=nc),
                                                          pei, v, k, 2, N))
                                jac = M.topk_jaccard(clean_tk[ename], tk2)
                                rows.append({
                                    "dataset": dataset, "backbone": backbone, "seed": seed,
                                    "explainer": ename, "node": v, "attack": attack, "radius": r,
                                    "pred_conditioned": ok, "jaccard": jac,
                                    "p_clean": p_clean, "p_pert": float(p_pert_vec[c]),
                                    "homophily": meta["homophily"], "acc": acc,
                                })
    df = pd.DataFrame(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, out)
    df.to_parquet(path)
    return df, path
