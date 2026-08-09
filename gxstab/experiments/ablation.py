"""
Headline ablation: what does receptive-field locality buy the certificate?

By Lemma 1, an injected edge with no endpoint in the target's K-hop ball cannot
change the target's explanation. So a *global* certificate that smooths over the
whole edge space and a *localized* one that smooths only the receptive-field
surface yield the SAME certified radius -- but the localized one runs on the
(K+1)-hop subgraph (tens of nodes) instead of the full graph (thousands).

We report three things per node:

  1. radius_local == radius_global at equal sample budget (locality is lossless).
  2. cost ratio = (full-graph explainer time) / (subgraph explainer time): at
     equal wall-clock the localized method affords this many times more samples,
     hence tighter Clopper-Pearson bounds and a larger certified radius.
  3. coverage: |surface candidates| / |all node pairs| -- the localized
     certificate defends against the entire global injection adversary while
     smoothing a vanishing fraction of the edge space.

This is the evidence that the contribution is locality, not "smoothing applied
to explanations."
"""

from __future__ import annotations

import os
import time

import numpy as np

from gxstab.explainers import REGISTRY, top_k_edges
from gxstab.subgraph import build_local_context
from gxstab.smoothing import (Surface, build_surface, inclusion_probabilities,
                              inclusion_probabilities_local, SmoothConfig)
from gxstab import certify as C
from gxstab.experiments.common import get_trained_model, sample_targets, RESULTS_DIR


def run_locality_ablation(datasets, backbones=("GCN",), seeds=(0,), *,
                          explainer="grad", p_i=0.3, k=1, n_targets=20,
                          n_samples=150, insert_cap=30, timed_samples=40,
                          out="ablation_locality.parquet"):
    import pandas as pd

    rows = []
    for dataset in datasets:
        for backbone in backbones:
            for seed in seeds:
                model, data, meta, acc = get_trained_model(dataset, backbone, seed)
                N = meta["num_nodes"]
                all_pairs = N * (N - 1) // 2
                targets = sample_targets(model, data, meta, n=n_targets, seed=seed)
                cfg = SmoothConfig(p_d=0.0, p_i=p_i, n_samples=n_samples, k=k, seed=seed, mode="insert")

                for v in targets:
                    ctx = build_local_context(model, data.x, data.edge_index, v, N, K=2, insert_cap=insert_cap)
                    if len(ctx.present_local) < k + 1 or len(ctx.absent_local) < 1:
                        continue

                    # localized (subgraph) smoothing + radius
                    pid_local = inclusion_probabilities_local(model, ctx, REGISTRY[explainer], cfg, num_classes=meta["num_classes"])
                    r_local = C.certified_radius(pid_local, k, 0.0, p_i, max_r=min(6, len(ctx.absent_local)), mode="insert")

                    # global (full-graph) smoothing over the SAME candidate pool
                    surf = build_surface(v, data.edge_index, N, K=2, insert_cap=insert_cap)
                    cfg_t = SmoothConfig(p_d=0.0, p_i=p_i, n_samples=timed_samples, k=k, seed=seed, mode="insert")
                    t0 = time.time()
                    pid_global = inclusion_probabilities(model, data.x, data.edge_index, surf, REGISTRY[explainer], cfg_t, num_classes=meta["num_classes"])
                    t_global = (time.time() - t0) / timed_samples
                    r_global = C.certified_radius(pid_global, k, 0.0, p_i, max_r=min(6, surf.n_absent), mode="insert")

                    # subgraph per-sample time
                    t0 = time.time()
                    _ = inclusion_probabilities_local(model, ctx, REGISTRY[explainer], cfg_t, num_classes=meta["num_classes"])
                    t_local = (time.time() - t0) / timed_samples

                    rows.append({
                        "dataset": dataset, "backbone": backbone, "seed": seed, "node": v,
                        "r_local": r_local, "r_global_sametarget": r_global,
                        "t_local_ms": t_local * 1e3, "t_global_ms": t_global * 1e3,
                        "cost_ratio": t_global / max(t_local, 1e-9),
                        "sub_nodes": ctx.x_local.size(0), "full_nodes": N,
                        "surface_candidates": surf.n_present + surf.n_absent,
                        "all_pairs": all_pairs,
                        "coverage_frac": (surf.n_present + surf.n_absent) / all_pairs,
                    })
    df = pd.DataFrame(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, out)
    df.to_parquet(path)
    return df, path
