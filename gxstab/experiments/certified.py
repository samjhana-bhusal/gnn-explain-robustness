"""
Study 2: certified radius of the smoothed explainer.

For each (dataset, backbone, seed, target) we build the local context, estimate
smoothed inclusion probabilities with a base explainer, and compute the certified
insertion radius of the smoothed top-k. One row per target.

Frugal by design: everything runs on the (K+1)-hop subgraph (subgraph.py), and
grad is the default base (one backward per draw). occlusion is available for a
sharper-but-slower comparison on a node subset.
"""

from __future__ import annotations

import os
import numpy as np

from gxstab.explainers import REGISTRY
from gxstab.subgraph import build_local_context
from gxstab.smoothing import inclusion_probabilities_local, smoothed_topk, SmoothConfig
from gxstab import certify as C
from gxstab.experiments.common import get_trained_model, sample_targets, RESULTS_DIR


def run_certified(datasets, backbones, seeds, *,
                  explainers=("grad",), p_is=(0.3,), k=1, n_targets=40,
                  conf_min=0.7, deg_max=12, n_samples=200, insert_cap=30,
                  max_r=6, alpha=0.05, out="certified.parquet"):
    import pandas as pd

    rows = []
    for dataset in datasets:
        for backbone in backbones:
            for seed in seeds:
                model, data, meta, acc = get_trained_model(dataset, backbone, seed)
                targets = sample_targets(model, data, meta, n=n_targets,
                                         conf_min=conf_min, deg_max=deg_max, seed=seed)
                for v in targets:
                    ctx = build_local_context(model, data.x, data.edge_index, v,
                                              meta["num_nodes"], K=2, insert_cap=insert_cap)
                    n_present, n_absent = len(ctx.present_local), len(ctx.absent_local)
                    if n_present < k + 1 or n_absent < 1:
                        continue
                    for ename in explainers:
                        for p_i in p_is:
                            cfg = SmoothConfig(p_d=0.0, p_i=p_i, n_samples=n_samples,
                                               k=k, seed=seed, mode="insert")
                            pid = inclusion_probabilities_local(
                                model, ctx, REGISTRY[ename], cfg, num_classes=meta["num_classes"])
                            pis = sorted(pid["pi_hat"].values(), reverse=True)
                            r = C.certified_radius(pid, k, 0.0, p_i,
                                                   max_r=min(max_r, n_absent),
                                                   alpha=alpha, mode="insert")
                            rows.append({
                                "dataset": dataset, "backbone": backbone, "seed": seed,
                                "explainer": ename, "p_i": p_i, "k": k, "node": v,
                                "cert_radius": r,
                                "pi_top": pis[0] if pis else np.nan,
                                "pi_gap": (pis[0] - pis[1]) if len(pis) > 1 else pis[0],
                                "n_present": n_present, "n_absent": n_absent,
                                "ball_nodes": ctx.x_local.size(0),
                                "homophily": meta["homophily"], "acc": acc,
                            })
    df = pd.DataFrame(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, out)
    df.to_parquet(path)
    return df, path
