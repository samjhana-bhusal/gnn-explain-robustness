"""
Driver for the two auxiliary studies (run after run_all): locality ablation and
the empirical-vs-certified attack comparison. Kept separate so the main sweep and
these can run sequentially without competing for the laptop's cores.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

from gxstab.experiments.ablation import run_locality_ablation
from gxstab.experiments.common import get_trained_model, sample_targets, RESULTS_DIR
from gxstab.explainers import REGISTRY
from gxstab.subgraph import build_local_context
from gxstab.smoothing import inclusion_probabilities_local, SmoothConfig
from gxstab import certify as C
from gxstab.attack import empirical_attack_radius


def run_attack_comparison(datasets, seeds, *, p_i=0.3, k=1, n_targets=15,
                          n_samples=120, out="attack_vs_certified.parquet"):
    import pandas as pd
    rows = []
    for dataset in datasets:
        for seed in seeds:
            model, data, meta, acc = get_trained_model(dataset, "GCN", seed)
            targets = sample_targets(model, data, meta, n=n_targets, seed=seed)
            for v in targets:
                ctx = build_local_context(model, data.x, data.edge_index, v, meta["num_nodes"], K=2, insert_cap=30)
                if len(ctx.present_local) < k + 1 or len(ctx.absent_local) < 1:
                    continue
                cfg = SmoothConfig(p_d=0.0, p_i=p_i, n_samples=n_samples, k=k, seed=seed, mode="insert")
                pid = inclusion_probabilities_local(model, ctx, REGISTRY["grad"], cfg, num_classes=meta["num_classes"])
                rc = C.certified_radius(pid, k, 0.0, p_i, max_r=min(6, len(ctx.absent_local)), mode="insert")
                re = empirical_attack_radius(model, data, meta, v, explainer="grad", p_i=p_i,
                                             k=k, n_samples=n_samples, max_r=6, seed=seed)
                rows.append({"dataset": dataset, "seed": seed, "node": v,
                             "certified": rc, "empirical": re, "sound": rc <= re,
                             "gap": (re - rc)})
    df = pd.DataFrame(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, out)
    df.to_parquet(path)
    return df, path


def main(quick=False):
    seeds = [0] if quick else [0, 1, 2]
    t0 = time.time()

    def log(m):
        print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

    log("locality ablation (cora, citeseer, roman-empire)")
    df, p = run_locality_ablation(["cora", "citeseer", "roman-empire"], seeds=seeds[:2],
                                  n_targets=15 if not quick else 6,
                                  n_samples=150, out="ablation_locality.parquet")
    log(f"  -> {p} ({len(df)} rows, median cost_ratio={df.cost_ratio.median():.0f}x, "
        f"radii_equal={(df.r_local==df.r_global_sametarget).mean():.2f})")

    log("attack vs certified (cora, citeseer)")
    df, p = run_attack_comparison(["cora", "citeseer"], seeds[:2],
                                  n_targets=15 if not quick else 6, out="attack_vs_certified.parquet")
    log(f"  -> {p} ({len(df)} rows, sound={df.sound.mean():.2f}, median gap={df.gap.median():.1f})")
    log("DONE")


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
