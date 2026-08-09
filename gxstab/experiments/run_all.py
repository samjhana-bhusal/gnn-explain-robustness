"""
Driver: run every sweep the paper needs and write parquet artifacts to results/.

Usage:
    python -m gxstab.experiments.run_all --quick    # small, for iteration
    python -m gxstab.experiments.run_all            # full paper run
"""

from __future__ import annotations

import os
import sys
import time

from gxstab.experiments.certified import run_certified
from gxstab.experiments.fragility import run_fragility
from gxstab.experiments.common import RESULTS_DIR


def _exists(name):
    return os.path.exists(os.path.join(RESULTS_DIR, name))

CITATION = ["cora", "citeseer", "pubmed"]
HETERO = ["roman-empire", "amazon-ratings", "minesweeper"]
ALL_DATA = CITATION + HETERO


def main(quick=False):
    seeds = [0] if quick else [0, 1, 2]
    n_targets = 12 if quick else 40
    n_samples = 100 if quick else 250
    t0 = time.time()

    def log(msg):
        print(f"[{time.time()-t0:6.0f}s] {msg}", flush=True)

    def step(name, out, fn):
        if _exists(out):
            log(f"SKIP {name} (results/{out} exists)")
            return
        log(name)
        df, p = fn()
        log(f"  -> {p} ({len(df)} rows)")

    # 1. Main certified-radius table: GCN across all datasets, grad base, p_i=0.3.
    step("certified: main (GCN, grad, all datasets)", "certified_main.parquet",
         lambda: run_certified(ALL_DATA, ["GCN"], seeds, explainers=["grad"], p_is=[0.3],
                               k=1, n_targets=n_targets, n_samples=n_samples, out="certified_main.parquet"))

    # 2. Smoothing-strength ablation: cora GCN, grad, p_i sweep.
    step("certified: p_i sweep (cora GCN, grad)", "certified_pi_sweep.parquet",
         lambda: run_certified(["cora"], ["GCN"], seeds, explainers=["grad"],
                               p_is=[0.1, 0.2, 0.3, 0.4], k=1, n_targets=n_targets,
                               n_samples=n_samples, out="certified_pi_sweep.parquet"))

    # 3. Base-explainer comparison: cora GCN, grad vs occlusion. Occlusion is
    # O(surface) forwards/sample, so use a small node set + sample budget.
    occ_nodes = 6 if quick else 12
    occ_samples = 100 if quick else 120
    step("certified: base explainer (cora GCN, grad vs occlusion)", "certified_explainer.parquet",
         lambda: run_certified(["cora"], ["GCN"], seeds[:2], explainers=["grad", "occlusion"],
                               p_is=[0.3], k=1, n_targets=occ_nodes, n_samples=occ_samples,
                               out="certified_explainer.parquet"))

    # 4. Backbone comparison: cora, {GCN, SGC, APPNP} (all support fast grad).
    step("certified: backbones (cora, GCN/SGC/APPNP, grad)", "certified_backbone.parquet",
         lambda: run_certified(["cora"], ["GCN", "SGC", "APPNP"], seeds, explainers=["grad"],
                               p_is=[0.3], k=1, n_targets=n_targets, n_samples=n_samples,
                               out="certified_backbone.parquet"))

    # 5a. Fragility: fast explainers on all datasets, subgraph-local (fast, exact).
    fast = ["grad", "degree", "cosine", "random"]
    step("fragility: fast explainers (GCN, all datasets)", "fragility_main.parquet",
         lambda: run_fragility(ALL_DATA, ["GCN"], fast, seeds,
                               radii=[1, 2, 3, 4], attacks=["random", "degree", "homophily"],
                               k=5, n_targets=min(n_targets, 20), use_local=True,
                               out="fragility_main.parquet"))

    # 5b. Fragility: GNNExplainer (self-extracts, full graph) on a small subset for
    # the key stable-vs-fragile contrast.
    step("fragility: gnnexplainer (GCN, subset)", "fragility_slow.parquet",
         lambda: run_fragility(["cora", "citeseer", "roman-empire"], ["GCN"],
                               ["gnnexplainer"], seeds[:1],
                               radii=[1, 4], attacks=["random", "homophily"],
                               k=5, n_targets=8 if not quick else 6, use_local=False,
                               out="fragility_slow.parquet"))

    log("DONE")


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
