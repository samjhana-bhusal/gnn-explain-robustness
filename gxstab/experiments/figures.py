"""
Generate the paper's figures from the parquet artifacts in results/.
Writes PDFs to report/figs/. Theme-agnostic, colorblind-safe palette.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gxstab.experiments.common import RESULTS_DIR

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "report", "figs")
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]


def _load(name):
    p = os.path.join(RESULTS_DIR, name)
    return pd.read_parquet(p) if os.path.exists(p) else None


def _ci(x, n_boot=2000, seed=0):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    b = [rng.choice(x, len(x), replace=True).mean() for _ in range(n_boot)]
    return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def fig_pi_sweep():
    df = _load("certified_pi_sweep.parquet")
    if df is None:
        return
    ps = sorted(df.p_i.unique())
    means, los, his = [], [], []
    for p in ps:
        m, lo, hi = _ci(df[df.p_i == p].cert_radius)
        means.append(m); los.append(lo); his.append(hi)
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.plot(ps, means, "-o", color=PALETTE[0], lw=2)
    ax.fill_between(ps, los, his, color=PALETTE[0], alpha=0.2)
    ax.set_xlabel("smoothing insertion prob. $p_i$")
    ax.set_ylabel("mean certified radius $r^\\star$")
    ax.set_title("Smoothing strength (Cora/GCN)", fontsize=10)
    ax.grid(alpha=0.3)
    _save(fig, "pi_sweep")


def fig_by_dataset():
    df = _load("certified_main.parquet")
    if df is None:
        return
    rows = []
    for ds, g in df.groupby("dataset"):
        m, lo, hi = _ci(g.cert_radius)
        rows.append((ds, g.homophily.iloc[0], m, lo, hi))
    rows.sort(key=lambda r: -r[1])
    labels = [r[0] for r in rows]
    means = [r[2] for r in rows]
    err = [[r[2] - r[3] for r in rows], [r[4] - r[2] for r in rows]]
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.bar(range(len(labels)), means, yerr=err, color=PALETTE[:len(labels)], capsize=3)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("mean certified radius $r^\\star$")
    ax.set_title("Certified radius by dataset (GCN, $p_i{=}0.3$)", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "by_dataset")


def fig_attack_vs_certified():
    df = _load("attack_vs_certified.parquet")
    if df is None:
        return
    fig, ax = plt.subplots(figsize=(4.0, 3.6))
    jitter = np.random.default_rng(0).normal(0, 0.06, len(df))
    ax.scatter(df.certified + jitter, df.empirical + jitter, s=18,
               color=PALETTE[0], alpha=0.6, edgecolor="none")
    m = max(df.empirical.max(), df.certified.max()) + 1
    ax.plot([0, m], [0, m], "--", color="gray", lw=1, label="certified = empirical")
    ax.set_xlabel("certified radius $r^\\star$")
    ax.set_ylabel("empirical attack radius")
    ax.set_title("Certificate is a sound lower bound")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    _save(fig, "attack_vs_certified")


def fig_locality_cost():
    df = _load("ablation_locality.parquet")
    if df is None:
        return
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for i, (ds, g) in enumerate(df.groupby("dataset")):
        ax.scatter(g.full_nodes, g.cost_ratio, s=20, color=PALETTE[i % len(PALETTE)], label=ds, alpha=0.7)
    ax.set_xlabel("graph size $|V|$")
    ax.set_ylabel("speedup (full / subgraph, per sample)")
    ax.set_title("Locality: cost reduction")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    _save(fig, "locality_cost")


def _save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, name + ".pdf"))
    fig.savefig(os.path.join(FIG_DIR, name + ".png"), dpi=130)
    plt.close(fig)
    print("wrote", name)


def main():
    fig_pi_sweep()
    fig_by_dataset()
    fig_attack_vs_certified()
    fig_locality_cost()


if __name__ == "__main__":
    main()
