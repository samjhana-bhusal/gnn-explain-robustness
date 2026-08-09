"""
Turn the raw parquet artifacts in results/ into the paper's tables, with
bootstrap confidence intervals. Prints markdown; also returns DataFrames.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from gxstab.experiments.common import RESULTS_DIR


def _ci(x, n_boot=2000, seed=0):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = [rng.choice(x, len(x), replace=True).mean() for _ in range(n_boot)]
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _load(name):
    p = os.path.join(RESULTS_DIR, name)
    return pd.read_parquet(p) if os.path.exists(p) else None


def certified_by_dataset(fname="certified_main.parquet"):
    df = _load(fname)
    if df is None:
        return None
    rows = []
    for ds, g in df.groupby("dataset"):
        m, lo, hi = _ci(g.cert_radius)
        rows.append({
            "dataset": ds, "homophily": round(g.homophily.iloc[0], 2),
            "n_nodes": len(g), "mean_r*": round(m, 2),
            "ci95": f"[{lo:.2f},{hi:.2f}]",
            "frac>=1": round((g.cert_radius >= 1).mean(), 2),
            "frac>=2": round((g.cert_radius >= 2).mean(), 2),
            "median_surface": int(g.n_present.median()),
        })
    return pd.DataFrame(rows).sort_values("homophily", ascending=False)


def pi_sweep(fname="certified_pi_sweep.parquet"):
    df = _load(fname)
    if df is None:
        return None
    rows = []
    for p_i, g in df.groupby("p_i"):
        m, lo, hi = _ci(g.cert_radius)
        rows.append({"p_i": p_i, "mean_r*": round(m, 2), "ci95": f"[{lo:.2f},{hi:.2f}]",
                     "frac>=1": round((g.cert_radius >= 1).mean(), 2),
                     "frac>=2": round((g.cert_radius >= 2).mean(), 2)})
    return pd.DataFrame(rows)


def by_group(fname, col):
    df = _load(fname)
    if df is None:
        return None
    rows = []
    for key, g in df.groupby(col):
        m, lo, hi = _ci(g.cert_radius)
        rows.append({col: key, "mean_r*": round(m, 2), "ci95": f"[{lo:.2f},{hi:.2f}]",
                     "frac>=1": round((g.cert_radius >= 1).mean(), 2), "n": len(g)})
    return pd.DataFrame(rows)


def fragility_table(fname="fragility_main.parquet"):
    df = _load(fname)
    if df is None:
        return None
    df = df[df.pred_conditioned]
    rows = []
    for (expl, attack), g in df[df.radius == df.radius.max()].groupby(["explainer", "attack"]):
        m, lo, hi = _ci(g.jaccard)
        rows.append({"explainer": expl, "attack": attack, "mean_jaccard": round(m, 2),
                     "ci95": f"[{lo:.2f},{hi:.2f}]", "n": len(g)})
    piv = pd.DataFrame(rows).pivot(index="explainer", columns="attack", values="mean_jaccard")
    return piv


def _md(df, title):
    print(f"\n### {title}\n")
    if df is None:
        print("_(no data yet)_")
    else:
        print(df.to_markdown(index=isinstance(df.index, pd.Index) and df.index.name is not None))


def print_all():
    _md(certified_by_dataset(), "Certified radius by dataset (GCN, grad, p_i=0.3)")
    _md(pi_sweep(), "Smoothing-strength ablation (cora GCN)")
    _md(by_group("certified_explainer.parquet", "explainer"), "Base-explainer comparison (cora GCN)")
    _md(by_group("certified_backbone.parquet", "backbone"), "Backbone comparison (cora)")
    _md(fragility_table(), "Explanation fragility: mean top-k Jaccard at max radius (pred-conditioned)")


if __name__ == "__main__":
    print_all()
