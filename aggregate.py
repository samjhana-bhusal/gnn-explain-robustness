"""
Aggregate a saved evaluation run into summary tables.

evaluate.py writes per-node records; this script reduces them to the mean
metrics the report and README actually cite, so those numbers are reproducible
from a run artifact instead of being transcribed by hand.

Usage:
    python aggregate.py [path/to/run.json]

Defaults to web/data.json (the latest run). Writes:
    web/results_summary.json   -- machine-readable summary
    SUMMARY.md                 -- human-readable tables
"""

import os
import sys
import json

EXPLAINERS = ["gnn_explainer", "subgraph_mcts", "counterfactual"]
METHODS = ["random", "degree", "homophily"]
PRETTY = {
    "gnn_explainer": "GNNExplainer",
    "subgraph_mcts": "Subgraph MCTS",
    "counterfactual": "Counterfactual",
}


def mean(values):
    """Mean over non-None values, or None if the list is empty after filtering."""
    vals = [v for v in values if v is not None]
    return (sum(vals) / len(vals)) if vals else None


def fmt(v, nd=3):
    return f"{v:.{nd}f}" if v is not None else "n/a"


def summarize(run):
    nodes = run["nodes"]
    rates = sorted({e["rate"] for e in nodes[0]["robustness"][EXPLAINERS[0]][METHODS[0]]})

    summary = {
        "dataset": run.get("dataset"),
        "num_nodes": len(nodes),
        "rates": rates,
        "fidelity": {},
        "hai_clean": {},
        "robustness": {},   # [explainer][method][rate] -> mean jaccard
        "hai_shift": {},    # [explainer][method][rate] -> mean hai shift
        "flip_rate": {},    # [method][rate] -> fraction of nodes whose prediction flipped
        "defense": {},      # [method] -> mean clean/attacked/defended probs
    }

    # Fidelity + clean HAI (per explainer, averaged over nodes)
    for ex in EXPLAINERS:
        summary["fidelity"][ex] = {
            "fid_minus": mean([n["fidelity"][ex]["fid_minus"] for n in nodes]),
            "fid_plus": mean([n["fidelity"][ex]["fid_plus"] for n in nodes]),
        }
        summary["hai_clean"][ex] = mean([n.get("hai_clean", {}).get(ex) for n in nodes])

    # Robustness (Jaccard) + HAI shift per explainer/method/rate
    for ex in EXPLAINERS:
        summary["robustness"][ex] = {}
        summary["hai_shift"][ex] = {}
        for m in METHODS:
            summary["robustness"][ex][m] = {}
            summary["hai_shift"][ex][m] = {}
            for r in rates:
                jaccs, shifts = [], []
                for n in nodes:
                    for e in n["robustness"][ex][m]:
                        if e["rate"] == r:
                            jaccs.append(e.get("jaccard"))
                            shifts.append(e.get("hai_shift"))
                summary["robustness"][ex][m][str(r)] = mean(jaccs)
                summary["hai_shift"][ex][m][str(r)] = mean(shifts)

    # Prediction flip rate (a graph property, identical across explainers)
    for m in METHODS:
        summary["flip_rate"][m] = {}
        for r in rates:
            flips = []
            for n in nodes:
                for e in n["robustness"][EXPLAINERS[0]][m]:
                    if e["rate"] == r:
                        flips.append(1.0 if e.get("pred_flipped") else 0.0)
            summary["flip_rate"][m][str(r)] = mean(flips)

    # Defense recovery (per attack method, averaged over nodes)
    if nodes and nodes[0].get("defense"):
        for m in METHODS:
            entries = [n["defense"][m] for n in nodes if m in n.get("defense", {})]
            if not entries:
                continue
            summary["defense"][m] = {
                "clean_prob": mean([e.get("clean_prob") for e in entries]),
                "attacked_prob": mean([e["attacked_prob"] for e in entries]),
                "defended_prob": mean([e["defended_prob"] for e in entries]),
                "pruned_edges": mean([e["pruned_edges"] for e in entries]),
                "neighborhood_edges": mean([e["neighborhood_edges"] for e in entries]),
            }

    return summary


def render_markdown(s):
    lines = []
    lines.append("# Results Summary\n")
    lines.append(f"Dataset: **{s['dataset']}** | Nodes averaged: **{s['num_nodes']}** "
                 f"| Perturbation rates: {', '.join(str(r) for r in s['rates'])}\n")

    lines.append("\n## Fidelity (clean graph, top-k=5, mean over nodes)\n")
    lines.append("| Method | Fidelity-minus (F-) ↑ | Fidelity-plus (F+) ↓ | Clean HAI |")
    lines.append("|---|---|---|---|")
    for ex in EXPLAINERS:
        f = s["fidelity"][ex]
        lines.append(f"| {PRETTY[ex]} | {fmt(f['fid_minus'])} | {fmt(f['fid_plus'])} "
                     f"| {fmt(s['hai_clean'][ex])} |")

    lines.append("\n## Explanation robustness — mean Jaccard vs. clean explanation\n")
    for m in METHODS:
        lines.append(f"\n**{m.capitalize()} perturbation**\n")
        header = "| Method | " + " | ".join(f"ε={r}" for r in s["rates"]) + " |"
        lines.append(header)
        lines.append("|" + "---|" * (len(s["rates"]) + 1))
        for ex in EXPLAINERS:
            row = [PRETTY[ex]] + [fmt(s["robustness"][ex][m][str(r)], 2) for r in s["rates"]]
            lines.append("| " + " | ".join(row) + " |")

    lines.append("\n## Prediction flip rate (fraction of target nodes whose label changed)\n")
    header = "| Attack | " + " | ".join(f"ε={r}" for r in s["rates"]) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(s["rates"]) + 1))
    for m in METHODS:
        row = [m.capitalize()] + [fmt(s["flip_rate"][m][str(r)], 2) for r in s["rates"]]
        lines.append("| " + " | ".join(row) + " |")

    if s["defense"]:
        lines.append(f"\n## XAI-Guided Topological Defense (attack rate applied per node)\n")
        lines.append("Target-class probability, averaged over nodes: clean → attacked → defended.\n")
        lines.append("| Attack | Clean prob | Attacked prob | Defended prob | Recovered | Edges pruned |")
        lines.append("|---|---|---|---|---|---|")
        for m in METHODS:
            if m not in s["defense"]:
                continue
            d = s["defense"][m]
            recovered = None
            if d["defended_prob"] is not None and d["attacked_prob"] is not None:
                recovered = d["defended_prob"] - d["attacked_prob"]
            lines.append(f"| {m.capitalize()} | {fmt(d['clean_prob'])} | {fmt(d['attacked_prob'])} "
                         f"| {fmt(d['defended_prob'])} | {fmt(recovered)} "
                         f"| {fmt(d['pruned_edges'], 1)} / {fmt(d['neighborhood_edges'], 1)} |")

    lines.append("\n---\n_Generated by aggregate.py from the saved run artifact._\n")
    return "\n".join(lines)


def main():
    run_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join("web", "data.json")
    if not os.path.exists(run_path):
        print(f"Error: run file '{run_path}' not found. Run evaluate.py first.")
        sys.exit(1)

    with open(run_path) as f:
        run = json.load(f)

    summary = summarize(run)

    out_json = os.path.join("web", "results_summary.json")
    os.makedirs("web", exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    md = render_markdown(summary)
    with open("SUMMARY.md", "w") as f:
        f.write(md)

    print(md)
    print(f"\nWrote {out_json} and SUMMARY.md")


if __name__ == "__main__":
    main()
