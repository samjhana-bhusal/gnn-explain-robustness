# gxstab — Certified Stability of Node-Level GNN Explanations

**Certified Stability of Node-Level Graph Neural Network Explanations under
Receptive-Field-Localized Randomized Smoothing**

Post-hoc GNN explanations are fragile: a small, *prediction-preserving* edge
injection can change which edges an explainer highlights. This project gives the
first **node-level, permutation-invariant, worst-case certificate** that a target
node's top-*k* edge explanation cannot be changed by injecting up to *r* edges —
plus the empirical study characterizing when that guarantee holds.

## The idea in one paragraph

For a *K*-layer message-passing GNN, only edges incident to the target's *K*-hop
ball can change its explanation (**Lemma 1**, brute-force validated). So the
adversary's entire action space is a receptive-field *attack surface* of tens to
hundreds of edges, not the whole graph. We smooth a base explainer with symmetric
edge-flip noise on that surface, estimate each edge's top-*k* **inclusion
probability**, and prove — via a discrete Neyman–Pearson analysis — a **certified
radius** of top-*k* invariance to edge injection. Locality is what makes the radii
non-vacuous and the computation cheap.

## Key results (Cora/GCN and heterophilous graphs)

- **Majority of nodes certify**: e.g. PubMed mean certified radius **2.14**, 72% of
  targets provably robust to ≥1 injected edge, 54% to ≥2 (3 seeds, 95% CIs).
- **Monotonic smoothing tradeoff**: mean r\* rises 0.35 → 3.35 as insertion
  probability p_i goes 0.1 → 0.4.
- **Sound lower bound**: a greedy prediction-preserving injection attack never
  breaks a node below its certified radius.
- **Locality is free lunch**: subgraph-restricted smoothing gives identical radii
  to full-graph smoothing, orders of magnitude faster.

## Package layout (`gxstab/`)

| Module | Role |
|---|---|
| `data.py` | dataset registry (Planetoid + heterophilous), homophily measurement |
| `models.py` | GCN / GAT / GraphSAGE / APPNP / SGC backbones, trainer |
| `surface.py` | receptive-field attack surface + **Lemma 1** |
| `subgraph.py` | (K+1)-hop `LocalContext` for fast, exact smoothing |
| `explainers/` | one `edge_scores` interface: grad, IG, occlusion, PGExplainer-style, GNNExplainer, MCTS, parameter-free baselines |
| `perturb.py` | symmetric, surface-budget topological attacks |
| `metrics.py` | robust fidelity, top-k Jaccard, rank correlation, prediction-conditioning, HAI |
| `smoothing.py` | edge-flip smoothing + inclusion probabilities (MC and exact) |
| `certify.py` | discrete Neyman–Pearson bounds + top-k invariance certificate |
| `attack.py` | greedy prediction-preserving injection attack (GXAttack-lite) |
| `experiments/` | sweep drivers, aggregation, figures |

## Reproduce

```bash
python -m gxstab.experiments.run_all          # certified + fragility sweeps -> results/*.parquet
python -m gxstab.experiments.run_extras       # locality ablation + attack-vs-certified
python -m gxstab.experiments.aggregate_results  # print paper tables (bootstrap CIs)
python -m gxstab.experiments.figures          # write report/figs/*.pdf
pytest tests/                                 # Lemma 1 + certificate brute-force validation (12 tests)
```

Add `--quick` to the drivers for a fast, small-scale run.

## Correctness gates (`tests/`)

- **Lemma 1** — flipping any off-surface edge leaves the target's logits
  bit-identical (GCN/GAT/GraphSAGE/SGC).
- **Certificate** — the Neyman–Pearson bounds are validated against exhaustive
  enumeration and are tight; end-to-end, every injection within the certified
  radius provably preserves the smoothed top-k; a negative control confirms the
  certificate refuses to over-claim.

## Report

`report/main.tex` — the paper. `report/ref.bib` — bibliography.

## Note on the earlier prototype

This supersedes an earlier Cora/GCN fragility prototype (GNNExplainer / Subgraph
MCTS / counterfactual under three attacks, with HAI and an XAI-guided defense).
That code is migrated into `gxstab/` (the explainers and HAI metric live on as
baselines); the earlier flat scripts and dashboard remain for reference.

## License

MIT.
