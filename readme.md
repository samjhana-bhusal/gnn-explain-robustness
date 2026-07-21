# gnn-explain-robustness

**Robustness and Fidelity Analysis of Graph Neural Network Attribution Methods Under Topological Perturbations**

A benchmark study comparing how well popular GNN explainability methods hold up when the underlying graph topology is adversarially perturbed, plus two novel extensions for defense and homophily-aware attribution analysis.

## Overview

Graph Neural Networks (GNNs) are widely used for node classification, link prediction, and graph classification, but their message-passing mechanics make predictions hard to interpret. Explainability methods (e.g., GNNExplainer, SubgraphX/MCTS, counterfactual explanations) attempt to surface the subgraphs driving a prediction — but their **robustness to small topological changes** is largely untested. If a tiny, irrelevant edge perturbation flips the explanation while the model's prediction stays the same, the explanation cannot be trusted.

This project benchmarks three explanation paradigms under three adversarial perturbation models on the Cora citation network, and introduces two new research directions.

## Explanation Methods Evaluated

| Method | Paradigm | Description |
|---|---|---|
| GNNExplainer | Perturbation-based | Learns a continuous edge mask via mutual information maximization |
| Subgraph MCTS (SubgraphX) | Search-based | Uses Monte Carlo Tree Search to find high-scoring connected subgraphs |
| Counterfactual Explanations | Contrastive | Finds the minimal edge edit that flips a node's predicted label |

## Perturbation Models

- **Random Edge Perturbations** — random edge add/delete within the local neighborhood at rates ε ∈ {0.01, 0.05, 0.10, 0.15}
- **Degree-Targeted Perturbations** — connects local nodes to high-degree global hubs to dilute aggregation signal
- **Homophily-Disrupting Perturbations** — connects local nodes to cross-class nodes to directly attack label homophily

## Metrics

- **Fidelity-minus (F⁻)** — prediction drop when explanatory edges are removed
- **Fidelity-plus (F⁺)** — prediction drop when *only* explanatory edges are kept
- **Jaccard Robustness** — overlap between clean-graph and perturbed-graph explanations across perturbation rates

## Extensions (implemented in `extensions.py`)

1. **Homophily Attribution Invariance (HAI)** — a metric quantifying how much of an explanation's attribution weight sits on homophilous (same-label) edges, and how far that shifts under structural attack.
2. **XAI-Guided Topological Defense** — a sparsifier that prunes low-attribution neighborhood edges after an attack, testing whether explanation scores can filter adversarial injections.

## Key Findings (Cora, GCN, top-k=5, mean over 5 nodes)

All numbers below are regenerated from a saved run by `aggregate.py` (see [SUMMARY.md](SUMMARY.md)), not hand-written. Explanations are restricted to the target's 2-hop receptive field.

| Method | F⁻ ↑ | F⁺ ↓ | Clean HAI | Structure |
|---|---|---|---|---|
| GNNExplainer | 0.013 | 0.138 | 0.84 | Sparse edges |
| Subgraph MCTS | 0.197 | −0.039 | 0.88 | Connected subgraph |
| Counterfactual | 0.298 | −0.058 | 0.96 | Contrastive edges |

- **Predictions never flip** under any attack at any rate (flip rate 0.00) — so explanation instability is a *distinct* failure mode from prediction instability.
- **Robustness (mean top-k Jaccard @ 10% random perturbation):** Counterfactual ≈ **0.75**, Subgraph MCTS ≈ **0.62** (strikingly flat across all attacks), GNNExplainer ≈ **0.18**. Explainers with structural priors are far more stable than the continuous per-edge mask.
- **GNNExplainer's low fidelity is real:** its highest-weight mask edges are frequently *not* the causally load-bearing ones (target-incident edges rank ~95th percentile but a few other edges outrank them and dominate the top-k).
- **Defense is a negative result:** naive median-threshold pruning removes ~half the neighborhood and *lowers* the target-class probability rather than restoring it — a selective, injection-aware sparsifier is needed.

## Dataset

- **Cora** citation network, 2-layer GCN backbone, test accuracy 81.30%.

## Reproduce

```bash
python setup_env.py        # create .venv and install requirements
python train_gnn.py        # train the GCN, saves data/gcn_cora.pt
python evaluate.py         # run explainers + attacks + extensions -> web/data.json
python aggregate.py        # reduce the run to SUMMARY.md + web/results_summary.json
python run_dashboard.py    # serve the interactive dashboard at localhost:8000
```

`evaluate.py` is stochastic (MCTS rollouts and perturbation sampling are random), so exact numbers vary run to run; `aggregate.py` always reflects the latest `web/data.json`.

## Future Directions

- A selective, injection-aware sparsifier (score only *added* edges) to turn the defense into a positive result
- Multiple seeds and confidence intervals; heterophilous datasets; a GAT backbone
- Contrastive explanation regularization to penalize instability under perturbation
- Dynamic, explanation-guided message passing (self-explaining GNN layers)

## Citation

If you use this work, please cite the accompanying report (see `report/`).

## License

MIT (or your preferred license — update before publishing).