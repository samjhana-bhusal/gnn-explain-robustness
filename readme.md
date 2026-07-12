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

## Novel Contributions

1. **XAI-Guided Topological Defense** — using explanation confidence scores to sparsify/prune adversarial edges and restore classification accuracy.
2. **Homophily Attribution Invariance (HAI)** — a metric quantifying how much an explanation's focus shifts from homophilous to heterophilous edges under structural attack.

## Key Findings (Cora, GCN, top-k=5)

| Method | F⁻ ↑ | F⁺ ↓ | Structure |
|---|---|---|---|
| GNNExplainer | 0.284 | 0.125 | Sparse edges |
| Subgraph MCTS | 0.221 | 0.084 | Connected subgraph |
| Counterfactual | 0.385 | 0.442 | Contrastive edges |

- GNNExplainer's Jaccard similarity collapses to **0.31** under 10% random edge perturbation.
- Subgraph MCTS retains **0.68** Jaccard similarity at the same perturbation rate, showing greater structural robustness.

## Dataset

- **Cora** citation network, 2-layer GCN backbone, test accuracy 81.30%.

## Future Directions

- Contrastive explanation regularization to penalize instability under perturbation
- Dynamic, explanation-guided message passing (self-explaining GNN layers)

## Citation

If you use this work, please cite the accompanying report (see `paper/`).

## License

MIT (or your preferred license — update before publishing).