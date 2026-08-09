"""
Prototype explainers, migrated to the unified interface.

  gnnexplainer -- PyG GNNExplainer restricted to the target's computation
                  subgraph (canonical usage), mask mapped back to global edges.
  mcts         -- the prototype's simplified subgraph search. Relabelled
                  honestly: this is UCT over node-removal states with a
                  probability-minus-size reward, NOT the Shapley-based SubgraphX.

Both are expensive; they live in the fragility comparison, not the smoothing
base set.
"""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig


def gnnexplainer_edge_scores(
    model, x, edge_index, node_idx, *, K=2, num_classes=None, epochs=200
):
    model.eval()
    subset, sub_ei, mapping, _ = k_hop_subgraph(
        node_idx=node_idx, num_hops=K, edge_index=edge_index, relabel_nodes=True
    )
    sub_x = x[subset]
    local_target = int(mapping.item())

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=epochs),
        explanation_type="model",
        node_mask_type=None,
        edge_mask_type="object",
        model_config=ModelConfig(
            mode="multiclass_classification",
            task_level="node",
            return_type="log_probs",
        ),
    )
    expl = explainer(sub_x, sub_ei, target_index=local_target)
    sub_mask = expl.edge_mask.detach().cpu().numpy()

    global_mask = np.zeros(edge_index.size(1))
    subset_list = subset.cpu().numpy().tolist()
    gu = edge_index[0].cpu().numpy()
    gv = edge_index[1].cpu().numpy()
    lookup = {(int(gu[i]), int(gv[i])): i for i in range(edge_index.size(1))}
    su = sub_ei[0].cpu().numpy()
    sv = sub_ei[1].cpu().numpy()
    for j in range(sub_ei.size(1)):
        g = (subset_list[su[j]], subset_list[sv[j]])
        gi = lookup.get(g)
        if gi is not None:
            global_mask[gi] = sub_mask[j]
    return global_mask


def subgraph_mcts_edge_scores(
    model, x, edge_index, node_idx, *, K=2, num_classes=None,
    max_rollouts=25, target_size=5, c_puct=1.0,
):
    model.eval()
    subset, sub_ei, mapping, _ = k_hop_subgraph(
        node_idx=node_idx, num_hops=K, edge_index=edge_index, relabel_nodes=True
    )
    sub_x = x[subset]
    target_sub = int(mapping.item())
    num_nodes = subset.size(0)
    with torch.no_grad():
        pred_class = int(model(x, edge_index)[node_idx].argmax())

    root = set(range(num_nodes))
    visits, rewards = {}, {}

    for _ in range(max_rollouts):
        state = root.copy()
        path = [tuple(sorted(state))]
        while len(state) > target_size:
            children = [state - {n} for n in state if n != target_sub]
            if not children:
                break
            unvisited = [c for c in children if tuple(sorted(c)) not in visits]
            if unvisited:
                state = unvisited[np.random.choice(len(unvisited))]
                path.append(tuple(sorted(state)))
                break
            pk = tuple(sorted(state))
            pn = visits[pk]
            best_uct, best = -1, None
            for c in children:
                ck = tuple(sorted(c))
                q = rewards[ck] / visits[ck]
                uct = q + c_puct * np.sqrt(np.log(pn) / visits[ck])
                if uct > best_uct:
                    best_uct, best = uct, c
            state = best
            path.append(tuple(sorted(state)))

        while len(state) > target_size:
            cands = [n for n in state if n != target_sub]
            if not cands:
                break
            state.remove(np.random.choice(cands))

        active = [
            i for i in range(sub_ei.size(1))
            if int(sub_ei[0, i]) in state and int(sub_ei[1, i]) in state
        ]
        with torch.no_grad():
            out = model(sub_x, sub_ei[:, active])
            prob = float(torch.exp(out[target_sub, pred_class]))
        reward = prob - 0.1 * (len(state) / num_nodes)
        for p in path:
            visits[p] = visits.get(p, 0) + 1
            rewards[p] = rewards.get(p, 0.0) + reward

    best_state, best_q = None, -1e9
    for sk, cnt in visits.items():
        if cnt > 0 and len(sk) <= target_size:
            q = rewards[sk] / cnt
            if q > best_q:
                best_q, best_state = q, set(sk)
    if best_state is None:
        best_state = root

    mask = np.zeros(edge_index.size(1))
    subset_list = subset.cpu().numpy().tolist()
    local = {g: i for i, g in enumerate(subset_list)}
    gu = edge_index[0].cpu().numpy()
    gv = edge_index[1].cpu().numpy()
    for i in range(edge_index.size(1)):
        a, b = int(gu[i]), int(gv[i])
        if a in local and b in local and local[a] in best_state and local[b] in best_state:
            mask[i] = 1.0
    return mask
