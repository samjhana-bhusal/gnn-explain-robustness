"""
Subgraph restriction for fast smoothing.

By Lemma 1, a target's K-layer output depends only on its K-hop ball, and the
degrees of those nodes depend only on edges within the (K+1)-hop ball. So we can
extract the (K+1)-hop computation subgraph once, relabel it into local node ids,
and run every smoothing draw on that small graph -- identical target logits and
explanations, but on ~50-200 nodes instead of the whole graph.

``LocalContext`` holds the relabeled subgraph plus the global<->local maps and
the surface expressed in local ids, so smoothing.py can operate entirely locally
and translate the final inclusion-probability keys back to global edges.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch_geometric.utils import k_hop_subgraph

from gxstab.surface import surface_edges, surface_ball


@dataclass
class LocalContext:
    v_global: int
    v_local: int
    K: int
    subset: torch.Tensor          # local -> global node id
    global_to_local: dict
    x_local: torch.Tensor
    edge_index_local: torch.Tensor
    present_local: list           # surface edges as local (min,max) pairs
    absent_local: list            # candidate insertions as local (min,max) pairs
    present_global: list          # aligned with present_local
    absent_global: list


def build_local_context(model, x, edge_index, v, num_nodes, K=2, insert_cap=None):
    # (K+1)-hop subgraph preserves B_K node degrees, hence the target's logits.
    subset, sub_ei, mapping, _ = k_hop_subgraph(
        node_idx=v, num_hops=K + 1, edge_index=edge_index,
        relabel_nodes=True, num_nodes=num_nodes,
    )
    g2l = {int(g): i for i, g in enumerate(subset.tolist())}
    v_local = int(mapping.item())
    x_local = x[subset]

    # Surface computed on the local subgraph (identical ball, since <=K hops).
    present_local = surface_edges(v_local, K, sub_ei, subset.size(0))
    ball_local = sorted(surface_ball(v_local, K, sub_ei, subset.size(0)))
    present_set = set(present_local)
    absent_local = []
    for i in range(len(ball_local)):
        for j in range(i + 1, len(ball_local)):
            e = (ball_local[i], ball_local[j])
            if e not in present_set:
                absent_local.append(e)
                if insert_cap is not None and len(absent_local) >= insert_cap:
                    break
        if insert_cap is not None and len(absent_local) >= insert_cap:
            break

    l2g = subset.tolist()

    def to_global(e):
        a, b = l2g[e[0]], l2g[e[1]]
        return (min(a, b), max(a, b))

    return LocalContext(
        v_global=v, v_local=v_local, K=K, subset=subset, global_to_local=g2l,
        x_local=x_local, edge_index_local=sub_ei,
        present_local=present_local, absent_local=absent_local,
        present_global=[to_global(e) for e in present_local],
        absent_global=[to_global(e) for e in absent_local],
    )
