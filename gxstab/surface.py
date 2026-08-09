"""
Receptive-field attack surface  (the locality contribution).

For a K-layer message-passing GNN and a target node ``v``, the output logits
``h_v^(K)`` depend only on a bounded computation graph. Concretely:

    Lemma 1 (locality). An undirected edge {a, b} can influence h_v^(K) only if
    dist_G(v, a) <= K or dist_G(v, b) <= K.

Define the K-hop ball B = B_K(v). The **attack surface** A_v is the set of node
pairs {a, b} with at least one endpoint in B: these are the only edges whose
deletion, and the only non-edges whose insertion, can change h_v^(K).

Why the ball is B_K and not the messaging depth B_{K-1}. Two distinct channels
couple an edge into h_v^(K):

  1. Messages. x_q for a node q at distance d <= K reaches v; the edges carrying
     it are incident to B_{K-1}(v).
  2. Degree normalization. GCN's symmetric coefficient 1/sqrt(deg(p) deg(q))
     makes h_v^(K) depend on the *degrees* of nodes at distance exactly K
     (they appear in the innermost layer-1 coefficient). An edge incident to a
     distance-K node changes that degree, hence h_v^(K), even though it carries
     no message inward. This is the channel the brute-force test in
     tests/test_surface.py caught; it forces the surface out to B_K.

Using B_K is exact for GCN and a sound superset for aggregators without
degree coupling (e.g. mean-SAGE), so certified radii computed over it are never
optimistic.

Two consequences the rest of the package relies on:

  * Any perturbation disjoint from A_v leaves the prediction *and* any
    receptive-field-restricted explanation exactly unchanged (verified by
    brute force in tests/test_surface.py).
  * Inserting an edge with *both* endpoints outside B cannot pull any node into
    B: the new edge only shortcuts two already-distant nodes, so distances from
    v to nodes originally within K are unchanged. Hence the surface computed on
    the clean graph is sound for insertions too, and certified radii can be
    counted over |A_v| (tens–hundreds) instead of |E| (thousands).
"""

from __future__ import annotations

from collections import deque

import torch


def k_hop_ball(v: int, k: int, edge_index, num_nodes: int) -> set:
    """
    Nodes within ``k`` hops of ``v`` (BFS on the undirected graph), inclusive of
    ``v`` itself. ``k = 0`` returns ``{v}``.
    """
    if k < 0:
        return set()
    adj = _adjacency(edge_index, num_nodes)
    seen = {v}
    frontier = deque([(v, 0)])
    while frontier:
        node, d = frontier.popleft()
        if d == k:
            continue
        for nb in adj[node]:
            if nb not in seen:
                seen.add(nb)
                frontier.append((nb, d + 1))
    return seen


def _adjacency(edge_index, num_nodes: int):
    adj = [[] for _ in range(num_nodes)]
    u = edge_index[0].tolist()
    w = edge_index[1].tolist()
    for a, b in zip(u, w):
        adj[a].append(b)
    return adj


def surface_ball(v: int, K: int, edge_index, num_nodes: int) -> set:
    """
    The K-hop ball B_K(v) whose incidence defines the attack surface. K (not
    K-1) because degree normalization couples distance-K node degrees into
    h_v^(K); see the module docstring.
    """
    return k_hop_ball(v, K, edge_index, num_nodes)


def in_surface(a: int, b: int, ball: set) -> bool:
    """A pair {a, b} is on the surface iff at least one endpoint is in the ball."""
    return (a in ball) or (b in ball)


def surface_edges(v: int, K: int, edge_index, num_nodes: int):
    """
    Existing undirected edges on the attack surface, as a sorted list of
    (min, max) tuples. These are the edges a deletion attack (or the smoothing
    deletion noise) may touch.
    """
    ball = surface_ball(v, K, edge_index, num_nodes)
    u = edge_index[0].tolist()
    w = edge_index[1].tolist()
    out = set()
    for a, b in zip(u, w):
        if a == b:
            continue
        if in_surface(a, b, ball):
            out.add((min(a, b), max(a, b)))
    return sorted(out)


def surface_nonedges(v: int, K: int, edge_index, num_nodes: int, cap: int | None = None):
    """
    Non-edges on the attack surface: pairs {a, b} with an endpoint in the ball
    that are not currently edges. This set can be large (|B| x N in the worst
    case), so ``cap`` optionally bounds how many are returned (deterministic
    order). Used for the insertion side of the perturbation model and smoothing.
    """
    ball = surface_ball(v, K, edge_index, num_nodes)
    existing = set()
    u = edge_index[0].tolist()
    w = edge_index[1].tolist()
    for a, b in zip(u, w):
        existing.add((min(a, b), max(a, b)))

    out = []
    ball_sorted = sorted(ball)
    for a in ball_sorted:
        for b in range(num_nodes):
            if a == b:
                continue
            e = (min(a, b), max(a, b))
            if e in existing:
                continue
            out.append(e)
            if cap is not None and len(out) >= cap:
                return out
    # De-duplicate pairs where both endpoints are in the ball (added twice).
    return sorted(set(out))


def surface_size(v: int, K: int, edge_index, num_nodes: int) -> dict:
    """Report |B|, #surface edges, and (uncapped) #surface non-edges."""
    ball = surface_ball(v, K, edge_index, num_nodes)
    n_edges = len(surface_edges(v, K, edge_index, num_nodes))
    # |non-edges| = sum over a in ball of (N - 1 - deg-within-count), but we count
    # unordered pairs with an endpoint in ball, minus existing surface edges.
    n_pairs_with_ball_endpoint = _count_pairs_with_endpoint(ball, num_nodes)
    return {
        "ball_size": len(ball),
        "surface_edges": n_edges,
        "surface_nonedges": n_pairs_with_ball_endpoint - n_edges,
        "surface_total": n_pairs_with_ball_endpoint,
    }


def _count_pairs_with_endpoint(ball: set, num_nodes: int) -> int:
    """Number of unordered pairs {a,b}, a!=b, with at least one endpoint in ball."""
    b = len(ball)
    # pairs with >=1 endpoint in ball = C(N,2) - C(N-b,2)
    def c2(n):
        return n * (n - 1) // 2
    return c2(num_nodes) - c2(num_nodes - b)
