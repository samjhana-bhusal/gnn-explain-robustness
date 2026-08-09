"""
Topological perturbation model  (rewritten).

Fixes three defects in the prototype's ``perturb_graph_topology``:

  * Budget is counted over the **attack surface** |A_v| of the target, not over
    |E_global|. "Flip r% of the graph" injected ~1580 edges into a ~275-edge
    neighborhood in the old code; here a radius-r attack flips exactly r surface
    entries, which is what a small, meaningful perturbation actually is.
  * Every flip is **symmetric** (both directions added/removed together), so the
    graph stays undirected as the model assumes.
  * Perturbations are confined to A_v, so they can actually influence the target
    (the old code mostly hit distance>=K edges that provably cannot, which is
    why its prediction-flip rate was a spurious 0.00).

An "edge flip" is an involution on unordered pairs: an existing surface edge is
deleted, a surface non-edge is inserted. The perturbation *radius* is the number
of flips; the ``method`` chooses which pairs are eligible.
"""

from __future__ import annotations

import numpy as np
import torch

from gxstab.surface import surface_ball, surface_edges, in_surface


def _edge_set(edge_index):
    return {
        (min(a, b), max(a, b))
        for a, b in zip(edge_index[0].tolist(), edge_index[1].tolist())
        if a != b
    }


def _to_edge_index(pairs, device):
    """Undirected pair set -> directed edge_index (both orientations)."""
    if not pairs:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    rows, cols = [], []
    for a, b in pairs:
        rows += [a, b]
        cols += [b, a]
    return torch.tensor([rows, cols], dtype=torch.long, device=device)


def eligible_flips(v, K, edge_index, num_nodes, y=None, method="random", rng=None):
    """
    Return (deletable, insertable): two lists of unordered surface pairs the
    given attack family may flip.

      random    -- any surface edge deletable; any surface non-edge insertable.
      degree    -- insertions connect the ball to high-degree global hubs
                   (dilution attack); deletions from surface edges.
      homophily -- insertions connect the ball to different-label nodes
                   (homophily-disruption); deletions target *same-label*
                   surface edges (removing supporting evidence).

    Insertable lists are capped implicitly by the candidate construction so the
    dense-graph case does not blow up memory.
    """
    rng = rng or np.random.default_rng()
    ball = surface_ball(v, K, edge_index, num_nodes)
    ball_list = sorted(ball)
    existing = _edge_set(edge_index)
    deletable = surface_edges(v, K, edge_index, num_nodes)

    if method == "random":
        insertable = _random_surface_nonedges(ball_list, existing, num_nodes, rng, cap=2000)

    elif method == "degree":
        deg = np.zeros(num_nodes)
        for a in edge_index[0].tolist():
            deg[a] += 1
        hubs = np.argsort(deg)[-50:]
        insertable = []
        for a in ball_list:
            for h in hubs:
                h = int(h)
                if a == h:
                    continue
                e = (min(a, h), max(a, h))
                if e not in existing:
                    insertable.append(e)

    elif method == "homophily":
        if y is None:
            raise ValueError("homophily attack needs labels y")
        labels = y.cpu().numpy()
        insertable = []
        cand = _random_surface_nonedges(ball_list, existing, num_nodes, rng, cap=4000)
        for (a, b) in cand:
            # keep only cross-label insertions
            if labels[a] != labels[b]:
                insertable.append((a, b))
        # deletions bias toward removing same-label (supporting) edges
        deletable = [e for e in deletable if labels[e[0]] == labels[e[1]]] or deletable

    else:
        raise ValueError(f"unknown method {method!r}")

    return deletable, list(dict.fromkeys(insertable))


def _random_surface_nonedges(ball_list, existing, num_nodes, rng, cap):
    """Sample up to ``cap`` surface non-edges (>=1 endpoint in the ball)."""
    out, seen = [], set()
    tries = 0
    max_tries = cap * 20
    while len(out) < cap and tries < max_tries:
        tries += 1
        a = int(ball_list[rng.integers(len(ball_list))])
        b = int(rng.integers(num_nodes))
        if a == b:
            continue
        e = (min(a, b), max(a, b))
        if e in existing or e in seen:
            continue
        seen.add(e)
        out.append(e)
    return out


def perturb(
    v, edge_index, num_nodes, radius, *, K=2, y=None, method="random",
    add_frac=0.5, rng=None,
):
    """
    Apply a radius-``radius`` surface perturbation around target ``v``.

    ``radius`` flips are split into insertions (``add_frac``) and deletions.
    Returns the perturbed directed ``edge_index`` and the set of flipped pairs
    (so the certificate/attack can account for exactly what changed).
    """
    rng = rng or np.random.default_rng()
    device = edge_index.device
    if radius <= 0:
        return edge_index.clone(), set()

    deletable, insertable = eligible_flips(
        v, K, edge_index, num_nodes, y=y, method=method, rng=rng
    )

    n_add = min(int(round(radius * add_frac)), len(insertable))
    n_del = min(radius - n_add, len(deletable))

    add = _sample(insertable, n_add, rng)
    dele = _sample(deletable, n_del, rng)

    base = _edge_set(edge_index)
    new_pairs = (base | set(add)) - set(dele)
    flipped = set(add) | set(dele)

    return _to_edge_index(sorted(new_pairs), device), flipped


def _sample(pool, n, rng):
    if n <= 0 or not pool:
        return []
    idx = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
    return [pool[i] for i in idx]
