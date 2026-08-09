"""
Brute-force validation of Lemma 1 (locality of the attack surface).

If this test ever fails, the certificate's radius accounting is unsound, so it is
a hard correctness gate. We check, on several backbones and several target nodes:

  (a) NEGATIVE / soundness: flipping edges that are OUTSIDE the surface
      (deleting off-surface edges, inserting off-surface non-edges) leaves the
      target's logits bit-identical.

  (b) POSITIVE / teeth: at least some ON-surface edge deletions DO change the
      target's logits, proving the test is not vacuously passing because the
      model ignores its input.

Everything runs on CPU for exact float reproducibility.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gxstab.data import load_dataset
from gxstab.models import GNN, train_model
from gxstab.surface import surface_ball, in_surface, surface_edges
from gxstab.utils import set_seed

BACKBONES = ["GCN", "GAT", "GraphSAGE", "SGC"]
TOL = 1e-5


def _edge_set(edge_index):
    return {(min(a, b), max(a, b)) for a, b in zip(edge_index[0].tolist(), edge_index[1].tolist())}


def _add_undirected(edge_index, pairs):
    """Return a new edge_index with each {a,b} in pairs added in both directions."""
    if not pairs:
        return edge_index.clone()
    extra = []
    for a, b in pairs:
        extra.append([a, b])
        extra.append([b, a])
    extra = torch.tensor(extra, dtype=torch.long).t()
    return torch.cat([edge_index, extra], dim=1)


def _remove_undirected(edge_index, pairs):
    """Return a new edge_index with each {a,b} in pairs removed (both directions)."""
    drop = set()
    for a, b in pairs:
        drop.add((a, b))
        drop.add((b, a))
    keep = [
        i for i in range(edge_index.size(1))
        if (edge_index[0, i].item(), edge_index[1, i].item()) not in drop
    ]
    return edge_index[:, keep]


@pytest.fixture(scope="module")
def cora():
    set_seed(0)
    data, meta = load_dataset("cora")
    data = data.to("cpu")
    return data, meta


@pytest.mark.parametrize("backbone", BACKBONES)
def test_lemma1_offsurface_is_inert(cora, backbone):
    data, meta = cora
    set_seed(0)
    model = GNN(meta["num_features"], 64, meta["num_classes"], backbone, K=2)
    train_model(model, data, epochs=60)
    model.eval()

    K = model.K
    num_nodes = meta["num_nodes"]
    existing = _edge_set(data.edge_index)

    # A few representative targets: pick nodes with non-trivial neighborhoods.
    degrees = torch.zeros(num_nodes)
    for a in data.edge_index[0].tolist():
        degrees[a] += 1
    targets = torch.argsort(degrees, descending=True)[[5, 50, 200, 800]].tolist()

    with torch.no_grad():
        base = model(data.x, data.edge_index)

    checked_positive = False
    for v in targets:
        ball = surface_ball(v, K, data.edge_index, num_nodes)

        # --- (a) delete OFF-surface existing edges ---
        off_del = [
            e for e in existing
            if e[0] != e[1] and not in_surface(e[0], e[1], ball)
        ][:200]
        if off_del:
            ei = _remove_undirected(data.edge_index, off_del)
            with torch.no_grad():
                out = model(data.x, ei)
            assert torch.allclose(out[v], base[v], atol=TOL), (
                f"{backbone}: deleting off-surface edges changed target {v}"
            )

        # --- (a) insert OFF-surface non-edges ---
        off_ins, tries = [], 0
        while len(off_ins) < 50 and tries < 5000:
            tries += 1
            a = torch.randint(0, num_nodes, (1,)).item()
            b = torch.randint(0, num_nodes, (1,)).item()
            if a == b:
                continue
            e = (min(a, b), max(a, b))
            if e in existing or in_surface(a, b, ball):
                continue
            off_ins.append(e)
        if off_ins:
            ei = _add_undirected(data.edge_index, off_ins)
            with torch.no_grad():
                out = model(data.x, ei)
            assert torch.allclose(out[v], base[v], atol=TOL), (
                f"{backbone}: inserting off-surface non-edges changed target {v}"
            )

        # --- (b) teeth: some ON-surface deletion must matter ---
        on_edges = [e for e in surface_edges(v, K, data.edge_index, num_nodes)]
        for e in on_edges[:30]:
            ei = _remove_undirected(data.edge_index, [e])
            with torch.no_grad():
                out = model(data.x, ei)
            if not torch.allclose(out[v], base[v], atol=TOL):
                checked_positive = True
                break

    assert checked_positive, (
        f"{backbone}: no on-surface edge deletion changed any target -- "
        "test has no teeth, investigate"
    )
