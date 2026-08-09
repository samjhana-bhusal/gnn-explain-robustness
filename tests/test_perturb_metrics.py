"""Invariants for the perturbation model and metric edge cases."""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gxstab.data import load_dataset
from gxstab import metrics as M
from gxstab.perturb import perturb, _edge_set
from gxstab.surface import surface_ball, in_surface
from gxstab.utils import set_seed


def _load():
    set_seed(0)
    data, meta = load_dataset("cora")
    return data.to("cpu"), meta


def test_perturbation_is_symmetric_and_on_surface():
    data, meta = _load()
    N = meta["num_nodes"]
    rng = np.random.default_rng(0)
    v = 2661
    for method in ("random", "degree", "homophily"):
        pei, flipped = perturb(v, data.edge_index, N, radius=6, K=2, y=data.y, method=method, rng=rng)
        pairs = set(map(tuple, pei.t().tolist()))
        # undirected: every (a,b) has its (b,a)
        assert all((b, a) in pairs for a, b in pairs), f"{method}: not symmetric"
        # every flipped pair is on the surface
        ball = surface_ball(v, 2, data.edge_index, N)
        assert all(in_surface(a, b, ball) for a, b in flipped), f"{method}: off-surface flip"


def test_perturbation_radius_zero_is_identity():
    data, meta = _load()
    pei, flipped = perturb(2661, data.edge_index, meta["num_nodes"], radius=0, K=2, y=data.y)
    assert len(flipped) == 0
    assert _edge_set(pei) == _edge_set(data.edge_index)


def test_jaccard_edge_cases():
    assert M.topk_jaccard([], []) == 1.0
    assert M.topk_jaccard([(1, 2)], []) == 0.0
    # orientation-insensitive
    assert M.topk_jaccard([(1, 2)], [(2, 1)]) == 1.0
    assert M.topk_jaccard([(1, 2), (3, 4)], [(2, 1)]) == 0.5


def test_hai_undefined_when_empty():
    y = torch.tensor([0, 0, 1, 1])
    assert M.homophily_attribution_index([], y) is None
    assert M.homophily_attribution_index([(0, 1)], y) == 1.0  # same label
    assert M.homophily_attribution_index([(0, 2)], y) == 0.0  # cross label
    assert M.hai_shift(None, 0.5) is None
    assert abs(M.hai_shift(0.8, 0.6) - (-0.2)) < 1e-9
