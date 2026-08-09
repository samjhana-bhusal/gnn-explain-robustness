"""
Brute-force validation of the certificate. Hard correctness gate.

Two levels:

  1. NP shift bounds (pure math): for small radii, enumerate the flipped-
     coordinate space, and for many random events A verify
         shift_lower(mu(A)) <= mu'(A) <= shift_upper(mu(A)),
     plus tightness (threshold events achieve the bounds) and that region
     masses sum to 1.

  2. End-to-end soundness: on a tiny graph with a real explainer, take the
     certified radius r*, then enumerate EVERY adversarial graph within r* and
     assert the smoothed top-k is unchanged. If this fails, the certificate is
     unsound.
"""

import os
import sys
import itertools

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gxstab import certify as C
from gxstab.certify import _regions, shift_lower, shift_upper
from gxstab.utils import set_seed


# --------------------------------------------------------------------------- #
# Level 1: NP shift bounds
# --------------------------------------------------------------------------- #
def _enumerate_atoms(r_d, r_i, p_d, p_i):
    """All 2^(r_d+r_i) atoms with (mu, mu') probabilities under the symmetric model."""
    atoms = []
    for bits in itertools.product([0, 1], repeat=r_d + r_i):
        mu = mu2 = 1.0
        for c in range(r_d):           # deleted coord: clean=1, adv=0
            s = bits[c]
            mu *= (1 - p_d) if s == 1 else p_d
            mu2 *= p_d if s == 1 else (1 - p_d)
        for c in range(r_i):           # inserted coord: clean=0, adv=1
            s = bits[r_d + c]
            mu *= p_i if s == 1 else (1 - p_i)
            mu2 *= (1 - p_i) if s == 1 else p_i
        atoms.append((bits, mu, mu2))
    return atoms


def test_regions_are_distributions():
    for (r_d, r_i, p_d, p_i) in [(2, 1, 0.2, 0.05), (3, 2, 0.3, 0.1), (1, 3, 0.15, 0.15)]:
        regs = _regions(r_d, r_i, p_d, p_i)
        assert abs(sum(m for m, _, _ in regs) - 1.0) < 1e-9
        assert abs(sum(m2 for _, m2, _ in regs) - 1.0) < 1e-9


def test_np_bounds_hold_and_are_tight():
    rng = np.random.default_rng(0)
    for (r_d, r_i, p_d, p_i) in [(2, 1, 0.2, 0.05), (2, 2, 0.3, 0.1), (3, 0, 0.25, 0.05)]:
        atoms = _enumerate_atoms(r_d, r_i, p_d, p_i)
        n = len(atoms)
        # random events
        for _ in range(300):
            sub = rng.random(n) < rng.random()
            muA = sum(atoms[i][1] for i in range(n) if sub[i])
            mu2A = sum(atoms[i][2] for i in range(n) if sub[i])
            lo = shift_lower(muA, r_d, r_i, p_d, p_i)
            hi = shift_upper(muA, r_d, r_i, p_d, p_i)
            assert lo - 1e-9 <= mu2A <= hi + 1e-9, (
                f"NP bound violated: {lo} <= {mu2A} <= {hi} at muA={muA}"
            )
        # tightness: threshold-on-eta events achieve the bounds exactly
        order = sorted(range(n), key=lambda i: atoms[i][2] / atoms[i][1] if atoms[i][1] > 0 else float("inf"))
        cum_mu = 0.0
        for t in range(n + 1):
            chosen = order[:t]
            muA = sum(atoms[i][1] for i in chosen)
            mu2A = sum(atoms[i][2] for i in chosen)
            assert abs(shift_lower(muA, r_d, r_i, p_d, p_i) - mu2A) < 1e-9


# --------------------------------------------------------------------------- #
# Level 2: end-to-end soundness on a tiny graph
# --------------------------------------------------------------------------- #
def _tiny_setup(absent_winner=False):
    """
    A small graph plus a *deterministic, well-separated* edge scorer exposed as
    topk(bitmask). We use a fixed-weight scorer rather than a trained GNN here on
    purpose: this test validates the pi -> certified-radius -> invariance chain,
    which must hold for ANY deterministic explainer. Explainer realism (real
    grad/occlusion/GNNExplainer over trained models) is exercised by the other
    tests. A random-init GNN gives near-uniform scores that certify nothing, so
    it cannot validate the enumeration branch.

    ``absent_winner=True`` makes one injectable candidate outrank the dominant
    edge when inserted, so a single injection flips the top-1. The certificate
    must then REFUSE (radius 0) -- the negative control against over-claiming.
    """
    from gxstab.smoothing import build_surface

    v, K, k = 0, 2, 1
    N = 9
    base_pairs = [(0, 1), (0, 2), (1, 3), (2, 3), (2, 4), (3, 5), (4, 6), (1, 2)]

    def edge_index_from(pairs):
        if not pairs:
            return torch.empty((2, 0), dtype=torch.long)
        r, c = [], []
        for a, b in pairs:
            r += [a, b]; c += [b, a]
        return torch.tensor([r, c], dtype=torch.long)

    clean_ei = edge_index_from(base_pairs)
    surface = build_surface(v, clean_ei, N, K=K, insert_cap=3)
    coords = list(surface.present) + list(surface.absent)
    n = len(coords)
    present_clean = set(surface.present)

    # One strongly dominant target-incident edge: under an insertion-only
    # (injection) attack it stays top-1 whenever present, so the smoothed top-1
    # is stable and certifiable. Inserted edges never outrank it.
    weight = {}
    for e in coords:
        if e == (0, 1):
            weight[e] = 100.0
        elif e == (0, 2):
            weight[e] = 90.0
        elif 0 in e:
            weight[e] = 5.0
        else:
            weight[e] = 1.0 + 0.05 * (e[0] + e[1])

    # Negative control: when a specific candidate is injected, it makes genuine
    # edge (0,2) rank above (0,1) -- so injecting it flips the smoothed top-1
    # from (0,1) to (0,2). The certificate must then refuse (radius 0).
    trigger = surface.absent[0] if (absent_winner and surface.absent) else None

    cache = {}

    def topk_for_mask(mask_bits):
        key = tuple(mask_bits)
        if key in cache:
            return cache[key]
        present = [coords[i] for i in range(n) if mask_bits[i]]
        w = dict(weight)
        if trigger is not None and trigger in present:
            w[(0, 2)] = 1000.0
        ranked = sorted(present, key=lambda e: (w[e], e), reverse=True)
        tk = {e for e in ranked[:k] if e in present_clean}
        cache[key] = tk
        return tk

    return coords, present_clean, surface, topk_for_mask, k


def _exact_pi(coords, center_present, topk_for_mask, p_d, p_i, present_clean):
    """Exact inclusion prob of each clean-present edge under smoothing centered
    at ``center_present`` (a set of coords currently present)."""
    n = len(coords)
    pi = {e: 0.0 for e in present_clean}
    for code in range(1 << n):
        bits = [(code >> i) & 1 for i in range(n)]
        prob = 1.0
        for i, c in enumerate(coords):
            is_genuine = c in present_clean
            present_here = c in center_present
            if is_genuine and p_d == 0.0:
                # insertion mode: genuine edges are never noised -> always present
                if bits[i] != 1:
                    prob = 0.0
                    break
                continue
            flip_p = p_d if is_genuine else p_i   # symmetric flip from current state
            if present_here:
                prob *= (1 - flip_p) if bits[i] == 1 else flip_p
            else:
                prob *= flip_p if bits[i] == 1 else (1 - flip_p)
        if prob == 0.0:
            continue
        tk = topk_for_mask(bits)
        for e in tk:
            pi[e] += prob
    return pi


def test_certificate_soundness_end_to_end():
    # Insertion smoothing mirrors the insertion (injection) threat: no deletion
    # noise (p_d=0), random edge additions at p_i. A genuine dominant edge is
    # then never noised out, so the certificate has power against injections.
    p_d, p_i = 0.0, 0.3
    coords, present_clean, surface, topk_for_mask, k = _tiny_setup()

    # exact pi at the clean center
    pi_clean = _exact_pi(coords, set(present_clean), topk_for_mask, p_d, p_i, present_clean)
    clean_topk = set(sorted(pi_clean, key=lambda e: (pi_clean[e], e), reverse=True)[:k])

    # Insertion-only (injection) threat model, matching certified_radius default.
    r_star = C.certified_radius(pi_clean, k, p_d, p_i, max_r=3, alpha=0.05, mode="insert")
    assert r_star >= 1, "expected a non-trivial certified radius on this setup"

    # enumerate every insertion-only adversarial graph within r_star (add up to
    # r absent candidate edges) and check the smoothed top-k is unchanged.
    absent_idx = [i for i, c in enumerate(coords) if c not in present_clean]
    for r in range(1, r_star + 1):
        for flip in itertools.combinations(absent_idx, r):
            center = set(present_clean)
            for i in flip:
                center.add(coords[i])       # insertion: add the absent candidate
            pi_adv = _exact_pi(coords, center, topk_for_mask, p_d, p_i, present_clean)
            adv_topk = set(sorted(pi_adv, key=lambda e: (pi_adv[e], e), reverse=True)[:k])
            assert adv_topk == clean_topk, (
                f"certified r*={r_star} but top-k changed at insertion radius {r}, "
                f"flip={flip}: {clean_topk} -> {adv_topk}"
            )


def test_certificate_refuses_when_injection_breaks_topk():
    """Negative control: if a single injection genuinely flips the top-1, the
    certificate must return radius 0 (never over-claim)."""
    p_d, p_i = 0.0, 0.3
    coords, present_clean, surface, topk_for_mask, k = _tiny_setup(absent_winner=True)
    pi_clean = _exact_pi(coords, set(present_clean), topk_for_mask, p_d, p_i, present_clean)

    # sanity: injecting the winner really does change the smoothed top-1
    winner = surface.absent[0]
    center = set(present_clean) | {winner}
    pi_adv = _exact_pi(coords, center, topk_for_mask, p_d, p_i, present_clean)
    clean_topk = set(sorted(pi_clean, key=lambda e: (pi_clean[e], e), reverse=True)[:k])
    adv_topk = set(sorted(pi_adv, key=lambda e: (pi_adv[e], e), reverse=True)[:k])
    assert adv_topk != clean_topk, "test setup failed to create a breaking injection"

    r_star = C.certified_radius(pi_clean, k, p_d, p_i, max_r=3, alpha=0.05, mode="insert")
    assert r_star == 0, f"certificate over-claimed r*={r_star} on a breakable top-k"
