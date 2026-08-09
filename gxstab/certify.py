"""
Top-k invariance certificate for the smoothed explainer.

Threat model. The adversary edits the *clean* graph within a radius (r_d, r_i):
up to r_d deletions of present surface edges and up to r_i insertions from the
declared candidate pool (smoothing.Surface.absent). By Lemma 1 nothing outside
the surface can matter, so this is the full adversary for the target.

Mechanism. Asymmetric sparse-smoothing (delete present w.p. p_d, insert absent
w.p. p_i) gives, for each present edge e, an inclusion probability pi_e. For a
binary event A = "e in top-k", the Neyman-Pearson lemma bounds how far the
smoothed probability can move between the smoothing distribution centered at the
clean graph (mu) and at any adversarial graph within radius (mu'):

    shift_lower(q; r_d, r_i) <= mu'(A) <= shift_upper(q; r_d, r_i)

where q is a (confidence) bound on mu(A) = pi_e. Both are exact discrete
water-filling over the likelihood-ratio-sorted regions of the flipped
coordinates -- see _regions below for the derivation.

Certificate. The smoothed top-k set S is provably invariant at radius (r_d, r_i)
iff the worst-case-lowest member probability still exceeds the
worst-case-highest non-member probability:

    min_{e in S} shift_lower(pi_low_e)  >  max_{e' notin S} shift_upper(pi_high_e')

The whole construction is validated against brute force in tests/test_certify.py.
"""

from __future__ import annotations

from math import comb

import numpy as np


# --------------------------------------------------------------------------- #
# Confidence bounds
# --------------------------------------------------------------------------- #
def clopper_pearson(count: int, n: int, alpha: float):
    """Exact (1-alpha) two-sided Clopper-Pearson interval for a binomial rate."""
    from scipy.stats import beta

    if n == 0:
        return 0.0, 1.0
    lo = 0.0 if count == 0 else beta.ppf(alpha / 2, count, n - count + 1)
    hi = 1.0 if count == n else beta.ppf(1 - alpha / 2, count + 1, n - count)
    return float(lo), float(hi)


# --------------------------------------------------------------------------- #
# Neyman-Pearson discrete shift bounds
# --------------------------------------------------------------------------- #
def _regions(r_d, r_i, p_d, p_i):
    """
    Enumerate likelihood-ratio regions of the flipped coordinates under the
    *symmetric* smoothing model: each smoothing coordinate is flipped from its
    current value with a fixed probability (p_d for present-edge coordinates,
    p_i for candidate-insertion coordinates). Genuine present edges the adversary
    does not touch are never noised, so they factor out (eta = 1) and do not
    appear here -- only the r_d deleted and r_i inserted coordinates do.

    For a deleted coordinate (clean value 1, adversarial value 0):
        mu(sample=1)=1-p_d, mu'(sample=1)=p_d.
    For an inserted coordinate (clean value 0, adversarial value 1):
        mu(sample=1)=p_i,   mu'(sample=1)=1-p_i.

    A region is indexed by (j_d, j_i) = (#deleted coords sampled=1, #inserted
    coords sampled=1). Symmetric, so mu' is mu with the roles swapped -- never
    degenerate for p in (0, 1/2]. Returns list of (mu_prob, mu2_prob, eta).
    """
    out = []
    for j_d in range(r_d + 1):
        mu_d = comb(r_d, j_d) * (1 - p_d) ** j_d * p_d ** (r_d - j_d)
        mu2_d = comb(r_d, j_d) * p_d ** j_d * (1 - p_d) ** (r_d - j_d)
        for j_i in range(r_i + 1):
            mu_i = comb(r_i, j_i) * p_i ** j_i * (1 - p_i) ** (r_i - j_i)
            mu2_i = comb(r_i, j_i) * (1 - p_i) ** j_i * p_i ** (r_i - j_i)
            mu = mu_d * mu_i
            mu2 = mu2_d * mu2_i
            eta = (mu2 / mu) if mu > 0 else float("inf")
            out.append((mu, mu2, eta))
    return out


def shift_lower(q, r_d, r_i, p_d, p_i):
    """min mu'(A) s.t. mu(A) >= q. Fill mu-mass q on smallest-eta regions."""
    if q <= 0:
        return 0.0
    regions = sorted(_regions(r_d, r_i, p_d, p_i), key=lambda t: t[2])
    remaining, acc = q, 0.0
    for mu, mu2, _ in regions:
        if remaining <= 0:
            break
        take = min(mu, remaining)
        acc += mu2 * (take / mu) if mu > 0 else 0.0
        remaining -= take
    return acc


def shift_upper(q, r_d, r_i, p_d, p_i):
    """max mu'(A) s.t. mu(A) <= q. Fill mu-mass q on largest-eta regions."""
    if q <= 0:
        return 0.0
    regions = sorted(_regions(r_d, r_i, p_d, p_i), key=lambda t: t[2], reverse=True)
    remaining, acc = q, 0.0
    for mu, mu2, _ in regions:
        if remaining <= 0:
            break
        take = min(mu, remaining)
        acc += mu2 * (take / mu) if mu > 0 else 0.0
        remaining -= take
    return acc


# --------------------------------------------------------------------------- #
# Top-k invariance certificate
# --------------------------------------------------------------------------- #
def _pi_bounds(pi_data, alpha):
    """
    Return {edge: (pi_low, pi_high)}. Accepts either exact pi (dict edge->float,
    used in validation) or MC data with counts (dict with 'counts','n_samples').
    Confidence alpha is split across edges by Bonferroni.
    """
    if "counts" in pi_data:
        counts, n = pi_data["counts"], pi_data["n_samples"]
        m = max(len(counts), 1)
        a = alpha / m
        return {e: clopper_pearson(counts[e], n, a) for e in counts}
    # exact pi
    return {e: (float(p), float(p)) for e, p in pi_data.items()}


def certify_topk(pi_data, k, p_d, p_i, radii, alpha=0.05):
    """
    Certify the smoothed top-k at each (r_d, r_i) in ``radii``.

    Returns dict with the smoothed top-k set and, per radius, whether the set is
    provably invariant plus the decision margin.
    """
    bounds = _pi_bounds(pi_data, alpha)
    pi_hat = pi_data["pi_hat"] if "pi_hat" in pi_data else {e: (lo + hi) / 2 for e, (lo, hi) in bounds.items()}
    ranked = sorted(pi_hat.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    S = [e for e, _ in ranked[:k]]
    comp = [e for e, _ in ranked[k:]]

    results = {}
    for (r_d, r_i) in radii:
        member_low = min((shift_lower(bounds[e][0], r_d, r_i, p_d, p_i) for e in S), default=1.0)
        nonmember_high = max((shift_upper(bounds[e][1], r_d, r_i, p_d, p_i) for e in comp), default=0.0)
        results[(r_d, r_i)] = {
            "certified": member_low > nonmember_high,
            "member_low": member_low,
            "nonmember_high": nonmember_high,
            "margin": member_low - nonmember_high,
        }
    return {"topk": S, "complement": comp, "radii": results}


def _allocations(r, mode, n_present):
    """Radius-r splits (r_d, r_i) for the chosen threat model."""
    if mode == "insert":                      # injection attack: add edges only
        return [(0, r)]
    if mode == "delete":                      # evidence-removal attack
        return [(r, 0)] if r <= n_present else []
    # both: adversary picks the worst split of a total budget r
    return [(rd, r - rd) for rd in range(r + 1) if rd <= n_present]


def certified_radius(pi_data, k, p_d, p_i, max_r=6, alpha=0.05, mode="insert"):
    """
    Largest budget r for which the smoothed top-k is certified against every
    allocation permitted by ``mode``:

      "insert" (default) -- injection attacks that only add up to r edges from
                            the candidate pool. This is the paper's primary
                            threat model (can an attacker inject edges to change
                            which genuine edges are highlighted?). Deleting an
                            explanation edge is out of scope here, which is why
                            insertion-only radii are meaningfully certifiable
                            while deletion of the certified edge is not.
      "delete"           -- remove up to r present surface edges.
      "both"             -- worst split of a total budget r.
    """
    n_present = len(pi_data["counts"]) if "counts" in pi_data else len(pi_data)
    best = 0
    for r in range(1, max_r + 1):
        allocations = _allocations(r, mode, n_present)
        if not allocations:
            break
        out = certify_topk(pi_data, k, p_d, p_i, allocations, alpha=alpha)
        if all(out["radii"][a]["certified"] for a in allocations):
            best = r
        else:
            break
    return best
