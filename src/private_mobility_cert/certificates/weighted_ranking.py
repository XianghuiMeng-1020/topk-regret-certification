"""Rank-weighted multi-exchange certificates and adaptive-K.

Primary objects:
- position-weighted ranking utility U_{w,k}
- observed deficit D_{w,k}
- noise contrast Q_{w,k} via weighted order statistics
- full-ranking envelope and tighter moved-mass multi-exchange envelope
- simultaneous adaptive prefix K*

Probability statements are over mechanism randomness for fixed latent x.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from private_mobility_cert.certificates.rank_adaptive import (
    allocate_eta,
    solve_order_stat_threshold_continuous,
    solve_order_stat_threshold_integer,
)
from private_mobility_cert.certificates.topk_regret import TopKCertError


def validate_weights(w: np.ndarray) -> np.ndarray:
    """Require finite, nonnegative, nonincreasing weights."""
    arr = np.asarray(w, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise TopKCertError("weights must be nonempty")
    if not np.all(np.isfinite(arr)):
        raise TopKCertError("weights must be finite")
    if np.any(arr < 0):
        raise TopKCertError("weights must be nonnegative")
    if np.any(arr[1:] - arr[:-1] > 1e-12):
        raise TopKCertError("weights must be nonincreasing")
    return arr


def weight_family(name: str, k_max: int, *, lam: float = 0.9) -> np.ndarray:
    """Frozen empirical weight families (protocol-level)."""
    if k_max <= 0:
        raise TopKCertError("k_max must be positive")
    r = np.arange(1, k_max + 1, dtype=np.float64)
    if name == "dcg":
        return 1.0 / np.log2(r + 1.0)
    if name == "uniform":
        return np.ones(k_max, dtype=np.float64)
    if name == "reciprocal":
        return 1.0 / r
    if name == "geometric":
        return (lam ** (r - 1.0)).astype(np.float64)
    raise TopKCertError(f"unknown weight family {name}")


def clip_nonnegative(y: np.ndarray) -> np.ndarray:
    return np.maximum(np.asarray(y, dtype=np.float64), 0.0)


def stable_ranking(y_plus: np.ndarray, k: int) -> np.ndarray:
    """Stable descending sort indices; ties broken by smaller index first."""
    y = np.asarray(y_plus, dtype=np.float64)
    n = y.size
    k_eff = min(int(k), n)
    # lexsort: primary key last; sort by (-y, index)
    idx = np.arange(n)
    order = np.lexsort((idx, -y))
    return order[:k_eff].astype(np.int64)


def ranking_utility(x: np.ndarray, pi: np.ndarray, w: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    pi = np.asarray(pi, dtype=np.int64)
    ww = validate_weights(w)[: pi.size]
    return float(np.dot(ww, x[pi]))


def oracle_ranking(x: np.ndarray, k: int, w: np.ndarray | None = None) -> np.ndarray:
    """Oracle ranking for scalar item values: top-k by value (rearrangement)."""
    x = np.asarray(x, dtype=np.float64)
    if w is not None:
        validate_weights(w)
    return stable_ranking(x, k)


def ranking_regret(x: np.ndarray, y: np.ndarray, k: int, w: np.ndarray) -> float:
    """R_{w,k}(x,y) = U(oracle;x) - U(deployed from clip(y);x)."""
    ww = validate_weights(w)
    y_plus = clip_nonnegative(y)
    pi_hat = stable_ranking(y_plus, k)
    pi_star = oracle_ranking(x, k, ww)
    k_eff = pi_hat.size
    return ranking_utility(x, pi_star, ww[:k_eff]) - ranking_utility(x, pi_hat, ww[:k_eff])


def observed_deficit(y: np.ndarray, pi_hat: np.ndarray, w: np.ndarray) -> float:
    """D_{w,k} = max_σ Σ w_r (y_σr - y_π̂r) = Σ w_r y_(n-r+1) - Σ w_r y_π̂r.

    For scalar y values, the maximizing σ is the top-k by y (rearrangement).
    """
    y = np.asarray(y, dtype=np.float64)
    pi_hat = np.asarray(pi_hat, dtype=np.int64)
    ww = validate_weights(w)[: pi_hat.size]
    k = pi_hat.size
    # top-k values of y (signed), stable
    top = stable_ranking(y, k)
    return float(np.dot(ww, y[top]) - np.dot(ww, y[pi_hat]))


def noise_contrast_order_stat(z: np.ndarray, w: np.ndarray, k: int) -> float:
    """Q_{w,k}(z) = Σ w_r z_(n-r+1) - Σ w_r z_(r) for nonincreasing w."""
    z = np.asarray(z, dtype=np.float64)
    ww = validate_weights(w)[:k]
    if k > z.size:
        raise TopKCertError("k exceeds dimension")
    zs = np.sort(z)  # ascending: zs[0]=smallest
    low = zs[:k]
    high = zs[-k:][::-1]  # largest to k-th largest
    return float(np.dot(ww, high) - np.dot(ww, low))


def noise_contrast_assignment_ref(z: np.ndarray, w: np.ndarray, k: int) -> float:
    """Reference: max_{π,σ} Σ w_r (z_πr - z_σr) by brute force for tiny n."""
    z = np.asarray(z, dtype=np.float64)
    n = z.size
    ww = validate_weights(w)[:k]
    if n > 10:
        raise TopKCertError("assignment reference only for n<=10")
    from itertools import permutations

    best = -np.inf
    for pi in permutations(range(n), k):
        for sigma in permutations(range(n), k):
            val = float(np.dot(ww, z[list(pi)] - z[list(sigma)]))
            if val > best:
                best = val
    return float(best)


@dataclass(frozen=True)
class WeightedEnvelopeTable:
    mechanism: str
    epsilon: float
    n: int
    k_max: int
    alpha: float
    weights: np.ndarray
    u_plus: np.ndarray  # length k_max, 1-indexed conceptually at [0]=rank1
    u_minus: np.ndarray
    q_full: np.ndarray  # q[k] for k=0..k_max


def build_rank_thresholds(
    *,
    mechanism: str,
    n: int,
    k_max: int,
    epsilon: float,
    alpha: float,
    scheme: str = "harmonic",
) -> tuple[np.ndarray, np.ndarray]:
    """Simultaneous one-sided thresholds u_r± for r=1..k_max."""
    eta_plus, eta_minus = allocate_eta(k_max, alpha, scheme=scheme)
    u_plus = np.zeros(k_max, dtype=np.float64)
    u_minus = np.zeros(k_max, dtype=np.float64)
    mech = mechanism.lower()
    for r in range(1, k_max + 1):
        if "laplace" in mech or mech in {"m1", "m1_flat_laplace"}:
            u = solve_order_stat_threshold_continuous(
                n=n, r=r, eta=float(eta_plus[r - 1]), epsilon=epsilon, mechanism="M1_flat_laplace"
            )
            u_plus[r - 1] = float(u)
            u_minus[r - 1] = float(u)
        elif "geometric" in mech or mech in {"m2", "m2_two_sided_geometric"}:
            u = solve_order_stat_threshold_integer(
                n=n, r=r, eta=float(eta_plus[r - 1]), epsilon=epsilon, convention="strict"
            )
            u_plus[r - 1] = float(u)
            u_minus[r - 1] = float(u)
        else:
            raise TopKCertError(f"unsupported mechanism {mechanism}")
    return u_plus, u_minus


def weighted_full_envelope_table(
    *,
    mechanism: str,
    n: int,
    k_max: int,
    epsilon: float,
    alpha: float,
    weights: np.ndarray,
    scheme: str = "harmonic",
) -> WeightedEnvelopeTable:
    ww = validate_weights(weights)
    if ww.size < k_max:
        raise TopKCertError("weights shorter than k_max")
    ww = ww[:k_max]
    u_plus, u_minus = build_rank_thresholds(
        mechanism=mechanism, n=n, k_max=k_max, epsilon=epsilon, alpha=alpha, scheme=scheme
    )
    q = np.zeros(k_max + 1, dtype=np.float64)
    for k in range(1, k_max + 1):
        q[k] = float(np.dot(ww[:k], u_plus[:k] + u_minus[:k]))
    return WeightedEnvelopeTable(
        mechanism=mechanism,
        epsilon=float(epsilon),
        n=int(n),
        k_max=int(k_max),
        alpha=float(alpha),
        weights=ww.copy(),
        u_plus=u_plus,
        u_minus=u_minus,
        q_full=q,
    )


def certificate_full(
    y: np.ndarray,
    *,
    k: int,
    table: WeightedEnvelopeTable,
) -> tuple[float, np.ndarray, float, float]:
    """Return (C, pi_hat, D, q)."""
    ww = table.weights[:k]
    y_plus = clip_nonnegative(y)
    pi_hat = stable_ranking(y_plus, k)
    # deficit uses signed y (conservative)
    d = observed_deficit(y, pi_hat, ww)
    q = float(table.q_full[k])
    c = max(d + q, 0.0)
    return c, pi_hat, d, q


def moved_mass_depths(pi_hat: np.ndarray, y: np.ndarray, k: int) -> int:
    """Effective exchange depth: how many top-k oracle slots differ from pi_hat on y."""
    top = set(stable_ranking(y, k).tolist())
    hat = set(pi_hat.tolist())
    return int(len(top - hat))


def allocate_eta_weight_proportional(
    w: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    """Weight-proportional Bonferroni split: eta_r ∝ w_r (normalized), r=1..k.

    Rationale (multi-exchange tightening): the per-rank tail
    threshold u_r(η) is nonincreasing and convex-ish in the allocated budget η
    (order-statistic tails shrink slower than linearly as η grows). A
    scheme-agnostic split (e.g. harmonic-in-r, ignoring w) spends the same
    marginal budget at every rank regardless of how much that rank contributes
    to Σ_r w_r(u_r+ + u_r-). Concentrating budget where w_r is large trades
    unnecessary precision at low-weight ranks for needed precision at
    high-weight ranks, strictly reducing the weighted sum whenever w is
    non-uniform. Any fixed (data-independent) split is valid; this one is a
    second, independently valid construction — not a post-hoc min of two.
    """
    ww = validate_weights(w)
    if not (0.0 < alpha < 1.0):
        raise TopKCertError("alpha must be in (0,1)")
    total = float(ww.sum())
    if total <= 0:
        raise TopKCertError("weights must have positive total mass")
    mass = alpha / 2.0
    p = ww / total
    return p.copy() * mass, p.copy() * mass


def build_rank_thresholds_weighted(
    *,
    mechanism: str,
    n: int,
    k_max: int,
    epsilon: float,
    alpha: float,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-rank u_r± under the weight-proportional allocation (multi-exchange envelope)."""
    ww = validate_weights(weights)[:k_max]
    eta_plus, eta_minus = allocate_eta_weight_proportional(ww, alpha)
    u_plus = np.zeros(k_max, dtype=np.float64)
    u_minus = np.zeros(k_max, dtype=np.float64)
    mech = mechanism.lower()
    for r in range(1, k_max + 1):
        if "laplace" in mech or mech in {"m1", "m1_flat_laplace"}:
            u = solve_order_stat_threshold_continuous(
                n=n, r=r, eta=float(eta_plus[r - 1]), epsilon=epsilon, mechanism="M1_flat_laplace"
            )
            u_plus[r - 1] = float(u)
            u_minus[r - 1] = float(u)
        elif "geometric" in mech or mech in {"m2", "m2_two_sided_geometric"}:
            u = solve_order_stat_threshold_integer(
                n=n, r=r, eta=float(eta_plus[r - 1]), epsilon=epsilon, convention="strict"
            )
            u_plus[r - 1] = float(u)
            u_minus[r - 1] = float(u)
        else:
            raise TopKCertError(f"unsupported mechanism {mechanism}")
    return u_plus, u_minus


def weighted_multiexchange_envelope_table(
    *,
    mechanism: str,
    n: int,
    k_max: int,
    epsilon: float,
    alpha: float,
    weights: np.ndarray,
) -> WeightedEnvelopeTable:
    """Precompute weight-proportional multi-exchange envelope (valid standalone)."""
    ww = validate_weights(weights)
    if ww.size < k_max:
        raise TopKCertError("weights shorter than k_max")
    ww = ww[:k_max]
    u_plus, u_minus = build_rank_thresholds_weighted(
        mechanism=mechanism,
        n=n,
        k_max=k_max,
        epsilon=epsilon,
        alpha=alpha,
        weights=ww,
    )
    q = np.zeros(k_max + 1, dtype=np.float64)
    for k in range(1, k_max + 1):
        q[k] = float(np.dot(ww[:k], u_plus[:k] + u_minus[:k]))
    return WeightedEnvelopeTable(
        mechanism=mechanism,
        epsilon=float(epsilon),
        n=int(n),
        k_max=int(k_max),
        alpha=float(alpha),
        weights=ww.copy(),
        u_plus=u_plus,
        u_minus=u_minus,
        q_full=q,
    )


def certificate_multiexchange_tight(
    y: np.ndarray,
    *,
    k: int,
    table: WeightedEnvelopeTable,
    tight_table: WeightedEnvelopeTable | None = None,
) -> tuple[float, np.ndarray, float, float, int]:
    """Multi-exchange weighted envelope: weight-proportional per-rank allocation.

    Prefer passing a precomputed `tight_table` from
    `weighted_multiexchange_envelope_table` (O(1) lookup). If omitted, builds
    thresholds for this k only (slower; for tests/diagnostics).

        C_tight = D_{w,k}(y, π̂) + Σ_{r=1}^k w_r (ũ_r+ + ũ_r-)

    Standalone valid 1-α certificate (single pre-registered public event).
    """
    ww = table.weights[:k]
    y_plus = clip_nonnegative(y)
    pi_hat = stable_ranking(y_plus, k)
    d = observed_deficit(y, pi_hat, ww)
    if tight_table is not None:
        if tight_table.k_max < k:
            raise TopKCertError("tight_table.k_max < k")
        q_tight = float(tight_table.q_full[k])
    else:
        u_plus, u_minus = build_rank_thresholds_weighted(
            mechanism=table.mechanism,
            n=table.n,
            k_max=k,
            epsilon=table.epsilon,
            alpha=table.alpha,
            weights=ww,
        )
        q_tight = float(np.dot(ww, u_plus + u_minus))
    c = max(d + q_tight, 0.0)
    m_star = max(moved_mass_depths(pi_hat, y, k), 1)
    return float(c), pi_hat, float(d), float(q_tight), m_star


def range_baseline_q(weights: np.ndarray, u1_plus: float, u1_minus: float) -> float:
    """Global-range information-class envelope: B Σ w_r with B=u1++u1-."""
    ww = validate_weights(weights)
    b = float(u1_plus + u1_minus)
    return b * float(ww.sum())


def box_baseline_q(weights: np.ndarray, r: float) -> float:
    """Coordinate-box information-class envelope: 2 r Σ w_r.

    Equal to ``r * max_pi ||a(pi)||_1`` when the support condition holds
    (sufficiently, ``n >= 2k`` with positive weights). Otherwise it is only
    an upper bound on that maximum.
    """
    ww = validate_weights(weights)
    return 2.0 * float(r) * float(ww.sum())


@dataclass(frozen=True)
class AdaptiveKResult:
    k_star: int
    k_tilde: int
    certificates: np.ndarray
    normalized: np.ndarray
    ranking: np.ndarray
    determining_prefix: int
    determining_depth: int
    abstain: bool


def _prefix_weighted_cumsum(values: np.ndarray, weights: np.ndarray, k_max: int) -> np.ndarray:
    prefix = np.zeros(k_max + 1, dtype=np.float64)
    if k_max <= 0:
        return prefix
    prefix[1:] = np.cumsum(weights[:k_max] * values[:k_max], dtype=np.float64)
    return prefix


def _prefix_moved_mass_depths(
    top_signed: np.ndarray,
    deployed: np.ndarray,
    n_items: int,
) -> np.ndarray:
    depths = np.zeros(top_signed.size + 1, dtype=np.int64)
    in_top = np.zeros(n_items, dtype=bool)
    in_hat = np.zeros(n_items, dtype=bool)
    missing = 0
    for idx, (top_i, hat_i) in enumerate(zip(top_signed.tolist(), deployed.tolist()), start=1):
        if not in_top[top_i]:
            in_top[top_i] = True
            if not in_hat[top_i]:
                missing += 1
        if not in_hat[hat_i]:
            in_hat[hat_i] = True
            if in_top[hat_i]:
                missing -= 1
        depths[idx] = max(missing, 1)
    return depths


def adaptive_k_star(
    y: np.ndarray,
    *,
    table: WeightedEnvelopeTable,
    scales: np.ndarray,
    taus: np.ndarray,
    tight_table: WeightedEnvelopeTable | None = None,
) -> AdaptiveKResult:
    """Longest fully certified prefix (primary) and individually certified max (secondary)."""
    k_max = table.k_max
    scales = np.asarray(scales, dtype=np.float64)
    taus = np.asarray(taus, dtype=np.float64)
    if scales.size < k_max or taus.size < k_max:
        raise TopKCertError("scales/taus shorter than k_max")
    if np.any(scales[:k_max] <= 0):
        raise TopKCertError("scales must be positive")
    # Prefer precomputed weight-proportional table; else build once here.
    tt = tight_table
    if tt is None:
        tt = weighted_multiexchange_envelope_table(
            mechanism=table.mechanism,
            n=table.n,
            k_max=k_max,
            epsilon=table.epsilon,
            alpha=table.alpha,
            weights=table.weights,
        )
    certs = np.zeros(k_max + 1, dtype=np.float64)
    norms = np.full(k_max + 1, np.inf, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    y_plus = clip_nonnegative(y)
    ranking = stable_ranking(y_plus, k_max)
    top_signed = stable_ranking(y, k_max)
    ww = table.weights[:k_max]
    top_vals = y[top_signed]
    deployed_vals = y[ranking]
    d_prefix = _prefix_weighted_cumsum(top_vals, ww, k_max) - _prefix_weighted_cumsum(
        deployed_vals, ww, k_max
    )
    q_prefix = np.asarray(tt.q_full[: k_max + 1], dtype=np.float64)
    certs[1:] = np.maximum(d_prefix[1:] + q_prefix[1:], 0.0)
    norms[1:] = certs[1:] / scales[:k_max]
    depths = _prefix_moved_mass_depths(top_signed, ranking, y.size)
    # individual rule
    feasible = norms[1:] <= taus[:k_max]
    ok_indiv = np.flatnonzero(feasible)
    k_tilde = int(ok_indiv[-1] + 1) if ok_indiv.size > 0 else 0
    # nested prefix rule
    prefix_ok = np.logical_and.accumulate(feasible)
    ok_prefix = np.flatnonzero(prefix_ok)
    k_star = int(ok_prefix[-1] + 1) if ok_prefix.size > 0 else 0
    det = 0
    if k_star < k_max:
        # first failing prefix after k_star
        det = k_star + 1 if k_star < k_max else k_star
    elif k_star > 0:
        det = int(np.argmax(norms[1 : k_star + 1]) + 1)
    return AdaptiveKResult(
        k_star=int(k_star),
        k_tilde=int(k_tilde),
        certificates=certs,
        normalized=norms,
        ranking=ranking,
        determining_prefix=int(det),
        determining_depth=int(depths[det]) if det > 0 else 0,
        abstain=bool(k_star == 0),
    )


# ---------------------------------------------------------------------------
# Lower-bound constructors
# ---------------------------------------------------------------------------


def range_lower_bound(w: np.ndarray, k: int, B: float) -> float:
    """Closed-form LB1 rate: L_range(w,k,B) = B · Σ_{r=1}^k w_r.

    Sharp for the range class max z - min z <= B.
    """
    ww = validate_weights(w)[:k]
    return float(B) * float(ww.sum())


def box_lower_bound(w: np.ndarray, k: int, r: float) -> float:
    """Closed-form LB2 rate: L_box(w,k,r) = 2 r · Σ_{r=1}^k w_r.

    This is the coordinate-box class rate. It equals ``r * max_pi ||a(pi)||_1``
    when a competitor can avoid the deployed positive-weight support, and
    ``n >= 2k`` is a sufficient condition for that equality. It is an upper
    envelope on the noise term for every ``n >= k``, not an unconditional
    identity for the maximum L1 support.
    """
    ww = validate_weights(w)[:k]
    return 2.0 * float(r) * float(ww.sum())


def ratio_proposed_to_lower(C: float, L: float) -> float:
    """ρ = C_proposed / L_valid; return nan when L <= 0 or either input is non-finite."""
    c = float(C)
    ell = float(L)
    if not np.isfinite(c) or not np.isfinite(ell) or ell <= 0.0:
        return float("nan")
    return c / ell


def lb_range_attaining(n: int, k: int, w: np.ndarray, b: float) -> float:
    """Sharp worst-case weighted contrast under max z - min z <= B."""
    # n is unused for the closed form; attainment witnesses require n >= 2k.
    _ = n
    return range_lower_bound(w, k, b)


def lb_box_attaining(k: int, w: np.ndarray, r: float) -> float:
    return box_lower_bound(w, k, r)


def lb1_range_class_witness(
    n: int, k: int, w: np.ndarray, b: float
) -> tuple[np.ndarray, np.ndarray, float]:
    """LB1 constructive witness: attains R_{w,k} = B·Σw_r exactly on the range-[0,B] class.

    x has a "true top-k" block of value B (indices 0..k-1) and n-k zeros; y is
    constructed so the deployed ranking selects exactly the WRONG k items (the
    zero block at indices k..2k-1 outranks the true block). Any certificate
    scheme claiming validity for ALL x ∈ [0,B]^n with no other information
    must therefore have worst-case width ≥ B·Σw_r for this k — this rate is
    not improvable by a range-only certificate. See LOWER_BOUND_RANGE_CLASS.md.
    """
    if n < 2 * k:
        raise TopKCertError("need n >= 2k for LB1 witness")
    if b < 0:
        raise TopKCertError("b must be nonnegative")
    ww = validate_weights(w)[:k]
    x = np.zeros(n, dtype=np.float64)
    x[:k] = float(b)
    y = np.zeros(n, dtype=np.float64)
    y[k : 2 * k] = float(b) + 1.0
    r = ranking_regret(x, y, k, ww)
    achieved = float(b) * float(ww.sum())
    if abs(r - achieved) > 1e-9:
        raise TopKCertError(f"LB1 witness did not attain the claimed rate: {r} vs {achieved}")
    return x, y, float(r)


def lb2_box_class_witness(n: int, k: int, w: np.ndarray, r: float) -> tuple[np.ndarray, float]:
    """LB2 constructive witness: noise z consistent with max|z_i|<=r attaining Q_{w,k}=2r·Σw_r.

    z places k coordinates at exactly -r (bottom order statistics 1..k) and k
    coordinates at exactly +r (top order statistics n-k+1..n); this respects
    max|z_i|<=r yet achieves the full box-style tail term exactly, showing a
    certificate relying only on a single global radius cannot be tightened
    below 2r·Σw_r without per-rank (order-statistic) information. See
    LOWER_BOUND_BOX_CLASS.md.
    """
    if n < 2 * k:
        raise TopKCertError("need n >= 2k for LB2 witness")
    if r < 0:
        raise TopKCertError("r must be nonnegative")
    ww = validate_weights(w)[:k]
    z = np.zeros(n, dtype=np.float64)
    z[:k] = -float(r)
    z[n - k :] = float(r)
    q = noise_contrast_order_stat(z, ww, k)
    achieved = float(box_baseline_q(ww, r))
    if abs(q - achieved) > 1e-9:
        raise TopKCertError(f"LB2 witness did not attain the claimed rate: {q} vs {achieved}")
    return z, float(q)


def ranking_regret_assignment_ref(
    x: np.ndarray, pi_hat: np.ndarray, w: np.ndarray, k: int
) -> float:
    """Reference solver for the exact decomposition R = max_σ Σ w_r(x_σr - x_π̂r).

    Full enumeration over injections σ:{1..k}->[n]; intended for tiny n (<=8)
    identity-agreement tests only, cost O(n!/(n-k)!).
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n > 9:
        raise TopKCertError("assignment reference only for n<=9")
    ww = validate_weights(w)[:k]
    pi_hat = np.asarray(pi_hat, dtype=np.int64)
    u_hat = float(np.dot(ww[: pi_hat.size], x[pi_hat]))
    from itertools import permutations

    best = -np.inf
    for sigma in permutations(range(n), k):
        u_sigma = float(np.dot(ww, x[list(sigma)]))
        best = max(best, u_sigma - u_hat)
    return float(best)


# ---------------------------------------------------------------------------
# LB4 — near-optimality of the per-rank thresholds (finite-parameter, computer-verified)
# ---------------------------------------------------------------------------


def rank_threshold_tail_at(u: float, *, n: int, r: int, epsilon: float, mechanism: str) -> float:
    """Exact Pr(Z_(n-r+1) >= u) = P(Bin(n, p_+(u)) >= r) for the given mechanism.

    Reuses the exact coordinate tails and exact/SciPy binomial survival
    function already defined in rank_adaptive.py (no new tail formulas).
    """
    from private_mobility_cert.certificates.rank_adaptive import (
        binomial_sf,
        geometric_positive_tail_strict,
        laplace_positive_tail,
    )

    mech = mechanism.lower()
    if "laplace" in mech or mech in {"m1", "m1_flat_laplace"}:
        p = laplace_positive_tail(float(u), epsilon)
    elif "geometric" in mech or mech in {"m2", "m2_two_sided_geometric"}:
        p = geometric_positive_tail_strict(int(u), epsilon)
    else:
        raise TopKCertError(f"unsupported mechanism {mechanism}")
    return float(binomial_sf(r, n, p))


@dataclass(frozen=True)
class NearOptimalityAudit:
    mechanism: str
    n: int
    r: int
    eta: float
    epsilon: float
    u_r: float
    tail_at_u_r: float
    tail_if_reduced: float
    reduction_probe: float
    is_near_optimal: bool


def near_optimality_audit(
    *, n: int, r: int, eta: float, epsilon: float, mechanism: str, continuous_probe: float = 1e-6
) -> NearOptimalityAudit:
    """LB4 — verify u_r cannot be reduced without violating the eta-tail guarantee.

    M2 (integer): u_r is the solver's minimal feasible integer by construction;
    verify EXACTLY that u_r - 1 fails (tail(u_r-1) > eta), i.e. u_r is optimal
    among integers, not merely a conservative round-up.

    M1 (continuous): the bisection solver converges to the tail=eta boundary up
    to its tolerance; verify that decreasing u_r by `continuous_probe` (larger
    than the solver tolerance) already exceeds eta, i.e. u_r is optimal up to
    a small, explicit, finite margin. See NEAR_OPTIMALITY_AUDIT.md.
    """
    mech = mechanism.lower()
    if "geometric" in mech or mech in {"m2", "m2_two_sided_geometric"}:
        u_r = float(
            solve_order_stat_threshold_integer(
                n=n, r=r, eta=eta, epsilon=epsilon, convention="strict"
            )
        )
        probe = max(u_r - 1.0, 0.0)
    else:
        u_r = float(
            solve_order_stat_threshold_continuous(n=n, r=r, eta=eta, epsilon=epsilon)
        )
        probe = max(u_r - float(continuous_probe), 0.0)
    tail_at_u = rank_threshold_tail_at(u_r, n=n, r=r, epsilon=epsilon, mechanism=mechanism)
    tail_reduced = rank_threshold_tail_at(probe, n=n, r=r, epsilon=epsilon, mechanism=mechanism)
    near_optimal = bool(tail_at_u <= eta + 1e-9 and (tail_reduced > eta - 1e-9 or probe == u_r))
    return NearOptimalityAudit(
        mechanism=mechanism,
        n=int(n),
        r=int(r),
        eta=float(eta),
        epsilon=float(epsilon),
        u_r=float(u_r),
        tail_at_u_r=float(tail_at_u),
        tail_if_reduced=float(tail_reduced),
        reduction_probe=float(probe),
        is_near_optimal=near_optimal,
    )


@dataclass(frozen=True)
class ScalarFidelityWitness:
    x: np.ndarray
    y: np.ndarray
    fidelity: float
    regret_w: float


def scalar_fidelity_pair(
    k: int = 1, *, w: np.ndarray | None = None
) -> tuple[ScalarFidelityWitness, ScalarFidelityWitness]:
    """LB3 — scalar-fidelity information-class lower bound (constructive).

    Two releases (x, y1) and (x, y2) with IDENTICAL normalized-L1 fidelity
    F = ||x - y^+||_1 / max(||x||_1, 1) but different rank-weighted regret for
    ANY nonincreasing w with w_1 > 0: instance 1 has R_{w,k} = 0 (deployed
    order already matches the oracle order on x) and instance 2 has
    R_{w,k} = w_1 * (x_(1) - x_(2)) > 0 (the top-1 item is swapped out).

    This shows a scalar summary of the release error (any single number, not
    just L1) cannot certify R_{w,k} without access to the order-statistic
    structure used by `noise_contrast_order_stat` — reuses the
    construction in `scalar_limitation.equal_fidelity_discordant_regret_pair`
    and lifts its discordant-Top-K-regret witness to arbitrary rank weights.
    """
    from private_mobility_cert.certificates.scalar_limitation import (
        equal_fidelity_discordant_regret_pair,
    )

    inst1, inst2 = equal_fidelity_discordant_regret_pair(k=k)
    ww = validate_weights(w) if w is not None else weight_family("uniform", k)
    if abs(inst1.fidelity - inst2.fidelity) > 1e-12:
        raise TopKCertError("LB3 construction lost equal-fidelity property")
    r1 = ranking_regret(inst1.x, inst1.y, k, ww)
    r2 = ranking_regret(inst2.x, inst2.y, k, ww)
    if not (r1 == 0.0 and r2 > 0.0):
        raise TopKCertError("LB3 construction failed to separate rank-weighted regret")
    return (
        ScalarFidelityWitness(
            x=inst1.x, y=inst1.y, fidelity=float(inst1.fidelity), regret_w=float(r1)
        ),
        ScalarFidelityWitness(
            x=inst2.x, y=inst2.y, fidelity=float(inst2.fidelity), regret_w=float(r2)
        ),
    )
