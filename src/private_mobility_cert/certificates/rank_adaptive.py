"""Rank-adaptive multi-swap noise envelopes.

Primary validity route: exact binomial order-statistic tails for M1/M2.
Does NOT use W = max_m Q_m/(2m), which collapses to pairwise scaling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from private_mobility_cert.certificates.joint_calibration import binomial_sf_exact
from private_mobility_cert.certificates.topk_regret import TopKCertError
from private_mobility_cert.privacy.rng import spawn_generator

try:
    from scipy.stats import binom as _scipy_binom  # type: ignore[import-untyped]
except Exception:  # pragma: no cover
    _scipy_binom = None


def binomial_sf(k: int, n: int, p: float) -> float:
    """P(X >= k) for X~Bin(n,p). Prefer SciPy; fallback to exact log-sum."""
    if _scipy_binom is not None:
        if k <= 0:
            return 1.0
        if k > n:
            return 0.0
        return float(_scipy_binom.sf(k - 1, n, p))
    return binomial_sf_exact(k, n, p)


# ---------------------------------------------------------------------------
# Exact coordinate tails
# ---------------------------------------------------------------------------


def laplace_positive_tail(u: float, epsilon: float) -> float:
    """P(Z > u) for Z ~ Laplace(0, 1/ε). Convention: strict >."""
    if epsilon <= 0:
        raise TopKCertError("epsilon must be positive")
    if u < 0:
        # Laplace is symmetric; P(Z>u) = 1 - 0.5 e^{ε u} for u<0
        return float(1.0 - 0.5 * np.exp(epsilon * u))
    return float(0.5 * np.exp(-epsilon * u))


def geometric_positive_tail_strict(u: int, epsilon: float) -> float:
    """One-sided strict tail ``P(Z > u)`` for integer ``u``.

    PMF ``P(Z=j) = ((1-rho)/(1+rho)) rho^{|j|}`` with ``rho = exp(-epsilon)``.
    For integer ``u >= 0``, ``P(Z > u) = rho^{u+1}/(1+rho)``.
    This is the one-sided tail. The two-sided tails are
    ``P(|Z| > t) = 2 rho^{t+1}/(1+rho)`` and, for ``t >= 1``,
    ``P(|Z| >= t) = 2 rho^t/(1+rho)``.
    """
    if epsilon <= 0:
        raise TopKCertError("epsilon must be positive")
    rho = float(np.exp(-epsilon))
    if u < 0:
        # P(Z > u) = 1 - P(Z <= u) = 1 - P(Z < -|u|+... complex; use complement
        # For u=-1: P(Z>-1)=P(Z>=0)= (1+α)/(1+α)? Use series.
        # P(Z <= u) for u<0: sum_{k=-∞}^u c ρ^{|k|}
        uu = int(u)
        # P(Z > uu) = 1 - P(Z <= uu)
        # P(Z <= uu) = sum_{k=-∞}^{uu} c ρ^{-k} for uu<0
        c = (1.0 - rho) / (1.0 + rho)
        # sum_{k=-∞}^{uu} c ρ^{-k} = c ρ^{-uu} / (1-ρ) * ρ^0 wait
        # For k<=uu<0: |k|=-k, sum_{k=-∞}^{uu} c ρ^{-k} = c ρ^{-uu} sum_{j=0}^∞ ρ^j
        # where j=-k+uu... Let t=-uu>=1: sum_{k=-∞}^{-t} c ρ^{-k} = c ρ^t /(1-ρ)
        t = -uu
        p_le = c * (rho**t) / (1.0 - rho)
        return float(1.0 - p_le)
    # u >= 0
    return float((rho ** (int(u) + 1)) / (1.0 + rho))


def geometric_positive_tail_geq(u: int, epsilon: float) -> float:
    """One-sided tail ``P(Z >= u)`` for integer ``u``.

    For ``u >= 1`` this is ``rho^u / (1+rho)``, which equals ``P(Z > u-1)``.
    It is not the two-sided probability ``P(|Z| >= u)``.
    """
    if u <= 0:
        # P(Z >= 0) = 1/(1+rho).
        rho = float(np.exp(-epsilon))
        c = (1.0 - rho) / (1.0 + rho)
        return float(c + c * rho / (1.0 - rho))
    return geometric_positive_tail_strict(u - 1, epsilon)


# ---------------------------------------------------------------------------
# Alpha allocation (public only)
# ---------------------------------------------------------------------------


def allocate_eta(
    k: int,
    alpha: float,
    *,
    scheme: str = "harmonic",
) -> tuple[np.ndarray, np.ndarray]:
    """Return (eta_plus[1..K], eta_minus[1..K]) with sum(eta++eta-) <= alpha.

    Schemes: uniform, harmonic, geometric. Symmetric: half alpha each side.
    """
    if k <= 0 or not (0 < alpha < 1):
        raise TopKCertError("invalid k/alpha")
    mass = alpha / 2.0
    if scheme == "uniform":
        w = np.full(k, 1.0, dtype=np.float64)
    elif scheme == "harmonic":
        w = 1.0 / np.arange(1, k + 1, dtype=np.float64)
    elif scheme == "geometric":
        gamma = 0.7
        w = gamma ** np.arange(k, dtype=np.float64)
    else:
        raise TopKCertError(f"unknown allocation scheme {scheme}")
    w = w / w.sum() * mass
    return w.copy(), w.copy()


def public_objective_weighted_sum_u(
    u: np.ndarray,
    k: int,
) -> float:
    """Protected-data-independent objective: sum_r (K-r+1) * 2 u_r."""
    return float(sum((k - r + 1) * 2.0 * float(u[r - 1]) for r in range(1, k + 1)))


# ---------------------------------------------------------------------------
# Threshold solvers
# ---------------------------------------------------------------------------


def solve_order_stat_threshold_continuous(
    *,
    n: int,
    r: int,
    eta: float,
    epsilon: float,
    mechanism: str = "M1_flat_laplace",
    tol: float = 1e-10,
    max_iter: int = 80,
) -> float:
    """Smallest u >= 0 s.t. Pr(Bin(n, p_+(u)) >= r) <= eta."""
    if n < 1 or r < 1 or r > n or not (0 < eta < 1) or epsilon <= 0:
        raise TopKCertError("invalid order-stat threshold parameters")
    if eta >= 1.0 - 1e-15:
        return 0.0

    def p_plus(u: float) -> float:
        if mechanism == "M1_flat_laplace":
            return laplace_positive_tail(u, epsilon)
        raise TopKCertError("continuous solver only for M1")

    # Fail-closed upper search bound
    hi = float((np.log(max(n / eta, 2.0)) + 40.0) / epsilon)
    lo = 0.0
    # If even u=0 fails to meet eta, inflate
    if binomial_sf(r, n, p_plus(0.0)) <= eta + 1e-15:
        return 0.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        sf = binomial_sf(r, n, p_plus(mid))
        if sf <= eta + 1e-15:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    # Fail-closed: return hi (the feasible side)
    if binomial_sf(r, n, p_plus(hi)) > eta + 1e-12:
        return float(hi * 2.0 + 1e6)  # fail-closed sentinel
    return float(hi)


def solve_order_stat_threshold_integer(
    *,
    n: int,
    r: int,
    eta: float,
    epsilon: float,
    convention: str = "strict",
) -> int:
    """Minimal integer u >= 0 s.t. Binomial SF condition holds for M2."""
    if n < 1 or r < 1 or r > n or not (0 < eta < 1) or epsilon <= 0:
        raise TopKCertError("invalid integer threshold parameters")

    def p_plus(u: int) -> float:
        if convention == "strict":
            return geometric_positive_tail_strict(u, epsilon)
        if convention == "geq":
            return geometric_positive_tail_geq(u, epsilon)
        raise TopKCertError(f"unknown convention {convention}")

    # Upper bound search
    rho = float(np.exp(-epsilon))
    if rho <= 0:
        return 10**9
    u_hi = int(max(0, np.ceil(np.log(max(eta / max(n, 1), 1e-300)) / np.log(rho)))) + 50
    u_hi = max(u_hi, 1)
    if binomial_sf(r, n, p_plus(0)) <= eta + 1e-15:
        return 0
    for u in range(0, u_hi + 1):
        if binomial_sf(r, n, p_plus(u)) <= eta + 1e-15:
            return int(u)
    return int(u_hi * 2 + 10**6)  # fail-closed


@dataclass(frozen=True)
class RankAdaptiveEnvelope:
    mechanism: str
    epsilon: float
    n: int
    k: int
    alpha: float
    allocation_scheme: str
    integer_convention: str
    eta_plus: np.ndarray
    eta_minus: np.ndarray
    u_plus: np.ndarray
    u_minus: np.ndarray
    q_rank: np.ndarray
    method: str
    guarantee: str


def build_rank_adaptive_envelope(
    *,
    mechanism: str,
    n: int,
    k: int,
    epsilon: float,
    alpha: float,
    allocation_scheme: str = "harmonic",
    integer_convention: str = "strict",
) -> RankAdaptiveEnvelope:
    """Construct q_m^rank = sum_{r=1}^m (u_r^+ + u_r^-)."""
    if k < 0 or n < 1:
        raise TopKCertError("invalid n/k")
    eta_p, eta_m = allocate_eta(k, alpha, scheme=allocation_scheme)
    # verify budget
    if float(eta_p.sum() + eta_m.sum()) > alpha + 1e-12:
        raise TopKCertError("alpha allocation exceeds budget")
    u_p = np.zeros(k, dtype=np.float64)
    u_m = np.zeros(k, dtype=np.float64)
    max_r = min(k, n)
    for r in range(1, max_r + 1):
        if mechanism == "M1_flat_laplace":
            u_p[r - 1] = solve_order_stat_threshold_continuous(
                n=n, r=r, eta=float(eta_p[r - 1]), epsilon=epsilon, mechanism=mechanism
            )
            # symmetric Laplace: same for lower tail of -Z_{(r)}
            u_m[r - 1] = u_p[r - 1]
        elif mechanism == "M2_two_sided_geometric":
            u_p[r - 1] = float(
                solve_order_stat_threshold_integer(
                    n=n,
                    r=r,
                    eta=float(eta_p[r - 1]),
                    epsilon=epsilon,
                    convention=integer_convention,
                )
            )
            u_m[r - 1] = u_p[r - 1]  # symmetric
        else:
            raise TopKCertError(f"unknown mechanism {mechanism}")
    q = np.zeros(k + 1, dtype=np.float64)
    for m in range(1, k + 1):
        q[m] = float(np.sum(u_p[:m] + u_m[:m]))
    return RankAdaptiveEnvelope(
        mechanism=mechanism,
        epsilon=float(epsilon),
        n=int(n),
        k=int(k),
        alpha=float(alpha),
        allocation_scheme=allocation_scheme,
        integer_convention=integer_convention,
        eta_plus=eta_p,
        eta_minus=eta_m,
        u_plus=u_p,
        u_minus=u_m,
        q_rank=q,
        method="rank_adaptive_analytic_binomial_tails",
        guarantee="simultaneous_Qm_leq_q_rank_ge_1_minus_alpha",
    )


def pairwise_range_envelope_matched(
    *,
    mechanism: str,
    n: int,
    k: int,
    epsilon: float,
    alpha: float,
    allocation_scheme: str = "harmonic",
    integer_convention: str = "strict",
) -> np.ndarray:
    """Matched equal-guarantee pairwise scaling: q_m = m * q_1^range.

    q_1^range uses the same total alpha on the (r=1) upper+lower tails only
    (eta^+_1=eta^-_1=alpha/2), i.e. the classical range envelope at level alpha.
    This is the correct baseline for equal simultaneous validity comparisons
    against rank-adaptive (which spreads alpha across ranks).
    """
    # For equal-guarantee comparison of *range-scaled* baseline at full alpha on m=1:
    eta_p = np.array([alpha / 2.0])
    if mechanism == "M1_flat_laplace":
        u = solve_order_stat_threshold_continuous(
            n=n, r=1, eta=float(eta_p[0]), epsilon=epsilon
        )
    elif mechanism == "M2_two_sided_geometric":
        u = float(
            solve_order_stat_threshold_integer(
                n=n, r=1, eta=float(eta_p[0]), epsilon=epsilon, convention=integer_convention
            )
        )
    else:
        raise TopKCertError(mechanism)
    q1 = 2.0 * u
    q = np.zeros(k + 1, dtype=np.float64)
    for m in range(1, k + 1):
        q[m] = m * q1
    return q


def old_c2_joint_scalar_collapses(z: np.ndarray, k: int) -> tuple[float, float, bool]:
    """Return (W, Q1/2, identity_holds) verifying W = max_m Q_m/(2m) = Q1/2."""
    from private_mobility_cert.certificates.subset_swap import q_m_order_statistic

    zz = np.asarray(z, dtype=np.float64)
    max_m = min(int(k), zz.size // 2)
    q1 = q_m_order_statistic(zz, 1)
    w = 0.0
    for m in range(1, max_m + 1):
        qm = q_m_order_statistic(zz, m)
        w = max(w, qm / (2.0 * m))
        if qm > m * q1 + 1e-9:
            return w, q1 / 2.0, False
    return float(w), float(q1 / 2.0), bool(abs(w - q1 / 2.0) <= 1e-9)


# ---------------------------------------------------------------------------
# Shape-adaptive public calibration (secondary; must not collapse to m=1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShapeAdaptiveCalibrated:
    mechanism: str
    epsilon: float
    n: int
    k: int
    alpha: float
    n_sims: int
    seed: int
    b: np.ndarray
    s: np.ndarray
    t_hat: float
    q_cal: np.ndarray
    order_index_0based: int
    method: str
    guarantee: str


def _analytic_centers_scales(env: RankAdaptiveEnvelope) -> tuple[np.ndarray, np.ndarray]:
    """Public (b_m, s_m) from analytic rank structure — not fit to protected data."""
    k = env.k
    b = np.zeros(k + 1, dtype=np.float64)
    s = np.ones(k + 1, dtype=np.float64)
    # Center at 0; scale by marginal rank increment so T does not collapse to m=1
    for m in range(1, k + 1):
        inc = float(env.u_plus[m - 1] + env.u_minus[m - 1])
        s[m] = max(inc, 1e-12)
        b[m] = 0.0
    return b, s


def calibrate_shape_adaptive(
    *,
    mechanism: str,
    n: int,
    k: int,
    epsilon: float,
    alpha: float,
    n_sims: int,
    seed: int,
    allocation_scheme: str = "harmonic",
    integer_convention: str = "strict",
) -> ShapeAdaptiveCalibrated:
    """Calibrate T = max_m (Q_m - b_m)/s_m via exact exchangeable rank.

    Index j = ceil((N+1)*(1-alpha)) - 1; if j >= N, fail-closed inflate.
    No delta_cal for primary joint rank guarantee.
    """
    from math import ceil

    from private_mobility_cert.certificates.joint_calibration import _sample_noise_matrix

    env = build_rank_adaptive_envelope(
        mechanism=mechanism,
        n=n,
        k=k,
        epsilon=epsilon,
        alpha=alpha,
        allocation_scheme=allocation_scheme,
        integer_convention=integer_convention,
    )
    b, s = _analytic_centers_scales(env)
    rng = spawn_generator(seed)
    noise = _sample_noise_matrix(mechanism, n, epsilon, n_sims, rng)
    sorted_n = np.sort(noise, axis=1)
    max_m = min(k, n // 2)
    t_samples = np.zeros(n_sims, dtype=np.float64)
    for m in range(1, max_m + 1):
        qm = sorted_n[:, -m:].sum(axis=1) - sorted_n[:, :m].sum(axis=1)
        t_samples = np.maximum(t_samples, (qm - b[m]) / s[m])
    # Verify non-collapse: if score always determined only by m=1 on noise, flag
    # (checked in tests symbolically / numerically)
    j = int(ceil((n_sims + 1) * (1.0 - alpha)) - 1)
    if j >= n_sims:
        t_hat = float(np.max(t_samples) * 10.0 + 1e9)
        j_used = n_sims - 1
    else:
        t_hat = float(np.sort(t_samples)[j])
        j_used = j
    q = np.zeros(k + 1, dtype=np.float64)
    for m in range(1, k + 1):
        q[m] = float(b[m] + s[m] * t_hat)
    return ShapeAdaptiveCalibrated(
        mechanism=mechanism,
        epsilon=float(epsilon),
        n=int(n),
        k=int(k),
        alpha=float(alpha),
        n_sims=int(n_sims),
        seed=int(seed),
        b=b,
        s=s,
        t_hat=float(t_hat),
        q_cal=q,
        order_index_0based=int(j_used),
        method="shape_adaptive_exact_rank",
        guarantee="exchangeable_rank_joint_Pr_Tnew_leq_that_ge_1_minus_alpha",
    )


def score_collapses_to_m1(b: np.ndarray, s: np.ndarray, k: int) -> bool:
    """Algebraic collapse: q_m = b_m + s_m t  equals m*(b_1+s_1 t) for all t>0."""
    # If s_m = m s_1 and b_m = m b_1, then collapses to pairwise scaling of scalar t.
    if k < 2:
        return True
    for m in range(2, k + 1):
        if abs(s[m] - m * s[1]) > 1e-12 or abs(b[m] - m * b[1]) > 1e-12:
            return False
    return True
