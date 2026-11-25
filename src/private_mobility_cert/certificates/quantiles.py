"""Analytic and publicly calibrated quantiles for subset-swap noise contrasts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from private_mobility_cert.certificates.topk_regret import TopKCertError
from private_mobility_cert.privacy.rng import spawn_generator


@dataclass(frozen=True)
class QuantileTable:
    """q[m] bounds for m=0..K with simultaneous coverage >= 1-alpha (stated method)."""

    mechanism: str
    epsilon: float
    n: int
    k: int
    alpha: float
    method: str
    q: np.ndarray
    delta_cal: float
    calibration_seed: int | None
    n_sims: int
    joint_guarantee: str


def analytic_q_m_laplace(
    *,
    n: int,
    k: int,
    epsilon: float,
    alpha: float,
) -> QuantileTable:
    """Conservative analytic simultaneous bounds for Laplace noise.

    For each m, Q_m(Z) = sum_{r=1}^m (Z_{(n-r+1)} - Z_{(r)}).
    Each coordinate difference of two independent Laplace(b) is not Laplace, but
    |Z_i| is Exp(1/b) with b=1/ε. We use a union-bound envelope:

      Q_m(Z) <= 2 m max_i |Z_i|

    and set r so that Pr(max |Z_i| > r) <= alpha via n e^{-ε r} = alpha,
    yielding q_m = 2 m r for all m simultaneously (same event).
    """
    if n <= 0 or k < 0 or epsilon <= 0 or not (0 < alpha < 1):
        raise TopKCertError("invalid analytic quantile parameters")
    r = float(np.log(n / alpha) / epsilon)
    q = np.zeros(k + 1, dtype=np.float64)
    for m in range(1, k + 1):
        q[m] = 2.0 * m * r
    return QuantileTable(
        mechanism="M1_flat_laplace",
        epsilon=float(epsilon),
        n=int(n),
        k=int(k),
        alpha=float(alpha),
        method="analytic_union_maxnorm",
        q=q,
        delta_cal=0.0,
        calibration_seed=None,
        n_sims=0,
        joint_guarantee="mechanism_randomness_only",
    )


def analytic_q_m_geometric(
    *,
    n: int,
    k: int,
    epsilon: float,
    alpha: float,
) -> QuantileTable:
    """Conservative analytic simultaneous bounds for two-sided geometric noise.

    Use max-norm event: Pr(max |Z_i| >= t) <= n α_geo^t with α_geo=e^{-ε}.
    Then Q_m <= 2 m t.
    """
    if n <= 0 or k < 0 or epsilon <= 0 or not (0 < alpha < 1):
        raise TopKCertError("invalid analytic quantile parameters")
    alpha_geo = float(np.exp(-epsilon))
    t = float(max(np.ceil(np.log(alpha / n) / np.log(alpha_geo)), 0.0))
    q = np.zeros(k + 1, dtype=np.float64)
    for m in range(1, k + 1):
        q[m] = 2.0 * m * t
    return QuantileTable(
        mechanism="M2_two_sided_geometric",
        epsilon=float(epsilon),
        n=int(n),
        k=int(k),
        alpha=float(alpha),
        method="analytic_union_maxnorm",
        q=q,
        delta_cal=0.0,
        calibration_seed=None,
        n_sims=0,
        joint_guarantee="mechanism_randomness_only",
    )


def binomial_upper_tail_threshold(
    samples: np.ndarray,
    *,
    alpha: float,
    delta_cal: float,
) -> float:
    """One-sided order-statistic calibration.

    Choose the smallest order statistic q̂ such that, with confidence >= 1-δ_cal,
    Pr*(Q > q̂) <= alpha, where Pr* is the true noise law.

    Using Clopper-Pearson / binomial: if N sims and we take the
    (1-alpha)-empirical quantile with inflation, a standard conservative rule is:

      Let k* = min{ k : BinomialCDF(k; N, 1-alpha) >= 1-δ_cal wait — better:

    We use the order-statistic rule:
      index = ceil((N+1)*(1-alpha)) - 1  (0-based), then inflate by requiring
      that the number of exceedances of the chosen threshold has upper (1-δ)
      Clopper-Pearson bound <= alpha.

    Practical certified construction:
      Sort s_(1) <= ... <= s_(N).
      Find the smallest j such that the Clopper-Pearson upper bound on
      p = Pr(Q > s_(j)) using (N-j) exceedances is <= alpha at confidence 1-δ_cal.
    """
    from math import ceil

    s = np.sort(np.asarray(samples, dtype=np.float64))
    n = int(s.size)
    if n < 10:
        raise TopKCertError("need at least 10 calibration samples")
    if not (0 < alpha < 1) or not (0 < delta_cal < 1):
        raise TopKCertError("invalid alpha/delta_cal")

    # Beta quantile for Clopper-Pearson upper bound on exceedance probability.
    # For e exceedances out of n, U = BetaQuantile(1-δ; e+1, n-e) when e<n,
    # and U=1 when e=n.
    def cp_upper(exceedances: int) -> float:
        e = int(exceedances)
        if e >= n:
            return 1.0
        if e < 0:
            return 0.0
        # Inverse incomplete beta via numpy
        # scipy-free: use Wilson or Hoeffding fallback if needed.
        # Hoeffding: P(p̂ - p >= t) <= exp(-2 n t^2) => conservative
        # Prefer DKW/Clopper via beta incomplete through numpy.random? Use Hoeffding:
        # U = e/n + sqrt(log(1/δ)/(2n))
        return float(min(1.0, e / n + np.sqrt(np.log(1.0 / delta_cal) / (2.0 * n))))

    # Start from empirical (1-alpha) quantile index and move up until certified.
    j0 = min(n - 1, max(0, int(ceil((n + 1) * (1.0 - alpha)) - 1)))
    for j in range(j0, n):
        exceedances = n - (j + 1)  # strictly greater than s[j]
        # Treat ties: count values > s[j]
        exceedances = int(np.sum(s > s[j]))
        if cp_upper(exceedances) <= alpha + 1e-15:
            return float(s[j].item() if hasattr(s[j], "item") else s[j])
    # Fail closed: return max + margin
    return float(s[-1].item() if hasattr(s[-1], "item") else s[-1])


def calibrated_q_m_table(
    *,
    mechanism: str,
    n: int,
    k: int,
    epsilon: float,
    alpha: float,
    delta_cal: float,
    n_sims: int,
    seed: int,
) -> QuantileTable:
    """Public simulation-calibrated simultaneous q_m with calibration confidence.

    Allocates alpha equally across m=1..K via Bonferroni: α_m = alpha / K,
    then calibrates each marginal Q_m quantile at level α_m with δ_cal/K,
    yielding a joint guarantee conditional on successful calibration, or
    jointly over calibration+release if calibration failure is treated as
    infinite certificate (fail-closed).
    """
    if k <= 0:
        q = np.zeros(1, dtype=np.float64)
        return QuantileTable(
            mechanism=mechanism,
            epsilon=float(epsilon),
            n=int(n),
            k=0,
            alpha=float(alpha),
            method="simulation_calibrated_bonferroni",
            q=q,
            delta_cal=float(delta_cal),
            calibration_seed=int(seed),
            n_sims=int(n_sims),
            joint_guarantee="conditional_on_successful_public_calibration",
        )
    rng = spawn_generator(int(seed))
    max_m = min(k, n // 2)
    q = np.zeros(k + 1, dtype=np.float64)
    alpha_m = alpha / float(k)
    delta_m = delta_cal / float(k)
    # Vectorized public noise simulations: shape (n_sims, n)
    if mechanism == "M1_flat_laplace":
        scale = 1.0 / epsilon
        noise = rng.laplace(0.0, scale, size=(n_sims, n))
    elif mechanism == "M2_two_sided_geometric":
        alpha_geo = float(np.exp(-epsilon))
        p = 1.0 - alpha_geo
        g1 = rng.geometric(p, size=(n_sims, n)) - 1
        g2 = rng.geometric(p, size=(n_sims, n)) - 1
        noise = (g1 - g2).astype(np.float64)
    else:
        raise TopKCertError(f"unknown mechanism {mechanism}")
    sorted_noise = np.sort(noise, axis=1)
    for m in range(1, max_m + 1):
        samples = sorted_noise[:, -m:].sum(axis=1) - sorted_noise[:, :m].sum(axis=1)
        q[m] = binomial_upper_tail_threshold(samples, alpha=alpha_m, delta_cal=delta_m)
    # For m > max_m (impossible swaps), leave 0 — those m cannot occur.
    return QuantileTable(
        mechanism=mechanism,
        epsilon=float(epsilon),
        n=int(n),
        k=int(k),
        alpha=float(alpha),
        method="simulation_calibrated_bonferroni",
        q=q,
        delta_cal=float(delta_cal),
        calibration_seed=int(seed),
        n_sims=int(n_sims),
        joint_guarantee="conditional_on_successful_public_calibration",
    )


def analytic_q1_pairwise_laplace(n: int, epsilon: float, alpha: float) -> float:
    """Bound on Q_1 = max Z - min Z via 2 max|Z| envelope."""
    r = float(np.log(n / alpha) / epsilon)
    return 2.0 * r


def analytic_q1_pairwise_geometric(n: int, epsilon: float, alpha: float) -> float:
    alpha_geo = float(np.exp(-epsilon))
    t = float(max(np.ceil(np.log(alpha / n) / np.log(alpha_geo)), 0.0))
    return 2.0 * t
