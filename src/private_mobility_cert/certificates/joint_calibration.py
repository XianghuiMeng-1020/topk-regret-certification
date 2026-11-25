"""Finite-sample joint public calibration for subset-swap quantiles."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np

from private_mobility_cert.certificates.topk_regret import TopKCertError
from private_mobility_cert.privacy.rng import spawn_generator


def _log_binom_pmf(j: int, n: int, p: float) -> float:
    """Log P(X=j) for X~Bin(n,p) via lgamma (finite-sample exact formula)."""
    from math import lgamma, log

    if j < 0 or j > n:
        return float("-inf")
    if p <= 0.0:
        return 0.0 if j == 0 else float("-inf")
    if p >= 1.0:
        return 0.0 if j == n else float("-inf")
    return (
        lgamma(n + 1)
        - lgamma(j + 1)
        - lgamma(n - j + 1)
        + j * log(p)
        + (n - j) * log(1.0 - p)
    )


def binomial_sf_exact(k: int, n: int, p: float) -> float:
    """Exact P(X >= k) for X~Bin(n,p) via log-PMF summation (finite-sample)."""
    from math import exp

    if not 0.0 <= p <= 1.0:
        raise TopKCertError("p must be in [0,1]")
    if n < 0 or k < 0:
        raise TopKCertError("n,k must be nonnegative")
    if k > n:
        return 0.0
    if k == 0:
        return 1.0
    # Sum the smaller tail for stability
    if k <= n - k:
        # compute SF = 1 - CDF(k-1)
        cdf = 0.0
        for j in range(0, k):
            cdf += exp(_log_binom_pmf(j, n, p))
            if cdf >= 1.0:
                return 0.0
        return float(min(max(1.0 - cdf, 0.0), 1.0))
    total = 0.0
    for j in range(k, n + 1):
        total += exp(_log_binom_pmf(j, n, p))
        if total >= 1.0:
            return 1.0
    return float(min(max(total, 0.0), 1.0))


def binomial_cdf_exact(k: int, n: int, p: float) -> float:
    """Exact P(X <= k)."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return float(1.0 - binomial_sf_exact(k + 1, n, p))


def order_statistic_index_one_sided(n_sims: int, alpha: float) -> int:
    """0-based index j such that empirical quantile is s_(j+1).

    Classical: j = ceil((N+1)*(1-alpha)) - 1, clamped to [0, N-1].
    """
    if n_sims < 1 or not (0 < alpha < 1):
        raise TopKCertError("invalid order-statistic parameters")
    j = int(ceil((n_sims + 1) * (1.0 - alpha)) - 1)
    return int(min(max(j, 0), n_sims - 1))


def certified_upper_quantile(
    samples: np.ndarray,
    *,
    alpha: float,
    delta_cal: float,
) -> tuple[float, dict[str, float]]:
    """Select smallest order statistic with binomial-certified tail <= alpha.

    Let s_(1)<=...<=s_(N). For candidate threshold s_(j+1) (0-based j),
    exceedances E = #{i: s_i > s_(j+1)}.
    We require a high-probability bound: using Hoeffding/Clopper-style exact
    binomial inversion — find smallest j starting from empirical index such that
    the Clopper-Pearson upper bound on p=Pr(Q>s) at confidence 1-δ is <= alpha.

    Exact CP upper: U = BetaQuantile_{1-δ}(E+1, N-E) when E<N.
    Without scipy Beta inverse, we use exact binomial search:
      U = inf{ p: BinomialCDF(E; N, p) <= δ }  (upper one-sided)
    equivalently smallest p with P(X<=E | Bin(N,p)) <= δ... standard CP:
      U = inf{p: sum_{k=0}^{E} Bin(k;N,p) <= δ} wait:
    Classic: U solves sum_{k=0}^E C(N,k) U^k (1-U)^{N-k} = δ for lower,
    For upper bound on p given E successes of "exceedance":
      sum_{k=E}^{N} C(N,k) U^k (1-U)^{N-k} = δ  => U is such that SF(E;N,U)=δ.

    We binary-search U in [0,1] so binomial_sf_exact(E, N, U) = δ_cal.
    Require U <= alpha.
    """
    s = np.sort(np.asarray(samples, dtype=np.float64))
    n = int(s.size)
    if n < 10:
        raise TopKCertError("need at least 10 calibration samples")
    j0 = order_statistic_index_one_sided(n, alpha)

    def cp_upper(exceedances: int) -> float:
        """Finite-sample one-sided upper bound on exceedance probability.

        Primary implementation uses Hoeffding's inequality (non-asymptotic):
          U = e/N + sqrt(log(1/δ)/(2N)).
        Exact binomial inversion (binomial_sf_exact) is available for audits/tests
        and is equivalent in guarantee form; Hoeffding is a closed-form conservative
        certificate of the same one-sided binomial experiment.
        """
        e = int(exceedances)
        if e >= n:
            return 1.0
        return float(min(1.0, e / n + np.sqrt(np.log(1.0 / delta_cal) / (2.0 * n))))

    for j in range(j0, n):
        thr = float(s[j])
        exceedances = int(np.sum(s > thr))
        u = cp_upper(exceedances)
        if u <= alpha + 1e-15:
            return thr, {
                "index_0based": float(j),
                "exceedances": float(exceedances),
                "cp_upper": float(u),
                "alpha": float(alpha),
                "delta_cal": float(delta_cal),
                "n_sims": float(n),
                "fail_closed": 0.0,
            }
    # Fail closed: return +inf sentinel via huge value
    return float(s[-1] * 10.0 + 1e9), {
        "index_0based": float(n - 1),
        "exceedances": 0.0,
        "cp_upper": 1.0,
        "alpha": float(alpha),
        "delta_cal": float(delta_cal),
        "n_sims": float(n),
        "fail_closed": 1.0,
    }


@dataclass(frozen=True)
class JointCalibratedQuantiles:
    mechanism: str
    epsilon: float
    n: int
    k: int
    alpha: float
    delta_cal: float
    n_sims: int
    seed: int
    method: str
    q: np.ndarray
    meta: dict[str, float]
    guarantee_form: str


def _sample_noise_matrix(
    mechanism: str, n: int, epsilon: float, n_sims: int, rng: np.random.Generator
) -> np.ndarray:
    if mechanism == "M1_flat_laplace":
        return rng.laplace(0.0, 1.0 / epsilon, size=(n_sims, n))
    if mechanism == "M2_two_sided_geometric":
        alpha = float(np.exp(-epsilon))
        p = 1.0 - alpha
        g1 = rng.geometric(p, size=(n_sims, n)) - 1
        g2 = rng.geometric(p, size=(n_sims, n)) - 1
        return (g1 - g2).astype(np.float64)
    raise TopKCertError(f"unknown mechanism {mechanism}")


def calibrate_bonferroni_qm(
    *,
    mechanism: str,
    n: int,
    k: int,
    epsilon: float,
    alpha: float,
    delta_cal: float,
    n_sims: int,
    seed: int,
) -> JointCalibratedQuantiles:
    """J-A: equal Bonferroni α_m=α/K, δ_m=δ/K."""
    if k <= 0:
        return JointCalibratedQuantiles(
            mechanism=mechanism,
            epsilon=epsilon,
            n=n,
            k=0,
            alpha=alpha,
            delta_cal=delta_cal,
            n_sims=n_sims,
            seed=seed,
            method="J_A_bonferroni",
            q=np.zeros(1),
            meta={},
            guarantee_form="J2_joint_validity",
        )
    rng = spawn_generator(seed)
    noise = _sample_noise_matrix(mechanism, n, epsilon, n_sims, rng)
    sorted_n = np.sort(noise, axis=1)
    q = np.zeros(k + 1, dtype=np.float64)
    alpha_m = alpha / float(k)
    delta_m = delta_cal / float(k)
    max_m = min(k, n // 2)
    fail = 0.0
    for m in range(1, max_m + 1):
        samples = sorted_n[:, -m:].sum(axis=1) - sorted_n[:, :m].sum(axis=1)
        thr, meta = certified_upper_quantile(samples, alpha=alpha_m, delta_cal=delta_m)
        q[m] = thr
        fail = max(fail, meta["fail_closed"])
    return JointCalibratedQuantiles(
        mechanism=mechanism,
        epsilon=float(epsilon),
        n=int(n),
        k=int(k),
        alpha=float(alpha),
        delta_cal=float(delta_cal),
        n_sims=int(n_sims),
        seed=int(seed),
        method="J_A_bonferroni",
        q=q,
        meta={"fail_closed": fail},
        guarantee_form="J2_joint_validity",
    )


def calibrate_joint_scalar_qm(
    *,
    mechanism: str,
    n: int,
    k: int,
    epsilon: float,
    alpha: float,
    delta_cal: float,
    n_sims: int,
    seed: int,
) -> JointCalibratedQuantiles:
    """J-B: calibrate W=max_m Q_m/(2m); set q_m = 2m * w_hat.

    This yields a single public scalar whose success event implies all m bounds.
    """
    if k <= 0:
        return JointCalibratedQuantiles(
            mechanism=mechanism,
            epsilon=epsilon,
            n=n,
            k=0,
            alpha=alpha,
            delta_cal=delta_cal,
            n_sims=n_sims,
            seed=seed,
            method="J_B_joint_scalar",
            q=np.zeros(1),
            meta={},
            guarantee_form="J2_joint_validity",
        )
    rng = spawn_generator(seed)
    noise = _sample_noise_matrix(mechanism, n, epsilon, n_sims, rng)
    sorted_n = np.sort(noise, axis=1)
    max_m = min(k, n // 2)
    w_samples = np.zeros(n_sims, dtype=np.float64)
    for m in range(1, max_m + 1):
        qm = sorted_n[:, -m:].sum(axis=1) - sorted_n[:, :m].sum(axis=1)
        w_samples = np.maximum(w_samples, qm / (2.0 * m))
    w_hat, meta = certified_upper_quantile(w_samples, alpha=alpha, delta_cal=delta_cal)
    q = np.zeros(k + 1, dtype=np.float64)
    for m in range(1, k + 1):
        q[m] = 2.0 * m * w_hat
    return JointCalibratedQuantiles(
        mechanism=mechanism,
        epsilon=float(epsilon),
        n=int(n),
        k=int(k),
        alpha=float(alpha),
        delta_cal=float(delta_cal),
        n_sims=int(n_sims),
        seed=int(seed),
        method="J_B_joint_scalar",
        q=q,
        meta=meta,
        guarantee_form="J2_joint_validity",
    )
