"""Release-conditioned weighted Top-K regret certificates.

Scientific object
-----------------
True weighted regret admits the exact identity

    R_{w,k}(π̂; x) = max_{π ∈ A_k} [ G_Y(π, π̂) + N_Z(π, π̂) ],

where A_k is the set of injective ordered k-prefixes,
G_Y is observable from the raw release Y, and N_Z is the noise contrast.
The separable upper bound max G + max N collapses to a public constant whenever
π̂ already maximizes weighted released utility (typical when Y = Y^+). The
certificate below keeps the released margin inside the max:

    C(Y) = max_{π ∈ A_k} [ G_Y(π, π̂) + B(π) ],

with B(π) a mechanism-valid envelope for N_Z(π, π̂) on a single public event.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Literal

import numpy as np

from private_mobility_cert.certificates.rank_adaptive import (
    allocate_eta,
    solve_order_stat_threshold_continuous,
    solve_order_stat_threshold_integer,
)
from private_mobility_cert.certificates.subset_swap import (
    all_observed_deficits,
    certificate_c0_box,
    certificate_c2_subset_swap,
    coordinate_box_radius_geometric,
    coordinate_box_radius_laplace,
    deployable_selection_from_y_act,
)
from private_mobility_cert.certificates.topk_regret import TopKCertError
from private_mobility_cert.certificates.weighted_ranking import (
    build_rank_thresholds,
    build_rank_thresholds_weighted,
    clip_nonnegative,
    oracle_ranking,
    ranking_utility,
    stable_ranking,
    validate_weights,
    weight_family,
)


MechanismName = Literal["M1", "M2", "laplace", "geometric"]


def action_set_size(n: int, k: int) -> int:
    if k < 0 or k > n:
        raise TopKCertError("require 0 <= k <= n")
    out = 1
    for i in range(k):
        out *= n - i
    return int(out)


def weight_map(pi: np.ndarray, w: np.ndarray, n: int) -> np.ndarray:
    """W_π(i) = w_r if π(r)=i else 0."""
    ww = validate_weights(w)
    pi = np.asarray(pi, dtype=np.int64)
    if pi.size > ww.size:
        raise TopKCertError("weights shorter than prefix")
    out = np.zeros(n, dtype=np.float64)
    for r, idx in enumerate(pi.tolist()):
        out[int(idx)] += float(ww[r])
    return out


def coefficient_map(pi: np.ndarray, pi_hat: np.ndarray, w: np.ndarray, n: int) -> np.ndarray:
    """a = W_hat - W_π, so N = <a, Z>."""
    return weight_map(pi_hat, w, n) - weight_map(pi, w, n)


def observable_margin(y: np.ndarray, pi: np.ndarray, pi_hat: np.ndarray, w: np.ndarray) -> float:
    """G_Y(π, π̂) = Σ_r w_r (Y_π(r) - Y_π̂(r)). Uses raw Y (not Y^+)."""
    y = np.asarray(y, dtype=np.float64)
    ww = validate_weights(w)
    pi = np.asarray(pi, dtype=np.int64)
    pi_hat = np.asarray(pi_hat, dtype=np.int64)
    k = pi_hat.size
    if pi.size != k:
        raise TopKCertError("pi and pi_hat depth mismatch")
    return float(np.dot(ww[:k], y[pi] - y[pi_hat]))


def noise_contrast(z: np.ndarray, pi: np.ndarray, pi_hat: np.ndarray, w: np.ndarray) -> float:
    """N_Z(π, π̂) = Σ_r w_r (Z_π̂(r) - Z_π(r))."""
    z = np.asarray(z, dtype=np.float64)
    return observable_margin(z, pi_hat, pi, w)  # note swapped roles


def exact_weighted_regret(x: np.ndarray, pi_hat: np.ndarray, w: np.ndarray) -> float:
    """R = max_π Σ w_r (x_π(r) - x_π̂(r)) by rearrangement (oracle ranking)."""
    x = np.asarray(x, dtype=np.float64)
    ww = validate_weights(w)
    pi_hat = np.asarray(pi_hat, dtype=np.int64)
    k = pi_hat.size
    pi_star = oracle_ranking(x, k, ww)
    return ranking_utility(x, pi_star, ww[:k]) - ranking_utility(x, pi_hat, ww[:k])


def exact_weighted_regret_enum(x: np.ndarray, pi_hat: np.ndarray, w: np.ndarray) -> float:
    """Brute-force reference for R over A_k (small n)."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    pi_hat = np.asarray(pi_hat, dtype=np.int64)
    k = pi_hat.size
    ww = validate_weights(w)[:k]
    if n > 9:
        raise TopKCertError("enumeration regret only for n<=9")
    base = float(np.dot(ww, x[pi_hat]))
    best = 0.0
    for pi in permutations(range(n), k):
        best = max(best, float(np.dot(ww, x[list(pi)])) - base)
    return float(best)


def orderstat_noise_bound(a: np.ndarray, u_plus: np.ndarray, u_minus: np.ndarray) -> float:
    """Valid bound on <a,Z> under Z_(n-r+1)<=u_r^+ and Z_(r)>=-u_r^- for all used r.

    Positive coefficients are charged to upper order-stat thresholds in decreasing
    coefficient order; negative coefficients to lower order-stat thresholds.
    """
    a = np.asarray(a, dtype=np.float64)
    pos = np.sort(a[a > 0.0])[::-1]
    neg = np.sort(-a[a < 0.0])[::-1]
    if pos.size > u_plus.size or neg.size > u_minus.size:
        raise TopKCertError("order-stat table shorter than coefficient support")
    bound = 0.0
    if pos.size:
        bound += float(np.dot(pos, u_plus[: pos.size]))
    if neg.size:
        bound += float(np.dot(neg, u_minus[: neg.size]))
    return float(bound)


def l1_noise_bound(a: np.ndarray, half_width: float) -> float:
    """Bound <a,Z> under max_i |Z_i| <= half_width: |<a,Z>| <= half_width * ||a||_1."""
    return float(half_width) * float(np.sum(np.abs(a)))


def simplified_l1_support_is_exact(w_hat: np.ndarray, weights: np.ndarray) -> bool:
    """Whether ``max_pi ||W_hat - W_pi||_1`` equals ``2 Σ_r w_r``.

    The identity
    ``||W_hat - W_pi||_1 = 2 Σ w_r - 2 Σ_r min(W_hat(pi(r)), w_r)``
    shows the maximum equals ``2 Σ w_r`` if and only if some injection places
    every positive-weight rank on an item with ``W_hat(i) = 0``. That is
    possible exactly when the number of zero coordinates of ``W_hat`` is at
    least the number of strictly positive weights. ``n >= 2k`` is a sufficient
    condition for a deployed injection, because then at least ``k`` coordinates
    of ``W_hat`` are zero. It is not necessary when some weights are zero.
    """
    hat = np.asarray(w_hat, dtype=np.float64).reshape(-1)
    ww = np.asarray(weights, dtype=np.float64).reshape(-1)
    n_positive = int(np.count_nonzero(ww > 0.0))
    n_zero = int(np.count_nonzero(hat == 0.0))
    return n_zero >= n_positive


def max_l1_support(w_hat: np.ndarray, weights: np.ndarray) -> float:
    """Exact ``max_pi ||W_hat - W_pi||_1`` over injections.

    Uses
    ``max ||a||_1 = 2 Σ w - 2 min_pi Σ_r min(W_hat(pi(r)), w_r)``.
    The inner problem is a rectangular minimum-cost assignment. When
    ``simplified_l1_support_is_exact`` holds, the minimum overlap is zero and
    the closed form ``2 Σ w`` is returned without a solve.
    """
    hat = np.asarray(w_hat, dtype=np.float64).reshape(-1)
    ww = np.asarray(weights, dtype=np.float64).reshape(-1)
    if ww.size == 0:
        return 0.0
    if hat.ndim != 1:
        raise TopKCertError("w_hat must be 1-D")
    if ww.size > hat.size:
        raise TopKCertError("cannot place more ranks than items")
    two = 2.0 * float(np.sum(ww))
    if simplified_l1_support_is_exact(hat, ww):
        return float(two)
    # cost(r, i) = min(W_hat(i), w_r); minimize over injections.
    overlap_cost = np.minimum(hat[np.newaxis, :], ww[:, np.newaxis])
    overlap, _ = _assignment_minimize(overlap_cost)
    overlap = float(overlap)
    if overlap < 0.0:
        if overlap < -1e-8 * (1.0 + abs(two)):
            raise TopKCertError(f"negative overlap {overlap} in L1 support")
        overlap = 0.0
    support = two - 2.0 * overlap
    # Proven bounds: 0 <= max ||a||_1 <= 2 Σ w.
    if support < 0.0:
        if support < -1e-8 * (1.0 + abs(two)):
            raise TopKCertError(f"negative L1 support {support}")
        support = 0.0
    if support > two:
        if support > two + 1e-8 * (1.0 + abs(two)):
            raise TopKCertError(f"L1 support {support} exceeds 2 Σ w")
        support = two
    return float(support)


@dataclass(frozen=True)
class ThresholdTable:
    mechanism: str
    epsilon: float
    n: int
    k_max: int
    alpha: float
    scheme: str
    u_plus: np.ndarray
    u_minus: np.ndarray

    @property
    def range_width(self) -> float:
        return float(self.u_plus[0] + self.u_minus[0])


def build_threshold_table(
    *,
    mechanism: str,
    n: int,
    k_max: int,
    epsilon: float,
    alpha: float,
    weights: np.ndarray | None = None,
    scheme: str = "harmonic",
) -> ThresholdTable:
    if weights is None:
        u_plus, u_minus = build_rank_thresholds(
            mechanism=mechanism, n=n, k_max=k_max, epsilon=epsilon, alpha=alpha, scheme=scheme
        )
        used_scheme = scheme
    else:
        u_plus, u_minus = build_rank_thresholds_weighted(
            mechanism=mechanism,
            n=n,
            k_max=k_max,
            epsilon=epsilon,
            alpha=alpha,
            weights=weights,
        )
        used_scheme = "weight_proportional"
    return ThresholdTable(
        mechanism=str(mechanism),
        epsilon=float(epsilon),
        n=int(n),
        k_max=int(k_max),
        alpha=float(alpha),
        scheme=used_scheme,
        u_plus=np.asarray(u_plus, dtype=np.float64),
        u_minus=np.asarray(u_minus, dtype=np.float64),
    )


def deploy_ranking(y: np.ndarray, k: int) -> np.ndarray:
    """Deployed ranking from clipped release with stable ties (smaller index wins)."""
    return stable_ranking(clip_nonnegative(y), k)


@dataclass(frozen=True)
class CertificateOutcome:
    value: float
    pi_hat: np.ndarray
    best_pi: np.ndarray
    best_g: float
    best_b: float
    method: str


def certificate_coupled_exact(
    y: np.ndarray,
    *,
    k: int,
    weights: np.ndarray,
    u_plus: np.ndarray,
    u_minus: np.ndarray,
    bound: Literal["orderstat", "l1_range", "l1_box"] = "orderstat",
    box_half_width: float | None = None,
) -> CertificateOutcome:
    """Exact max over A_k of G_Y(π)+B(π) for small n."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n = y.size
    ww = validate_weights(weights)[:k]
    if n > 9:
        raise TopKCertError("exact coupled solver only for n<=9")
    pi_hat = deploy_ranking(y, k)
    u_plus = np.asarray(u_plus, dtype=np.float64)
    u_minus = np.asarray(u_minus, dtype=np.float64)

    best = -np.inf
    best_pi = pi_hat.copy()
    best_g = 0.0
    best_b = 0.0
    for pi_t in permutations(range(n), k):
        pi = np.asarray(pi_t, dtype=np.int64)
        g = observable_margin(y, pi, pi_hat, ww)
        a = coefficient_map(pi, pi_hat, ww, n)
        if bound == "orderstat":
            # Prefix-minimum support. Equal to the raw charge on a nonincreasing table.
            from private_mobility_cert.certificates.os_envelope import os_support_effective

            b = os_support_effective(a, u_plus, u_minus)
        elif bound == "l1_range":
            b = l1_noise_bound(a, 0.5 * float(u_plus[0] + u_minus[0]))
        elif bound == "l1_box":
            if box_half_width is None:
                raise TopKCertError("box_half_width required for l1_box")
            b = l1_noise_bound(a, float(box_half_width))
        else:
            raise TopKCertError(f"unknown bound {bound}")
        val = g + b
        if val > best:
            best = val
            best_pi = pi
            best_g = g
            best_b = b
    return CertificateOutcome(
        value=float(max(best, 0.0)),
        pi_hat=pi_hat,
        best_pi=best_pi,
        best_g=float(best_g),
        best_b=float(best_b),
        method=f"exact_{bound}",
    )


def _assignment_extremum(cost: np.ndarray, *, maximize: bool) -> tuple[float, np.ndarray]:
    """Exact rectangular assignment. ``selected[r]`` is the item assigned to rank ``r``.

    SciPy's Hungarian solver is exact for a finite rectangular cost matrix.
    An incomplete or colliding return is a solver failure. This function does
    not replace that failure with a greedy selection: a greedy value is not the
    defined assignment optimum.
    """
    try:
        from scipy.optimize import linear_sum_assignment
    except Exception as exc:  # pragma: no cover
        raise TopKCertError("scipy is required for the scalable assignment solver") from exc
    matrix = np.asarray(cost, dtype=np.float64)
    if matrix.ndim != 2:
        raise TopKCertError("assignment cost must be two-dimensional")
    k, n = matrix.shape
    if k == 0:
        return 0.0, np.zeros(0, dtype=np.int64)
    if k < 0 or n < 0 or k > n:
        raise TopKCertError(f"no injection of {k} ranks into {n} items")
    if not np.all(np.isfinite(matrix)):
        raise TopKCertError("assignment cost must be finite")
    objective = -matrix if maximize else matrix
    row_ind, col_ind = linear_sum_assignment(objective)
    selected = np.full(k, -1, dtype=np.int64)
    seen_rows: set[int] = set()
    seen_cols: set[int] = set()
    total = 0.0
    for r_raw, i_raw in zip(row_ind.tolist(), col_ind.tolist()):
        r = int(r_raw)
        i = int(i_raw)
        if r < 0 or r >= k or i < 0 or i >= n:
            raise TopKCertError(
                "assignment solver returned an out-of-range index; "
                "refusing a greedy approximation"
            )
        if r in seen_rows or i in seen_cols:
            raise TopKCertError(
                "assignment solver returned a non-injection; "
                "refusing a greedy approximation"
            )
        seen_rows.add(r)
        seen_cols.add(i)
        selected[r] = i
        total += float(matrix[r, i])
    if len(seen_rows) != k or int(np.any(selected < 0)):
        raise TopKCertError(
            "assignment solver did not assign every rank; "
            "refusing a greedy approximation"
        )
    return float(total), selected


def _assignment_maximize(cost: np.ndarray) -> tuple[float, np.ndarray]:
    """Maximize ``sum_r cost[r, i_r]`` over injections."""
    return _assignment_extremum(cost, maximize=True)


def _assignment_minimize(cost: np.ndarray) -> tuple[float, np.ndarray]:
    """Minimize ``sum_r cost[r, i_r]`` over injections."""
    return _assignment_extremum(cost, maximize=False)


def certificate_l1_assignment(
    y: np.ndarray,
    *,
    k: int,
    weights: np.ndarray,
    half_width: float,
) -> CertificateOutcome:
    """Scalable valid certificate under max_i |Z_i| <= half_width.

    Uses
        G + t||a||_1
        = Σ_r w_r Y_π(r) - Σ_r w_r Y_π̂(r) + 2 t Σ w - 2 t Σ_i min(W_hat(i), W_π(i)),
    maximized by bipartite assignment with
        cost(r,i) = w_r Y_i - 2 t min(W_hat(i), w_r).
    Complexity: O(n^3) Hungarian on a k×n rectangular problem.
    """
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n = y.size
    ww = validate_weights(weights)[:k]
    t = float(half_width)
    if t < 0:
        raise TopKCertError("half_width must be nonnegative")
    pi_hat = deploy_ranking(y, k)
    w_hat = weight_map(pi_hat, ww, n)
    cost = np.zeros((k, n), dtype=np.float64)
    for r in range(k):
        wr = float(ww[r])
        cost[r, :] = wr * y - 2.0 * t * np.minimum(w_hat, wr)
    best_var, best_pi = _assignment_maximize(cost)
    const = -float(np.dot(ww, y[pi_hat])) + 2.0 * t * float(ww.sum())
    value = float(best_var + const)
    g = observable_margin(y, best_pi, pi_hat, ww)
    a = coefficient_map(best_pi, pi_hat, ww, n)
    b = l1_noise_bound(a, t)
    # Algebraic identity for this same injection: K(Y) + Σ c = G + t||a||_1.
    # Do not replace a mismatch by the larger of the two evaluations.
    value_gb = float(g + b)
    scale = 1.0 + abs(value) + abs(value_gb)
    if abs(value_gb - value) > 1e-6 * scale:
        raise TopKCertError(
            "coupled-box assignment identity failed: "
            f"K+Σc={value} versus G+t||a||_1={value_gb}"
        )
    # The deployed injection attains objective 0, so the maximum is nonnegative.
    # A tiny negative value is float undershoot of that attainable zero, not a
    # second candidate being mixed into the maximum.
    certified = value_gb
    if certified < 0.0:
        if certified < -1e-8 * scale:
            raise TopKCertError(f"coupled certificate negative beyond tolerance: {certified}")
        certified = 0.0
    return CertificateOutcome(
        value=float(certified),
        pi_hat=pi_hat,
        best_pi=best_pi,
        best_g=float(g),
        best_b=float(b),
        method="l1_assignment",
    )


def certificate_orderstat_exact(
    y: np.ndarray,
    *,
    k: int,
    weights: np.ndarray,
    table: ThresholdTable,
) -> CertificateOutcome:
    return certificate_coupled_exact(
        y,
        k=k,
        weights=weights,
        u_plus=table.u_plus,
        u_minus=table.u_minus,
        bound="orderstat",
    )


def certificate_range_l1_assignment(
    y: np.ndarray,
    *,
    k: int,
    weights: np.ndarray,
    table: ThresholdTable,
) -> CertificateOutcome:
    """Valid scalable relaxation using only the rank-1 width B = u_1^+ + u_1^-."""
    half = 0.5 * table.range_width
    out = certificate_l1_assignment(y, k=k, weights=weights, half_width=half)
    return CertificateOutcome(
        value=out.value,
        pi_hat=out.pi_hat,
        best_pi=out.best_pi,
        best_g=out.best_g,
        best_b=out.best_b,
        method="range_l1_assignment",
    )


def old_degenerate_certificate(
    y: np.ndarray,
    *,
    k: int,
    weights: np.ndarray,
    table: ThresholdTable,
) -> tuple[float, float, float]:
    """Diagnostic only: C = D + q with D = max_π G_Y (often identically 0)."""
    y = np.asarray(y, dtype=np.float64)
    ww = validate_weights(weights)[:k]
    pi_hat = deploy_ranking(y, k)
    # Maximize G over A_k by rearrangement on raw Y
    top = stable_ranking(y, k)
    d = float(np.dot(ww, y[top] - y[pi_hat]))
    q = float(np.dot(ww, table.u_plus[:k] + table.u_minus[:k]))
    return float(max(d + q, 0.0)), float(d), float(q)


def range_baseline_coupled(
    y: np.ndarray,
    *,
    k: int,
    weights: np.ndarray,
    table: ThresholdTable,
) -> CertificateOutcome:
    """Matched range baseline: same coupled max, B(π)=(B/2)||a||_1 with B=u1++u1-."""
    return certificate_coupled_exact(
        y,
        k=k,
        weights=weights,
        u_plus=table.u_plus,
        u_minus=table.u_minus,
        bound="l1_range",
    ) if y.size <= 9 else certificate_range_l1_assignment(
        y, k=k, weights=weights, table=table
    )


def box_half_width(mechanism: str, n: int, epsilon: float, alpha: float) -> float:
    mech = mechanism.lower()
    if "geo" in mech or mech in {"m2", "m2_two_sided_geometric"}:
        return coordinate_box_radius_geometric(n, epsilon, alpha)
    return coordinate_box_radius_laplace(n, epsilon, alpha)


def box_baseline_coupled(
    y: np.ndarray,
    *,
    k: int,
    weights: np.ndarray,
    mechanism: str,
    epsilon: float,
    alpha: float,
) -> CertificateOutcome:
    t = box_half_width(mechanism, y.size, epsilon, alpha)
    if y.size <= 9:
        # reuse l1_box exact
        # fabricate dummy u for API; unused
        u = np.zeros(k, dtype=np.float64)
        return certificate_coupled_exact(
            y, k=k, weights=weights, u_plus=u, u_minus=u, bound="l1_box", box_half_width=t
        )
    return certificate_l1_assignment(y, k=k, weights=weights, half_width=t)


def unordered_subset_swap_certificate(
    y: np.ndarray,
    *,
    k: int,
    table: ThresholdTable,
) -> float:
    """Uniform-weight recovery target: classical max_m [d_m + q_m]_+."""
    selected = deployable_selection_from_y_act(y, k)
    q = np.zeros(k + 1, dtype=np.float64)
    for m in range(1, k + 1):
        q[m] = float(np.sum(table.u_plus[:m] + table.u_minus[:m]))
    return float(certificate_c2_subset_swap(y, selected, k=k, q_m=q, alpha=table.alpha).value)


def pathwise_identity_holds(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    k: int,
    *,
    atol: float = 1e-9,
) -> tuple[bool, float, float]:
    """Check R = max_π (G_Y + N_Z) pathwise with Z = Y - x."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = y - x
    ww = validate_weights(w)[:k]
    pi_hat = deploy_ranking(y, k)
    r_true = exact_weighted_regret(x, pi_hat, ww)
    if x.size > 9:
        r_enum = r_true
        coupled = r_true
    else:
        r_enum = exact_weighted_regret_enum(x, pi_hat, ww)
        best = -np.inf
        for pi_t in permutations(range(x.size), k):
            pi = np.asarray(pi_t, dtype=np.int64)
            best = max(best, observable_margin(y, pi, pi_hat, ww) + noise_contrast(z, pi, pi_hat, ww))
        coupled = float(max(best, 0.0))
    ok = abs(r_true - r_enum) <= atol and abs(r_true - coupled) <= atol
    return ok, float(r_true), float(coupled)


def monotonicity_probe(
    *,
    n: int,
    k: int,
    weights: np.ndarray,
    table: ThresholdTable,
    margin_grid: np.ndarray,
) -> list[dict[str, float]]:
    """Synthetic head/tail geometry: larger boundary margins should not increase C."""
    ww = validate_weights(weights)[:k]
    rows: list[dict[str, float]] = []
    for margin in np.asarray(margin_grid, dtype=np.float64).tolist():
        y = np.zeros(n, dtype=np.float64)
        # top-k large, next items margin below the k-th
        for r in range(k):
            y[r] = 100.0 - r
        for j in range(k, n):
            y[j] = y[k - 1] - float(margin) - 0.01 * (j - k)
        if n <= 9:
            exact = certificate_orderstat_exact(y, k=k, weights=ww, table=table)
            c_exact = float(exact.value)
            best_g = float(exact.best_g)
        else:
            exact = certificate_range_l1_assignment(y, k=k, weights=ww, table=table)
            c_exact = float("nan")
            best_g = float(exact.best_g)
        relax = certificate_range_l1_assignment(y, k=k, weights=ww, table=table)
        old_c, old_d, old_q = old_degenerate_certificate(y, k=k, weights=ww, table=table)
        rows.append(
            {
                "margin": float(margin),
                "C_exact": c_exact,
                "C_primary": float(exact.value if n <= 9 else relax.value),
                "C_range_l1": float(relax.value),
                "C_old_degenerate": float(old_c),
                "D_old": float(old_d),
                "q_old": float(old_q),
                "best_g": best_g,
            }
        )
    return rows


__all__ = [
    "ThresholdTable",
    "CertificateOutcome",
    "action_set_size",
    "build_threshold_table",
    "deploy_ranking",
    "exact_weighted_regret",
    "exact_weighted_regret_enum",
    "observable_margin",
    "noise_contrast",
    "coefficient_map",
    "orderstat_noise_bound",
    "certificate_coupled_exact",
    "certificate_orderstat_exact",
    "certificate_l1_assignment",
    "certificate_range_l1_assignment",
    "old_degenerate_certificate",
    "range_baseline_coupled",
    "box_baseline_coupled",
    "box_half_width",
    "unordered_subset_swap_certificate",
    "pathwise_identity_holds",
    "monotonicity_probe",
    "weight_family",
    "clip_nonnegative",
    "stable_ranking",
    "allocate_eta",
    "solve_order_stat_threshold_continuous",
    "solve_order_stat_threshold_integer",
    "certificate_c0_box",
    "all_observed_deficits",
]
