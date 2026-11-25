"""Valid full-universe upper bound on the order-statistic coupled certificate.

The bound is an envelope of rectangular assignment problems. It does not call
the stratified m-swap routine. The half-width used for an index s > 1 is an
algebraic device inside the support inequality. It is not a 1-alpha box.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from private_mobility_cert.certificates.release_conditioned import (
    ThresholdTable,
    certificate_l1_assignment,
    deploy_ranking,
)
from private_mobility_cert.certificates.topk_regret import TopKCertError
from private_mobility_cert.certificates.weighted_ranking import validate_weights

METHOD_OS_ENVELOPE = "os_envelope_v1"
METHOD_PRIMARY_OS_ENVELOPE = "primary_os_envelope"


def prefix_min_thresholds(thresholds: np.ndarray) -> np.ndarray:
    """Return U_j = min_{r <= j} u_r.

    The declared order-statistic event forces the j-th extreme coordinate to
    obey every tighter constraint among the first j ranks. The prefix minimum
    is therefore the effective threshold, and the resulting sequence is
    nonincreasing.
    """
    raw = np.asarray(thresholds, dtype=np.float64).reshape(-1)
    if raw.size == 0:
        raise TopKCertError("order-statistic threshold table is empty")
    if not np.all(np.isfinite(raw)):
        raise TopKCertError("order-statistic thresholds must be finite")
    if np.any(raw < 0.0):
        raise TopKCertError("order-statistic thresholds must be nonnegative")
    return np.minimum.accumulate(raw)


def os_support_effective(
    coefficients: np.ndarray, u_plus: np.ndarray, u_minus: np.ndarray
) -> float:
    """Exact support of <a, Z> on the continuous order-statistic event.

    Positive masses are paired with the descending upper extremes and negative
    masses with the descending lower extremes. Pairing uses prefix-minimum
    thresholds, which are feasible simultaneously: the r-th largest coordinate
    can sit at U_r^+ and the r-th smallest at -U_r^- because those sequences
    are nonnegative and nonincreasing, and the two supports occupy disjoint
    coordinates.
    """
    coefficients = np.asarray(coefficients, dtype=np.float64).reshape(-1)
    upper = prefix_min_thresholds(u_plus)
    lower = prefix_min_thresholds(u_minus)
    positive = np.sort(coefficients[coefficients > 0.0])[::-1]
    negative = np.sort(-coefficients[coefficients < 0.0])[::-1]
    if positive.size > upper.size or negative.size > lower.size:
        raise TopKCertError("order-stat table shorter than coefficient support")
    value = 0.0
    if positive.size:
        value += float(np.dot(positive, upper[: positive.size]))
    if negative.size:
        value += float(np.dot(negative, lower[: negative.size]))
    return float(value)


@dataclass(frozen=True)
class EnvelopeTerm:
    """One admissible assignment correction for a fixed event."""

    s: int
    half_width: float
    delta: float


def envelope_schedule(
    weights: np.ndarray, u_plus: np.ndarray, u_minus: np.ndarray, *, k: int
) -> tuple[np.ndarray, list[EnvelopeTerm]]:
    """Build t_s and delta_s for s = 1, ..., min(k, Kmax).

    With U the prefix minima and w_1 the largest weight,
    delta_s = w_1 * sum_{j < s} [(U_j^+ - U_s^+) + (U_j^- - U_s^-)]
    and t_s = (U_s^+ + U_s^-) / 2. The s = 1 term has delta_1 = 0.
    """
    if k < 1:
        raise TopKCertError("k must be positive")
    ww = validate_weights(weights)
    if ww.size < k:
        raise TopKCertError("weights shorter than k")
    upper = prefix_min_thresholds(u_plus)
    lower = prefix_min_thresholds(u_minus)
    if upper.size < k or lower.size < k:
        raise TopKCertError("threshold table shorter than k")
    upper = upper[:k]
    lower = lower[:k]
    w1 = float(ww[0])
    terms: list[EnvelopeTerm] = []
    for s in range(1, k + 1):
        gap = float(np.sum((upper[: s - 1] - upper[s - 1]) + (lower[: s - 1] - lower[s - 1])))
        terms.append(
            EnvelopeTerm(
                s=s,
                half_width=float(0.5 * (upper[s - 1] + lower[s - 1])),
                delta=float(w1 * gap),
            )
        )
    return ww[:k], terms


@dataclass(frozen=True)
class OSEnvelopeOutcome:
    """Coupled order-statistic envelope and the term that attained it."""

    value: float
    pi_hat: np.ndarray
    best_pi: np.ndarray
    best_g: float
    best_b: float
    method: str
    chosen_s: int
    half_width: float
    delta: float
    n_assignment_solves: int
    rank1_value: float
    fallback_used: bool


def certificate_os_envelope(
    y: np.ndarray,
    *,
    k: int,
    weights: np.ndarray,
    table: ThresholdTable,
) -> OSEnvelopeOutcome:
    """Return min_s [C_box(Y; t_s) + delta_s] on the order-statistic event.

    Each summand upper-bounds max_pi [G_Y + h_EOS] because
    h_EOS(a) <= t_s ||a||_1 + delta_s for every competitor coefficient a.
    The minimum needs no additional union bound: every summand is valid for
    the same event. s = 1 reproduces the rank-1 coupled box.
    """
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    _weights, terms = envelope_schedule(weights, table.u_plus, table.u_minus, k=k)
    solves: dict[float, object] = {}
    best_value = float("inf")
    best_term = terms[0]
    best_out = None
    for term in terms:
        if term.half_width not in solves:
            solves[term.half_width] = certificate_l1_assignment(
                y, k=k, weights=weights, half_width=term.half_width
            )
        out = solves[term.half_width]
        candidate = float(out.value + term.delta)
        if not np.isfinite(candidate):
            raise TopKCertError("non-finite order-statistic envelope candidate")
        if candidate < best_value:
            best_value = candidate
            best_term = term
            best_out = out
    if best_out is None:
        raise TopKCertError("order-statistic envelope produced no candidate")
    rank1 = float(solves[terms[0].half_width].value)
    # s = 1 is one of the candidates, so the minimum cannot exceed the rank-1 box
    # by more than a comparison of identical floats. A larger value is a bug.
    if best_value > rank1 + 1e-8 * (1.0 + abs(rank1)):
        raise TopKCertError(
            f"OS envelope {best_value} exceeds its rank-1 fallback {rank1}"
        )
    return OSEnvelopeOutcome(
        value=float(best_value),
        pi_hat=np.asarray(best_out.pi_hat, dtype=np.int64),
        best_pi=np.asarray(best_out.best_pi, dtype=np.int64),
        best_g=float(best_out.best_g),
        best_b=float(best_out.best_b),
        method=METHOD_OS_ENVELOPE,
        chosen_s=int(best_term.s),
        half_width=float(best_term.half_width),
        delta=float(best_term.delta),
        n_assignment_solves=int(len(solves)),
        rank1_value=rank1,
        fallback_used=bool(best_term.s == 1),
    )


def deployed_prefix(y: np.ndarray, k: int) -> np.ndarray:
    """Deployed ranking. Exposed for tests that must not reimplement clipping."""
    return deploy_ranking(y, k)
