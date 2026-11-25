"""Same-event structural envelope for the order-statistic support bound.

The coupled piece is the existing identity

    h_EOS(a) <= t_s ||a||_1 + delta_s,

so the exact order-statistic coupled certificate satisfies

    C_cpl,EOS(Y) <= C_box(Y; t_s) + delta_s

for every s. Each structural relaxation (separable, rank, pair, block) at that same auxiliary half-width is at least
the box certificate, and delta_s is added once to the finished relaxation:

    C_box(Y; t_s) + delta_s <= C_strong(Y; t_s) + delta_s.

The minimum over s is still one upper bound for the single event E_OS.
It is not a new union bound and it is not the exact OS optimum. The
half-width t_s for s > 1 is not claimed to be a 1-alpha box by itself.

For k <= 5 the block-5 partition is the full assignment, so the decision
copies satisfy C_strong(t_s) = C_box(t_s) and the two envelopes agree.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from private_mobility_cert.certificates.block_coupled_certificates import (
    block5_partition,
    certificate_block5_box,
    certificate_pair_block_box,
    certificate_rankwise_box,
    pair_partition,
)
from private_mobility_cert.certificates.direct_box_calibration import (
    canonicalize_k5_certificates,
)
from private_mobility_cert.certificates.os_envelope import (
    METHOD_OS_ENVELOPE,
    envelope_schedule,
)
from private_mobility_cert.certificates.release_conditioned import (
    ThresholdTable,
    _assignment_maximize,
    certificate_l1_assignment,
    deploy_ranking,
    max_l1_support,
    simplified_l1_support_is_exact,
    weight_map,
)
from private_mobility_cert.certificates.topk_regret import TopKCertError
from private_mobility_cert.certificates.weighted_ranking import validate_weights

METHOD_OS_MATCHED = "os_matched_strong_envelope_v1"
# A structural arm is skipped only when the already computed box arm exceeds
# the incumbent strong envelope by more than this relative gap. Under
# C_strong >= C_box the skipped arm cannot improve the minimum.
PRUNE_RELATIVE_GAP = 1e-7


@dataclass(frozen=True)
class OSMatchedOutcome:
    """Four same-event upper bounds and the envelope terms that attain them."""

    rank1_coupled: float
    rank1_strong: float
    coupled_envelope: float
    strong_envelope: float
    strong_envelope_independent: float
    chosen_s_coupled: int
    chosen_s_strong: int
    half_width_coupled: float
    half_width_strong: float
    delta_coupled: float
    delta_strong: float
    n_box_solves: int
    n_structural_solves: int
    n_unique_half_widths: int
    n_pruned_structural: int
    runtime_box_s: float
    runtime_structural_s: float
    k5_identity_applied: int
    coupled_method: str
    strong_method: str


def separable_pieces(y: np.ndarray, k: int, weights: np.ndarray) -> tuple[float, float]:
    """Return max_G and the unit L1 support so C_sep(t) = max(max_G + t h, 0)."""
    values = np.asarray(y, dtype=np.float64).reshape(-1)
    ww = validate_weights(weights)[:k]
    pi_hat = deploy_ranking(values, k)
    cost = np.vstack([float(wr) * values for wr in ww])
    best_var, _ = _assignment_maximize(cost)
    max_g = float(best_var - float(np.dot(ww, values[pi_hat])))
    w_hat = weight_map(pi_hat, ww, values.size)
    if simplified_l1_support_is_exact(w_hat, ww):
        h_unit = 2.0 * float(ww.sum())
    else:
        h_unit = float(max_l1_support(w_hat, ww))
    return max_g, h_unit


def _family_at(
    y: np.ndarray,
    *,
    k: int,
    weights: np.ndarray,
    half_width: float,
    max_g: float,
    h_unit: float,
) -> tuple[dict[str, float], float, float]:
    started = time.perf_counter()
    box = float(
        certificate_l1_assignment(
            y, k=k, weights=weights, half_width=half_width
        ).value
    )
    box_s = time.perf_counter() - started
    started = time.perf_counter()
    separable = float(max(max_g + half_width * h_unit, 0.0))
    rank = float(
        certificate_rankwise_box(
            y, k=k, weights=weights, half_width=half_width
        ).value
    )
    pair = float(
        certificate_pair_block_box(
            y, k=k, weights=weights, half_width=half_width
        ).value
    )
    block5 = float(
        certificate_block5_box(
            y, k=k, weights=weights, half_width=half_width
        ).value
    )
    strong = float(min(separable, rank, pair, block5))
    raw = {
        "C_box": box,
        "C_sep_general": separable,
        "C_rank": rank,
        "C_pair": pair,
        "C_block5": block5,
        "C_strong": strong,
    }
    decision = canonicalize_k5_certificates(raw, k=k)
    structural_s = time.perf_counter() - started
    return (
        {
            "C_box": float(decision["C_box"]),
            "C_strong": float(decision["C_strong"]),
            "C_strong_independent": float(decision["C_strong_independent"]),
            "k5_identity_applied": float(decision["k5_identity_applied"]),
        },
        box_s,
        structural_s,
    )


def certificate_os_matched(
    y: np.ndarray,
    *,
    k: int,
    weights: np.ndarray,
    table: ThresholdTable,
    prune: bool = True,
) -> OSMatchedOutcome:
    """Return the rank-1 bounds and both OS envelopes on one threshold table.

    ``delta_s`` is added to the completed box or strong value. It is not added
    inside each block. No minimum is taken with a union-bound or direct box.
    """
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    _weights, terms = envelope_schedule(weights, table.u_plus, table.u_minus, k=k)
    t0 = time.perf_counter()
    max_g, h_unit = separable_pieces(y, k, weights)
    separable_s = time.perf_counter() - t0
    cache: dict[float, dict[str, float]] = {}
    box_s = 0.0
    structural_s = separable_s
    best_coupled = float("inf")
    best_strong = float("inf")
    best_strong_ind = float("inf")
    term_coupled = terms[0]
    term_strong = terms[0]
    k5_flag = 0
    # Every distinct auxiliary width is solved. ``prune`` only counts arms
    # whose box-plus-delta cannot improve the strong envelope. Those arms are
    # still included in the minimum, so the count is not a skipped-solve claim.
    del prune
    for term in terms:
        key = float(term.half_width)
        if key not in cache:
            family, one_box_s, one_structural_s = _family_at(
                y, k=k, weights=weights, half_width=key, max_g=max_g, h_unit=h_unit
            )
            cache[key] = family
            box_s += one_box_s
            structural_s += one_structural_s
            k5_flag = int(family["k5_identity_applied"])
        family = cache[key]
        coupled_candidate = float(family["C_box"] + term.delta)
        strong_candidate = float(family["C_strong"] + term.delta)
        strong_ind = float(family["C_strong_independent"] + term.delta)
        if not np.isfinite(coupled_candidate) or not np.isfinite(strong_candidate):
            raise TopKCertError("non-finite OS matched envelope candidate")
        if coupled_candidate < best_coupled:
            best_coupled = coupled_candidate
            term_coupled = term
        if strong_candidate < best_strong:
            best_strong = strong_candidate
            term_strong = term
        if strong_ind < best_strong_ind:
            best_strong_ind = strong_ind
    n_pruned = 0
    incumbent = float("inf")
    for term in terms:
        family = cache[float(term.half_width)]
        coupled_candidate = float(family["C_box"] + term.delta)
        if incumbent < float("inf") and coupled_candidate > incumbent + PRUNE_RELATIVE_GAP * (
            1.0 + abs(incumbent)
        ):
            n_pruned += 1
        strong_candidate = float(family["C_strong"] + term.delta)
        if strong_candidate < incumbent:
            incumbent = strong_candidate
    rank1 = cache[float(terms[0].half_width)]
    rank1_coupled = float(rank1["C_box"])
    rank1_strong = float(rank1["C_strong"])
    scale = 1.0 + abs(rank1_coupled)
    if best_coupled > rank1_coupled + 1e-8 * scale:
        raise TopKCertError(
            f"OS coupled envelope {best_coupled} exceeds rank-1 box {rank1_coupled}"
        )
    if best_strong > rank1_strong + 1e-8 * (1.0 + abs(rank1_strong)):
        raise TopKCertError(
            f"OS strong envelope {best_strong} exceeds rank-1 strong {rank1_strong}"
        )
    if best_coupled > best_strong + 1e-6 * (1.0 + abs(best_strong)):
        raise TopKCertError(
            f"OS coupled envelope {best_coupled} exceeds OS strong envelope {best_strong}"
        )
    n_widths = len(cache)
    n_rank = int(k)
    n_pair = len(pair_partition(k).blocks)
    n_block = len(block5_partition(k).blocks)
    # One separable assignment, plus one box, rank, pair, and block-5 family
    # per distinct auxiliary half-width. Rank contributes k assignments.
    n_box = n_widths
    n_structural = 1 + n_widths * (n_rank + n_pair + n_block)
    return OSMatchedOutcome(
        rank1_coupled=rank1_coupled,
        rank1_strong=rank1_strong,
        coupled_envelope=float(best_coupled),
        strong_envelope=float(best_strong),
        strong_envelope_independent=float(best_strong_ind),
        chosen_s_coupled=int(term_coupled.s),
        chosen_s_strong=int(term_strong.s),
        half_width_coupled=float(term_coupled.half_width),
        half_width_strong=float(term_strong.half_width),
        delta_coupled=float(term_coupled.delta),
        delta_strong=float(term_strong.delta),
        n_box_solves=n_box,
        n_structural_solves=n_structural,
        n_unique_half_widths=n_widths,
        n_pruned_structural=int(n_pruned),
        runtime_box_s=float(box_s),
        runtime_structural_s=float(separable_s + structural_s),
        k5_identity_applied=k5_flag,
        coupled_method=METHOD_OS_ENVELOPE,
        strong_method=METHOD_OS_MATCHED,
    )
