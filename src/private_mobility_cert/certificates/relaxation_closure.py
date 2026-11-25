"""Optional monotone closure of a count-feasible structural relaxation.

A raw block relaxation can increase when the count-feasible box is a subset of
the original box. Cross-block item reuse makes the relaxed objective different
from the support of one globally injective competitor, so event inclusion does
not imply monotonicity.

If ``R_original`` is a valid upper bound on the original coupled certificate
and ``R_plus_raw`` is a valid upper bound on the count-feasible coupled
certificate, both dominate ``C_box_plus``:

    C_box_plus <= C_box_original <= R_original,
    C_box_plus <= R_plus_raw.

Their minimum is therefore a valid upper bound on ``C_box_plus``. It is a
hybrid of the two events. It is not the raw count-feasible relaxation, and it
does not make that raw relaxation monotone.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from private_mobility_cert.certificates.topk_regret import TopKCertError

CLOSURE_METHOD = "monotone_closed_min_original_and_raw_plus"


def monotone_closed_value(original: float, plus_raw: float) -> float:
    """Return ``min(R_plus_raw, R_original)`` for one named relaxation.

    Both arguments must already be valid upper bounds on the same
    count-feasible coupled certificate. This function does not recompute either
    bound and does not replace a stored raw value.
    """
    left = float(original)
    right = float(plus_raw)
    if left != left or right != right:
        raise TopKCertError("monotone closure requires finite certificate values")
    return float(min(left, right))


def close_matched_family(
    original: Mapping[str, float],
    plus_raw: Mapping[str, float],
    keys: Sequence[str],
) -> dict[str, float]:
    """Close each named raw relaxation. Keys absent from either map raise."""
    closed: dict[str, float] = {}
    for key in keys:
        if key not in original or key not in plus_raw:
            raise TopKCertError(f"monotone closure is missing {key}")
        closed[key] = monotone_closed_value(original[key], plus_raw[key])
    return closed
