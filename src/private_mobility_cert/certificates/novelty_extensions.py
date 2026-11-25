"""Extensions: margin-blindness, selective deploy, sensitivity-Delta.

These are mathematically checked helpers for propositions stated in the paper.
"""
from __future__ import annotations

import numpy as np

from private_mobility_cert.certificates.release_conditioned import (
    deploy_ranking,
    observable_margin,
)


def separable_is_margin_blind_after_yplus_sort(
    y: np.ndarray,
    k: int,
    weights: np.ndarray,
) -> bool:
    """Return True iff max_pi G_{Y^+}(pi, pi_hat) == 0 for pi_hat = argsort(Y^+).

    Under that condition any certificate of the form max G_{Y^+} + H(E) equals H(E)
    and loses dependence on realized released margins (margin-blindness).
    Coupling uses raw-Y margins G_Y inside the joint maximization and is not covered
    by this exact zero statement.
    """
    y = np.asarray(y, float)
    yplus = np.maximum(y, 0.0)
    ww = np.asarray(weights, float)[:k]
    pi_hat = deploy_ranking(y, k)  # uses Y+
    # maximize released deficit on Y+ by rearranging top-k of Y+
    top = np.argsort(-yplus, kind="stable")[:k]
    # if pi_hat already maximizes sum w Y+, deficit is 0
    max_util = float(np.dot(ww, yplus[top]))
    hat_util = float(np.dot(ww, yplus[pi_hat]))
    return abs(max_util - hat_util) <= 1e-12


def selective_deploy_normalized_bound(
    c: float,
    u: float,
    tau: float,
    *,
    stabilizer: float = 1e-6,
) -> bool:
    """Pathwise deploy predicate: deploy iff C / max(U, stabilizer) <= tau."""
    return (c / max(u, stabilizer)) <= tau


def laplace_scale_for_sensitivity(epsilon: float, delta_sensitivity: float) -> float:
    """Pure eps-DP Laplace scale for l1 sensitivity Delta."""
    if epsilon <= 0 or delta_sensitivity <= 0:
        raise ValueError("epsilon and Delta must be positive")
    return float(delta_sensitivity) / float(epsilon)
