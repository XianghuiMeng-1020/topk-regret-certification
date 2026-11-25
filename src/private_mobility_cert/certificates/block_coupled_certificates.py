"""Same-event block-coupled relaxations of the exact coordinate-box certificate.

On the symmetric coordinate-box event of half-width ``t``, the exact coupled
certificate equals

    C_box(Y) = const(Y) + max_{injective π} Σ_r cost(r, π(r)),

with

    cost(r, i) = w_r Y_i - 2 t min(W_hat(i), w_r),
    const(Y)   = -⟨w, Y[π̂]⟩ + 2 t Σ_r w_r.

For a partition P of the k rank positions, the block-coupled relaxation
enforces injectivity only within each block, solves a rectangular assignment
per block, and sums block objectives plus the shared constant. Every such
relaxation remains a valid box-event upper certificate; merging blocks is
monotone nonincreasing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from private_mobility_cert.certificates.release_conditioned import (
    CertificateOutcome,
    _assignment_maximize,
    coefficient_map,
    deploy_ranking,
    l1_noise_bound,
    max_l1_support,
    observable_margin,
    simplified_l1_support_is_exact,
    weight_map,
)
from private_mobility_cert.certificates.topk_regret import TopKCertError
from private_mobility_cert.certificates.weighted_ranking import validate_weights


@dataclass(frozen=True)
class BlockPartition:
    """Ordered blocks of rank indices in ``{0,...,k-1}`` covering each rank once."""

    blocks: tuple[tuple[int, ...], ...]

    def validate(self, k: int) -> None:
        seen: list[int] = []
        for b in self.blocks:
            if not b:
                raise TopKCertError("empty block")
            seen.extend(b)
        if sorted(seen) != list(range(k)):
            raise TopKCertError("partition must cover ranks 0..k-1 exactly once")


def consecutive_blocks(k: int, block_size: int) -> BlockPartition:
    if block_size <= 0:
        raise TopKCertError("block_size must be positive")
    blocks = []
    r = 0
    while r < k:
        blocks.append(tuple(range(r, min(k, r + block_size))))
        r += block_size
    return BlockPartition(tuple(blocks))


def singleton_partition(k: int) -> BlockPartition:
    return consecutive_blocks(k, 1)


def full_partition(k: int) -> BlockPartition:
    return BlockPartition((tuple(range(k)),))


def pair_partition(k: int) -> BlockPartition:
    return consecutive_blocks(k, 2)


def block5_partition(k: int) -> BlockPartition:
    return consecutive_blocks(k, 5)


def box_cost_matrix(
    y: np.ndarray, *, k: int, weights: np.ndarray, half_width: float
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Return ``(cost[k,n], const, pi_hat, ww)`` for the box assignment form."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n = y.size
    if not 0 < k <= n:
        raise TopKCertError("require 0 < k <= n")
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
    const = -float(np.dot(ww, y[pi_hat])) + 2.0 * t * float(ww.sum())
    return cost, const, pi_hat, ww


def certificate_block_coupled_box(
    y: np.ndarray,
    *,
    k: int,
    weights: np.ndarray,
    half_width: float,
    partition: BlockPartition,
) -> CertificateOutcome:
    """Valid box-event upper certificate under a rank-block injectivity partition."""
    cost, const, pi_hat, ww = box_cost_matrix(
        y, k=k, weights=weights, half_width=half_width
    )
    partition.validate(k)
    total_var = 0.0
    # Concatenate per-block chosen items only for diagnostics (may collide).
    chosen: list[int] = [-1] * k
    for block in partition.blocks:
        idx = np.asarray(block, dtype=np.int64)
        block_cost = cost[idx, :]
        val, sel = _assignment_maximize(block_cost)
        total_var += float(val)
        for local, item in enumerate(sel.tolist()):
            chosen[int(block[local])] = int(item)
    value = float(max(const + total_var, 0.0))
    best_pi = np.asarray(chosen, dtype=np.int64)
    # If collisions remain, g/b are diagnostic only; value is still a valid upper.
    if len(set(chosen)) == k and -1 not in chosen:
        g = observable_margin(y, best_pi, pi_hat, ww)
        a = coefficient_map(best_pi, pi_hat, ww, y.size)
        b = l1_noise_bound(a, float(half_width))
    else:
        g = float("nan")
        b = float("nan")
    return CertificateOutcome(
        value=value,
        pi_hat=pi_hat,
        best_pi=best_pi,
        best_g=g,
        best_b=b,
        method=f"block_coupled_sizes={[len(b) for b in partition.blocks]}",
    )


def certificate_full_coupled_box(
    y: np.ndarray, *, k: int, weights: np.ndarray, half_width: float
) -> CertificateOutcome:
    out = certificate_block_coupled_box(
        y, k=k, weights=weights, half_width=half_width, partition=full_partition(k)
    )
    # Align method name with the existing exact box assignment.
    from private_mobility_cert.certificates.release_conditioned import (
        certificate_l1_assignment,
    )

    ref = certificate_l1_assignment(y, k=k, weights=weights, half_width=half_width)
    return CertificateOutcome(
        value=float(ref.value),
        pi_hat=ref.pi_hat,
        best_pi=ref.best_pi,
        best_g=ref.best_g,
        best_b=ref.best_b,
        method="full_coupled_box",
    )


def certificate_rankwise_box(
    y: np.ndarray, *, k: int, weights: np.ndarray, half_width: float
) -> CertificateOutcome:
    return certificate_block_coupled_box(
        y, k=k, weights=weights, half_width=half_width, partition=singleton_partition(k)
    )


def certificate_pair_block_box(
    y: np.ndarray, *, k: int, weights: np.ndarray, half_width: float
) -> CertificateOutcome:
    return certificate_block_coupled_box(
        y, k=k, weights=weights, half_width=half_width, partition=pair_partition(k)
    )


def certificate_block5_box(
    y: np.ndarray, *, k: int, weights: np.ndarray, half_width: float
) -> CertificateOutcome:
    return certificate_block_coupled_box(
        y, k=k, weights=weights, half_width=half_width, partition=block5_partition(k)
    )


def matched_separable_box_value(
    y: np.ndarray, *, k: int, weights: np.ndarray, half_width: float
) -> float:
    """General matched separable certificate on the half-width-t coordinate box.

    ``C_sep(Y) = max_pi G_Y(pi, pi_hat) + t * max_pi ||W_hat - W_pi||_1``.
    The second factor equals ``2 Σ w`` exactly when a competitor can place
    every positive-weight rank outside the positive support of ``W_hat``.
    ``n >= 2k`` is a sufficient condition for that equality, not an identity
    for every ``n >= k``.
    """
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    ww = validate_weights(weights)[:k]
    t = float(half_width)
    if t < 0.0:
        raise TopKCertError("half_width must be nonnegative")
    pi_hat = deploy_ranking(y, k)
    cost = np.vstack([float(wr) * y for wr in ww])
    best_var, _ = _assignment_maximize(cost)
    max_g = float(best_var - float(np.dot(ww, y[pi_hat])))
    w_hat = weight_map(pi_hat, ww, y.size)
    if simplified_l1_support_is_exact(w_hat, ww):
        # Fixed product order keeps floating-point results deterministic.
        h_star = 2.0 * t * float(ww.sum())
    else:
        h_star = t * max_l1_support(w_hat, ww)
    return float(max(max_g + h_star, 0.0))


def strongest_nonexact_same_event(
    y: np.ndarray, *, k: int, weights: np.ndarray, half_width: float
) -> tuple[float, dict[str, float]]:
    """Pointwise minimum of valid same-event box upper certificates.

    The separable term is the general certificate, not the unconditional
    ``2 t Σ w`` envelope.
    """
    vals = {
        "separable": matched_separable_box_value(
            y, k=k, weights=weights, half_width=half_width
        ),
        "rankwise": float(
            certificate_rankwise_box(
                y, k=k, weights=weights, half_width=half_width
            ).value
        ),
        "pair": float(
            certificate_pair_block_box(
                y, k=k, weights=weights, half_width=half_width
            ).value
        ),
        "block5": float(
            certificate_block5_box(
                y, k=k, weights=weights, half_width=half_width
            ).value
        ),
    }
    return float(min(vals.values())), vals


def partition_complexity(k: int, n: int, partition: BlockPartition) -> dict[str, int]:
    """Hungarian cost model: each block of size b on n items is O(max(b,n)^3)."""
    partition.validate(k)
    per_block = []
    total = 0
    for b in partition.blocks:
        m = max(len(b), n)
        c = m * m * m
        per_block.append(c)
        total += c
    return {"per_block_cubic": per_block, "sum_cubic": total, "n": n, "k": k}
