"""Independent block-relaxation oracle for the monotonicity tests.

Costs, constants, and block enumerations are built here. This module does not
import the count-feasible certificate or the closure helper.
"""

from __future__ import annotations

import itertools
from decimal import Decimal

import numpy as np
from scipy.optimize import linear_sum_assignment


class RelaxationReferenceError(RuntimeError):
    pass


def stable_deployed(y: np.ndarray, k: int) -> list[int]:
    values = [float(v) for v in np.asarray(y, dtype=np.float64).reshape(-1)]
    order = sorted(range(len(values)), key=lambda i: (-max(values[i], 0.0), i))
    return order[:k]


def widths(y: np.ndarray, u_minus: float, u_plus: float) -> tuple[list[float], list[float], bool]:
    values = [float(v) for v in np.asarray(y, dtype=np.float64).reshape(-1)]
    lower = float(u_minus)
    upper = float(u_plus)
    if any(v < -lower for v in values):
        return [upper + lower] * len(values), [float("nan")] * len(values), False
    original = [upper + lower] * len(values)
    plus = [min(upper, v) + lower for v in values]
    return original, plus, True


def _weight_hat(pi_hat: list[int], weights: list[float], n: int) -> list[float]:
    hat = [0.0] * n
    for rank, item in enumerate(pi_hat):
        hat[item] = float(weights[rank])
    return hat


def _objective(
    pairs: list[tuple[int, int]],
    y: list[float],
    weights: list[float],
    pi_hat: list[int],
    hat: list[float],
    width: list[float],
) -> tuple[float, list[float]]:
    n = len(y)
    k = len(weights)
    constant = -sum(weights[r] * y[pi_hat[r]] for r in range(k))
    constant += sum(width[i] * hat[i] for i in range(n))
    variable = 0.0
    overlap = [0.0] * n
    for rank, item in pairs:
        variable += weights[rank] * y[item] - width[item] * min(hat[item], weights[rank])
        overlap[item] += min(hat[item], weights[rank])
    return constant + variable, overlap


def _best_block(
    ranks: tuple[int, ...],
    y: list[float],
    weights: list[float],
    hat: list[float],
    width: list[float],
) -> tuple[float, list[tuple[int, int]]]:
    best_value = -1e300
    best_pairs: list[tuple[int, int]] = []
    n = len(y)
    for items in itertools.permutations(range(n), len(ranks)):
        value = 0.0
        for rank, item in zip(ranks, items):
            value += weights[rank] * y[item] - width[item] * min(hat[item], weights[rank])
        if value > best_value:
            best_value = value
            best_pairs = list(zip(ranks, items))
    return best_value, best_pairs


def block_partition(k: int, size: int) -> list[tuple[int, ...]]:
    blocks = []
    rank = 0
    while rank < k:
        blocks.append(tuple(range(rank, min(k, rank + size))))
        rank += size
    return blocks


def evaluate_small(
    y: np.ndarray,
    weights: np.ndarray,
    *,
    u_minus: float,
    u_plus: float,
) -> dict:
    """Enumerate the exact certificate and the rank, pair, and block-5 relaxations."""
    values = [float(v) for v in np.asarray(y, dtype=np.float64).reshape(-1)]
    ww = [float(v) for v in np.asarray(weights, dtype=np.float64).reshape(-1)]
    n = len(values)
    k = len(ww)
    if n > 7 or k > n:
        raise RelaxationReferenceError("enumeration reference supports n<=7 and k<=n")
    pi_hat = stable_deployed(values, k)
    hat = _weight_hat(pi_hat, ww, n)
    original, plus, consistent = widths(values, u_minus, u_plus)
    if not consistent:
        raise RelaxationReferenceError("enumeration reference requires a nonempty intersection")
    partitions = {
        "box": [tuple(range(k))],
        "rank": block_partition(k, 1),
        "pair": block_partition(k, 2),
        "block5": block_partition(k, 5),
    }
    out: dict = {"pi_hat": pi_hat, "W_hat": hat, "d_original": original, "d_plus": plus}
    for label, width in (("original", original), ("plus", plus)):
        family = {}
        for name, blocks in partitions.items():
            pairs: list[tuple[int, int]] = []
            for block in blocks:
                _value, chosen = _best_block(block, values, ww, hat, width)
                pairs.extend(chosen)
            value, overlap = _objective(pairs, values, ww, pi_hat, hat, width)
            family[name] = {
                "value": value,
                "pairs": pairs,
                "overlap": overlap,
                "collisions": sorted(i for i, count in enumerate(overlap) if _reuse(pairs, i)),
            }
        # Separable: max margin and max support, each over global injections.
        base = sum(ww[r] * values[pi_hat[r]] for r in range(k))
        max_margin = -1e300
        max_support = -1e300
        support_width = width
        for items in itertools.permutations(range(n), k):
            margin = sum(ww[r] * values[items[r]] for r in range(k)) - base
            overlap = [0.0] * n
            for rank, item in enumerate(items):
                overlap[item] += min(hat[item], ww[rank])
            support = sum(support_width[i] * (hat[i] - overlap[i]) for i in range(n))
            max_margin = max(max_margin, margin)
            max_support = max(max_support, support)
        family["sep"] = {"value": max(max_margin + max_support, 0.0), "pairs": [], "overlap": []}
        family["strong"] = {
            "value": min(family[name]["value"] for name in ("sep", "rank", "pair", "block5"))
        }
        out[label] = family
    return out


def _reuse(pairs: list[tuple[int, int]], item: int) -> bool:
    return sum(1 for _rank, chosen in pairs if chosen == item) > 1


def fixed_assignment_gap(
    y: np.ndarray,
    weights: np.ndarray,
    pairs: list[tuple[int, int]],
    *,
    u_minus: float,
    u_plus: float,
) -> float:
    """Return R_plus - R_original for one fixed, possibly colliding assignment."""
    values = [float(v) for v in np.asarray(y, dtype=np.float64).reshape(-1)]
    ww = [float(v) for v in np.asarray(weights, dtype=np.float64).reshape(-1)]
    pi_hat = stable_deployed(values, len(ww))
    hat = _weight_hat(pi_hat, ww, len(values))
    original, plus, consistent = widths(values, u_minus, u_plus)
    if not consistent:
        raise RelaxationReferenceError("gap formula requires a nonempty intersection")
    overlap = [0.0] * len(values)
    for rank, item in pairs:
        overlap[item] += min(hat[item], ww[rank])
    return sum((plus[i] - original[i]) * (hat[i] - overlap[i]) for i in range(len(values)))


def decimal_assignment_family(y: list[str], weights: list[str], radius: str) -> dict[str, Decimal]:
    """High-precision evaluation of Hungarian assignments built from the definitions."""
    values = [Decimal(v) for v in y]
    ww = [Decimal(v) for v in weights]
    radius_d = Decimal(radius)
    n = len(values)
    k = len(ww)
    pi_hat = sorted(range(n), key=lambda i: (-max(values[i], 0), i))[:k]
    hat = [Decimal(0)] * n
    for rank, item in enumerate(pi_hat):
        hat[item] = ww[rank]
    families = {}
    for label, width in (
        ("original", [radius_d + radius_d] * n),
        ("plus", [min(radius_d, values[i]) + radius_d for i in range(n)]),
    ):
        constant = -sum(ww[r] * values[pi_hat[r]] for r in range(k))
        constant += sum(width[i] * hat[i] for i in range(n))
        y_float = np.array([float(v) for v in values])
        w_float = np.array([float(v) for v in ww])
        hat_float = np.array([float(v) for v in hat])
        width_float = np.array([float(v) for v in width])

        def solve(rows: range | list[int]) -> list[int]:
            cost = np.zeros((len(list(rows)), n))
            for local, rank in enumerate(rows):
                cost[local] = (
                    w_float[rank] * y_float
                    - width_float * np.minimum(hat_float, w_float[rank])
                )
            _row, col = linear_sum_assignment(-cost)
            return col.tolist()

        def block_sum(rows: list[int], items: list[int]) -> Decimal:
            total = Decimal(0)
            for rank, item in zip(rows, items):
                total += ww[rank] * values[item] - width[item] * min(hat[item], ww[rank])
            return total

        full_rows = list(range(k))
        box = constant + block_sum(full_rows, solve(full_rows))
        pieces = {
            "rank": [list(range(r, r + 1)) for r in range(k)],
            "pair": [list(range(r, min(k, r + 2))) for r in range(0, k, 2)],
            "block5": [list(range(r, min(k, r + 5))) for r in range(0, k, 5)],
        }
        out = {"box": box}
        for name, blocks in pieces.items():
            total = constant
            for block in blocks:
                total += block_sum(block, solve(block))
            out[name] = total
        # Separable support from two global assignments.
        margin_cost = np.vstack([float(wr) * y_float for wr in ww])
        _row, col = linear_sum_assignment(-margin_cost)
        margin = sum(ww[r] * values[int(col[r])] for r in range(k))
        margin -= sum(ww[r] * values[pi_hat[r]] for r in range(k))
        overlap = np.zeros((k, n))
        for rank in range(k):
            overlap[rank] = width_float * np.minimum(hat_float, w_float[rank])
        _row, col = linear_sum_assignment(overlap)
        min_overlap = sum(
            width[int(col[r])] * min(hat[int(col[r])], ww[r]) for r in range(k)
        )
        support = sum(width[i] * hat[i] for i in range(n)) - min_overlap
        out["sep"] = margin + support
        out["strong"] = min(out["sep"], out["rank"], out["pair"], out["block5"])
        families[label] = out
    return {
        "box_original": families["original"]["box"],
        "box_plus": families["plus"]["box"],
        "strong_original": families["original"]["strong"],
        "strong_plus": families["plus"]["strong"],
        "block5_original": families["original"]["block5"],
        "block5_plus": families["plus"]["block5"],
        "rank_original": families["original"]["rank"],
        "rank_plus": families["plus"]["rank"],
        "pair_original": families["original"]["pair"],
        "pair_plus": families["plus"]["pair"],
        "sep_original": families["original"]["sep"],
        "sep_plus": families["plus"]["sep"],
    }
