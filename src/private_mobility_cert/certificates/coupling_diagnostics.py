"""Exact coupling-necessity diagnostics for same-event block relaxations."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from private_mobility_cert.certificates.block_coupled_certificates import (
    BlockPartition,
    box_cost_matrix,
    certificate_block_coupled_box,
    certificate_full_coupled_box,
    consecutive_blocks,
    full_partition,
    matched_separable_box_value,
    pair_partition,
    singleton_partition,
    strongest_nonexact_same_event,
)
from private_mobility_cert.certificates.release_conditioned import _assignment_maximize
from private_mobility_cert.certificates.topk_regret import TopKCertError


def _bipartite_max_matching(adj: list[list[int]], n_right: int) -> list[int | None]:
    """DFS bipartite matching. Returns mate_right[item] = left rank or None."""
    mate_r = [-1] * n_right

    def dfs(u: int, seen: list[bool]) -> bool:
        for v in adj[u]:
            if seen[v]:
                continue
            seen[v] = True
            if mate_r[v] == -1 or dfs(mate_r[v], seen):
                mate_r[v] = u
                return True
        return False

    for u in range(len(adj)):
        dfs(u, [False] * n_right)
    return [None if m < 0 else m for m in mate_r]


def rankwise_argmax_sets(cost: np.ndarray, *, atol: float = 1e-10) -> list[list[int]]:
    """For each rank, all items attaining the row maximum (ties kept)."""
    k, n = cost.shape
    sets: list[list[int]] = []
    for r in range(k):
        mx = float(np.max(cost[r]))
        sets.append([i for i in range(n) if cost[r, i] >= mx - atol])
    return sets


def hall_sdr_exists(argmax_sets: list[list[int]], n_items: int) -> tuple[bool, int]:
    """Return (exists SDR, size of maximum matching)."""
    adj = [list(s) for s in argmax_sets]
    mate = _bipartite_max_matching(adj, n_items)
    matched = sum(1 for m in mate if m is not None)
    return matched == len(argmax_sets), matched


def block_optimal_value(cost: np.ndarray, block: tuple[int, ...]) -> float:
    idx = np.asarray(block, dtype=np.int64)
    val, _ = _assignment_maximize(cost[idx, :])
    return float(val)


def compatibility_penalty(
    cost: np.ndarray,
    partition: BlockPartition,
    *,
    atol: float = 1e-9,
) -> float:
    """Exact T9 penalty: R_P - F = min reconciliation loss (via R_P - F).

    Computed as sum of independent block optima minus the globally injective
    assignment value on ``cost``. Equivalent to the block reconciliation
    problem (T14: not a faster algorithm).
    """
    partition.validate(cost.shape[0])
    r_var = sum(block_optimal_value(cost, b) for b in partition.blocks)
    f_var, _ = _assignment_maximize(cost)
    pen = float(r_var - f_var)
    if pen < -atol:
        raise TopKCertError(f"negative compatibility penalty {pen}")
    return float(max(pen, 0.0))


def rankwise_deficit_matrix(cost: np.ndarray) -> np.ndarray:
    """Row-wise deficits d_ri = max_j c(r,j) - c(r,i) (T11)."""
    m = np.max(cost, axis=1, keepdims=True)
    return m - cost


def rankwise_reconciliation_penalty(
    cost: np.ndarray, *, atol: float = 1e-9
) -> tuple[float, np.ndarray]:
    """T11: min injective sum d_r,pi(r); returns (penalty, witness pi)."""
    d = rankwise_deficit_matrix(cost)
    # minimize sum d = maximize sum (-d)
    neg_val, pi = _assignment_maximize(-d)
    pen = float(-neg_val)
    # cross-check vs T9 on singletons
    pen2 = compatibility_penalty(cost, singleton_partition(cost.shape[0]), atol=atol)
    if abs(pen - pen2) > atol:
        raise TopKCertError(f"T11/T9 mismatch: {pen} vs {pen2}")
    return float(max(pen, 0.0)), np.asarray(pi, dtype=np.int64)


def exists_compatible_block_optima(
    cost: np.ndarray,
    partition: BlockPartition,
    *,
    atol: float = 1e-9,
) -> bool:
    """Exact check for small problems: enumerate optima per block when tiny.

    For general sizes, falls back to a sufficient constructive check:
    solve full assignment; if its block restrictions attain each block optimum,
    compatibility holds. If not, try a sequential greedy claim of unused items
    among near-optimal block assignments via reassignment with forbidden items.
    """
    partition.validate(cost.shape[0])
    # Sufficient: global optimum restricts to block optima.
    full_val, full_pi = _assignment_maximize(cost)
    used_ok = True
    for block in partition.blocks:
        idx = list(block)
        block_cost = cost[np.asarray(idx, dtype=np.int64), :]
        opt = block_optimal_value(cost, block)
        achieved = float(sum(cost[r, full_pi[local]] for local, r in enumerate(idx)))
        # full_pi is ordered by ranks 0..k-1
        achieved = float(sum(float(cost[r, full_pi[r]]) for r in idx))
        if achieved + atol < opt:
            used_ok = False
            break
    if used_ok and abs(full_val - sum(block_optimal_value(cost, b) for b in partition.blocks)) <= atol:
        return True

    # Exact for singleton: Hall
    if all(len(b) == 1 for b in partition.blocks):
        sets = rankwise_argmax_sets(cost, atol=atol)
        ok, _ = hall_sdr_exists(sets, cost.shape[1])
        return ok

    # Constructive sequential: for each block, maximize under forbidden items
    forbidden: set[int] = set()
    for block in partition.blocks:
        idx = np.asarray(block, dtype=np.int64)
        block_cost = cost[idx, :].copy()
        if forbidden:
            block_cost[:, list(forbidden)] = -1e300
        val, sel = _assignment_maximize(block_cost)
        opt = block_optimal_value(cost, block)
        if val + atol < opt:
            return False
        for item in sel.tolist():
            if item in forbidden:
                return False
            forbidden.add(int(item))
    return True


@dataclass
class CouplingDiagnostics:
    C_full: float
    C_sep: float
    C_rankwise: float
    C_pair: float
    C_block5: float
    C_strong: float
    G_rankwise: float
    G_pair: float
    G_block5: float
    G_strong: float
    G_tilde_strong: float
    hall_feasible: bool
    matching_size: int
    n_ranks: int
    duplicate_unconstrained: int
    max_reuse_multiplicity: int
    conflict_positions: int
    S: float
    rankwise_equals_full: bool
    strong_equals_full: bool

    def threshold_category(self, tau: float, ct_full: float, ct_strong: float) -> str:
        """A/B1/B2/C categories from T7 (structural vs saturation)."""
        if self.strong_equals_full or abs(self.G_strong) <= 1e-12:
            return "A_structural_irrelevance"
        if ct_strong <= tau + 1e-15:
            return "B_easy_saturation"  # full < strong <= tau
        if ct_full > tau + 1e-15:
            return "B_hard_abstention"  # tau < full <= strong
        if ct_full <= tau + 1e-15 < ct_strong:
            return "C_threshold_crossing"
        return "B_same_side_other"


def diagnose_coupling(
    y: np.ndarray,
    *,
    k: int,
    weights: np.ndarray,
    half_width: float,
    S: float,
    atol: float = 1e-9,
) -> CouplingDiagnostics:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    cost, const, _, _ = box_cost_matrix(y, k=k, weights=weights, half_width=half_width)
    c_full = float(
        certificate_full_coupled_box(y, k=k, weights=weights, half_width=half_width).value
    )
    c_rank = float(
        certificate_block_coupled_box(
            y, k=k, weights=weights, half_width=half_width, partition=singleton_partition(k)
        ).value
    )
    c_pair = float(
        certificate_block_coupled_box(
            y, k=k, weights=weights, half_width=half_width, partition=pair_partition(k)
        ).value
    )
    c_b5 = float(
        certificate_block_coupled_box(
            y, k=k, weights=weights, half_width=half_width, partition=consecutive_blocks(k, 5)
        ).value
    )
    c_sep = matched_separable_box_value(y, k=k, weights=weights, half_width=half_width)
    c_strong, _ = strongest_nonexact_same_event(
        y, k=k, weights=weights, half_width=half_width
    )

    sets = rankwise_argmax_sets(cost, atol=atol)
    hall_ok, match_sz = hall_sdr_exists(sets, y.size)

    # unconstrained multiplicity: pick one arbitrary argmax per rank
    picks = [s[0] for s in sets if s]
    from collections import Counter

    cnt = Counter(picks)
    dup = sum(1 for v in cnt.values() if v > 1)
    max_mult = max(cnt.values()) if cnt else 0
    # conflict positions: ranks whose entire argmax set intersects another rank's pick set poorly
    # Use: ranks that cannot all be matched = k - matching_size positions involved in deficit
    conflict_positions = int(k - match_sz)

    S = float(max(S, 1e-6))
    return CouplingDiagnostics(
        C_full=c_full,
        C_sep=float(c_sep),
        C_rankwise=c_rank,
        C_pair=c_pair,
        C_block5=c_b5,
        C_strong=float(c_strong),
        G_rankwise=float(max(c_rank - c_full, 0.0)),
        G_pair=float(max(c_pair - c_full, 0.0)),
        G_block5=float(max(c_b5 - c_full, 0.0)),
        G_strong=float(max(c_strong - c_full, 0.0)),
        G_tilde_strong=float(max(c_strong - c_full, 0.0) / S),
        hall_feasible=bool(hall_ok),
        matching_size=int(match_sz),
        n_ranks=int(k),
        duplicate_unconstrained=int(dup),
        max_reuse_multiplicity=int(max_mult),
        conflict_positions=conflict_positions,
        S=S,
        rankwise_equals_full=bool(abs(c_rank - c_full) <= atol),
        strong_equals_full=bool(abs(c_strong - c_full) <= atol),
    )


def diagnostics_to_dict(d: CouplingDiagnostics) -> dict:
    return asdict(d)
