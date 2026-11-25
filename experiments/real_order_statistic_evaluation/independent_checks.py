"""Independent small-instance checks for the OS matched envelope.

The exact order-statistic objective is enumerated by the reference module.
Box and block objectives are enumerated here from the cost matrix. Neither
enumeration calls certificate_os_matched to obtain its reference value.
"""

from __future__ import annotations

import math
import sys
from itertools import permutations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from private_mobility_cert.certificates.block_coupled_certificates import (  # noqa: E402
    block5_partition,
    box_cost_matrix,
    certificate_block5_box,
    certificate_pair_block_box,
    certificate_rankwise_box,
    pair_partition,
    singleton_partition,
)
from private_mobility_cert.certificates.coupled_certificate_ref import (  # noqa: E402
    bos_lp_verify,
    certificate_coupled_ref_exact,
    coefficient_map_ref,
)
from private_mobility_cert.certificates.os_envelope import (  # noqa: E402
    certificate_os_envelope,
    envelope_schedule,
    os_support_effective,
    prefix_min_thresholds,
)
from private_mobility_cert.certificates.os_matched_envelope import (  # noqa: E402
    certificate_os_matched,
)
from private_mobility_cert.certificates.release_conditioned import (  # noqa: E402
    ThresholdTable,
    deploy_ranking,
)
from private_mobility_cert.certificates.weighted_ranking import (  # noqa: E402
    build_rank_thresholds_weighted,
    weight_family,
)


def _table(u_plus, u_minus) -> ThresholdTable:
    upper = np.asarray(u_plus, dtype=np.float64)
    lower = np.asarray(u_minus, dtype=np.float64)
    return ThresholdTable(
        mechanism="fixture",
        epsilon=1.0,
        n=0,
        k_max=int(upper.size),
        alpha=0.05,
        scheme="fixture",
        u_plus=upper,
        u_minus=lower,
    )


def _brute_partition(y, k, weights, half_width, blocks):
    cost, const, _pi, _ww = box_cost_matrix(
        y, k=k, weights=weights, half_width=half_width
    )
    total = 0.0
    n_assign = 0
    n = int(np.asarray(y).size)
    for block in blocks:
        rows = [int(r) for r in block]
        best = -np.inf
        for injection in permutations(range(n), len(rows)):
            n_assign += 1
            value = 0.0
            for local, item in enumerate(injection):
                value += float(cost[rows[local], item])
            if value > best:
                best = value
        total += best
    return float(max(const + total, 0.0)), n_assign


def _hand_delta(weights, u_plus, u_minus, s: int):
    upper = prefix_min_thresholds(u_plus)[: len(weights)]
    lower = prefix_min_thresholds(u_minus)[: len(weights)]
    w1 = float(weights[0])
    gap = 0.0
    for j in range(s - 1):
        gap += float(upper[j] - upper[s - 1]) + float(lower[j] - lower[s - 1])
    return float(w1 * gap), float(0.5 * (upper[s - 1] + lower[s - 1]))


def run_suite() -> dict:
    failures = []
    n_fixtures = 0
    n_injections = 0
    n_block_assignments = 0
    max_exact_above_coupled = 0.0
    max_coupled_above_strong = 0.0
    max_strong_above_rank1 = 0.0
    max_envelope_vs_reference = 0.0
    max_block_vs_brute = 0.0
    max_lp_gap = 0.0
    max_support_above_arm = 0.0
    n_k5 = 0
    n_k5_mismatch = 0
    n_lp = 0

    shapes = [(3, 3), (5, 3), (6, 3), (5, 5), (6, 5), (7, 6), (8, 4), (8, 6)]
    families = ("uniform", "dcg", "reciprocal")
    releases = ("separated", "negative", "ties", "near_zero")
    kinds = ("laplace_shape", "asymmetric", "flat")

    def make_y(name: str, n: int) -> np.ndarray:
        if name == "separated":
            return np.arange(n, 0, -1, dtype=np.float64)
        if name == "negative":
            return np.linspace(-2.0, 1.5, n)
        if name == "ties":
            return np.array([1.0 if i < max(n // 2, 1) else 0.0 for i in range(n)])
        return np.full(n, 1e-8)

    def make_thresholds(kind: str, k: int):
        if kind == "laplace_shape":
            upper = np.array([3.0 / (r + 1) + 0.4 for r in range(k)])
            return upper, upper.copy()
        if kind == "asymmetric":
            base_u = np.array([2.5, 2.0, 1.2, 1.2, 0.8, 0.5, 0.5, 0.4])
            base_l = np.array([3.0, 1.5, 1.5, 0.9, 0.9, 0.7, 0.2, 0.2])
            return base_u[:k].astype(np.float64), base_l[:k].astype(np.float64)
        return np.full(k, 1.25), np.full(k, 0.75)

    planned = []
    for n, k in shapes:
        for wf in families:
            if n == 8 and k == 6 and wf != "uniform":
                continue
            for release_name in releases:
                if n >= 8 and release_name not in {"separated", "negative"}:
                    continue
                for kind in kinds:
                    if n >= 8 and kind != "asymmetric":
                        continue
                    planned.append((n, k, wf, release_name, kind))

    for n, k, wf, release_name, kind in planned:
        y = make_y(release_name, n)
        weights = weight_family(wf, k)
        upper, lower = make_thresholds(kind, k)
        table = _table(upper, lower)
        exact = certificate_coupled_ref_exact(
            y,
            k,
            weights,
            prefix_min_thresholds(upper),
            prefix_min_thresholds(lower),
        )
        matched = certificate_os_matched(y, k=k, weights=weights, table=table)
        reference = certificate_os_envelope(y, k=k, weights=weights, table=table)
        n_fixtures += 1
        n_injections += math.perm(n, k)
        gap_exact = float(exact.value - matched.coupled_envelope)
        gap_cs = float(matched.coupled_envelope - matched.strong_envelope)
        gap_sr = float(matched.strong_envelope - matched.rank1_strong)
        gap_reference = float(matched.coupled_envelope - reference.value)
        max_exact_above_coupled = max(max_exact_above_coupled, gap_exact)
        max_coupled_above_strong = max(max_coupled_above_strong, gap_cs)
        max_strong_above_rank1 = max(max_strong_above_rank1, gap_sr)
        max_envelope_vs_reference = max(max_envelope_vs_reference, abs(gap_reference))
        if gap_exact > 1e-6 * (1.0 + abs(float(exact.value))):
            failures.append(f"exact above coupled n={n} k={k} {wf} {release_name} {kind}")
        if gap_cs > 1e-6 * (1.0 + abs(matched.strong_envelope)):
            failures.append(f"coupled above strong n={n} k={k} {wf} {release_name}")
        if gap_sr > 1e-6 * (1.0 + abs(matched.rank1_strong)):
            failures.append(f"strong envelope above rank1 n={n} k={k}")
        if abs(gap_reference) > 1e-7 * (1.0 + abs(reference.value)):
            failures.append(f"reference envelope mismatch n={n} k={k} {wf} {release_name} {kind}")
        if min(
            matched.rank1_coupled,
            matched.rank1_strong,
            matched.coupled_envelope,
            matched.strong_envelope,
        ) < -1e-10:
            failures.append(f"negative certificate n={n} k={k}")
        if k <= 5:
            n_k5 += 1
            if abs(matched.strong_envelope - matched.coupled_envelope) > 1e-8:
                n_k5_mismatch += 1
                failures.append(f"k<=5 envelope mismatch n={n} k={k} {wf}")
        for s in range(1, k + 1):
            hand_delta, hand_t = _hand_delta(weights, upper, lower, s)
            _ww, terms = envelope_schedule(weights, upper, lower, k=k)
            term = terms[s - 1]
            if abs(hand_delta - term.delta) > 1e-9 or abs(hand_t - term.half_width) > 1e-9:
                failures.append(f"schedule mismatch s={s} n={n} k={k}")
                break

    for mechanism in ("M1_flat_laplace", "M2_two_sided_geometric"):
        for wf in families:
            weights = weight_family(wf, 4)
            built_plus, built_minus = build_rank_thresholds_weighted(
                mechanism=mechanism,
                n=7,
                k_max=4,
                epsilon=1.0,
                alpha=0.05,
                weights=weights,
            )
            y = np.array([4.0, -1.0, 0.0, 2.0, 2.0, 0.5, -3.0])
            table = _table(built_plus, built_minus)
            exact = certificate_coupled_ref_exact(
                y,
                4,
                weights,
                prefix_min_thresholds(built_plus),
                prefix_min_thresholds(built_minus),
            )
            matched = certificate_os_matched(y, k=4, weights=weights, table=table)
            n_fixtures += 1
            n_injections += math.perm(7, 4)
            if exact.value > matched.coupled_envelope + 1e-6 * (1.0 + abs(exact.value)):
                failures.append(f"solver-table exact above coupled {mechanism} {wf}")
            if abs(matched.strong_envelope - matched.coupled_envelope) > 1e-8:
                failures.append(f"solver-table k<=5 mismatch {mechanism} {wf}")

    y_block = np.array([0.2, 3.0, -1.0, 3.0, 0.4, 1.5])
    weights = weight_family("dcg", 5)
    for half in (0.3, 1.7):
        for name, blocks, production in (
            ("rank", singleton_partition(5).blocks, certificate_rankwise_box),
            ("pair", pair_partition(5).blocks, certificate_pair_block_box),
            ("block5", block5_partition(5).blocks, certificate_block5_box),
        ):
            brute, n_assign = _brute_partition(y_block, 5, weights, half, blocks)
            n_block_assignments += n_assign
            got = float(production(y_block, k=5, weights=weights, half_width=half).value)
            max_block_vs_brute = max(max_block_vs_brute, abs(got - brute))
            if abs(got - brute) > 1e-6 * (1.0 + abs(brute)):
                failures.append(f"block enumeration mismatch {name} t={half}")

    rng = np.random.default_rng(20261124)
    for _ in range(24):
        n = 8
        k = 5
        weights = weight_family("dcg", k)
        upper = np.array([4.0, 3.0, 1.5, 1.5, 0.6])
        lower = np.array([2.0, 2.0, 1.0, 0.4, 0.4])
        pi_hat = np.arange(k)
        rival = rng.permutation(n)[:k]
        a = coefficient_map_ref(rival, pi_hat, weights, n)
        closed = os_support_effective(a, upper, lower)
        lp = bos_lp_verify(
            a, prefix_min_thresholds(upper), prefix_min_thresholds(lower), n
        )
        n_lp += 1
        max_lp_gap = max(max_lp_gap, abs(closed - lp))
        if abs(closed - lp) > 1e-6 * (1.0 + abs(lp)):
            failures.append("LP support mismatch")
        _ww, terms = envelope_schedule(weights, upper, lower, k=k)
        l1 = float(np.abs(a).sum())
        for term in terms:
            slack = term.half_width * l1 + term.delta - closed
            max_support_above_arm = max(max_support_above_arm, -float(slack))
            if slack < -1e-6 * (1.0 + abs(closed)):
                failures.append(f"support exceeds arm s={term.s}")
                break

    y = np.array([5.0, 4.0, 1.0, 0.0, -2.0])
    weights = weight_family("reciprocal", 3)
    pi_hat = deploy_ranking(y, 3)
    explicit = (
        np.array([pi_hat[1], pi_hat[0], pi_hat[2]]),
        np.array([pi_hat[0], pi_hat[1], 4]),
    )
    upper = np.array([2.0, 1.0, 0.5])
    lower = np.array([3.0, 0.5, 0.5])
    for rival in explicit:
        a = coefficient_map_ref(rival, pi_hat, weights, y.size)
        closed = os_support_effective(a, upper, lower)
        lp = bos_lp_verify(
            a, prefix_min_thresholds(upper), prefix_min_thresholds(lower), y.size
        )
        n_lp += 1
        max_lp_gap = max(max_lp_gap, abs(closed - lp))
        if abs(closed - lp) > 1e-6 * (1.0 + abs(lp)):
            failures.append("explicit competitor LP mismatch")

    return {
        "n_fixtures": n_fixtures,
        "n_injections": n_injections,
        "n_block_assignments": n_block_assignments,
        "n_lp_supports": n_lp,
        "n_k5_fixtures": n_k5,
        "n_k5_mismatches": n_k5_mismatch,
        "max_exact_minus_coupled": max_exact_above_coupled,
        "max_coupled_minus_strong": max_coupled_above_strong,
        "max_strong_minus_rank1": max_strong_above_rank1,
        "max_abs_matched_minus_reference": max_envelope_vs_reference,
        "max_abs_block_minus_brute": max_block_vs_brute,
        "max_abs_closed_minus_lp": max_lp_gap,
        "max_support_above_linear_arm": max_support_above_arm,
        "n_failures": len(failures),
        "failures": failures,
    }


if __name__ == "__main__":
    import json

    result = run_suite()
    print(json.dumps({k: result[k] for k in result if k != "failures"}, indent=2))
    if result["n_failures"]:
        print(result["failures"][:20])
        raise SystemExit(1)
