"""Independent enumeration checks for the certificate identities.

The reference loops over injections. It does not call the library objective
helpers to generate the numbers it compares against the optimized code.
"""
from __future__ import annotations

from itertools import permutations
from pathlib import Path

import numpy as np
import pytest

from private_mobility_cert.certificates.block_coupled_certificates import (
    block5_partition,
    certificate_block5_box,
    certificate_full_coupled_box,
    certificate_pair_block_box,
    certificate_rankwise_box,
    matched_separable_box_value,
    pair_partition,
    singleton_partition,
    strongest_nonexact_same_event,
)
from private_mobility_cert.certificates.coupling_diagnostics import (
    compatibility_penalty,
    hall_sdr_exists,
    rankwise_argmax_sets,
)
from private_mobility_cert.certificates.release_conditioned import (
    _assignment_maximize,
    certificate_l1_assignment,
    deploy_ranking,
    max_l1_support,
    simplified_l1_support_is_exact,
    weight_map,
)
from private_mobility_cert.certificates.topk_regret import TopKCertError
from private_mobility_cert.certificates.weighted_ranking import validate_weights, weight_family

ROOT = Path(__file__).resolve().parents[1]
ATOL = 1e-8
SEED = 20261124


def _deploy(y: np.ndarray, k: int) -> tuple[int, ...]:
    y_plus = np.maximum(np.asarray(y, dtype=np.float64), 0.0)
    order = sorted(range(y_plus.size), key=lambda i: (-float(y_plus[i]), i))
    return tuple(order[:k])


def _g(y: np.ndarray, pi: tuple[int, ...], pi_hat: tuple[int, ...], w: np.ndarray) -> float:
    return float(sum(float(w[r]) * (float(y[pi[r]]) - float(y[pi_hat[r]])) for r in range(len(w))))


def _l1(pi: tuple[int, ...], pi_hat: tuple[int, ...], w: np.ndarray, n: int) -> float:
    a = [0.0] * n
    for r, item in enumerate(pi_hat):
        a[item] += float(w[r])
    for r, item in enumerate(pi):
        a[item] -= float(w[r])
    return float(sum(abs(v) for v in a))


def _overlap(pi: tuple[int, ...], pi_hat: tuple[int, ...], w: np.ndarray, n: int) -> float:
    hat = [0.0] * n
    for r, item in enumerate(pi_hat):
        hat[item] += float(w[r])
    return float(sum(min(hat[pi[r]], float(w[r])) for r in range(len(w))))


def enumerate_case(y: np.ndarray, k: int, w: np.ndarray, t: float) -> dict[str, float]:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    w = np.asarray(w, dtype=np.float64).reshape(-1)[:k]
    pi_hat = _deploy(y, k)
    max_g = 0.0
    max_l1 = 0.0
    max_box = 0.0
    two = 2.0 * float(np.sum(w))
    for pi in permutations(range(y.size), k):
        g = _g(y, pi, pi_hat, w)
        ell = _l1(pi, pi_hat, w, y.size)
        identity = two - 2.0 * _overlap(pi, pi_hat, w, y.size)
        if abs(ell - identity) > 1e-8 * (1.0 + abs(ell) + abs(identity)):
            raise AssertionError(f"L1 identity failed: {ell} vs {identity}")
        max_g = max(max_g, g)
        max_l1 = max(max_l1, ell)
        max_box = max(max_box, g + float(t) * ell)
    return {
        "max_g": max_g,
        "max_l1": max_l1,
        "sep_general": max_g + float(t) * max_l1,
        "sep_simplified": max_g + float(t) * two,
        "box": max_box,
        "two": two,
    }


def _cases() -> list[dict]:
    rows: list[dict] = []

    def add(y, k, w, t, tag):
        rows.append(
            {
                "y": np.asarray(y, dtype=np.float64),
                "k": int(k),
                "w": np.asarray(w, dtype=np.float64),
                "t": float(t),
                "tag": tag,
            }
        )

    add(np.arange(5.0), 5, weight_family("uniform", 5), 1.0, "n=k=5 uniform")
    add(np.arange(7.0), 5, weight_family("uniform", 5), 1.0, "n=7 k=5 uniform")
    add(np.arange(10.0), 5, weight_family("uniform", 5), 1.0, "n=10 k=5 uniform")
    add(np.arange(6.0), 3, weight_family("uniform", 3), 1.0, "n=2k uniform")
    add(np.arange(8.0), 3, weight_family("dcg", 3), 0.5, "n>2k dcg")
    add(np.array([1.0]), 1, weight_family("uniform", 1), 2.0, "k=1 n=k")
    add(np.array([4.0, -2.0, 0.0, 1.0, 3.0]), 1, weight_family("reciprocal", 1), 1e-12, "k=1 tiny t")
    add(np.arange(8.0), 6, weight_family("uniform", 6), 1.0, "k=6 n=8 uniform")
    add(np.linspace(-2.0, 3.0, 9), 6, weight_family("dcg", 6), 0.7, "k=6 n=9 dcg")
    add(np.array([1.0, 1.0, 1.0, 0.0, 0.0]), 4, np.array([1.0, 0.5, 0.0, 0.0]), 1.0, "zeros in weights")
    add(np.array([2.0, 2.0, 2.0, 2.0, 1.0]), 3, np.ones(3), 1.5, "repeated counts")
    add(np.array([-5.0, -1.0, 3.0, 0.0]), 2, np.ones(2), 1.0, "negative raw and clipping")
    add(np.array([-4.0, -3.0, -2.0, -1.0]), 3, weight_family("reciprocal", 3), 0.0, "t=0 all negative")
    add(np.array([0.2, -1.5, 4.0, 4.0, -0.3, 1.0]), 4, weight_family("dcg", 4), 3.0, "larger t")
    add(np.arange(4.0), 4, weight_family("uniform", 4), 1e-12, "n=k tiny t")
    add(np.arange(5.0), 3, weight_family("reciprocal", 3), 0.0, "k<n<2k t=0")

    rng = np.random.default_rng(SEED)
    for i in range(24):
        n = int(rng.integers(2, 8))
        k = int(rng.integers(1, n + 1))
        family = ("uniform", "dcg", "reciprocal")[int(rng.integers(0, 3))]
        w = weight_family(family, k)
        if int(rng.integers(0, 4)) == 0 and k >= 2:
            w = w.copy()
            w[-1] = 0.0
        y = rng.normal(size=n)
        if int(rng.integers(0, 3)) == 0:
            y = y.copy()
            y[int(rng.integers(0, n))] *= -1.0
            y[int(rng.integers(0, n))] = -abs(float(y[int(rng.integers(0, n))]))
        t = float(rng.choice(np.array([0.0, 1e-12, 0.05, 1.0, 2.5])))
        add(y, k, w, t, f"seed-{SEED}-{i}-{family}")
    return rows


CASES = _cases()


def test_required_support_counterexamples():
    y = np.arange(5.0)
    w = weight_family("uniform", 5)
    ref = enumerate_case(y, 5, w, 1.0)
    assert ref["max_l1"] == pytest.approx(0.0, abs=1e-9)
    assert ref["two"] == pytest.approx(10.0, abs=1e-12)
    assert ref["sep_general"] < ref["sep_simplified"] - 1.0

    y7 = np.arange(7.0)
    ref7 = enumerate_case(y7, 5, w, 1.0)
    assert ref7["max_l1"] < ref7["two"] - 1.0
    assert ref7["max_l1"] == pytest.approx(4.0, abs=1e-9)

    y10 = np.arange(10.0)
    ref10 = enumerate_case(y10, 5, weight_family("dcg", 5), 1.0)
    assert ref10["max_l1"] == pytest.approx(ref10["two"], abs=1e-9)


@pytest.mark.parametrize("case", CASES, ids=[c["tag"] for c in CASES])
def test_optimized_matches_enumeration(case):
    y, k, w, t = case["y"], case["k"], case["w"], case["t"]
    ref = enumerate_case(y, k, w, t)
    pi_hat = deploy_ranking(y, k)
    assert tuple(int(i) for i in pi_hat) == _deploy(y, k)
    hat = weight_map(pi_hat, validate_weights(w)[:k], y.size)
    got_l1 = max_l1_support(hat, validate_weights(w)[:k])
    got_sep = matched_separable_box_value(y, k=k, weights=w, half_width=t)
    got_box = certificate_l1_assignment(y, k=k, weights=w, half_width=t).value
    got_full = certificate_full_coupled_box(y, k=k, weights=w, half_width=t).value
    assert got_l1 == pytest.approx(ref["max_l1"], abs=ATOL)
    assert got_sep == pytest.approx(ref["sep_general"], abs=ATOL)
    assert got_box == pytest.approx(ref["box"], abs=ATOL)
    assert got_full == pytest.approx(ref["box"], abs=ATOL)
    assert got_box >= -ATOL
    if simplified_l1_support_is_exact(hat, validate_weights(w)[:k]):
        assert ref["max_l1"] == pytest.approx(ref["two"], abs=ATOL)
        assert got_sep == pytest.approx(ref["sep_simplified"], abs=ATOL)
    else:
        assert ref["max_l1"] <= ref["two"] + ATOL
    assert ref["box"] <= ref["sep_general"] + ATOL
    assert ref["sep_general"] <= ref["sep_simplified"] + ATOL


def test_headline_n_at_least_2k_matches_simplified_product():
    rng = np.random.default_rng(SEED + 7)
    for n, k in ((40, 5), (50, 10), (500, 25), (16, 8)):
        for family in ("uniform", "dcg", "reciprocal"):
            w = weight_family(family, k)
            y = rng.normal(size=n)
            t = 1.25
            got = matched_separable_box_value(y, k=k, weights=w, half_width=t)
            # Closed-form simplified value with the same max-G solve.
            ww = validate_weights(w)[:k]
            pi_hat = deploy_ranking(y, k)
            cost = np.vstack([float(wr) * y for wr in ww])
            best_var, _ = _assignment_maximize(cost)
            max_g = float(best_var - float(np.dot(ww, y[pi_hat])))
            simplified = float(max(max_g + 2.0 * t * float(ww.sum()), 0.0))
            assert got == simplified


def test_solver_refuses_incomplete_assignment(monkeypatch):
    def incomplete(cost):
        return np.array([0]), np.array([0])

    monkeypatch.setattr("scipy.optimize.linear_sum_assignment", incomplete)
    y = np.array([1.0, 2.0, 0.5, 0.0])
    with pytest.raises(TopKCertError, match="refusing a greedy approximation"):
        certificate_l1_assignment(y, k=2, weights=np.ones(2), half_width=0.5)


def test_assignment_beats_row_greedy():
    # Rank 0's greedy pick blocks rank 1's much larger entry.
    cost = np.array([[3.0, 2.0], [100.0, 0.0]])
    total, selected = _assignment_maximize(cost)
    assert list(selected) == [1, 0]
    assert total == pytest.approx(102.0, abs=1e-12)


def test_structural_dominance_and_block5_equality():
    rng = np.random.default_rng(SEED + 3)
    max_violation = 0.0
    max_block_gap = 0.0
    checked = 0
    specs = [(5, 5), (7, 5), (6, 3), (8, 3), (8, 6), (4, 1), (5, 4)]
    families = ("uniform", "dcg", "reciprocal")
    for n, k in specs:
        for family in families:
            y = rng.normal(size=n)
            y[0] = -abs(float(y[0]))
            w = weight_family(family, k)
            t = float(rng.uniform(0.0, 2.0))
            box = certificate_full_coupled_box(y, k=k, weights=w, half_width=t).value
            rank = certificate_rankwise_box(y, k=k, weights=w, half_width=t).value
            pair = certificate_pair_block_box(y, k=k, weights=w, half_width=t).value
            block = certificate_block5_box(y, k=k, weights=w, half_width=t).value
            strong, parts = strongest_nonexact_same_event(y, k=k, weights=w, half_width=t)
            values = (rank, pair, block, parts["separable"], strong)
            max_violation = max(max_violation, max(box - v for v in values))
            for v in values:
                assert box <= v + 1e-7
            assert strong == pytest.approx(min(parts.values()), abs=1e-12)
            if k <= 5:
                max_block_gap = max(max_block_gap, abs(block - box), abs(strong - box))
                assert block == pytest.approx(box, abs=1e-7)
                assert strong == pytest.approx(box, abs=1e-7)
            checked += 1
    assert checked == len(specs) * len(families)
    assert max_violation <= 1e-7
    assert max_block_gap <= 1e-7


def test_compatibility_penalty_matches_enumeration():
    rng = np.random.default_rng(SEED + 11)
    max_gap = 0.0
    hall_checks = 0
    for _ in range(12):
        n = int(rng.integers(3, 7))
        k = int(rng.integers(1, min(5, n) + 1))
        cost = rng.normal(size=(k, n))
        for partition in (singleton_partition(k), pair_partition(k), block5_partition(k)):
            enum_blocks = 0.0
            for block in partition.blocks:
                best = -np.inf
                for choice in permutations(range(n), len(block)):
                    best = max(
                        best,
                        sum(float(cost[block[j], choice[j]]) for j in range(len(block))),
                    )
                enum_blocks += float(best)
            enum_full = max(
                sum(float(cost[r, pi[r]]) for r in range(k))
                for pi in permutations(range(n), k)
            )
            got = compatibility_penalty(cost, partition)
            gap = abs(got - (enum_blocks - enum_full))
            max_gap = max(max_gap, gap)
            assert gap <= 1e-7
        row_max = cost.max(axis=1)
        enum_hall = any(
            all(float(cost[r, pi[r]]) >= float(row_max[r]) - 1e-10 for r in range(k))
            for pi in permutations(range(n), k)
        )
        ok, _ = hall_sdr_exists(rankwise_argmax_sets(cost), n)
        assert ok == enum_hall
        penalty = compatibility_penalty(cost, singleton_partition(k))
        assert (penalty <= 1e-8) == enum_hall
        hall_checks += 1
    assert hall_checks == 12
    assert max_gap <= 1e-7


def test_case_inventory_covers_required_regimes():
    tags = [c["tag"] for c in CASES]
    assert any("n=k=5" in t for t in tags)
    assert any("n=7 k=5" in t for t in tags)
    assert any("n=2k" in t for t in tags)
    assert any("n>2k" in t for t in tags)
    assert any("k=1" in t for t in tags)
    assert any("k=6" in t for t in tags)
    assert any("zeros" in t for t in tags)
    assert any("repeated" in t for t in tags)
    assert any("clipping" in t for t in tags)
    assert any("t=0" in t for t in tags)
    assert any("tiny t" in t for t in tags)
    assert any("larger t" in t for t in tags)
    assert sum(1 for t in tags if t.startswith("seed-")) == 24
    assert len(CASES) >= 40
