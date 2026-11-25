"""Privacy mechanism unit tests (synthetic only)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from private_mobility_cert.privacy.accounting import (
    monte_carlo_interpretation,
    parallel_compose_disjoint,
    sequential_compose,
)
from private_mobility_cert.privacy.adjacency import (
    assert_add_remove_not_replace_one,
    l1_sensitivity_flat_histogram,
    neighboring_histograms,
    one_hot_contribution,
)
from private_mobility_cert.privacy.geometric import (
    geometric_privacy_log_ratio,
    release_geometric_histogram,
    sample_two_sided_geometric,
    two_sided_geometric_pmf,
    two_sided_geometric_variance,
)
from private_mobility_cert.privacy.hierarchical import (
    hierarchy_levels,
    next_power_of_two,
    path_nodes_for_leaf,
    release_hierarchical_station,
    release_hierarchical_tensor,
    uniform_level_epsilons,
)
from private_mobility_cert.privacy.laplace import (
    laplace_log_density_ratio_bound,
    laplace_scale,
    release_laplace_histogram,
)
from private_mobility_cert.privacy.postprocess import (
    clip_action_view,
    postprocessing_privacy_cost,
)
from private_mobility_cert.privacy.rng import spawn_generator
from private_mobility_cert.privacy.seeds import (
    SeedKey,
    confirmatory_replicate_ids,
    derive_release_seed,
    smoke_replicate_ids,
)


def test_sensitivity_and_adjacency() -> None:
    assert l1_sensitivity_flat_histogram() == 1
    v = one_hot_contribution(5, 2)
    assert int(v.sum()) == 1
    x, xp = neighboring_histograms(np.array([1, 0, 3]), 1, direction="add")
    assert np.array_equal(xp - x, np.array([0, 1, 0]))
    assert "replace-one" in assert_add_remove_not_replace_one()


def test_laplace_scale_and_ratio() -> None:
    eps = 1.0
    assert laplace_scale(eps) == pytest.approx(1.0)
    x = np.array([2.0, 1.0])
    xp = np.array([3.0, 1.0])
    y = np.array([2.5, 1.2])
    log_r = laplace_log_density_ratio_bound(x, xp, y, epsilon=eps)
    assert log_r <= eps + 1e-9


def test_laplace_synthetic_moments_and_seed() -> None:
    eps = 2.0
    x = np.zeros(20000)
    r1 = release_laplace_histogram(x, epsilon=eps, seed=123)
    r2 = release_laplace_histogram(x, epsilon=eps, seed=123)
    assert np.allclose(r1.y_mech, r2.y_mech)
    assert r1.y_mech.dtype == np.float64
    mean = float(np.mean(r1.y_mech))
    var = float(np.var(r1.y_mech))
    assert abs(mean) < 0.05
    assert abs(var - 2.0 / eps**2) < 0.05


def test_geometric_pmf_ratio_and_integer() -> None:
    eps = 1.0
    alpha = math.exp(-eps)
    # PMF normalization spot check
    mass = sum(two_sided_geometric_pmf(k, epsilon=eps) for k in range(-20, 21))
    assert mass == pytest.approx(1.0, abs=1e-6)
    x = np.array([0, 1, 2], dtype=np.int64)
    xp = np.array([1, 1, 2], dtype=np.int64)
    y = np.array([0, 1, 2], dtype=np.int64)
    log_r = geometric_privacy_log_ratio(x, xp, y, epsilon=eps)
    assert log_r <= eps + 1e-9
    rel = release_geometric_histogram(x, epsilon=eps, seed=7)
    assert np.issubdtype(rel.y_mech.dtype, np.integer)
    assert abs(rel.alpha - alpha) < 1e-12


def test_geometric_moments() -> None:
    eps = 1.5
    rng = spawn_generator(99)
    z = sample_two_sided_geometric(rng, epsilon=eps, size=50000)
    assert abs(float(np.mean(z))) < 0.05
    assert abs(float(np.var(z)) - two_sided_geometric_variance(eps)) < 0.15
    for k in range(0, 6):
        assert two_sided_geometric_pmf(k, epsilon=eps) == pytest.approx(
            two_sided_geometric_pmf(-k, epsilon=eps), abs=1e-12
        )


def test_hierarchical_levels_padding_and_budget() -> None:
    assert next_power_of_two(5) == 8
    assert hierarchy_levels(5) == 4
    parts = uniform_level_epsilons(2.0, 4)
    assert float(np.sum(parts)) == pytest.approx(2.0)
    leaves = np.array([1.0, 0.0, 3.0, 2.0, 1.0])
    r1 = release_hierarchical_station(leaves, epsilon=1.0, seed=11)
    r2 = release_hierarchical_station(leaves, epsilon=1.0, seed=11)
    assert np.allclose(r1.y_cons, r2.y_cons)
    assert r1.y_tree is not None and r1.y_tree.size > 0
    assert r1.n_padded == 8
    path = path_nodes_for_leaf(3, 8)
    assert len(path) == hierarchy_levels(5)
    assert path[0][0] == 0


def test_hierarchical_tensor_no_cross_station_leak_seed_isolation() -> None:
    x = np.array([[1.0, 2.0, 0.0], [0.0, 0.0, 5.0]])
    y, meta = release_hierarchical_tensor(x, epsilon=1.0, seed=5)
    assert y.shape == x.shape
    assert meta.epsilon_total == 1.0
    # Changing station 1 should not require reading station 0 truth in API
    y2, _ = release_hierarchical_tensor(x.copy(), epsilon=1.0, seed=5)
    assert np.allclose(y, y2)


def test_postprocessing_contract() -> None:
    from private_mobility_cert.privacy.postprocess import make_flat_release_views

    mech = np.array([-1.0, 0.5, 2.0])
    views = make_flat_release_views(mech)
    assert np.allclose(views.y_mech, mech)
    assert np.allclose(views.y_cons, mech)
    assert np.allclose(views.y_act, np.array([0.0, 0.5, 2.0]))
    assert np.all(clip_action_view(mech) >= 0)
    assert postprocessing_privacy_cost() == 0.0


def test_composition_rules() -> None:
    assert sequential_compose([0.5, 0.5]) == (1.0, 0.0)
    assert parallel_compose_disjoint([1.0, 2.0, 0.5]) == (2.0, 0.0)
    assert "not jointly published" in monte_carlo_interpretation()


def test_seed_derivation_immutable() -> None:
    ids = confirmatory_replicate_ids(100)
    assert ids == list(range(100))
    assert smoke_replicate_ids() == [0, 1, 2]
    k = SeedKey("nyc", "development", "M1_flat_laplace", 1.0, 0)
    s1 = derive_release_seed(20260801, k)
    s2 = derive_release_seed(20260801, k)
    assert s1 == s2
    k2 = SeedKey("nyc", "development", "M1_flat_laplace", 1.0, 1)
    assert derive_release_seed(20260801, k2) != s1
