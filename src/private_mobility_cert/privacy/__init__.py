"""Differential privacy mechanisms and accounting.

Primary model: record-level add/remove pure ε-DP (δ=0) over eligible completed trips.
Mechanisms: M1 flat Laplace, M2 two-sided geometric, M3 hierarchical Laplace.
"""

from private_mobility_cert.privacy.adjacency import (
    assert_add_remove_not_replace_one,
    l1_sensitivity_flat_histogram,
    neighboring_histograms,
    one_hot_contribution,
)
from private_mobility_cert.privacy.geometric import (
    GeometricRelease,
    release_geometric_histogram,
    two_sided_geometric_pmf,
    two_sided_geometric_variance,
)
from private_mobility_cert.privacy.hierarchical import (
    HierarchicalRelease,
    hierarchy_levels,
    release_hierarchical_station,
    release_hierarchical_tensor,
    uniform_level_epsilons,
)
from private_mobility_cert.privacy.laplace import (
    LaplaceRelease,
    laplace_scale,
    release_laplace_histogram,
)
from private_mobility_cert.privacy.postprocess import (
    ReleaseViews,
    certificate_input_registry,
    clip_action_view,
    make_flat_release_views,
    make_hierarchical_release_views,
    make_release_views,
    postprocessing_privacy_cost,
)

__all__ = [
    "GeometricRelease",
    "HierarchicalRelease",
    "LaplaceRelease",
    "ReleaseViews",
    "assert_add_remove_not_replace_one",
    "certificate_input_registry",
    "clip_action_view",
    "hierarchy_levels",
    "l1_sensitivity_flat_histogram",
    "laplace_scale",
    "make_flat_release_views",
    "make_hierarchical_release_views",
    "make_release_views",
    "neighboring_histograms",
    "one_hot_contribution",
    "postprocessing_privacy_cost",
    "release_geometric_histogram",
    "release_hierarchical_station",
    "release_hierarchical_tensor",
    "release_laplace_histogram",
    "two_sided_geometric_pmf",
    "two_sided_geometric_variance",
    "uniform_level_epsilons",
]
