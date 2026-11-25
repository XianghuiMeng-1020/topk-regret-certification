"""Post-processing contract: mechanism / consistent / action views.

Terminology:
- Y_mech: signed flat mechanism output (M1/M2)
- Y_tree: complete noisy tree-node mechanism output (M3)
- Y_cons: consistent leaf view (M1/M2: Y_cons = Y_mech; M3: g(Y_tree))
- Y_act: max(Y_cons, 0) action input

Deprecated: ambiguous Y_raw / Y_plus aliases kept only as thin wrappers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReleaseViews:
    """Frozen distinction between mechanism, consistent, and action views."""

    y_mech: np.ndarray
    y_cons: np.ndarray
    y_act: np.ndarray
    y_tree: np.ndarray | None = None

    # Deprecated aliases for transitional tests
    @property
    def y_raw(self) -> np.ndarray:
        return self.y_cons

    @property
    def y_plus(self) -> np.ndarray:
        return self.y_act


def clip_action_view(y_cons: np.ndarray) -> np.ndarray:
    """Deterministic public post-processing Y_act = max(Y_cons, 0)."""
    arr = np.asarray(y_cons, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError("y_cons must be finite")
    return np.maximum(arr, 0.0)


def make_flat_release_views(y_mech: np.ndarray) -> ReleaseViews:
    """M1/M2: Y_cons = Y_mech; Y_act = max(Y_cons, 0)."""
    mech = np.asarray(y_mech, dtype=np.float64).copy()
    cons = mech
    return ReleaseViews(y_mech=mech, y_cons=cons, y_act=clip_action_view(cons), y_tree=None)


def make_hierarchical_release_views(
    y_tree: np.ndarray,
    y_cons_leaves: np.ndarray,
) -> ReleaseViews:
    """M3: certificate may read Y_tree and/or Y_cons; action uses Y_act."""
    tree = np.asarray(y_tree, dtype=np.float64).copy()
    cons = np.asarray(y_cons_leaves, dtype=np.float64).copy()
    return ReleaseViews(
        y_mech=cons,  # leaf mechanism-equivalent for flat-compatible APIs
        y_cons=cons,
        y_act=clip_action_view(cons),
        y_tree=tree,
    )


def make_release_views(y_mech: np.ndarray) -> ReleaseViews:
    """Backward-compatible alias: treat input as flat Y_mech."""
    return make_flat_release_views(y_mech)


def postprocessing_privacy_cost() -> float:
    """Deterministic post-processing consumes no additional privacy budget."""
    return 0.0


def certificate_input_registry() -> dict[str, str]:
    """What future certificates may read (no current truth)."""
    return {
        "M1_flat_laplace": "Y_cons (=Y_mech) and/or Y_act for action-conditioned objects",
        "M2_two_sided_geometric": "Y_cons (=Y_mech) and/or Y_act",
        "M3_hierarchical_laplace": "Y_tree and/or Y_cons; both are reported separately",
    }
