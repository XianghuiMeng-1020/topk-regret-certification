"""M3 — Hierarchical temporal Laplace release with level composition."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from private_mobility_cert.privacy.laplace import laplace_scale
from private_mobility_cert.privacy.rng import spawn_generator


def next_power_of_two(n: int) -> int:
    if n <= 0:
        raise ValueError("n must be positive")
    return 1 << (n - 1).bit_length()


def hierarchy_levels(leaf_count: int) -> int:
    """Number of released levels for a padded binary tree (root..leaf)."""
    padded = next_power_of_two(leaf_count)
    return int(np.log2(padded)) + 1


def uniform_level_epsilons(total_epsilon: float, n_levels: int) -> np.ndarray:
    if total_epsilon <= 0 or not np.isfinite(total_epsilon):
        raise ValueError("total_epsilon must be positive")
    if n_levels <= 0:
        raise ValueError("n_levels must be positive")
    parts = np.full(n_levels, total_epsilon / n_levels, dtype=np.float64)
    # Repair floating drift on last coordinate
    parts[-1] = total_epsilon - float(np.sum(parts[:-1]))
    if abs(float(np.sum(parts)) - total_epsilon) > 1e-12:
        raise ValueError("epsilon allocation failed to sum to total")
    if np.any(parts <= 0):
        raise ValueError("level epsilons must be positive")
    return parts


@dataclass(frozen=True)
class HierarchicalRelease:
    """Per-station hierarchical release.

    y_tree: concatenated noisy tree-node mechanism output (all levels).
    y_cons: consistency-reconstructed leaf estimates g(Y_tree).
    y_raw_leaves: deprecated alias for y_cons (transitional).
    """

    y_cons: np.ndarray
    y_tree: np.ndarray
    epsilon_total: float
    epsilon_levels: np.ndarray
    n_leaves: int
    n_padded: int
    n_levels: int
    mechanism_id: str = "M3_hierarchical_laplace"

    @property
    def y_raw_leaves(self) -> np.ndarray:
        return self.y_cons


def _build_true_nodes(leaves: np.ndarray, n_padded: int) -> list[np.ndarray]:
    """nodes[level] has length 2^level; level 0 is root."""
    n_levels = int(np.log2(n_padded)) + 1
    padded = np.zeros(n_padded, dtype=np.float64)
    padded[: leaves.size] = leaves.astype(np.float64)
    nodes: list[np.ndarray] = [np.zeros(0) for _ in range(n_levels)]
    nodes[n_levels - 1] = padded.copy()
    for level in range(n_levels - 2, -1, -1):
        child = nodes[level + 1]
        parent = child.reshape(-1, 2).sum(axis=1)
        nodes[level] = parent
    return nodes


def _add_level_noise(
    nodes: list[np.ndarray],
    epsilon_levels: np.ndarray,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    noisy: list[np.ndarray] = []
    for level, eps in enumerate(epsilon_levels):
        scale = laplace_scale(float(eps), sensitivity=1.0)
        noise = rng.laplace(0.0, scale, size=nodes[level].shape)
        noisy.append(nodes[level] + noise)
    return noisy


def _consistent_reconstruct(noisy: list[np.ndarray]) -> np.ndarray:
    """Bottom-up average then top-down sum consistency (deterministic)."""
    n_levels = len(noisy)
    cons = [arr.astype(np.float64).copy() for arr in noisy]
    # Bottom-up: blend node with children sum
    for level in range(n_levels - 2, -1, -1):
        child = cons[level + 1]
        child_sum = child.reshape(-1, 2).sum(axis=1)
        cons[level] = 0.5 * cons[level] + 0.5 * child_sum
    # Top-down: force children to sum to parent
    for level in range(n_levels - 1):
        parent = cons[level]
        child = cons[level + 1].reshape(-1, 2)
        sums = child.sum(axis=1)
        for i in range(parent.size):
            target = parent[i]
            s = sums[i]
            if abs(s) < 1e-15:
                child[i, 0] = 0.5 * target
                child[i, 1] = 0.5 * target
            else:
                scale = target / s
                child[i, 0] *= scale
                child[i, 1] *= scale
        cons[level + 1] = child.reshape(-1)
        # recompute sums for safety
    return cons[-1]


def release_hierarchical_station(
    leaf_counts: np.ndarray,
    *,
    epsilon: float,
    seed: int,
) -> HierarchicalRelease:
    """Release one station's temporal hierarchy under total ε (sequential levels)."""
    leaves = np.asarray(leaf_counts, dtype=np.float64)
    if leaves.ndim != 1:
        raise ValueError("leaf_counts must be 1-D")
    if not np.all(np.isfinite(leaves)):
        raise ValueError("leaf_counts must be finite")
    n_leaves = int(leaves.size)
    n_padded = next_power_of_two(n_leaves)
    n_levels = hierarchy_levels(n_leaves)
    eps_levels = uniform_level_epsilons(epsilon, n_levels)
    true_nodes = _build_true_nodes(leaves, n_padded)
    rng = spawn_generator(seed)
    noisy = _add_level_noise(true_nodes, eps_levels, rng)
    recon = _consistent_reconstruct(noisy)
    y_cons = recon[:n_leaves].copy()
    y_tree = np.concatenate([np.asarray(level, dtype=np.float64).ravel() for level in noisy])
    return HierarchicalRelease(
        y_cons=y_cons,
        y_tree=y_tree,
        epsilon_total=float(epsilon),
        epsilon_levels=eps_levels,
        n_leaves=n_leaves,
        n_padded=n_padded,
        n_levels=n_levels,
    )


def release_hierarchical_tensor(
    station_hour: np.ndarray,
    *,
    epsilon: float,
    seed: int,
) -> tuple[np.ndarray, HierarchicalRelease]:
    """Vectorized hierarchical Laplace across stations (shared public tree geometry)."""
    x = np.asarray(station_hour, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("station_hour must be 2-D")
    n_stations, n_hours = x.shape
    n_padded = next_power_of_two(n_hours)
    n_levels = hierarchy_levels(n_hours)
    eps_levels = uniform_level_epsilons(epsilon, n_levels)
    padded = np.zeros((n_stations, n_padded), dtype=np.float64)
    padded[:, :n_hours] = x
    # true nodes[level]: (n_stations, 2^level)
    nodes: list[np.ndarray] = [np.zeros((n_stations, 0)) for _ in range(n_levels)]
    nodes[n_levels - 1] = padded
    for level in range(n_levels - 2, -1, -1):
        child = nodes[level + 1]
        nodes[level] = child.reshape(n_stations, -1, 2).sum(axis=2)
    rng = spawn_generator(seed)
    noisy: list[np.ndarray] = []
    for level, eps in enumerate(eps_levels):
        scale = laplace_scale(float(eps), sensitivity=1.0)
        noise = rng.laplace(0.0, scale, size=nodes[level].shape)
        noisy.append(nodes[level] + noise)
    # consistency per station via shared geometry
    cons = [arr.copy() for arr in noisy]
    for level in range(n_levels - 2, -1, -1):
        child = cons[level + 1]
        child_sum = child.reshape(n_stations, -1, 2).sum(axis=2)
        cons[level] = 0.5 * cons[level] + 0.5 * child_sum
    for level in range(n_levels - 1):
        parent = cons[level]
        child = cons[level + 1].reshape(n_stations, -1, 2)
        sums = child.sum(axis=2)
        scale = np.ones_like(sums)
        nonzero = np.abs(sums) >= 1e-15
        scale[nonzero] = parent[nonzero] / sums[nonzero]
        child[..., 0] = np.where(nonzero, child[..., 0] * scale, 0.5 * parent)
        child[..., 1] = np.where(nonzero, child[..., 1] * scale, 0.5 * parent)
        cons[level + 1] = child.reshape(n_stations, -1)
    y_cons = cons[-1][:, :n_hours].copy()
    # compact tree checksum object from station 0
    y_tree0 = np.concatenate([level[0].ravel() for level in noisy])
    meta = HierarchicalRelease(
        y_cons=y_cons[0].copy(),
        y_tree=y_tree0,
        epsilon_total=float(epsilon),
        epsilon_levels=eps_levels,
        n_leaves=n_hours,
        n_padded=n_padded,
        n_levels=n_levels,
    )
    return y_cons, meta


def path_nodes_for_leaf(leaf_index: int, n_padded: int) -> list[tuple[int, int]]:
    """Return [(level, node_index), ...] on the root-to-leaf path."""
    n_levels = int(np.log2(n_padded)) + 1
    path: list[tuple[int, int]] = []
    idx = leaf_index
    for level in range(n_levels - 1, -1, -1):
        path.append((level, idx))
        idx //= 2
    path.reverse()
    return path
