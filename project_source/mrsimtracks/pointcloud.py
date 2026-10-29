"""Approximate particle tracking on CFD point clouds without cell connectivity."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


class PointCloudFlow:
    """Time-resolved velocity field sampled from an unstructured point cloud.

    This adapter is intended for CFD exports that retain coordinates and nodal
    velocities but not the original volume-cell connectivity. Spatial velocity
    interpolation uses inverse-distance weighted k-nearest neighbours. A query
    is considered outside the represented domain when its nearest source point
    is farther than ``support_radius``.

    Unlike tetrahedral interpolation, k-NN interpolation cannot represent the
    vessel boundary exactly. Results from this class should therefore be
    treated as approximate, particularly near walls and open boundaries.

    Args:
        coordinates: Source coordinates with shape ``(n_points, 3)``.
        velocities: Velocity samples with shape ``(n_points, 3, n_frames)``.
            A memory-mapped NumPy array is supported and recommended.
        frame_dt: Physical time between stored velocity frames.
        k: Number of spatial neighbours used for interpolation.
        power: Inverse-distance weighting exponent.
        support_radius: Maximum nearest-point distance for a valid query.
            When omitted, it is estimated from source-point spacing.
        support_scale: Multiplier applied to the estimated 99.5th percentile
            nearest-neighbour spacing.
        start_frame: Frame corresponding to tracking time zero.
        periodic: Wrap temporal interpolation from the last frame to the first.
    """

    def __init__(
        self,
        coordinates,
        velocities,
        *,
        frame_dt,
        k=8,
        power=2.0,
        support_radius=None,
        support_scale=2.5,
        spacing_samples=50_000,
        start_frame=0,
        periodic=True,
    ):
        self.coordinates = np.ascontiguousarray(coordinates, dtype=np.float32)
        self.velocities = velocities
        self.dtype = np.dtype(np.float32)
        self.frame_dt = float(frame_dt)
        self.k = int(k)
        self.power = float(power)
        self.periodic = bool(periodic)

        if self.coordinates.ndim != 2 or self.coordinates.shape[1] != 3:
            raise ValueError("coordinates must have shape (n_points, 3)")
        if len(self.coordinates) < 2:
            raise ValueError("point-cloud tracking requires at least two points")
        if (
            np.ndim(velocities) != 3
            or velocities.shape[0] != len(self.coordinates)
            or velocities.shape[1] != 3
        ):
            raise ValueError(
                "velocities must have shape (n_points, 3, n_frames)"
            )
        if velocities.shape[2] < 2:
            raise ValueError("velocities must contain at least two frames")
        if self.frame_dt <= 0:
            raise ValueError("frame_dt must be > 0")
        if self.k < 1 or self.k > len(self.coordinates):
            raise ValueError("k must be between 1 and n_points")
        if self.power <= 0:
            raise ValueError("power must be > 0")
        if support_scale <= 0:
            raise ValueError("support_scale must be > 0")

        self.n_frames = int(velocities.shape[2])
        self.start_frame = int(start_frame) % self.n_frames
        self.tmax = self.n_frames * self.frame_dt
        self.tree = cKDTree(self.coordinates, copy_data=False)

        if support_radius is None:
            spacing = self.estimate_spacing(spacing_samples)
            support_radius = support_scale * spacing
        self.support_radius = float(support_radius)
        if self.support_radius <= 0:
            raise ValueError("support_radius must be > 0")

    @classmethod
    def from_npy(cls, coordinates_path, velocities_path, **kwargs):
        """Create a flow while memory-mapping coordinate and velocity arrays."""
        coordinates = np.load(Path(coordinates_path), mmap_mode="r")
        velocities = np.load(Path(velocities_path), mmap_mode="r")
        return cls(coordinates, velocities, **kwargs)

    def estimate_spacing(self, sample_count=50_000):
        """Estimate high-end local source spacing from a deterministic sample."""
        sample_count = min(max(int(sample_count), 2), len(self.coordinates))
        indices = np.linspace(
            0, len(self.coordinates) - 1, sample_count, dtype=np.int64
        )
        distances, _ = self.tree.query(self.coordinates[indices], k=2, workers=-1)
        nearest = np.asarray(distances)[:, 1]
        nearest = nearest[np.isfinite(nearest) & (nearest > 0)]
        if not nearest.size:
            raise ValueError("could not estimate spacing from duplicate coordinates")
        return float(np.percentile(nearest, 99.5))

    def _frame_pair(self, time):
        phase = float(time) / self.frame_dt + self.start_frame
        if self.periodic:
            phase %= self.n_frames
            i0 = int(np.floor(phase))
            return i0, (i0 + 1) % self.n_frames, phase - i0

        phase = np.clip(phase, 0.0, self.n_frames - 1.0)
        i0 = int(np.floor(phase))
        i1 = min(i0 + 1, self.n_frames - 1)
        return i0, i1, phase - i0

    def sample_v(self, points_xyz, time, guess=None):
        """Return interpolated velocity, validity mask, and nearest point IDs."""
        del guess  # API compatibility with the tetrahedral sampler.
        points = np.asarray(points_xyz, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points_xyz must have shape (n, 3)")
        if not len(points):
            return (
                np.empty((0, 3), dtype=self.dtype),
                np.empty(0, dtype=bool),
                np.empty(0, dtype=np.int64),
            )

        distances, indices = self.tree.query(points, k=self.k, workers=-1)
        if self.k == 1:
            distances = distances[:, None]
            indices = indices[:, None]
        distances = np.asarray(distances)
        indices = np.asarray(indices, dtype=np.int64)

        valid = np.isfinite(distances[:, 0]) & (
            distances[:, 0] <= self.support_radius
        )
        i0, i1, alpha = self._frame_pair(time)
        v0 = np.asarray(self.velocities[indices, :, i0], dtype=self.dtype)
        if alpha:
            v1 = np.asarray(self.velocities[indices, :, i1], dtype=self.dtype)
            node_velocity = (1.0 - alpha) * v0 + alpha * v1
        else:
            node_velocity = v0

        epsilon = max(self.support_radius * 1e-6, np.finfo(np.float32).eps)
        weights = 1.0 / np.maximum(distances, epsilon) ** self.power
        weights /= weights.sum(axis=1, keepdims=True)
        velocity = np.einsum(
            "nk,nkj->nj", weights, node_velocity, optimize=True
        ).astype(self.dtype, copy=False)
        velocity[~valid] = 0.0
        nearest = np.where(valid, indices[:, 0], -1)
        return velocity, valid, nearest

    def seed_points(
        self,
        n_points,
        *,
        frame=None,
        rng=None,
        min_speed=1e-6,
    ):
        """Draw repeatable source points with non-negligible local velocity."""
        n_points = int(n_points)
        if n_points < 1:
            raise ValueError("n_points must be >= 1")
        rng = rng if rng is not None else np.random.default_rng()
        frame = self.start_frame if frame is None else int(frame) % self.n_frames

        chosen = []
        remaining = n_points
        for _ in range(20):
            candidate = rng.integers(
                0, len(self.coordinates), size=max(remaining * 3, 1024)
            )
            velocity = np.asarray(self.velocities[candidate, :, frame])
            keep = np.linalg.norm(velocity, axis=1) >= min_speed
            accepted = candidate[keep][:remaining]
            chosen.append(accepted)
            remaining -= len(accepted)
            if remaining == 0:
                break
        if remaining:
            raise ValueError(
                f"could only find {n_points - remaining} moving seed points"
            )
        indices = np.concatenate(chosen)
        return self.coordinates[indices].copy()
