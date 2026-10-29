"""Geometry- and timing-bound CFD references for the E-022 comparison.

This is a resolution-matched *kinematic* reference, not an MRI signal model,
independent validation, or a new Navier--Stokes solution.  Existing periodic
CFD fields already contain the synthetic tail: this module never closes them
again.  Subvoxel midpoint sampling estimates fluid fractions and fluid-only
mean velocities.  It is not exact voxel/tetrahedron clipping.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
import tomllib
from typing import Any

import h5py
import numpy as np
import pyvista as pv

from mrsimtracks.io import load_flow
from mrsimtracks.sampler import _TetSampler, _tet_volumes, _condition_mesh


@dataclass(frozen=True)
class MRIGrid:
    """Array axes S/P/R, physical/vector axes R/P/S, all lengths in metres."""

    shape_spr: tuple[int, int, int]
    fov_rps_m: tuple[float, float, float]
    source_center_m: tuple[float, float, float]
    source_axis_order: tuple[int, int, int] = (2, 0, 1)

    def __post_init__(self) -> None:
        if len(self.shape_spr) != 3 or any(int(n) != n or n < 1 for n in self.shape_spr):
            raise ValueError("shape_spr must contain three positive integers")
        if len(self.fov_rps_m) != 3 or not np.isfinite(self.fov_rps_m).all() or min(self.fov_rps_m) <= 0:
            raise ValueError("FOV must contain three finite positive lengths")
        if len(self.source_center_m) != 3 or not np.isfinite(self.source_center_m).all():
            raise ValueError("Source center must contain three finite lengths")
        if sorted(self.source_axis_order) != [0, 1, 2]:
            raise ValueError("source_axis_order must be a permutation of (0,1,2)")

    @property
    def spacing_rps_m(self) -> np.ndarray:
        return np.asarray(self.fov_rps_m) / np.asarray(self.shape_spr[::-1])

    @property
    def voxel_volume_m3(self) -> float:
        return float(np.prod(self.spacing_rps_m))

    def axis_centers_m(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return R/P/S centers following the existing support/plot contract."""
        return tuple(
            (np.arange(n) + 0.5) * d - f / 2
            for n, d, f in zip(self.shape_spr[::-1], self.spacing_rps_m, self.fov_rps_m, strict=True)
        )

    def mri_to_source(self, points_rps_m: np.ndarray) -> np.ndarray:
        points = np.asarray(points_rps_m, dtype=np.float64)
        if points.shape[-1:] != (3,):
            raise ValueError("Points must have final dimension 3")
        return points[..., np.argsort(self.source_axis_order)] + self.source_center_m

    def source_to_mri(self, points_xyz_m: np.ndarray) -> np.ndarray:
        points = np.asarray(points_xyz_m, dtype=np.float64)
        if points.shape[-1:] != (3,):
            raise ValueError("Points must have final dimension 3")
        return (points - self.source_center_m)[..., self.source_axis_order]

    def velocity_to_mri(self, velocity_xyz: np.ndarray) -> np.ndarray:
        value = np.asarray(velocity_xyz)
        if value.shape[-1:] != (3,):
            raise ValueError("Velocity must have final dimension 3")
        return value[..., self.source_axis_order]


def read_e022_contract(
    tracking_report: str | Path,
    simulation_report: str | Path,
    support_phase_h5: str | Path,
) -> tuple[MRIGrid, dict[str, Any]]:
    """Read the spatial and temporal contract without loading velocity data.

    Returned timing keys: source_cycle_s, effective_motion_period_s,
    velocity_scale, cardiac_bins, synthetic_tail_start_source_s,
    bin_spacing_s, source_frame_spacing_s, source_to_mri_axis_order.
    Artifact hashing and production-file identity are the caller's provenance
    responsibility; this function checks their mutually consistent metadata.
    """
    tracking = json.loads(Path(tracking_report).read_text(encoding="utf-8"))
    with Path(simulation_report).open("rb") as stream:
        simulation = tomllib.load(stream)
    run = tracking["run"]
    if run["axis_order_zero_based"] != [2, 0, 1] or run["time_interpolation"] != "cubic":
        raise ValueError("Not the E-022 source-axis/cubic-interpolation contract")
    if simulation.get("status") != "PASS" or simulation.get("seam_safe_motion") is not True:
        raise ValueError("Expected the accepted seam-safe simulation")
    source_cycle = float(run["duration_s"])
    effective_period = float(simulation["motion_timing"]["effective_period_s"])
    if not np.isfinite([source_cycle, effective_period]).all() or min(source_cycle, effective_period) <= 0:
        raise ValueError("Periods must be finite and positive")
    with h5py.File(support_phase_h5, "r") as h5:
        shape = h5["velocity_corrected_m_per_s"].shape
        matrix = tuple(int(n) for n in h5.attrs["matrix_readout_phase_slice"])
        fov = tuple(float(f) for f in h5.attrs["FOV_m"])
        if shape != (3, 15, 32, 64, 400) or matrix != shape[-3:][::-1]:
            raise ValueError(f"Unexpected E-022 grid: {shape}, matrix={matrix}")
        if not np.isclose(float(h5.attrs["RR_s"]), effective_period, rtol=0, atol=1e-6):
            raise ValueError("Reconstruction RR and effective motion period disagree")
        if not np.allclose(np.asarray(fov) / matrix * 1e3, h5.attrs["voxel_size_mm_readout_phase_slice"]):
            raise ValueError("Reconstruction voxel sizes disagree with FOV/matrix")
    grid = MRIGrid(shape[-3:], fov, tuple(run["source_center_m"]))
    timing = {
        "source_cycle_s": source_cycle,
        "effective_motion_period_s": effective_period,
        "velocity_scale": source_cycle / effective_period,
        "cardiac_bins": shape[1],
        "synthetic_tail_start_source_s": float(tracking["cycle_scope"]["synthetic_tail_blend_start_s"]),
        "bin_spacing_s": effective_period / shape[1],
        "source_frame_spacing_s": 0.01,
        "source_to_mri_axis_order": list(grid.source_axis_order),
        "coordinate_center_convention": "(index+0.5)*FOV/matrix-FOV/2; existing support convention",
        "temporal_reference_semantics": "periodic Catmull-Rom source field, then velocity scaled by source_period/effective_period",
        "is_exact_mri_forward_response": False,
    }
    return grid, timing


def geometry_sha256(mesh: pv.UnstructuredGrid) -> str:
    """Fingerprint geometry/connectivity only; independent of active velocity."""
    digest = hashlib.sha256()
    for array in (mesh.points, mesh.cells, mesh.celltypes):
        value = np.ascontiguousarray(array)
        digest.update(str((value.shape, value.dtype.str)).encode("ascii"))
        digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def load_periodic_reference(pvd_path: str | Path):
    """Load the same conditioned mesh and f32 Catmull-Rom fields as tracking."""
    flow = load_flow(
        str(pvd_path), active_key="Velocity", precision="f32",
        time_interp="cubic", pbar=False, only_active_key=True,
        conform_mesh=True,
    )
    times = np.asarray(flow.times_shift_s)
    if times.size != 73 or not np.allclose(np.diff(times), 0.01, rtol=0, atol=1e-12):
        raise ValueError("Expected 73 native 10-ms dp150 frames")
    if not np.isclose(flow.tmax, 0.720, rtol=0, atol=1e-12):
        raise ValueError("Expected the dp150 0.720-s source cycle")
    if not np.array_equal(flow._frame_vel(0), flow._frame_vel(times.size - 1)):
        raise ValueError("Reference must already contain the closed periodic field")
    for index in range(times.size):
        values = np.asarray(flow._frame_vel(index))
        if values.shape != (flow.active_mesh.n_points, 3) or not np.isfinite(values).all():
            raise ValueError(f"Invalid nodal field at source frame {index}")
    return flow


@dataclass
class VoxelLookup:
    """Compact fixed-mesh interpolation; stores valid subvoxels only."""

    grid: MRIGrid
    subsamples: int | tuple[int, int, int]
    mesh_sha256: str
    mesh_n_points: int
    flat_indices: np.ndarray
    sample_voxel: np.ndarray
    vertex_indices: np.ndarray
    barycentric: np.ndarray
    counts: np.ndarray

    @property
    def sample_shape_rps(self) -> tuple[int, int, int]:
        return _sample_shape(self.subsamples)

    @property
    def samples_per_voxel(self) -> int:
        return int(np.prod(self.sample_shape_rps))

    def fraction_grid(self) -> np.ndarray:
        fraction = np.zeros(self.grid.shape_spr, dtype=np.float32)
        fraction.ravel()[self.flat_indices] = self.counts / self.samples_per_voxel
        return fraction

    def scatter(self, compact: np.ndarray, *, fill: float = np.nan) -> np.ndarray:
        """Scatter final compact axis M to S/P/R, keeping leading axes."""
        compact = np.asarray(compact)
        if compact.shape[-1] != self.flat_indices.size:
            raise ValueError("Compact value count does not match lookup")
        out = np.full((*compact.shape[:-1], int(np.prod(self.grid.shape_spr))), fill, dtype=compact.dtype)
        out[..., self.flat_indices] = compact
        return out.reshape(*compact.shape[:-1], *self.grid.shape_spr)

    def average(self, nodal_velocity_mri: np.ndarray, *, chunk_size: int = 100_000) -> dict[str, np.ndarray]:
        """Fluid-only <u> and <|u|²>; missing geometry remains NaN, not zero.

        The squared-speed moment is preserved separately from |<u>|².  The
        difference is spatial subvoxel variance, not a measurement of TKE.
        """
        values = np.asarray(nodal_velocity_mri)
        if values.shape != (self.mesh_n_points, 3) or not np.isfinite(values).all():
            raise ValueError("Nodal velocity shape/finiteness mismatch")
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        count = self.flat_indices.size
        summed = np.zeros((count, 3), dtype=np.float64)
        summed_squared = np.zeros(count, dtype=np.float64)
        for start in range(0, self.sample_voxel.size, chunk_size):
            sl = slice(start, start + chunk_size)
            sampled = np.einsum("ni,nij->nj", self.barycentric[sl], values[self.vertex_indices[sl]])
            ids = self.sample_voxel[sl]
            for component in range(3):
                summed[:, component] += np.bincount(ids, weights=sampled[:, component], minlength=count)
            summed_squared += np.bincount(ids, weights=np.sum(sampled**2, axis=1), minlength=count)
        mean = summed / self.counts[:, None]
        mean_squared = summed_squared / self.counts
        return {
            "velocity_mri": self.scatter(mean.T.astype(np.float32)),
            "mean_speed_squared": self.scatter(mean_squared.astype(np.float32)),
            "volume_fraction": self.fraction_grid(),
        }


def _candidate_indices(mesh: pv.UnstructuredGrid, grid: MRIGrid) -> np.ndarray:
    bounds = np.asarray(mesh.bounds).reshape(3, 2)
    bounds = bounds[list(grid.source_axis_order)] - np.asarray(grid.source_center_m)[list(grid.source_axis_order), None]
    spacing = grid.spacing_rps_m
    shape = np.asarray(grid.shape_spr[::-1])
    lower = np.maximum(np.floor((bounds[:, 0] + np.asarray(grid.fov_rps_m) / 2) / spacing).astype(int), 0)
    upper = np.minimum(np.floor((bounds[:, 1] + np.asarray(grid.fov_rps_m) / 2) / spacing).astype(int), shape - 1)
    if np.any(upper < lower):
        raise ValueError("CFD mesh does not intersect the requested MRI grid")
    s, p, r = np.meshgrid(
        np.arange(lower[2], upper[2] + 1), np.arange(lower[1], upper[1] + 1),
        np.arange(lower[0], upper[0] + 1), indexing="ij",
    )
    return np.ravel_multi_index((s.ravel(), p.ravel(), r.ravel()), grid.shape_spr)


def _sample_shape(subsamples: int | tuple[int, int, int]) -> tuple[int, int, int]:
    shape = (subsamples,) * 3 if np.isscalar(subsamples) else tuple(subsamples)
    if len(shape) != 3 or any(isinstance(n, bool) or int(n) != n or n < 1 for n in shape):
        raise ValueError("subsamples must be positive integer or three positive integers R/P/S")
    return tuple(int(n) for n in shape)


def build_voxel_lookup(
    mesh: pv.UnstructuredGrid, grid: MRIGrid, *, subsamples: int | tuple[int, int, int] = 2,
    chunk_voxels: int = 10_000,
) -> VoxelLookup:
    """Probe subvoxel midpoints and cache exact linear-tet barycentric weights.

    Use the tracking-conditioned tetrahedral mesh.  All geometry-intersecting
    bounding-box voxels are considered, independently of MRI signal masks.
    n=2 is the default quadrature; n=3 is a sensitivity, not a proven accuracy.
    A tuple (1,n,n) instead samples each readout-center *plane*, and then the
    returned fraction is an area fraction, NOT a three-dimensional fraction.
    """
    sample_shape = _sample_shape(subsamples)
    sample_count = int(np.prod(sample_shape))
    if chunk_voxels < 1:
        raise ValueError("chunk_voxels must be positive")
    sampler = _TetSampler(mesh, dtype=np.float64)
    if not sampler.ok:
        raise ValueError("Expected the tracking-conditioned tetrahedral mesh")
    indices = _candidate_indices(mesh, grid)
    axes = grid.axis_centers_m()
    offset_axes = [(np.arange(n) + 0.5) / n - 0.5 for n in sample_shape]
    offsets = np.stack(np.meshgrid(*offset_axes, indexing="ij"), axis=-1).reshape(-1, 3)
    offsets *= grid.spacing_rps_m
    all_ids, all_vertices, all_weights = [], [], []
    counts = np.zeros(indices.size, dtype=np.int32)
    for start in range(0, indices.size, chunk_voxels):
        chosen = indices[start:start + chunk_voxels]
        s, p, r = np.unravel_index(chosen, grid.shape_spr)
        centers = np.column_stack((axes[0][r], axes[1][p], axes[2][s]))
        points = grid.mri_to_source((centers[:, None, :] + offsets).reshape(-1, 3))
        cells = sampler.locate(points)
        valid = cells >= 0
        valid_positions = np.flatnonzero(valid)
        weights = sampler._bary(points[valid], cells[valid])
        # VTK's automatic probe tolerance may extend outside the actual cell.
        # A strict barycentric test prevents extrapolated boundary occupancy.
        inside = np.isfinite(weights).all(axis=1) & (weights >= -1e-9).all(axis=1) & (weights <= 1 + 1e-9).all(axis=1)
        valid_positions = valid_positions[inside]
        weights = np.clip(weights[inside], 0, 1)
        weights /= weights.sum(axis=1, keepdims=True)
        ids = start + valid_positions // sample_count
        all_ids.append(ids.astype(np.int32))
        all_vertices.append(sampler.conn[cells[valid_positions]].astype(np.int32))
        all_weights.append(weights)
        counts += np.bincount(ids, minlength=indices.size).astype(np.int32)
    keep = counts > 0
    if not keep.any():
        raise ValueError("No subvoxel sample was inside CFD geometry")
    remap = np.cumsum(keep) - 1
    return VoxelLookup(
        grid, subsamples, geometry_sha256(mesh), mesh.n_points,
        indices[keep], remap[np.concatenate(all_ids)].astype(np.int32),
        np.concatenate(all_vertices), np.concatenate(all_weights), counts[keep],
    )


def build_section_lookup(mesh: pv.UnstructuredGrid, grid: MRIGrid, *, transverse_subsamples: int = 2, chunk_voxels: int = 10_000) -> VoxelLookup:
    """All readout-center planes; fraction_grid() is *area* fraction here.

    The velocity shape is still (3,S,P,R), but each R is a zero-thickness
    plane sampled only transversely.  Multiply its fraction by dP*dS for
    section integration; do not use this lookup for volume-integrated KE.
    """
    return build_voxel_lookup(mesh, grid, subsamples=(1, transverse_subsamples, transverse_subsamples), chunk_voxels=chunk_voxels)


def bin_sample_times(n_bins: int, effective_period_s: float, *, samples_per_bin: int = 1) -> np.ndarray:
    """Sequence times (bin,sample), equal-weight midpoint quadrature.

    Three samples per bin are a temporal-averaging sensitivity, not simulated
    gradient/ADC weighting and not three additional acquired cardiac phases.
    """
    if isinstance(n_bins, bool) or int(n_bins) != n_bins or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")
    if samples_per_bin not in (1, 3):
        raise ValueError("Only 1 midpoint or 3 within-bin midpoint samples are supported")
    if not np.isfinite(effective_period_s) or effective_period_s <= 0:
        raise ValueError("effective_period_s must be finite and positive")
    offsets = (np.arange(samples_per_bin) + 0.5) / samples_per_bin
    return (np.arange(n_bins)[:, None] + offsets) * effective_period_s / n_bins


def transformed_nodal_velocity(flow, grid: MRIGrid, sequence_time_s: float, effective_period_s: float) -> np.ndarray:
    """Evaluate production Catmull-Rom interpolation then retime/reorder u.

    Time origin is the motion seam, not the scanner trigger or an RF event.
    """
    if not np.isfinite([sequence_time_s, effective_period_s]).all() or effective_period_s <= 0:
        raise ValueError("Time must be finite and effective period positive")
    scale = float(flow.tmax) / effective_period_s
    source_time = (sequence_time_s % effective_period_s) * scale
    flow.set_active_time(source_time)
    velocity = np.asarray(flow.active_mesh.point_data[flow.active_key])
    return grid.velocity_to_mri(velocity).astype(np.float64) * scale


def sample_reference(flow, lookup: VoxelLookup, sequence_time_s: float, effective_period_s: float) -> dict[str, np.ndarray]:
    values = transformed_nodal_velocity(flow, lookup.grid, sequence_time_s, effective_period_s)
    return lookup.average(values)


def native_tet_moments(mesh: pv.UnstructuredGrid, nodal_velocity_mri: np.ndarray, *, grid: MRIGrid | None = None) -> dict[str, np.ndarray | float]:
    """Exact integrals for the linear velocity field in each conditioned tet.

    These are native-resolution *converted/interpolated* CFD quantities, not
    original Fluent cell-centered quantities.  <|u|²> is exact for the linear
    field; evaluating only at tet centroids would underestimate that moment.
    No density or viscosity is assumed.  Regional cell selection by centroid
    is not exact clipping against a region's boundary.
    """
    values = np.asarray(nodal_velocity_mri, dtype=np.float64)
    if values.shape != (mesh.n_points, 3) or not np.isfinite(values).all():
        raise ValueError("Nodal velocity shape/finiteness mismatch")
    if not np.all(np.asarray(mesh.celltypes) == 10):
        original_points = np.asarray(mesh.points)
        mesh = _condition_mesh(mesh, verbose=False)
        if not np.array_equal(original_points, np.asarray(mesh.points)):
            raise ValueError("Mesh conditioning changed nodal identity; cannot map fields")
    conn = mesh.cells.reshape(-1, 5)[:, 1:]
    vertices = values[conn]
    volume = _tet_volumes(np.asarray(mesh.points, dtype=np.float64), conn)
    if np.any(volume <= 0) or not np.isfinite(volume).all():
        raise ValueError("Degenerate/nonfinite native tetrahedron")
    vertex_sum = vertices.sum(axis=1)
    mean_velocity = vertex_sum / 4
    mean_speed_squared = (np.sum(vertices**2, axis=(1, 2)) + np.sum(vertex_sum**2, axis=1)) / 20
    centers = np.asarray(mesh.points, dtype=np.float64)[conn].mean(axis=1)
    return {
        "cell_volume_m3": volume,
        "cell_centers_source_m": centers,
        "cell_centers_mri_m": grid.source_to_mri(centers) if grid is not None else None,
        "cell_mean_velocity_mri": mean_velocity,
        "cell_mean_speed_squared": mean_speed_squared,
        "total_volume_m3": float(volume.sum()),
        "integral_speed_squared_m5_s2": float(np.dot(volume, mean_speed_squared)),
        "integral_velocity_m4_s": np.einsum("i,ij->j", volume, mean_velocity),
    }


def save_lookup(path: str | Path, lookup: VoxelLookup) -> None:
    """Atomically publish a new provenance-bound NPZ; never replace a file."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    if not path.parent.is_dir():
        raise FileNotFoundError(path.parent)
    metadata = {
        "schema_version": 1, "grid": asdict(lookup.grid),
        "subsamples": lookup.subsamples, "mesh_sha256": lookup.mesh_sha256,
        "mesh_n_points": lookup.mesh_n_points,
        "quadrature": "fluid-only midpoint samples; not exact geometric clipping",
        "sample_shape_rps": list(lookup.sample_shape_rps),
        "fraction_semantics": "plane_area" if lookup.sample_shape_rps[0] == 1 and lookup.sample_shape_rps[1:] != (1, 1) else "volume",
    }
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w+b", dir=path.parent, prefix=path.name + ".", suffix=".partial", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(
                stream, metadata_json=json.dumps(metadata), flat_indices=lookup.flat_indices,
                sample_voxel=lookup.sample_voxel, vertex_indices=lookup.vertex_indices,
                barycentric=lookup.barycentric, counts=lookup.counts,
            )
            stream.flush()
            os.fsync(stream.fileno())
        # Hard-link publication is atomic and fails if a concurrent writer won.
        os.link(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def load_lookup(path: str | Path, *, expected_grid: MRIGrid, expected_mesh_sha256: str, expected_subsamples: int | tuple[int, int, int]) -> VoxelLookup:
    """Fail-closed lookup reuse: all geometry/quadrature fields must match."""
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["metadata_json"]))
        if meta.get("schema_version") != 1:
            raise ValueError("Unsupported lookup schema")
        raw_grid = meta["grid"]
        grid = MRIGrid(**{key: tuple(value) for key, value in raw_grid.items()})
        restored_subsamples = tuple(meta["subsamples"]) if isinstance(meta["subsamples"], list) else meta["subsamples"]
        if grid != expected_grid or meta["mesh_sha256"] != expected_mesh_sha256 or _sample_shape(restored_subsamples) != _sample_shape(expected_subsamples):
            raise ValueError("Lookup provenance mismatch")
        lookup = VoxelLookup(grid, restored_subsamples, meta["mesh_sha256"], meta["mesh_n_points"],
                             *[np.array(data[key]) for key in ("flat_indices", "sample_voxel", "vertex_indices", "barycentric", "counts")])
    m, k = lookup.flat_indices.size, lookup.sample_voxel.size
    if lookup.vertex_indices.shape != (k, 4) or lookup.barycentric.shape != (k, 4) or lookup.counts.shape != (m,):
        raise ValueError("Corrupt lookup array shapes")
    if not m or not k or np.any(lookup.counts <= 0) or np.any(lookup.counts > lookup.samples_per_voxel):
        raise ValueError("Corrupt lookup counts")
    if np.any(lookup.sample_voxel < 0) or np.any(lookup.sample_voxel >= m):
        raise ValueError("Corrupt lookup sample indices")
    if np.any(lookup.vertex_indices < 0) or np.any(lookup.vertex_indices >= lookup.mesh_n_points):
        raise ValueError("Corrupt lookup node indices")
    if np.any(lookup.flat_indices < 0) or np.any(lookup.flat_indices >= np.prod(grid.shape_spr)) or np.unique(lookup.flat_indices).size != m:
        raise ValueError("Corrupt lookup voxel indices")
    if not np.isfinite(lookup.barycentric).all() or np.any(lookup.barycentric < 0) or not np.allclose(lookup.barycentric.sum(axis=1), 1, rtol=0, atol=1e-12):
        raise ValueError("Corrupt lookup barycentric weights")
    if not np.array_equal(np.bincount(lookup.sample_voxel, minlength=m), lookup.counts):
        raise ValueError("Corrupt lookup sample counts")
    return lookup
