"""Audit approximate uniform aorta seeds without requiring a volume mesh."""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree


DEFAULT_POINTS = Path(
    "outputs/aorta_dp2_archive_2026-08-21/recovered_analysis/"
    "aorta_dp2_uniform_2M.h5"
)
DEFAULT_DATA_ROOT = Path("../external_inputs/aorta_pointcloud")
DEFAULT_SIMULATION = "dp2"
DEFAULT_OUTPUT_DIR = Path("outputs/aorta_dp2_mri_pipeline/uniform_seeding")
DEFAULT_TRACKING_REPORT = Path(
    "outputs/aorta_dp2_mri_pipeline/aorta_motion_1m_tracking_report.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=Path, default=DEFAULT_POINTS)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=(
            "Aorta package root containing processed/<simulation> "
            f"(default: {DEFAULT_DATA_ROOT})."
        ),
    )
    parser.add_argument("--simulation", default=DEFAULT_SIMULATION)
    parser.add_argument(
        "--coordinates",
        type=Path,
        default=None,
        help="Explicit point_coordinates.npy override.",
    )
    parser.add_argument(
        "--case",
        type=Path,
        default=None,
        help=(
            "Legacy compatibility: explicit processed/<simulation> directory; "
            "overrides --data-root and --simulation."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tracking-report", type=Path, default=DEFAULT_TRACKING_REPORT)
    parser.add_argument("--voxel-mm", type=float, default=2.0)
    parser.add_argument("--sample-points", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260827)
    return parser.parse_args()


@dataclass(frozen=True)
class AuditSources:
    """Resolved coordinate and velocity inputs for the seeding audit."""

    case_directory: Path
    coordinates: Path
    velocities: Path
    coordinates_format: str
    coordinates_provenance: str
    data_layout: str


def resolve_audit_sources(
    data_root: Path,
    simulation: str,
    coordinates: Path | None = None,
    case: Path | None = None,
) -> AuditSources:
    """Resolve the canonical arrays and preserve the old ``--case`` mode."""
    if case is not None:
        case_directory = Path(case).expanduser().resolve()
        data_layout = "explicit_case_directory"
    else:
        root = Path(data_root).expanduser().resolve()
        canonical = root / "processed" / simulation
        legacy_cli = root / simulation
        if (canonical / "velocities.npy").is_file() or not (
            legacy_cli / "velocities.npy"
        ).is_file():
            case_directory = canonical
            data_layout = "aorta_package_root"
        else:
            print(
                "WARNING: --data-root points directly to processed/. This "
                "legacy layout remains supported, but its parent is preferred."
            )
            case_directory = legacy_cli
            data_layout = "legacy_processed_root_argument"

    velocity_path = (case_directory / "velocities.npy").resolve()
    if not velocity_path.is_file():
        raise FileNotFoundError(f"velocity file not found: {velocity_path}")

    if coordinates is not None:
        coordinate_path = Path(coordinates).expanduser().resolve()
        if not coordinate_path.is_file():
            raise FileNotFoundError(
                f"coordinate file not found: {coordinate_path}"
            )
        if coordinate_path.suffix.lower() != ".npy":
            raise ValueError("--coordinates must point to a .npy array")
        coordinate_format = "npy"
        coordinate_provenance = "explicit_coordinates_override"
    else:
        coordinate_path = (case_directory / "point_coordinates.npy").resolve()
        if coordinate_path.is_file():
            coordinate_format = "npy"
            coordinate_provenance = "processed_simulation_directory"
        else:
            geometry_path = (case_directory / "geometry.pkl").resolve()
            if not geometry_path.is_file():
                raise FileNotFoundError(
                    "coordinate source not found; expected canonical array "
                    f"{coordinate_path} or legacy geometry {geometry_path}"
                )
            print(
                "WARNING: point_coordinates.npy is absent; auditing coordinates "
                f"from legacy geometry pickle {geometry_path}."
            )
            coordinate_path = geometry_path
            coordinate_format = "geometry_pickle"
            coordinate_provenance = "legacy_geometry_pickle_fallback"

    return AuditSources(
        case_directory=case_directory.resolve(),
        coordinates=coordinate_path,
        velocities=velocity_path,
        coordinates_format=coordinate_format,
        coordinates_provenance=coordinate_provenance,
        data_layout=data_layout,
    )


def load_audit_arrays(
    sources: AuditSources,
) -> tuple[np.ndarray, np.ndarray]:
    """Load coordinate/velocity inputs and validate their matched shapes."""
    if sources.coordinates_format == "npy":
        coordinates = np.load(sources.coordinates, mmap_mode="r")
    else:
        with sources.coordinates.open("rb") as stream:
            geometry = pickle.load(stream)
        if "mesh_coords" not in geometry:
            raise KeyError(
                f"legacy geometry has no 'mesh_coords': {sources.coordinates}"
            )
        coordinates = np.asarray(geometry["mesh_coords"], dtype=np.float64)
    velocities = np.load(sources.velocities, mmap_mode="r")

    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError(
            "point coordinates must have shape (n_points, 3), got "
            f"{coordinates.shape}"
        )
    if velocities.ndim != 3 or velocities.shape[1] != 3:
        raise ValueError(
            "velocities must have shape (n_points, 3, n_frames), got "
            f"{velocities.shape}"
        )
    if coordinates.shape[0] != velocities.shape[0]:
        raise ValueError(
            "coordinate/velocity row-count mismatch: "
            f"{coordinates.shape[0]} coordinates vs "
            f"{velocities.shape[0]} velocity rows"
        )
    return coordinates, velocities


def voxel_statistics(points: np.ndarray, voxel_m: float) -> dict[str, float | int]:
    origin = points.min(axis=0)
    index = np.floor((points - origin) / voxel_m).astype(np.int64)
    shape = index.max(axis=0) + 1
    counts = np.bincount(np.ravel_multi_index(index.T, shape))
    occupied = counts[counts > 0].astype(float)
    median = float(np.median(occupied))
    return {
        "occupied_voxels": int(len(occupied)),
        "mean": float(occupied.mean()),
        "median": median,
        "p01": float(np.percentile(occupied, 1)),
        "p99": float(np.percentile(occupied, 99)),
        "maximum": int(occupied.max()),
        "coefficient_of_variation": float(occupied.std() / occupied.mean()),
        "p99_to_median": float(np.percentile(occupied, 99) / median),
        "max_to_median": float(occupied.max() / median),
    }


def wall_mask(
    case: Path,
    velocities: np.ndarray,
    n_nodes: int,
    block_size: int = 100_000,
) -> tuple[np.ndarray, str]:
    wall_path = case / "wall_shear.npy"
    mask = np.zeros(n_nodes, dtype=bool)
    if wall_path.is_file():
        values = np.load(wall_path, mmap_mode="r")
        if values.shape[0] != n_nodes:
            raise ValueError(
                "wall_shear.npy node count does not match coordinates"
            )
        for start in range(0, n_nodes, block_size):
            stop = min(start + block_size, n_nodes)
            mask[start:stop] = (
                np.max(np.abs(values[start:stop, 0, :]), axis=1) > 0
            )
        return mask, "wall_shear.npy nonzero nodes"

    if velocities.shape[0] != n_nodes:
        raise ValueError("velocities.npy node count does not match coordinates")
    for start in range(0, n_nodes, block_size):
        stop = min(start + block_size, n_nodes)
        speed_sq = np.sum(np.square(velocities[start:stop]), axis=1)
        mask[start:stop] = np.max(speed_sq, axis=1) < 1e-12
    return mask, "velocities.npy zero-speed fallback"


def projection_histogram(points: np.ndarray, axes: tuple[int, int], bins: int = 180):
    return np.histogram2d(points[:, axes[0]], points[:, axes[1]], bins=bins)


def plot_qc(
    seeds: np.ndarray,
    raw_interior: np.ndarray,
    seed_voxels: dict[str, float | int],
    raw_voxels: dict[str, float | int],
    output: Path,
) -> None:
    labels = ((0, 1, "XY"), (0, 2, "XZ"), (1, 2, "YZ"))
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for column, (a, b, label) in enumerate(labels):
        hist, xedge, yedge = projection_histogram(seeds, (a, b))
        axes[0, column].imshow(
            np.log1p(hist.T),
            origin="lower",
            extent=(xedge[0] * 1e3, xedge[-1] * 1e3, yedge[0] * 1e3, yedge[-1] * 1e3),
            cmap="viridis",
            aspect="equal",
        )
        axes[0, column].set_title(f"Uniform seed projection - {label}")
        axes[0, column].set_xlabel(f"{label[0]} (mm)")
        axes[0, column].set_ylabel(f"{label[1]} (mm)")

    sample_limit = min(250_000, len(raw_interior), len(seeds))
    rng = np.random.default_rng(0)
    raw_sample = raw_interior[rng.choice(len(raw_interior), sample_limit, replace=False)]
    seed_sample = seeds[rng.choice(len(seeds), sample_limit, replace=False)]
    for column, (a, b, label) in enumerate(labels):
        axes[1, column].scatter(
            raw_sample[:, a] * 1e3,
            raw_sample[:, b] * 1e3,
            s=0.08,
            alpha=0.12,
            color="#6b7280",
            rasterized=True,
            label="raw interior nodes",
        )
        axes[1, column].scatter(
            seed_sample[:, a] * 1e3,
            seed_sample[:, b] * 1e3,
            s=0.08,
            alpha=0.12,
            color="#0f766e",
            rasterized=True,
            label="uniform seeds",
        )
        axes[1, column].set_title(f"Matched sample overlay - {label}")
        axes[1, column].set_xlabel(f"{label[0]} (mm)")
        axes[1, column].set_ylabel(f"{label[1]} (mm)")
        axes[1, column].set_aspect("equal")
    axes[1, 0].legend(markerscale=20, frameon=False, loc="best")
    figure.suptitle(
        "Aorta dp2 approximate uniform-seeding audit\n"
        f"2 mm voxel CV: raw {raw_voxels['coefficient_of_variation']:.3f}, "
        f"seeds {seed_voxels['coefficient_of_variation']:.3f}",
        fontsize=14,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.voxel_mm <= 0:
        raise ValueError("voxel-mm must be positive")
    if args.sample_points < 1:
        raise ValueError("sample-points must be positive")

    with h5py.File(args.points, "r") as source:
        seeds = np.asarray(source["points"], dtype=np.float64)
        seed_attributes = {
            key: value.item() if isinstance(value, np.generic) else value
            for key, value in source.attrs.items()
        }
    if seeds.ndim != 2 or seeds.shape[1] != 3:
        raise ValueError("points must have shape (n_points, 3)")

    sources = resolve_audit_sources(
        args.data_root,
        args.simulation,
        coordinates=args.coordinates,
        case=args.case,
    )
    coordinates, velocities = load_audit_arrays(sources)
    coordinates = np.asarray(coordinates, dtype=np.float64)
    is_wall, wall_source = wall_mask(
        sources.case_directory, velocities, len(coordinates)
    )
    interior = coordinates[~is_wall]

    tracking = json.loads(args.tracking_report.read_text(encoding="utf-8"))
    support_radius = float(tracking["tracking"]["support_radius_m"])

    rng = np.random.default_rng(args.seed)
    sample_count = min(args.sample_points, len(seeds))
    sample_index = rng.choice(len(seeds), sample_count, replace=False)
    sample = seeds[sample_index]
    source_tree = cKDTree(coordinates)
    support_distance, _ = source_tree.query(sample, workers=-1)

    sample_tree = cKDTree(sample)
    neighbor_distance, _ = sample_tree.query(sample, k=2, workers=-1)
    nearest = neighbor_distance[:, 1]
    unique_sample = np.unique(sample.astype(np.float32), axis=0)

    voxel_m = args.voxel_mm * 1e-3
    seed_voxels = voxel_statistics(seeds, voxel_m)
    raw_voxels = voxel_statistics(interior, voxel_m)
    extent = seeds.max(axis=0) - seeds.min(axis=0)
    axis_order = np.argsort(extent)[::-1]

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / "aorta_uniform_2m_seeding_qc.png"
    report_path = output_dir / "aorta_uniform_2m_seeding_report.json"
    plot_qc(seeds, interior, seed_voxels, raw_voxels, figure_path)

    checks = {
        "shape": seeds.shape[1] == 3 and len(seeds) > 0,
        "finite": bool(np.isfinite(seeds).all()),
        "sample_exact_duplicates": len(unique_sample) == sample_count,
        "sample_within_tracker_support": bool(
            np.all(support_distance <= support_radius)
        ),
    }
    passed = all(checks.values())
    report = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            "PASS_DESCRIPTIVE_UNIFORMITY_AUDIT"
            if passed
            else "FAIL_INTEGRITY_CHECK"
        ),
        "scope": (
            "Integrity and descriptive density audit for approximate local-spacing-cubed "
            "point-cloud seeding; no tetrahedral volume mesh is available."
        ),
        "input": {
            "points_hdf5": str(args.points.resolve()),
            "case": str(sources.case_directory),
            "data_layout": sources.data_layout,
            "coordinates": str(sources.coordinates),
            "coordinates_format": sources.coordinates_format,
            "coordinates_provenance": sources.coordinates_provenance,
            "velocities": str(sources.velocities),
            "particle_count": int(len(seeds)),
            "source_node_count": int(len(coordinates)),
            "interior_node_count": int(len(interior)),
            "wall_node_count": int(is_wall.sum()),
            "wall_node_definition": wall_source,
            "hdf5_attributes": seed_attributes,
        },
        "geometry": {
            "coordinate_min_m": seeds.min(axis=0).tolist(),
            "coordinate_max_m": seeds.max(axis=0).tolist(),
            "coordinate_extent_m": extent.tolist(),
            "largest_to_smallest_axis_order_zero_based": axis_order.tolist(),
        },
        "voxel_density": {
            "voxel_mm": args.voxel_mm,
            "raw_interior_nodes": raw_voxels,
            "uniform_seed_cloud": seed_voxels,
            "seed_to_raw_cv_ratio": float(
                seed_voxels["coefficient_of_variation"]
                / raw_voxels["coefficient_of_variation"]
            ),
            "interpretation": (
                "Occupied-voxel statistics include partial boundary voxels and are "
                "descriptive, not externally validated acceptance thresholds."
            ),
        },
        "sample_diagnostics": {
            "sample_count": sample_count,
            "rng_seed": args.seed,
            "tracker_support_radius_m": support_radius,
            "maximum_distance_to_source_node_m": float(support_distance.max()),
            "outside_tracker_support_fraction": float(
                np.mean(support_distance > support_radius)
            ),
            "exact_duplicate_fraction": float(1 - len(unique_sample) / sample_count),
            "nearest_neighbor_p05_m": float(np.percentile(nearest, 5)),
            "nearest_neighbor_median_m": float(np.median(nearest)),
            "nearest_neighbor_p95_m": float(np.percentile(nearest, 95)),
        },
        "checks": checks,
        "outputs": {
            "report": str(report_path),
            "qc_figure": str(figure_path),
        },
        "limitations": [
            "No tetrahedral connectivity is available, so exact cell-volume seeding cannot be verified.",
            "Wall nodes inferred from nonzero wall shear are not a substitute for an explicit closed wall surface.",
            "Tracker support is a numerical interpolation criterion, not proof of physical boundary containment.",
        ],
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(report_path), "figure": str(figure_path)}, indent=2))


if __name__ == "__main__":
    main()
