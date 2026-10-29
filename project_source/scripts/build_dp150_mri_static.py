r"""Generate the corrected two-million-particle dp150 static MRI input.

Seeds are drawn with MRSimTracks 0.2.1 from the periodic-flow volume, checked
against the tetrahedral locator, and transformed from CFD ``(x, y, z)`` into
MRI largest-to-smallest axis order. The geometry bounding-box centre, rather
than random sample extrema, defines the MRI origin.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pyvista as pv


DEFAULT_ROOT = Path("outputs/dp150_mri_pipeline")
EXPECTED_VERSION = "0.2.1"
ACTIVE_KEY = "Velocity"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--count", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--axial-bins", type=int, default=64)
    parser.add_argument("--fov", type=float, nargs=3, default=(0.512, 0.128, 0.128))
    parser.add_argument("--matrix", type=int, nargs=3, default=(256, 64, 64))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def package_provenance() -> dict[str, str]:
    version = importlib.metadata.version("mrsimtracks")
    import mrsimtracks

    module = Path(mrsimtracks.__file__).resolve()
    if version != EXPECTED_VERSION:
        raise RuntimeError(f"Expected MRSimTracks {EXPECTED_VERSION}, found {version}")
    if "site-packages" not in {part.lower() for part in module.parts}:
        raise RuntimeError("Run with Python -I; the checkout is shadowing the wheel")
    return {"distribution_version": version, "module_path": str(module)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_seed_mesh(mesh: pv.DataSet, count: int, rng: np.random.Generator) -> np.ndarray:
    from mrsimtracks.seeding import seed_mesh

    chunks: list[np.ndarray] = []
    total = 0
    for _ in range(10):
        if total >= count:
            break
        requested = max(10_000, count - total)
        points = np.asarray(seed_mesh(mesh, requested, rng=rng), dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3 or not points.size:
            raise RuntimeError("MRSimTracks seed_mesh returned invalid points")
        chunks.append(points)
        total += points.shape[0]
    if total < count:
        raise RuntimeError(f"Generated {total:,} seeds, fewer than {count:,}")
    points = np.vstack(chunks)
    if points.shape[0] > count:
        keep = rng.choice(points.shape[0], count, replace=False)
        points = points[keep]
    return np.ascontiguousarray(points, dtype=np.float32)


def tet_volume_distribution(
    mesh: pv.UnstructuredGrid, edges: np.ndarray
) -> np.ndarray:
    if not np.all(np.asarray(mesh.celltypes) == pv.CellType.TETRA):
        raise ValueError("Expected an all-tetrahedral conditioned mesh")
    conn = np.asarray(mesh.cells).reshape(-1, 5)[:, 1:]
    points = np.asarray(mesh.points, dtype=np.float64)
    vertices = points[conn]
    e0 = vertices[:, 0] - vertices[:, 3]
    e1 = vertices[:, 1] - vertices[:, 3]
    e2 = vertices[:, 2] - vertices[:, 3]
    volumes = np.abs(np.einsum("ij,ij->i", e0, np.cross(e1, e2))) / 6.0
    centres = vertices[:, :, 2].mean(axis=1)
    hist, _ = np.histogram(centres, bins=edges, weights=volumes)
    return hist / hist.sum()


def radial_qc(seeds: np.ndarray, wall: pv.PolyData) -> dict[str, float | int]:
    wall_points = np.asarray(wall.points, dtype=np.float64)
    wall_radius = np.linalg.norm(wall_points[:, :2], axis=1)
    global_radius = float(wall_radius.max())
    edges = np.linspace(float(wall.bounds[4]), float(wall.bounds[5]), 129)
    wall_bins = np.clip(
        np.searchsorted(edges, wall_points[:, 2], side="right") - 1,
        0,
        len(edges) - 2,
    )
    profile = np.full(len(edges) - 1, np.nan)
    for index in range(profile.size):
        values = wall_radius[wall_bins == index]
        if values.size:
            profile[index] = values.max()
    valid = np.isfinite(profile)
    profile = np.interp(np.arange(profile.size), np.flatnonzero(valid), profile[valid])
    seed_bins = np.clip(
        np.searchsorted(edges, seeds[:, 2], side="right") - 1,
        0,
        profile.size - 1,
    )
    nominal = profile[seed_bins] >= 0.98 * global_radius
    ratio = np.linalg.norm(seeds[nominal, :2], axis=1) / global_radius
    return {
        "nominal_section_seed_count": int(ratio.size),
        "median_r_over_R": float(np.median(ratio)),
        "fraction_gt_0_90": float(np.mean(ratio > 0.90)),
        "fraction_gt_0_95": float(np.mean(ratio > 0.95)),
        "expected_median_r_over_R": float(2**-0.5),
        "expected_fraction_gt_0_90": 0.19,
        "expected_fraction_gt_0_95": 0.0975,
    }


def main() -> None:
    args = parse_args()
    if args.count < 1 or args.axial_bins < 8:
        raise ValueError("count must be positive and axial-bins must be >= 8")
    fov = np.asarray(args.fov, dtype=np.float64)
    matrix = np.asarray(args.matrix, dtype=np.int64)
    if np.any(fov <= 0) or np.any(matrix <= 0):
        raise ValueError("FOV and matrix entries must be positive")
    resolution = fov / matrix
    if not np.allclose(resolution, resolution[0], rtol=0, atol=1e-12):
        raise ValueError("This controlled protocol requires isotropic resolution")

    provenance = package_provenance()
    import mrsimtracks as mt

    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    pvd_path = root / "flow_periodic" / "dp150_periodic.pvd"
    wall_path = root / "flow_periodic" / "wall.vtp"
    periodic_report_path = root / "periodic_flow_report.json"
    raw_path = root / f"dp150_uniform_seeds_{args.count // 1_000_000}m_raw.npy"
    h5_path = root / f"dp150_static_{args.count // 1_000_000}m_mri.h5"
    report_path = root / "static_2m_report.json"
    for path in (pvd_path, wall_path, periodic_report_path):
        if not path.exists():
            raise FileNotFoundError(path)
    for path in (raw_path, h5_path, report_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite to replace it")

    periodic_report = json.loads(periodic_report_path.read_text(encoding="utf-8"))
    if not periodic_report["validation"]["passed"]:
        raise RuntimeError("Periodic-flow input has not passed closure validation")
    print(f"Loading periodic flow with MRSimTracks {EXPECTED_VERSION} ...", flush=True)
    flow = mt.load_flow(
        pvd_path,
        active_key=ACTIVE_KEY,
        precision="f32",
        time_interp="linear",
        pbar=True,
    )
    mesh = flow.active_mesh
    bounds = np.asarray(mesh.bounds, dtype=np.float64).reshape(3, 2)
    raw_center = bounds.mean(axis=1)
    raw_extents = np.diff(bounds, axis=1).ravel()
    axis_order = np.argsort(-raw_extents, kind="stable")
    ordered_extents = raw_extents[axis_order]
    if not np.all(ordered_extents <= fov):
        raise ValueError(f"Ordered extents {ordered_extents} exceed FOV {fov}")

    print(f"Generating {args.count:,} corrected volume seeds ...", flush=True)
    rng = np.random.default_rng(args.seed)
    raw_seeds = exact_seed_mesh(mesh, args.count, rng)
    cells = flow._sampler.locate(raw_seeds, guess=None)
    valid_fraction = float(np.mean(cells >= 0))
    if valid_fraction != 1.0:
        raise RuntimeError(f"Only {valid_fraction:.9f} of seeds are inside the volume")
    np.save(raw_path, raw_seeds, allow_pickle=False)

    mri_seeds = np.ascontiguousarray(
        raw_seeds[:, axis_order] - raw_center[axis_order], dtype=np.float32
    )
    mri_geometry_bounds = np.column_stack(
        (-0.5 * ordered_extents, 0.5 * ordered_extents)
    )
    margins = 0.5 * (fov - ordered_extents)
    if np.any(np.abs(mri_seeds) > 0.5 * fov + 1e-7):
        raise RuntimeError("Centered static particles exceed the requested FOV")

    with h5py.File(h5_path, "w") as h5:
        h5.create_dataset(
            "position",
            data=mri_seeds[None, :, :],
            dtype=np.float32,
            chunks=(1, min(args.count, 65_536), 3),
        )
        h5.create_dataset(
            "reset",
            data=np.zeros((1, args.count), dtype=bool),
            chunks=(1, min(args.count, 65_536)),
        )
        h5.create_dataset("position_time_s", data=np.array([0.0], dtype=np.float64))
        h5.attrs.update(
            {
                "dt": 0.0,
                "n_steps": 1,
                "n_particles": args.count,
                "mrsimtracks_version": EXPECTED_VERSION,
                "coordinate_space": "MRI_centered_largest_to_smallest",
                "source_axis_order_zero_based": axis_order,
                "source_geometry_center_m": raw_center,
                "source_geometry_extents_m": raw_extents,
                "mri_geometry_extents_m": ordered_extents,
                "fov_m": fov,
                "matrix": matrix,
                "resolution_m": resolution,
                "rng_seed": args.seed,
                "static_no_motion": True,
                "periodic_flow_report_sha256": sha256(periodic_report_path),
            }
        )

    edges = np.linspace(bounds[2, 0], bounds[2, 1], args.axial_bins + 1)
    expected_axial = tet_volume_distribution(mesh, edges)
    observed_axial, _ = np.histogram(raw_seeds[:, 2], bins=edges)
    observed_axial = observed_axial / observed_axial.sum()
    axial_tv = float(0.5 * np.abs(observed_axial - expected_axial).sum())
    radial = radial_qc(raw_seeds, pv.read(wall_path))
    limits = {
        "axial_total_variation_max": 0.01,
        "radial_median_absolute_error_max": 0.01,
        "radial_tail_absolute_error_max": 0.015,
    }
    checks = {
        "all_inside": valid_fraction == 1.0,
        "fov_fit": bool(np.all(margins > 0)),
        "axial_uniformity": axial_tv <= limits["axial_total_variation_max"],
        "radial_median": abs(
            radial["median_r_over_R"] - radial["expected_median_r_over_R"]
        ) <= limits["radial_median_absolute_error_max"],
        "radial_gt_0_90": abs(
            radial["fraction_gt_0_90"] - radial["expected_fraction_gt_0_90"]
        ) <= limits["radial_tail_absolute_error_max"],
        "radial_gt_0_95": abs(
            radial["fraction_gt_0_95"] - radial["expected_fraction_gt_0_95"]
        ) <= limits["radial_tail_absolute_error_max"],
    }
    report = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {"python": sys.version, "platform": platform.platform()},
        "mrsimtracks": provenance,
        "input": {
            "pvd": str(pvd_path),
            "pvd_sha256": sha256(pvd_path),
            "periodic_flow_report": str(periodic_report_path),
        },
        "seeding": {
            "count": args.count,
            "rng_seed": args.seed,
            "valid_fraction": valid_fraction,
            "axial_total_variation": axial_tv,
            "radial": radial,
        },
        "coordinate_transform": {
            "source_axes": ["x", "y", "z"],
            "mri_source_axis_indices": axis_order.tolist(),
            "mri_axes_from_source": [["x", "y", "z"][index] for index in axis_order],
            "source_center_m": raw_center.tolist(),
            "source_extents_m": raw_extents.tolist(),
            "mri_geometry_bounds_m": mri_geometry_bounds.tolist(),
            "mri_geometry_extents_m": ordered_extents.tolist(),
            "fov_m": fov.tolist(),
            "matrix": matrix.tolist(),
            "resolution_mm": (resolution * 1e3).tolist(),
            "half_margin_mm": (margins * 1e3).tolist(),
        },
        "acceptance": {
            "passed": all(checks.values()),
            "checks": checks,
            "limits": limits,
        },
        "outputs": {
            "raw_seed_npy": str(raw_path),
            "raw_seed_sha256": sha256(raw_path),
            "static_hdf5": str(h5_path),
            "static_hdf5_sha256": sha256(h5_path),
            "static_hdf5_bytes": h5_path.stat().st_size,
        },
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Static 2M QC: {'PASS' if report['acceptance']['passed'] else 'FAIL'}")
    print(f"HDF5: {h5_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
