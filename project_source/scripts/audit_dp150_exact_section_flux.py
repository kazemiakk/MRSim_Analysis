"""Exact triangle integration of converted linear-tet CFD section flux.

This separates coarse plane quadrature from axial variation already present in
the converted nodal field. It is not an audit of Fluent face-flux conservation.
No fitted correction or replacement of existing comparison results is made.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pyvista as pv

from scripts.dp150_hemodynamic_reference import (
    MRIGrid, load_periodic_reference, transformed_nodal_velocity,
)


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/dp150_hemodynamic_comparison/e025_v1"


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def triangle_flux(points_m: np.ndarray, triangles: np.ndarray,
                  normal_velocity_m_s: np.ndarray) -> dict:
    """Exact signed flux of linear nodal normal velocity over planar triangles.

    Triangle winding does not change the prescribed normal-velocity sign. This
    routine does not approximate forward/reverse partitions across a triangle
    containing a velocity zero crossing.
    """
    points = np.asarray(points_m, dtype=float)
    triangles = np.asarray(triangles)
    velocity = np.asarray(normal_velocity_m_s, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError("Finite (N,3) points required.")
    if triangles.ndim != 2 or triangles.shape[1] != 3 or not len(triangles):
        raise ValueError("Nonempty (M,3) triangles required.")
    if not np.issubdtype(triangles.dtype, np.integer) or np.any(triangles < 0) or np.any(triangles >= len(points)):
        raise ValueError("Triangle indices must be valid integers.")
    if velocity.shape != (len(points),) or not np.all(np.isfinite(velocity)):
        raise ValueError("One finite normal velocity per point is required.")
    xyz = points[triangles]
    area = np.linalg.norm(np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]), axis=1) * .5
    if np.any(area <= 0):
        raise ValueError("Degenerate triangles cannot define section flux.")
    return {"area_m2": float(area.sum()),
            "net_m3_s": float(np.sum(area * velocity[triangles].mean(axis=1))),
            "triangle_count": len(triangles)}


def exact_mesh_section(mesh: pv.UnstructuredGrid, axial_velocity_m_s: np.ndarray,
                       source_z_m: float) -> dict:
    """Slice source +z plane, keeping linearly interpolated MRI-X velocity."""
    velocity = np.asarray(axial_velocity_m_s, dtype=float)
    if velocity.shape != (mesh.n_points,) or not np.isfinite(source_z_m):
        raise ValueError("Finite plane location and matching nodal velocity required.")
    mesh.point_data["E025_MRI_X_velocity_m_s"] = velocity
    section = mesh.slice(normal=(0, 0, 1), origin=(0, 0, float(source_z_m)),
                         generate_triangles=True).triangulate()
    faces = np.asarray(section.faces)
    if not faces.size or faces.size % 4 or not np.all(faces.reshape(-1, 4)[:, 0] == 3):
        raise ValueError("Slice did not produce a nonempty triangle section.")
    return triangle_flux(section.points, faces.reshape(-1, 4)[:, 1:],
                         section.point_data["E025_MRI_X_velocity_m_s"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=RUN / "results_v2/report.json")
    parser.add_argument("--output-dir", type=Path, default=RUN / "validation")
    args = parser.parse_args()
    report_path = args.report.resolve()
    output_dir = args.output_dir.resolve()
    json_path, csv_path = output_dir / "exact_section_flux.json", output_dir / "exact_section_flux.csv"
    for path in (json_path, csv_path):
        if path.exists():
            raise FileExistsError(f"Preserving existing output: {path}")
    started = time.monotonic()
    comparison = json.loads(report_path.read_text(encoding="utf-8"))
    grid = MRIGrid(**{key: tuple(value) for key, value in comparison["grid"].items()})
    timing = comparison["timing"]
    verified = []
    for record in comparison["inputs"]:
        path = ROOT / record["path"]
        if "flow_periodic" in str(path):
            if path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
                raise ValueError(f"Periodic input provenance mismatch: {path}")
            verified.append(record)
    frame_count = sum(str(record["path"]).endswith(".vtu") for record in verified)
    if frame_count != 73:
        raise ValueError(f"Expected 73 hash-verified periodic frames, found {frame_count}.")
    flow = load_periodic_reference(ROOT / "outputs/dp150_mri_pipeline/flow_periodic/dp150_periodic.pvd")
    samples = comparison["section_values"]
    rows = []
    for cardiac_bin in range(1, 16):
        time_s = (cardiac_bin - .5) * timing["effective_motion_period_s"] / 15
        velocity = transformed_nodal_velocity(flow, grid, time_s, timing["effective_motion_period_s"])
        for name, specification in comparison["flow_summary"].items():
            exact = exact_mesh_section(flow.active_mesh, velocity[:, 0], specification["source_z_mm"] / 1000)
            lookup = {row["dataset"]: row for row in samples if row["bin"] == cardiac_bin and row["section"] == name}
            row = {"bin": cardiac_bin, "time_s": time_s, "section": name,
                   "source_z_mm": specification["source_z_mm"], **exact}
            for dataset in ("cfd_plane_full", "cfd_voxel_common"):
                candidate = lookup[dataset]
                row[f"{dataset}_net_m3_s"] = candidate["net_m3_s"]
                row[f"{dataset}_minus_exact_m3_s"] = candidate["net_m3_s"] - exact["net_m3_s"]
                row[f"{dataset}_area_m2"] = candidate["sampled_area_m2"]
            rows.append(row)
        print(f"Exact converted-field sections: bin {cardiac_bin}/15", flush=True)
    summaries = {}
    for name in comparison["flow_summary"]:
        subset = [row for row in rows if row["section"] == name]
        exact_q = np.array([row["net_m3_s"] for row in subset])
        peak_abs = float(np.max(np.abs(exact_q)))
        summaries[name] = {"area_m2": subset[0]["area_m2"],
                           "cycle_volume_ml": float(exact_q.sum() * timing["bin_spacing_s"] * 1e6)}
        for dataset in ("cfd_plane_full", "cfd_voxel_common"):
            residual = np.array([row[f"{dataset}_minus_exact_m3_s"] for row in subset])
            summaries[name][dataset] = {
                "max_absolute_difference_ml_s": float(np.max(np.abs(residual)) * 1e6),
                "rmse_ml_s": float(np.sqrt(np.mean(residual**2)) * 1e6),
                "max_absolute_difference_fraction_of_exact_section_peak": float(np.max(np.abs(residual)) / peak_abs),
                "area_difference_fraction": float(subset[0][f"{dataset}_area_m2"] / subset[0]["area_m2"] - 1),
            }
    variation = []
    for cardiac_bin in range(1, 16):
        q = np.array([row["net_m3_s"] for row in rows if row["bin"] == cardiac_bin])
        variation.append({"bin": cardiac_bin, "minimum_ml_s": float(q.min() * 1e6),
                          "maximum_ml_s": float(q.max() * 1e6), "range_ml_s": float(np.ptp(q) * 1e6),
                          "range_over_mean_absolute": float(np.ptp(q) / np.mean(np.abs(q)))})
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_partial = csv_path.with_suffix(".csv.partial")
    with csv_partial.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(csv_partial, csv_path)
    result = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETE_EXACT_LINEAR_TET_SECTION_FLUX_AUDIT",
        "evidence_class": "converted_CFD_kinematic_reference_numerical_check_not_solver_mass_balance",
        "method": "Slice conditioned tetrahedral mesh at five fixed source-z planes; integrate linear MRI-X velocity as triangle area times mean vertex velocity.",
        "quadrature": "Triangle net flux is exact for the piecewise-linear nodal field up to floating-point slicing error. No voxel midpoint quadrature is used for this reference.",
        "limitations": [
            "Converted nodal velocity includes cell-to-point interpolation and imposed wall zeros; conditioning splits original wedge cells into tetrahedra.",
            "Axial flux variation here is not sufficient to diagnose original Fluent solver mass conservation; original face flux must be checked separately.",
            "cfd_voxel_common additionally includes finite readout thickness and common MRI support, so its difference is not a pure plane-quadrature error.",
            "All fields retain documented periodic closure and retiming; no amplitude fit, correction or production-data alteration is applied.",
        ],
        "sections": summaries, "exact_between_plane_variation": variation,
        "maximum_plane_quadrature_difference_ml_s": max(item["cfd_plane_full"]["max_absolute_difference_ml_s"] for item in summaries.values()),
        "maximum_exact_between_plane_range_ml_s": max(item["range_ml_s"] for item in variation),
        "bin4_exact_ml_s": {row["section"]: row["net_m3_s"] * 1e6 for row in rows if row["bin"] == 4},
        "input_report": {"path": str(report_path), "sha256": digest(report_path)},
        "verified_periodic_inputs": verified,
        "implementation": {"path": str(Path(__file__).resolve()), "sha256": digest(Path(__file__))},
        "reference_helper": {"path": str(ROOT / "scripts/dp150_hemodynamic_reference.py"), "sha256": digest(ROOT / "scripts/dp150_hemodynamic_reference.py")},
        "csv": {"path": str(csv_path), "sha256": digest(csv_path)},
        "rows": rows, "runtime_seconds": time.monotonic() - started,
    }
    json_partial = json_path.with_suffix(".json.partial")
    with json_partial.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    os.replace(json_partial, json_path)
    print(json.dumps({key: result[key] for key in ("bin4_exact_ml_s", "maximum_plane_quadrature_difference_ml_s",
                                                 "maximum_exact_between_plane_range_ml_s", "runtime_seconds")}, indent=2))


if __name__ == "__main__":
    main()
