"""Read-only source/support audit before dp150 wall-shear or pressure analysis.

This does not estimate WSS or pressure. It records which native solver fields
exist and why their original operating condition cannot silently be equated to
the periodically closed, retimed MRI motion. Outputs are a new atomic JSON.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pyvista as pv
from scipy.ndimage import binary_erosion


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = (ROOT / "outputs/dp150_mri_pipeline/koma"
              / "simulation_motion_1m_scanner_matched_venc5_seam_safe_v1")
DEFAULT_RECON = (PRODUCTION / "final_image_analysis/support_constrained_phase"
                 / "dp150_seam_safe_support_phase.h5")
DEFAULT_QUALITY = PRODUCTION / "final_image_analysis/quality_audit_e023/quality_metrics.json"


def artifact(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


def support_audit(support: np.ndarray, direct: np.ndarray) -> dict:
    support, direct = np.asarray(support, bool), np.asarray(direct, bool)
    if support.ndim != 3 or support.shape != direct.shape or not support.any():
        raise ValueError("Nonempty matching 3D support/direct masks are required.")
    if np.any(direct & ~support):
        raise ValueError("Direct mask must be contained in support.")
    eroded = binary_erosion(support)  # six axial neighbors: exactly E-023
    boundary = support & ~eroded
    direct_stencil = binary_erosion(direct)

    def summarize(mask):
        count = int(np.count_nonzero(mask))
        direct_count = int(np.count_nonzero(direct & mask))
        return {"support_voxels": count, "direct_voxels": direct_count,
                "regularized_voxels": count - direct_count,
                "direct_fraction": direct_count / count if count else None}

    return {
        "overall": summarize(support),
        "boundary_six_connected_one_voxel": summarize(boundary),
        "interior_six_connected_one_voxel": summarize(eroded),
        "all_direct_center_and_six_neighbor_gradient_stencils": int(np.count_nonzero(direct_stencil)),
        "all_direct_gradient_stencil_fraction_of_support": float(np.count_nonzero(direct_stencil) / np.count_nonzero(support)),
        "boundary_voxels_with_complete_in_lumen_central_stencil": int(np.count_nonzero(boundary & eroded)),
        "boundary_definition": "support minus one six-connected binary erosion; matches E-023",
        "gradient_interpretation": (
            "Central interior gradients do not estimate wall gradients. A wall fit, "
            "its distances/order, wall location, normals and regularization must be "
            "specified and validated separately; zero-filling outside lumen is not permitted."
        ),
    }


def geometry_audit(points: np.ndarray, voxel_xyz_mm: np.ndarray) -> dict:
    points = np.asarray(points, float)
    voxel = np.asarray(voxel_xyz_mm, float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError("Finite wall points with shape (N,3) required.")
    if not len(points) or voxel.shape != (3,) or np.any(voxel <= 0) or not np.all(np.isfinite(voxel)):
        raise ValueError("Nonempty wall and positive finite voxel sizes required.")
    center_xy = (points[:, :2].max(axis=0) + points[:, :2].min(axis=0)) / 2
    if np.max(np.abs(center_xy)) > 1e-6:
        raise ValueError("dp150 source radial axis is not centered at x=y=0.")
    radii_mm = np.linalg.norm(points[:, :2], axis=1) * 1000
    throat = int(np.argmin(radii_mm))
    radius_min, radius_max = float(radii_mm[throat]), float(np.max(radii_mm))
    return {
        "source_axis": "z; radius measured from source x=y=0",
        "wall_vertex_min_radius_mm": radius_min,
        "wall_vertex_max_radius_mm": radius_max,
        "minimum_radius_source_z_mm": float(points[throat, 2] * 1000),
        "minimum_diameter_from_wall_vertices_mm": 2 * radius_min,
        "maximum_diameter_from_wall_vertices_mm": 2 * radius_max,
        "voxel_size_mm_mri_readout_phase_slice": voxel.tolist(),
        "throat_diameter_samples_phase_slice": (2 * radius_min / voxel[1:]).tolist(),
        "throat_radius_samples_phase_slice": (radius_min / voxel[1:]).tolist(),
        "full_diameter_samples_phase_slice": (2 * radius_max / voxel[1:]).tolist(),
        "sampling_interpretation": (
            "Geometry-to-grid sampling ratios, not independent resolution elements or "
            "a universal WSS acceptance threshold. Thin wall-normal layers can be much "
            "less resolved than the complete vessel diameter."
        ),
    }


def source_field_inventory(path: Path) -> dict:
    fields = {}
    with h5py.File(path, "r") as h5:
        for label, dataset in {
            "cell_pressure": "results/1/phase-1/cells/SV_P/1",
            "face_pressure": "results/1/phase-1/faces/SV_P/1",
            "face_wall_shear": "results/1/phase-1/faces/SV_WALL_SHEAR/1",
            "cell_wall_distance": "results/1/phase-1/cells/SV_WALL_DIST/1",
        }.items():
            record = {"dataset": dataset, "present": dataset in h5}
            if dataset in h5:
                record.update({"shape": list(h5[dataset].shape), "dtype": str(h5[dataset].dtype)})
            fields[label] = record
    return {"scope": "One representative original Fluent frame; not a full-cycle export/mapping verification.",
            "fields": fields,
            "mapping_status": "Original face/cell IDs, units and shear-vector convention require explicit registration before quantitative reuse."}


def retiming_constraints(velocity_scale: float) -> dict:
    if not np.isfinite(velocity_scale) or velocity_scale <= 0:
        raise ValueError("Velocity scale must be finite and positive.")
    return {
        "velocity_time_scale_a": float(velocity_scale),
        "inertial_acceleration_scale_a_squared": float(velocity_scale**2),
        "viscous_acceleration_scale_same_viscosity_a": float(velocity_scale),
        "original_pressure_is_matched_reference": bool(np.isclose(velocity_scale, 1, rtol=0, atol=1e-12)),
        "pressure_scaling_applied": False,
        "explanation": (
            "For u_new(x,t)=a*u_old(x,a*t) at fixed geometry and fluid properties, "
            "unsteady and convective acceleration scale as a^2 while viscous "
            "acceleration scales as a. Thus multiplying original pressure by a^2 "
            "does not generally satisfy the same momentum equation. The synthetic "
            "tail blend adds another departure from the original solver state."
        ),
        "wall_shear_qualification": (
            "At fixed wall and viscosity, a kinematic shear estimate from a spatial "
            "velocity gradient scales linearly with a. This does not equate the "
            "retimed motion to a new Navier-Stokes solution. Original solver wall "
            "stress and gradients of converted wall-zeroed nodal velocity also use "
            "different discretizations; they cannot silently be substituted."
        ),
    }


def build_report(wall: Path, reconstruction: Path, quality: Path, source_frame: Path,
                 converted_frame: Path, flow_comparison: Path) -> dict:
    wall_mesh = pv.read(wall)
    converted_mesh = pv.read(converted_frame)
    quality_data = json.loads(quality.read_text(encoding="utf-8"))
    flow_data = json.loads(flow_comparison.read_text(encoding="utf-8"))
    with h5py.File(reconstruction, "r") as h5:
        coverage = support_audit(h5["support_mask"][...], h5["direct_mask"][...])
        geometry = geometry_audit(wall_mesh.points, h5.attrs["voxel_size_mm_readout_phase_slice"])
        sigma_mm = float(h5.attrs["gaussian_sigma_mm"])
    expected_boundary = next(item for item in quality_data["support_coverage"]["regions"]
                             if item["region"] == "One-voxel boundary")
    actual_boundary = coverage["boundary_six_connected_one_voxel"]
    for key in ("support_voxels", "direct_voxels", "regularized_voxels"):
        if actual_boundary[key] != expected_boundary[key]:
            raise ValueError(f"Recomputed E-023 boundary {key} disagrees with report.")
    scale = float(flow_data["timing"]["net_source_to_final_velocity_scale"])
    inputs = {name: artifact(path) for name, path in {
        "wall_surface": wall, "accepted_reconstruction": reconstruction,
        "e023_quality_metrics": quality, "representative_original_fluent_frame": source_frame,
        "representative_converted_velocity_frame": converted_frame,
        "e024_retiming_report": flow_comparison, "implementation": Path(__file__),
    }.items()}
    return {
        "schema_version": 1, "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETE_FEASIBILITY_AUDIT_NO_WALL_OR_PRESSURE_ESTIMATOR_APPLIED",
        "evidence_class": "local_source_inventory_and_internal_sampling_audit",
        "geometry": {**geometry, "surface_points": wall_mesh.n_points, "surface_triangles": wall_mesh.n_cells},
        "support": coverage, "e023_boundary_counts_reproduced": True,
        "regularization": {"gaussian_sigma_mm": sigma_mm, "gaussian_fwhm_mm": sigma_mm * np.sqrt(8 * np.log(2))},
        "native_solver_fields": source_field_inventory(source_frame),
        "converted_frame_fields": {"point_arrays": list(converted_mesh.point_data),
                                    "cell_arrays": list(converted_mesh.cell_data)},
        "retiming": retiming_constraints(scale),
        "wall_shear": {
            "status": "CONDITIONAL_EXPLORATORY_WALL_ESTIMATOR_REQUIRED",
            "available": "Connected wall geometry, original solver wall-shear array in inspected native frame, and reconstructed lumen velocity.",
            "not_claimed": "Validated WSS, TAWSS or OSI accuracy; MRI wall/background material realism.",
            "required_before_quantitative_comparison": [
                "Register original Fluent wall-face IDs, vector conventions and complete-cycle samples to the exported wall.",
                "Specify wall-normal fit distances/order and no-slip treatment; validate against analytical controls and matched-resolution CFD.",
                "Quantify direct-only versus regularized wall-fit support and fit-scale sensitivity at each wall location.",
                "Keep original solver stress separate from kinematic stress derived from the tail-closed, retimed input field.",
                "Report full-cycle TAWSS/OSI as including the synthetic closure; an excluded-tail interval is not a complete cycle.",
            ],
        },
        "pressure_drop": {
            "status": "ORIGINAL_PRESSURE_PRESENT_BUT_NOT_A_MATCHED_RETIMED_REFERENCE",
            "available": "Original Fluent cell/face pressure in inspected native frame; not exported in the converted velocity VTU.",
            "without_new_cfd": (
                "An explicitly chosen pressure estimator can be applied to both the retimed CFD velocity and MRI velocity as an estimator-consistency comparison. "
                "It would not establish agreement with a pressure solution for the transformed motion. No estimator is selected or executed here."
            ),
            "for_physical_pressure_accuracy": (
                "Use a solver-consistent reference at the intended waveform/period/fluid properties, or simulate MRI from the unchanged original CFD state; "
                "validate any pressure-estimation boundary/gauge assumptions. No scalar pressure rescaling is justified automatically."
            ),
        },
        "inputs": inputs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/dp150_hemodynamic_comparison/e025_v1/input_audit/wall_pressure_feasibility.json")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Preserving existing audit: {output}")
    report = build_report(
        ROOT / "outputs/dp150_volume_pipeline/flow/wall.vtp", DEFAULT_RECON, DEFAULT_QUALITY,
        ROOT.parent / "external_inputs/dp150_fluent/FFF-18-00018.dat.h5",
        ROOT / "outputs/dp150_volume_pipeline/flow/frames/dp150_00018.vtu",
        PRODUCTION / "final_image_analysis/cfd_koma_flow_waveform_e024/dp150_cfd_koma_flow_waveform_report.json",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    with partial.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    os.replace(partial, output)
    print(output)
    print(json.dumps({"geometry": report["geometry"], "support": report["support"],
                      "wall_status": report["wall_shear"]["status"],
                      "pressure_status": report["pressure_drop"]["status"]}, indent=2))


if __name__ == "__main__":
    main()
