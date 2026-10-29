r"""Audit the nine dp150 MRI-simulation closure items and write JSON/Markdown."""

from __future__ import annotations

import json
import tomllib
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_ROOT = Path("outputs/dp150_mri_pipeline")


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_toml(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as stream:
        return tomllib.load(stream)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def close(a: float, b: float, tolerance: float = 1e-9) -> bool:
    return abs(float(a) - float(b)) <= tolerance


def main() -> None:
    root = DEFAULT_ROOT.resolve()
    koma = root / "koma"
    periodic = read_json(root / "periodic_flow_report.json")
    static = read_json(root / "static_2m_report.json")
    convergence = read_json(root / "tracking_convergence_report.json")
    motion = read_json(root / "motion_1m_report.json")
    animation = read_json(root / "motion_1m_animation_report.json")
    static_phantom = read_toml(koma / "static_phantom_report.toml")
    motion_phantom = read_toml(koma / "motion_phantom_report.toml")
    static_sim = read_toml(
        koma / "simulation_static_2m_ref" / "simulation_report.toml"
    )
    motion_sim = read_toml(
        koma / "simulation_motion_1m" / "simulation_report.toml"
    )
    sign = read_toml(koma / "pc_velocity_sign_report.toml")
    batching = read_toml(koma / "batched_signal_validation.toml")

    fov = [0.512, 0.128, 0.128]
    matrix = [256, 64, 64]
    encodings = ["REF", "X", "Y", "Z"]
    sequence = motion_sim["sequence"]
    sequence_rr = [sequence[name]["RR_s"] for name in encodings]
    sequence_venc = [sequence[name]["Venc_m_per_s"] for name in encodings]
    raw_velocity = motion["quality_control"]["velocity_encoding"]
    component_stats = motion_sim["velocity_qc"]["component_statistics"]

    checks = []

    passed = (
        static["acceptance"]["passed"]
        and static["seeding"]["count"] == 2_000_000
        and motion["run"]["particle_count"] == 1_000_000
        and motion["quality_control"]["acceptance"]["passed"]
    )
    checks.append(
        {
            "point": 1,
            "title": "Production particle counts and uniform seeding",
            "status": "PASS" if passed else "FAIL",
            "evidence": {
                "static_particles": static["seeding"]["count"],
                "moving_particles": motion["run"]["particle_count"],
                "static_axial_total_variation": static["seeding"][
                    "axial_total_variation"
                ],
                "moving_tracking_qc": motion["quality_control"]["acceptance"],
            },
        }
    )

    static_p95 = static_sim["velocity_qc"]["static_zero_velocity_acceptance"]
    passed = (
        static_phantom["status"] == "PASS"
        and static_sim["status"] == "PASS"
        and static_sim["particle_count"] == 2_000_000
        and static_p95["passed"]
    )
    checks.append(
        {
            "point": 2,
            "title": "Static Koma simulation and reconstruction litmus test",
            "status": "PASS" if passed else "FAIL",
            "evidence": {
                "particle_count": static_sim["particle_count"],
                "maximum_component_p95_absolute_m_per_s": static_p95[
                    "maximum_component_p95_absolute_m_per_s"
                ],
                "acceptance_limit_m_per_s": static_p95["limit_m_per_s"],
                "reconstruction_file": static_sim["reconstruction_file"],
            },
        }
    )

    transform = static["coordinate_transform"]
    passed = (
        transform["mri_axes_from_source"] == ["z", "x", "y"]
        and transform["fov_m"] == fov
        and transform["matrix"] == matrix
        and min(transform["half_margin_mm"]) > 0
        and all(sequence[name]["FOV_m"] == fov for name in encodings)
        and all(sequence[name]["matrix"] == matrix for name in encodings)
    )
    checks.append(
        {
            "point": 3,
            "title": "FOV, resolution, and largest-to-smallest axis order",
            "status": "PASS" if passed else "FAIL",
            "evidence": {
                "source_geometry_extent_mm": [
                    value * 1e3 for value in transform["source_extents_m"]
                ],
                "mri_axes_from_source": transform["mri_axes_from_source"],
                "fov_m": transform["fov_m"],
                "matrix": transform["matrix"],
                "resolution_mm": transform["resolution_mm"],
                "minimum_half_margin_mm": min(transform["half_margin_mm"]),
            },
        }
    )

    passed = (
        periodic["validation"]["passed"]
        and periodic["validation"]["endpoint_exactly_equal"]
        and periodic["validation"]["endpoint_component_rmse_m_per_s"] == 0
    )
    checks.append(
        {
            "point": 4,
            "title": "One-cycle periodic endpoint closure",
            "status": "PASS_CONTROLLED_INPUT" if passed else "FAIL",
            "evidence": {
                "endpoint_component_rmse_m_per_s": periodic["validation"][
                    "endpoint_component_rmse_m_per_s"
                ],
                "blend_start_s": periodic["closure_method"]["blend_start_s"],
                "preserved_source_through_s": periodic["closure_method"][
                    "preserved_source_through_s"
                ],
                "scope": periodic["closure_method"]["scientific_scope"],
            },
        }
    )

    passed = (
        all(close(value, sequence_rr[0]) for value in sequence_rr)
        and close(motion_phantom["source_cycle_s"], 0.72)
        # The phantom stores RR as Float32, so compare at sub-microsecond precision.
        and close(motion_phantom["realized_sequence_rr_s"], sequence_rr[0], 1e-7)
    )
    checks.append(
        {
            "point": 5,
            "title": "Source-cycle to realizable sequence RR mapping",
            "status": "PASS" if passed else "FAIL",
            "evidence": {
                "source_cycle_s": motion_phantom["source_cycle_s"],
                "realized_sequence_rr_s": motion_phantom[
                    "realized_sequence_rr_s"
                ],
                "all_encoding_rr_s": sequence_rr,
                "mapping": motion_phantom["motion_time_mapping"],
            },
        }
    )

    ensemble_pass = convergence["comparison"]["ensemble_acceptance"]["passed"]
    selected_dt = convergence["decision"]["selected_full_tracking_dt_s"]
    passed = ensemble_pass and close(selected_dt, 0.0005, 1e-12)
    checks.append(
        {
            "point": 6,
            "title": "Tracking timestep convergence and CFD temporal resolution",
            "status": "ADDRESSED_WITH_SOURCE_LIMITATION" if passed else "FAIL",
            "evidence": {
                "selected_tracking_dt_s": selected_dt,
                "ensemble_distribution_convergence_passed": ensemble_pass,
                "individual_trajectory_convergence_passed": convergence["decision"][
                    "individual_trajectory_convergence_passed"
                ],
                "maximum_axial_total_variation": convergence["comparison"][
                    "ensemble_distribution"
                ]["maximum_axial_total_variation"],
                "maximum_radial_total_variation": convergence["comparison"][
                    "ensemble_distribution"
                ]["maximum_radial_total_variation"],
                "source_temporal_limitation": convergence["decision"][
                    "source_temporal_limitation"
                ],
            },
        }
    )

    passed = (
        static_phantom["coordinate_space"]
        == "MRI_centered_largest_to_smallest"
        and motion_phantom["coordinate_space"]
        == "MRI_centered_largest_to_smallest"
        and min(static_phantom["minimum_half_margin_m"]) > 0
        and min(motion_phantom["minimum_half_margin_m"]) > 0
    )
    checks.append(
        {
            "point": 7,
            "title": "MRI origin at the phantom center",
            "status": "PASS" if passed else "FAIL",
            "evidence": {
                "source_geometry_center_m": transform["source_center_m"],
                "static_minimum_half_margin_m": min(
                    static_phantom["minimum_half_margin_m"]
                ),
                "motion_minimum_half_margin_m": min(
                    motion_phantom["minimum_half_margin_m"]
                ),
            },
        }
    )

    venc = sequence_venc[0]
    passed = (
        all(close(value, venc) for value in sequence_venc)
        and periodic["validation"]["maximum_corrected_point_speed_m_per_s"] < venc
        and sum(raw_velocity["component_exceedance_count"]) == 0
    )
    checks.append(
        {
            "point": 8,
            "title": "VENC coverage",
            "status": "PASS" if passed else "FAIL",
            "evidence": {
                "venc_m_per_s": venc,
                "maximum_corrected_cfd_speed_m_per_s": periodic["validation"][
                    "maximum_corrected_point_speed_m_per_s"
                ],
                "maximum_tracked_component_m_per_s": max(
                    raw_velocity["maximum_absolute_component_m_per_s"]
                ),
                "tracked_component_exceedance_count": sum(
                    raw_velocity["component_exceedance_count"]
                ),
            },
        }
    )

    motion_x_p95 = component_stats["X"]["p95_absolute_m_per_s"]
    motion_lateral_p95 = max(
        component_stats["Y"]["p95_absolute_m_per_s"],
        component_stats["Z"]["p95_absolute_m_per_s"],
    )
    passed = (
        motion_sim["status"] == "PASS"
        and motion_sim["particle_count"] == 1_000_000
        and motion_sim["encodings"] == encodings
        and Path(motion_sim["reconstruction_file"]).is_file()
        and sign["status"] == "PASS"
        and batching["status"] == "PASS"
        and motion_x_p95 > motion_lateral_p95
        and animation["verification"]["passed"]
        and animation["render"]["tracked_particles"] == 1_000_000
    )
    checks.append(
        {
            "point": 9,
            "title": "REF/X/Y/Z phase-contrast reconstruction and motion output",
            "status": "PASS" if passed else "FAIL",
            "evidence": {
                "encodings": motion_sim["encodings"],
                "velocity_formula": motion_sim["velocity_qc"]["phase_formula"],
                "sign_validation": sign["status"],
                "batching_validation": batching["status"],
                "readout_p95_m_per_s": motion_x_p95,
                "maximum_lateral_p95_m_per_s": motion_lateral_p95,
                "reconstruction_file": motion_sim["reconstruction_file"],
                "gif": animation["outputs"]["gif"],
                "gif_frame_count": animation["verification"]["frame_count"],
            },
        }
    )

    failed = [item for item in checks if item["status"] == "FAIL"]
    report = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "overall_status": (
            "PASS_WITH_SOURCE_TEMPORAL_LIMITATION" if not failed else "FAIL"
        ),
        "fully_closed_point_count": sum(
            item["status"] in ("PASS", "PASS_CONTROLLED_INPUT")
            for item in checks
        ),
        "source_limited_point_count": sum(
            item["status"] == "ADDRESSED_WITH_SOURCE_LIMITATION"
            for item in checks
        ),
        "points": checks,
    }
    json_path = root / "nine_point_closure_report.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# dp150 MRI Nine-Point Closure",
        "",
        f"Overall status: **{report['overall_status']}**",
        "",
        "| # | Item | Status |",
        "|---:|---|---|",
    ]
    lines.extend(
        f"| {item['point']} | {item['title']} | {item['status']} |"
        for item in checks
    )
    lines.extend(
        [
            "",
            "## Scientific Boundary",
            "",
            (
                "All nine items were addressed, but point 6 cannot be fully closed "
                "from the available data: the CFD velocity fields were saved every "
                "10 ms. The 0.5 ms RK4 run uses cubic temporal interpolation and "
                "passed ensemble-density convergence against 1 ms, but this does "
                "not create or validate a CFD solution saved at 0.5-1 ms."
            ),
            "",
            (
                "The final 80 ms periodic-tail blend is a controlled MRI simulation "
                "input and must not be described as unmodified CFD."
            ),
            "",
            "Detailed numerical evidence is stored in `nine_point_closure_report.json`.",
        ]
    )
    markdown_path = root / "NINE_POINT_CLOSURE.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    require(not failed, f"Closure audit failed points: {[x['point'] for x in failed]}")
    print(f"Closure audit: {report['overall_status']}")
    print(f"JSON: {json_path}")
    print(f"Markdown: {markdown_path}")


if __name__ == "__main__":
    main()
