r"""Compare the dp150 Koma reconstruction with its CFD-driven inlet waveform.

This is an internal consistency analysis.  It does not compare against a
measured pump trace, PC-MRI, PIV, or another independent velocity reference.
The Fluent inlet monitor is transformed with the same periodic tail closure,
Catmull-Rom temporal interpolation, and net 0.720-s-to-Koma-period retiming
used to build the accepted motion simulation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = (
    ROOT
    / "outputs"
    / "dp150_mri_pipeline"
    / "koma"
    / "simulation_motion_1m_scanner_matched_venc5_seam_safe_v1"
)
DEFAULT_RECONSTRUCTION = (
    RUN_DIR
    / "final_image_analysis"
    / "support_constrained_phase"
    / "dp150_seam_safe_support_phase.h5"
)
DEFAULT_SUPPORT_REPORT = DEFAULT_RECONSTRUCTION.with_name(
    "dp150_seam_safe_support_phase_report.json"
)
DEFAULT_SIMULATION_REPORT = RUN_DIR / "simulation_report.toml"
DEFAULT_PHANTOM_REPORT = (
    ROOT / "outputs" / "dp150_mri_pipeline" / "koma" / "motion_phantom_report.toml"
)
DEFAULT_TRACKING_REPORT = (
    ROOT / "outputs" / "dp150_mri_pipeline" / "motion_1m_report.json"
)
DEFAULT_PERIODIC_REPORT = (
    ROOT / "outputs" / "dp150_mri_pipeline" / "periodic_flow_report.json"
)
DEFAULT_CONVERSION_REPORT = (
    ROOT / "outputs" / "dp150_volume_pipeline" / "conversion_report.json"
)
DEFAULT_FLUENT_DIR = ROOT.parent / "external_inputs" / "dp150_fluent"
DEFAULT_FLUENT_CASE = DEFAULT_FLUENT_DIR / "FFF-18.cas.h5"
DEFAULT_FLOW_MONITOR = DEFAULT_FLUENT_DIR / "vfr_inlet-rfile.out"
DEFAULT_FLUENT_SETTINGS = DEFAULT_FLUENT_DIR / "FFF.set"
DEFAULT_OUTPUT_DIR = (
    RUN_DIR / "final_image_analysis" / "cfd_koma_flow_waveform_e024"
)
DEFAULT_REFERENCE_INVENTORY = (
    DEFAULT_OUTPUT_DIR / "measured_reference_search_inventory.json"
)

REPORT_NAME = "dp150_cfd_koma_flow_waveform_report.json"
SUMMARY_NAME = "DP150_CFD_KOMA_FLOW_WAVEFORM.md"
FIGURE_NAME = "dp150_cfd_koma_flow_waveform.png"
EXPECTED_BINS = 15
EXPECTED_COMPONENT = "X"
DEFAULT_SECTION_INDEX = 70
DEFAULT_SECTION_WINDOW = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flow-monitor", type=Path, default=DEFAULT_FLOW_MONITOR)
    parser.add_argument("--fluent-case", type=Path, default=DEFAULT_FLUENT_CASE)
    parser.add_argument(
        "--fluent-settings", type=Path, default=DEFAULT_FLUENT_SETTINGS
    )
    parser.add_argument("--periodic-report", type=Path, default=DEFAULT_PERIODIC_REPORT)
    parser.add_argument(
        "--conversion-report", type=Path, default=DEFAULT_CONVERSION_REPORT
    )
    parser.add_argument("--tracking-report", type=Path, default=DEFAULT_TRACKING_REPORT)
    parser.add_argument("--phantom-report", type=Path, default=DEFAULT_PHANTOM_REPORT)
    parser.add_argument(
        "--simulation-report", type=Path, default=DEFAULT_SIMULATION_REPORT
    )
    parser.add_argument("--reconstruction", type=Path, default=DEFAULT_RECONSTRUCTION)
    parser.add_argument("--support-report", type=Path, default=DEFAULT_SUPPORT_REPORT)
    parser.add_argument(
        "--measured-reference-inventory",
        type=Path,
        default=DEFAULT_REFERENCE_INVENTORY,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--section-index", type=int, default=DEFAULT_SECTION_INDEX)
    parser.add_argument(
        "--section-window", type=int, default=DEFAULT_SECTION_WINDOW
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_record(path: Path, *, with_hash: bool = True) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
    }
    if with_hash:
        record["sha256"] = sha256(path)
    return record


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def smoothstep5(value: np.ndarray | float) -> np.ndarray:
    clipped = np.clip(np.asarray(value, dtype=np.float64), 0.0, 1.0)
    return clipped**3 * (10.0 - 15.0 * clipped + 6.0 * clipped**2)


def catmull_rom(
    p0: float, p1: float, p2: float, p3: float, fraction: float
) -> float:
    """Use the scalar form of MRSimTracks' uniform Catmull-Rom interpolator."""

    s = float(fraction)
    s2 = s * s
    s3 = s2 * s
    w0 = 0.5 * (-s + 2.0 * s2 - s3)
    w1 = 0.5 * (2.0 - 5.0 * s2 + 3.0 * s3)
    w2 = 0.5 * (s + 4.0 * s2 - 3.0 * s3)
    w3 = 0.5 * (-s2 + s3)
    return float(w0 * p0 + w1 * p1 + w2 * p2 + w3 * p3)


def periodic_cubic_sample(
    times_s: np.ndarray,
    values: np.ndarray,
    query_s: np.ndarray | float,
    *,
    period_s: float,
    knot_tolerance: float = 1e-3,
) -> np.ndarray:
    """Sample an endpoint-closed series like MRSimTracks ``time_interp='cubic'``."""

    times = np.asarray(times_s, dtype=np.float64)
    data = np.asarray(values, dtype=np.float64)
    query = np.atleast_1d(np.asarray(query_s, dtype=np.float64))
    if times.ndim != 1 or data.shape != times.shape or times.size < 5:
        raise ValueError("times and values must be matching one-dimensional series")
    if not np.all(np.diff(times) > 0):
        raise ValueError("flow-monitor times must be strictly increasing")
    if not np.allclose(np.diff(times), np.diff(times)[0], rtol=1e-6, atol=1e-12):
        raise ValueError("cubic interpolation requires uniform monitor spacing")
    if not np.isclose(times[0], 0.0, rtol=0, atol=1e-12):
        raise ValueError("periodic monitor must start at zero")
    if not np.isclose(times[-1], period_s, rtol=0, atol=1e-12):
        raise ValueError("periodic monitor endpoint does not equal its period")
    if not np.isclose(data[0], data[-1], rtol=0, atol=1e-15):
        raise ValueError("periodic monitor endpoints are not closed")

    distinct_count = times.size - 1
    output = np.empty(query.shape, dtype=np.float64)
    for index, requested in enumerate(query):
        wrapped = float(requested % period_s)
        next_index = int(np.argmax((times - wrapped) > 0))
        previous_index = next_index - 1
        fraction = float(
            (wrapped - times[previous_index])
            / (times[next_index] - times[previous_index])
        )
        if fraction < knot_tolerance:
            output[index] = data[previous_index]
        elif fraction > 1.0 - knot_tolerance:
            output[index] = data[next_index]
        else:
            output[index] = catmull_rom(
                data[(previous_index - 1) % distinct_count],
                data[previous_index % distinct_count],
                data[next_index % distinct_count],
                data[(next_index + 1) % distinct_count],
                fraction,
            )
    return output


def parse_fluent_monitor(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    records: list[tuple[int, float, float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.strip().split()
        if len(fields) != 3:
            continue
        try:
            step = int(fields[0])
            flow = float(fields[1])
            time_s = float(fields[2])
        except ValueError:
            continue
        records.append((step, flow, time_s))
    if not records:
        raise ValueError(f"No numeric Fluent monitor records found in {path}")
    array = np.asarray(records, dtype=np.float64)
    steps = array[:, 0].astype(np.int64)
    flow = array[:, 1]
    times = array[:, 2]
    if not np.array_equal(steps, np.arange(steps.size)):
        raise ValueError("Fluent monitor steps are not contiguous from zero")
    if not np.all(np.isfinite(flow)) or not np.all(np.isfinite(times)):
        raise ValueError("Fluent monitor contains non-finite values")
    return steps, times, flow


def close_monitor_tail(
    times_s: np.ndarray,
    raw_flow_m3_s: np.ndarray,
    *,
    blend_start_s: float,
    blend_end_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not (times_s[0] <= blend_start_s < blend_end_s == times_s[-1]):
        raise ValueError("tail-closure interval does not match the monitor range")
    fractions = (times_s - blend_start_s) / (blend_end_s - blend_start_s)
    weights = np.where(times_s >= blend_start_s, smoothstep5(fractions), 0.0)
    corrected = (1.0 - weights) * raw_flow_m3_s + weights * raw_flow_m3_s[0]
    corrected[-1] = corrected[0]
    return weights, corrected


def waveform_metrics(
    expected_m3_s: np.ndarray,
    reconstructed_m3_s: np.ndarray,
    *,
    bin_spacing_s: float,
) -> dict[str, Any]:
    expected = np.asarray(expected_m3_s, dtype=np.float64)
    reconstructed = np.asarray(reconstructed_m3_s, dtype=np.float64)
    if expected.shape != reconstructed.shape or expected.ndim != 1:
        raise ValueError("waveforms must be matching one-dimensional arrays")
    if expected.size < 3 or not np.all(np.isfinite(expected + reconstructed)):
        raise ValueError("waveforms must contain at least three finite samples")

    residual = reconstructed - expected
    expected_range = float(np.ptp(expected))
    rmse = float(np.sqrt(np.mean(residual**2)))
    origin_slope = float(np.dot(expected, reconstructed) / np.dot(expected, expected))
    free_slope, free_intercept = np.polyfit(expected, reconstructed, 1)
    affine_residual = reconstructed - (free_slope * expected + free_intercept)
    expected_peak = int(np.argmax(expected))
    reconstructed_peak = int(np.argmax(reconstructed))
    expected_sv = float(np.sum(expected) * bin_spacing_s)
    reconstructed_sv = float(np.sum(reconstructed) * bin_spacing_s)
    stroke_volume_ratio = reconstructed_sv / expected_sv

    return {
        "metric_pair_count": int(expected.size),
        "pearson_r": float(np.corrcoef(expected, reconstructed)[0, 1]),
        "rmse_m3_per_s": rmse,
        "rmse_ml_per_s": rmse * 1e6,
        "range_normalized_rmse": rmse / expected_range,
        "mae_m3_per_s": float(np.mean(np.abs(residual))),
        "mae_ml_per_s": float(np.mean(np.abs(residual)) * 1e6),
        "bias_m3_per_s": float(np.mean(residual)),
        "bias_ml_per_s": float(np.mean(residual) * 1e6),
        "range_normalization_denominator_m3_per_s": expected_range,
        "range_normalization_denominator_ml_per_s": expected_range * 1e6,
        "origin_constrained_slope": origin_slope,
        "free_affine_slope": float(free_slope),
        "free_affine_intercept_m3_per_s": float(free_intercept),
        "free_affine_intercept_ml_per_s": float(free_intercept * 1e6),
        "affine_shape_residual_range_normalized_rmse": float(
            np.sqrt(np.mean(affine_residual**2)) / expected_range
        ),
        "expected_peak_bin_one_based": expected_peak + 1,
        "reconstructed_peak_bin_one_based": reconstructed_peak + 1,
        "peak_bin_difference": reconstructed_peak - expected_peak,
        "expected_peak_ml_per_s": float(expected[expected_peak] * 1e6),
        "reconstructed_peak_ml_per_s": float(
            reconstructed[reconstructed_peak] * 1e6
        ),
        "reconstructed_to_expected_peak_ratio": float(
            reconstructed[reconstructed_peak] / expected[expected_peak]
        ),
        "expected_midpoint_stroke_volume_ml": expected_sv * 1e6,
        "reconstructed_midpoint_stroke_volume_ml": reconstructed_sv * 1e6,
        "reconstructed_to_expected_stroke_volume_ratio": stroke_volume_ratio,
        "stroke_volume_percent_error": (stroke_volume_ratio - 1.0) * 100.0,
        "stroke_volume_definition": (
            "midpoint Riemann sum over equally spaced reconstructed cardiac bins"
        ),
        "residual_sign_counts": {
            "negative": int(np.count_nonzero(residual < 0.0)),
            "zero": int(np.count_nonzero(residual == 0.0)),
            "positive": int(np.count_nonzero(residual > 0.0)),
        },
        "residual_ml_per_s": (residual * 1e6).tolist(),
    }


def integrate_section(
    velocity: np.ndarray,
    mask: np.ndarray,
    *,
    voxel_area_m2: float,
) -> np.ndarray:
    data = np.asarray(velocity, dtype=np.float64)
    selected = np.asarray(mask, dtype=bool)
    if data.ndim != 3 or selected.shape != data.shape[1:]:
        raise ValueError("velocity must be (bin, slice, phase) with a 2-D mask")
    if not np.all(np.isfinite(data[:, selected])):
        raise ValueError("selected reconstruction voxels contain non-finite velocity")
    return np.sum(data[:, selected], axis=1) * voxel_area_m2


def validate_inputs(
    *,
    flow_monitor: Path,
    fluent_settings: Path,
    fluent_case: Path,
    conversion_report: dict[str, Any],
    periodic_report: dict[str, Any],
    tracking_report: dict[str, Any],
    phantom_report: dict[str, Any],
    simulation_report: dict[str, Any],
    support_report: dict[str, Any],
    reconstruction: Path,
) -> dict[str, Any]:
    settings = fluent_settings.read_text(encoding="utf-8", errors="replace")
    required_settings = (
        '(case-file-units "si")',
        'name . "vfr_inlet"',
        "surface-volumeflowrate",
        'file-name ".\\\\vfr_inlet-rfile.out"',
    )
    missing_settings = [item for item in required_settings if item not in settings]
    if missing_settings:
        raise RuntimeError(f"Fluent monitor definition is incomplete: {missing_settings}")
    if flow_monitor.name != "vfr_inlet-rfile.out":
        raise RuntimeError("requested flow monitor is not the monitor bound in FFF.set")
    if not (flow_monitor.parent == fluent_settings.parent == fluent_case.parent):
        raise RuntimeError("Fluent monitor, settings, and case are not colocated")

    closure = periodic_report.get("closure_method", {})
    validation = periodic_report.get("validation", {})
    if closure.get("type") != "quintic_tail_blend_to_opening_field":
        raise RuntimeError("periodic-flow report uses an unexpected closure method")
    if validation.get("passed") is not True or validation.get("endpoint_exactly_equal") is not True:
        raise RuntimeError("periodic-flow closure did not pass")
    if tracking_report.get("run", {}).get("time_interpolation") != "cubic":
        raise RuntimeError("production tracking did not use cubic interpolation")
    tracking_run = tracking_report.get("run", {})
    if tracking_run.get("axis_order_zero_based") != [2, 0, 1]:
        raise RuntimeError("production tracking axis order is not [2, 0, 1]")
    if tracking_run.get("mri_axes_from_source") != ["z", "x", "y"]:
        raise RuntimeError("production tracking axis mapping is not MRI=(z,x,y)")
    if tracking_report.get("quality_control", {}).get("acceptance", {}).get("passed") is not True:
        raise RuntimeError("production particle tracking did not pass")
    if phantom_report.get("status") != "PASS":
        raise RuntimeError("motion-phantom report did not pass")
    if simulation_report.get("status") != "PASS":
        raise RuntimeError("accepted Koma simulation report did not pass")
    if simulation_report.get("seam_safe_motion") is not True:
        raise RuntimeError("Koma simulation is not seam-safe")
    timing = simulation_report.get("motion_timing", {})
    if timing.get("rescale_requested") is not True or timing.get("seam_safe_requested") is not True:
        raise RuntimeError("Koma motion timing lacks required retiming/seam alignment")
    if support_report.get("status") != "PASS_RECONSTRUCTION_DERIVATIVE":
        raise RuntimeError("support-constrained reconstruction did not pass")
    recorded = support_report.get("outputs", {}).get("reconstruction_h5", {})
    if Path(recorded.get("path", "")).name != reconstruction.name:
        raise RuntimeError("support report is not bound to the requested reconstruction")
    actual_hash = sha256(reconstruction)
    if recorded.get("sha256") != actual_hash:
        raise RuntimeError("reconstruction hash differs from its support report")

    conversion_source = conversion_report.get("source", {})
    conversion_output = conversion_report.get("output", {})
    local_case_hash = sha256(fluent_case)
    if conversion_source.get("case_sha256") != local_case_hash:
        raise RuntimeError("local Fluent case hash differs from conversion report")
    if Path(conversion_source.get("case_file", "")).name != fluent_case.name:
        raise RuntimeError("conversion report names a different Fluent case")
    if periodic_report.get("source", {}).get("pvd_sha256") != conversion_output.get(
        "pvd_sha256"
    ):
        raise RuntimeError("periodic-flow source is not bound to conversion output")
    periodic_hash = periodic_report.get("output", {}).get("pvd_sha256")
    if tracking_report.get("inputs", {}).get("periodic_pvd_sha256") != periodic_hash:
        raise RuntimeError("tracking input is not bound to periodic-flow output")

    source_cycle = float(tracking_report["run"]["duration_s"])
    if not np.isclose(source_cycle, phantom_report["source_cycle_s"], rtol=0, atol=1e-12):
        raise RuntimeError("tracking and motion-phantom source periods disagree")
    initial_period = float(phantom_report["realized_sequence_rr_s"])
    if not np.isclose(initial_period, timing["original_period_s"], rtol=0, atol=1e-12):
        raise RuntimeError("motion-phantom and Koma original periods disagree")
    effective_period = float(timing["effective_period_s"])
    sequence_rr = float(timing["sequence_rr_s"])
    return {
        "source_cycle_s": source_cycle,
        "initial_phantom_period_s": initial_period,
        "effective_motion_period_s": effective_period,
        "sequence_rr_s": sequence_rr,
        "blend_start_s": float(closure["blend_start_s"]),
        "blend_end_s": float(closure["blend_end_s"]),
        "reconstruction_sha256": actual_hash,
        "fluent_case_sha256": local_case_hash,
        "axis_and_sign": {
            "source_axis_order_zero_based": [2, 0, 1],
            "mri_axes_from_source": ["z", "x", "y"],
            "comparison_axis": "MRI +X/readout = CFD +z",
            "fluent_monitor_sign_handling": "used as reported",
            "sign_inversion_applied": False,
        },
        "run_linkage": {
            "local_fluent_case_matches_conversion_report": True,
            "conversion_output_matches_periodic_flow_source": True,
            "periodic_flow_output_matches_tracking_input": True,
            "settings_bind_monitor_to_inlet_surface": True,
            "monitor_definition_and_case_are_colocated": bool(
                flow_monitor.parent == fluent_settings.parent == fluent_case.parent
            ),
        },
    }


def load_reconstruction_sections(
    path: Path,
    *,
    section_index: int,
    section_window: int,
) -> dict[str, Any]:
    with h5py.File(path, "r") as h5:
        components = str(h5.attrs["velocity_components"]).split(",")
        if components != ["X", "Y", "Z"]:
            raise RuntimeError(f"Unexpected component order: {components}")
        component_index = components.index(EXPECTED_COMPONENT)
        corrected = h5["velocity_corrected_m_per_s"]
        raw = h5["velocity_raw_m_per_s"]
        if corrected.shape[1] != EXPECTED_BINS or corrected.shape != raw.shape:
            raise RuntimeError(f"Unexpected reconstructed velocity shape: {corrected.shape}")
        readout_size = corrected.shape[-1]
        if not 0 <= section_index < readout_size:
            raise ValueError("section index lies outside the reconstruction")
        support = np.asarray(h5["support_mask"], dtype=bool)
        direct = np.asarray(h5["direct_mask"], dtype=bool)
        regularized = np.asarray(h5["regularized_mask"], dtype=bool)
        if not np.array_equal(support, direct | regularized):
            raise RuntimeError("direct and regularized masks do not partition support")
        voxel_size_mm = np.asarray(
            h5.attrs["voxel_size_mm_readout_phase_slice"], dtype=np.float64
        )
        voxel_area_m2 = float(voxel_size_mm[1] * voxel_size_mm[2] * 1e-6)
        rr_h5 = float(h5.attrs["RR_s"])
        venc = float(h5.attrs["Venc_m_per_s"])

        start = max(0, section_index - section_window)
        stop = min(readout_size - 1, section_index + section_window)
        section_records: dict[int, dict[str, Any]] = {}
        for current in range(start, stop + 1):
            velocity = np.asarray(corrected[component_index, :, :, :, current])
            section_support = support[:, :, current]
            flow = integrate_section(
                velocity, section_support, voxel_area_m2=voxel_area_m2
            )
            section_records[current] = {
                "flow_m3_per_s": flow,
                "support_voxels": int(np.count_nonzero(section_support)),
                "direct_voxels": int(np.count_nonzero(direct[:, :, current])),
                "regularized_voxels": int(
                    np.count_nonzero(regularized[:, :, current])
                ),
            }

        selected_corrected = np.asarray(
            corrected[component_index, :, :, :, section_index]
        )
        selected_raw = np.asarray(raw[component_index, :, :, :, section_index])
        selected_support = support[:, :, section_index]
        selected_direct = direct[:, :, section_index]
        selected_regularized = regularized[:, :, section_index]

    return {
        "component": EXPECTED_COMPONENT,
        "component_index_zero_based": component_index,
        "section_index_zero_based": section_index,
        "section_records": section_records,
        "corrected_flow_m3_per_s": integrate_section(
            selected_corrected, selected_support, voxel_area_m2=voxel_area_m2
        ),
        "raw_flow_m3_per_s": integrate_section(
            selected_raw, selected_support, voxel_area_m2=voxel_area_m2
        ),
        "direct_flow_m3_per_s": integrate_section(
            selected_corrected, selected_direct, voxel_area_m2=voxel_area_m2
        ),
        "regularized_flow_m3_per_s": integrate_section(
            selected_corrected, selected_regularized, voxel_area_m2=voxel_area_m2
        ),
        "support_voxels": int(np.count_nonzero(selected_support)),
        "direct_voxels": int(np.count_nonzero(selected_direct)),
        "regularized_voxels": int(np.count_nonzero(selected_regularized)),
        "voxel_area_m2": voxel_area_m2,
        "voxelized_section_area_m2": float(
            np.count_nonzero(selected_support) * voxel_area_m2
        ),
        "voxel_size_mm_readout_phase_slice": voxel_size_mm.tolist(),
        "RR_h5_s": rr_h5,
        "Venc_m_per_s": venc,
    }


def make_figure(
    path: Path,
    *,
    sequence_centers_s: np.ndarray,
    dense_sequence_s: np.ndarray,
    dense_expected_m3_s: np.ndarray,
    expected_m3_s: np.ndarray,
    reconstructed_m3_s: np.ndarray,
    metrics: dict[str, Any],
    synthetic_tail_start_sequence_s: float,
) -> None:
    expected_ml_s = expected_m3_s * 1e6
    reconstructed_ml_s = reconstructed_m3_s * 1e6
    dense_ml_s = dense_expected_m3_s * 1e6
    residual_ml_s = reconstructed_ml_s - expected_ml_s

    figure = plt.figure(figsize=(12.0, 8.4))
    grid = figure.add_gridspec(2, 2, height_ratios=(1.05, 1.0))
    waveform_axis = figure.add_subplot(grid[0, :])
    residual_axis = figure.add_subplot(grid[1, 0])
    scatter_axis = figure.add_subplot(grid[1, 1])

    ax = waveform_axis
    ax.axvspan(
        synthetic_tail_start_sequence_s,
        dense_sequence_s[-1],
        color="#d9d9d9",
        alpha=0.45,
        label="controlled synthetic tail",
    )
    ax.plot(
        dense_sequence_s,
        dense_ml_s,
        color="#1f77b4",
        linewidth=2.0,
        label="Tail-closed, retimed CFD inlet target",
    )
    ax.plot(
        sequence_centers_s,
        expected_ml_s,
        "s",
        color="#1f77b4",
        markersize=4.5,
    )
    ax.plot(
        sequence_centers_s,
        reconstructed_ml_s,
        "o--",
        color="#d95f02",
        linewidth=1.6,
        markersize=4.5,
        label="Simulated Koma reconstruction",
    )
    ax.set_title("Upstream volumetric-flow waveform")
    ax.set_xlabel("Retimed simulation-cycle time from periodic seam (s)")
    ax.set_ylabel("Flow (mL/s)")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.grid(alpha=0.25)

    ax = residual_axis
    colors = np.where(residual_ml_s >= 0.0, "#2ca25f", "#756bb1")
    ax.bar(np.arange(1, expected_ml_s.size + 1), residual_ml_s, color=colors)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Binwise residual")
    ax.set_xlabel("Simulated phase bin")
    ax.set_ylabel("Reconstruction - retimed CFD target (mL/s)")
    ax.set_xticks(np.arange(1, expected_ml_s.size + 1, 2))
    ax.grid(axis="y", alpha=0.25)

    ax = scatter_axis
    limit = float(max(expected_ml_s.max(), reconstructed_ml_s.max()) * 1.08)
    ax.plot([0.0, limit], [0.0, limit], "--", color="0.45", label="identity")
    points = ax.scatter(
        expected_ml_s,
        reconstructed_ml_s,
        c=np.arange(1, expected_ml_s.size + 1),
        cmap="viridis",
        s=42,
        edgecolor="white",
        linewidth=0.4,
        zorder=3,
    )
    colorbar = figure.colorbar(points, ax=ax, fraction=0.047, pad=0.035)
    colorbar.set_label("Simulated phase bin")
    colorbar.set_ticks([1, 4, 8, 12, 15])
    ax.set_xlim(0.0, limit)
    ax.set_ylim(min(-2.0, float(reconstructed_ml_s.min()) - 1.0), limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Binwise same-source comparison")
    ax.set_xlabel("Tail-closed, retimed CFD target (mL/s)")
    ax.set_ylabel("Simulated Koma reconstruction (mL/s)")
    ax.grid(alpha=0.25)
    annotation = (
        f"n = {expected_ml_s.size}\n"
        f"r = {metrics['pearson_r']:.4f}\n"
        f"bias = {metrics['bias_ml_per_s']:.2f} mL/s\n"
        f"RMSE = {metrics['rmse_ml_per_s']:.2f} mL/s\n"
        f"RMSE / range = {100 * metrics['range_normalized_rmse']:.2f}%\n"
        f"midpoint SV ratio = "
        f"{metrics['reconstructed_to_expected_stroke_volume_ratio']:.3f}"
    )
    ax.text(
        0.04,
        0.96,
        annotation,
        transform=ax.transAxes,
        va="top",
        fontsize=8.5,
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.9},
    )

    figure.suptitle(
        "dp150 same-source internal flow comparison\n"
        "CFD inlet monitor to accepted 1M seam-safe Koma reconstruction",
        fontsize=14.5,
        y=0.985,
    )
    figure.text(
        0.5,
        0.01,
        "Internal consistency only — no qualifying measured waveform or velocity reference was found in the audited locations.",
        ha="center",
        color="#8c2d04",
        fontsize=9,
    )
    figure.tight_layout(rect=(0.02, 0.055, 0.99, 0.92))
    temporary = path.with_suffix(".partial.png")
    figure.savefig(temporary, dpi=220, facecolor="white", format="png")
    plt.close(figure)
    os.replace(temporary, path)


def markdown_summary(report: dict[str, Any]) -> str:
    metrics = report["comparison_metrics"]
    section = report["reconstruction_section"]
    sensitivity = report["section_sensitivity"]
    non_tail = report["non_synthetic_tail_sensitivity"]
    temporal = report["temporal_semantics"]
    interpolation = report["interpolation_diagnostics"]
    rows = report["binwise_values"]
    lines = [
        "# dp150 CFD-to-Koma Flow-Waveform Comparison",
        "",
        f"**Status:** `{report['status']}`",
        "",
        "This is a post hoc, same-source **internal-consistency** comparison. The "
        "target is a Fluent inlet surface-flow monitor from the same CFD run that "
        "supplied the motion field; it is neither a measured waveform nor independent "
        "ground truth.",
        "",
        "## Main result",
        "",
        f"At selected upstream zero-based readout index "
        f"{section['readout_index_zero_based']}, the 15 midpoint pairs show high "
        f"shape correlation (r = {metrics['pearson_r']:.4f}), but reconstructed "
        f"flow is systematically low: all "
        f"{metrics['residual_sign_counts']['negative']} residuals are negative, "
        f"bias is {metrics['bias_ml_per_s']:.2f} mL/s, RMSE is "
        f"{metrics['rmse_ml_per_s']:.2f} mL/s "
        f"({100 * metrics['range_normalized_rmse']:.2f}% of the discrete target "
        f"range), and the 15-bin midpoint stroke-volume estimate is "
        f"{metrics['reconstructed_midpoint_stroke_volume_ml']:.2f} versus "
        f"{metrics['expected_midpoint_stroke_volume_ml']:.2f} mL "
        f"(Koma/target = "
        f"{metrics['reconstructed_to_expected_stroke_volume_ratio']:.4f}; "
        f"{abs(metrics['stroke_volume_percent_error']):.2f}% low). Both discrete "
        f"cross-sectional-flow series peak at bin "
        f"{metrics['expected_peak_bin_one_based']}.",
        "",
        f"Across neighboring upstream planes {sensitivity['first_index_zero_based']}–"
        f"{sensitivity['last_index_zero_based']}, r spans "
        f"{sensitivity['pearson_r_min']:.4f}–{sensitivity['pearson_r_max']:.4f} "
        f"and the stroke-volume ratio spans "
        f"{sensitivity['stroke_volume_ratio_min']:.4f}–"
        f"{sensitivity['stroke_volume_ratio_max']:.4f}.",
        "",
        f"Excluding bins 14–15, whose midpoint targets lie inside the synthetic "
        f"tail, gives r = {non_tail['pearson_r']:.4f}, RMSE = "
        f"{non_tail['rmse_ml_per_s']:.2f} mL/s, and range-NRMSE = "
        f"{100 * non_tail['range_normalized_rmse']:.2f}%; the interpretation is "
        "unchanged.",
        "",
        "## Method",
        "",
        "The 73-point Fluent `surface-volumeflowrate` monitor (m³/s inferred from "
        "the SI Fluent case settings; its own header is unitless; 10-ms "
        "spacing, 0–0.720 s) was given the same 0.640–0.720-s quintic closure "
        "used by the tracked velocity fields. It was then sampled with the same "
        "periodic Catmull–Rom interpolation. The stored path traverses the original "
        "0.720-s source cycle over the final Koma motion period, so flow was scaled "
        "by source-period/final-period. Reconstructed X/readout velocity was "
        "integrated over the fixed seed-derived support mask (voxel occupancy with "
        "declared per-plane enclosed-hole processing; no dilation) using the "
        "phase-by-slice pixel area.",
        "",
        "Axis and sign are provenance-bound: tracking records source-axis order "
        "[2, 0, 1], so MRI +X/readout is source CFD +z; the positive Fluent monitor "
        "is used as reported with no sign inversion. The local Fluent case hash "
        "matches the conversion report, whose output is hash-linked through the "
        "periodic-flow derivative to the tracking input.",
        "",
        "The comparison uses two clocks. CFD target values are evaluated at "
        "source-cycle midpoints `t_k = (k - 0.5) × 0.720 / 15`, then mapped to "
        f"simulation-cycle coordinates `t'_k = (k - 0.5) × "
        f"{temporal['effective_motion_period_s']:.9f} / 15` after the artificial "
        "periodic motion seam. These plotted centers are not scanner trigger-delay "
        "labels. The dense curve is display sampling of Catmull–Rom interpolation "
        "over 10-ms source samples and does not increase source temporal resolution. "
        f"Bins 14–15 lie within the controlled closure interval, mapped to "
        f"{temporal['synthetic_tail_start_retimed_s']:.6f}–"
        f"{temporal['effective_motion_period_s']:.6f} s.",
        "",
        f"On a 0.1-ms diagnostic grid, the production Catmull-Rom interpolation "
        f"briefly undershoots to {interpolation['minimum_retimed_flow_ml_per_s']:.3f} "
        f"mL/s near closure ({100 * interpolation['minimum_magnitude_as_fraction_of_positive_peak']:.3f}% "
        "of the positive peak). The monitor knots are nonnegative, so this is a "
        "small interpolation artifact rather than resolved reverse flow; no values "
        "were clipped or changed. The target midpoint stroke volume differs from "
        f"dense-grid integration by only "
        f"{abs(interpolation['midpoint_vs_dense_stroke_volume_percent_difference']):.3f}%.",
        "",
        f"The highest sampled raw CFD monitor value occurs at source time "
        f"{temporal['raw_sampled_peak_source_s']:.3f} s and maps to simulation time "
        f"{temporal['raw_sampled_peak_retimed_s']:.4f} s. “Peak bin 4” refers only to "
        "the 15-point midpoint cross-sectional-flow sampling; it is a different "
        "quantity from E-023's support-wide scalar-speed peak at bin 5.",
        "",
        "## Binwise values",
        "",
        "| Bin | Retimed simulation center from seam (s) | Tail-closed, retimed CFD target (mL/s) | Simulated Koma reconstruction (mL/s) | Residual (mL/s) |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['bin_one_based']} | {row['sequence_center_s']:.6f} | "
            f"{row['cfd_expected_ml_per_s']:.3f} | "
            f"{row['koma_reconstructed_ml_per_s']:.3f} | "
            f"{row['residual_ml_per_s']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Claim boundary and next input",
            "",
            "These results show that the accepted reconstruction follows its own "
            "CFD-driven waveform closely. They do **not** validate the physical "
            "phantom or scanner velocity values because no qualifying measured "
            "pump/flowmeter waveform or PC-MRI/4D-flow/PIV velocity field was found "
            "in the audited locations. Absolute "
            "flow also depends on a binary, seed-derived voxel support; the selected "
            f"plane contains {section['support_voxels']} support voxels and "
            f"{section['direct_fraction'] * 100:.2f}% direct-phase coverage.",
            "",
            "The accepted simulation used VENC 5.0 m/s, whereas recovered scanner "
            "metadata reports VENC 3.0 m/s; this is a documented anti-aliasing "
            "adaptation, not an exact scanner-protocol match.",
            "",
            "Independent validation requires a deidentified measured waveform and "
            "measured PC-MRI/4D-flow or PIV velocities from the same phantom run, "
            "with units, sensor/section location, trigger and phase definition, VENC, "
            "mask and processing history, uncertainty, spatial/temporal registration, "
            "and run IDs/timestamps proving matched acquisition.",
            "",
            f"Figure: `{FIGURE_NAME}`  ",
            f"Machine-readable report: `{REPORT_NAME}`  ",
            "Scoped measured-reference audit: "
            "`measured_reference_search_inventory.json`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    paths = {
        "flow_monitor": args.flow_monitor.resolve(),
        "fluent_case": args.fluent_case.resolve(),
        "fluent_settings": args.fluent_settings.resolve(),
        "periodic_report": args.periodic_report.resolve(),
        "conversion_report": args.conversion_report.resolve(),
        "tracking_report": args.tracking_report.resolve(),
        "phantom_report": args.phantom_report.resolve(),
        "simulation_report": args.simulation_report.resolve(),
        "reconstruction": args.reconstruction.resolve(),
        "support_report": args.support_report.resolve(),
        "measured_reference_inventory": args.measured_reference_inventory.resolve(),
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing comparison inputs: {missing}")
    if args.section_window < 0:
        raise ValueError("section-window must be non-negative")

    output_dir = args.output_dir.resolve()
    report_path = output_dir / REPORT_NAME
    summary_path = output_dir / SUMMARY_NAME
    figure_path = output_dir / FIGURE_NAME
    existing = [path for path in (report_path, summary_path, figure_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Outputs already exist; use --overwrite: {existing}")
    output_dir.mkdir(parents=True, exist_ok=True)

    periodic_report = read_json(paths["periodic_report"])
    conversion_report = read_json(paths["conversion_report"])
    tracking_report = read_json(paths["tracking_report"])
    phantom_report = read_toml(paths["phantom_report"])
    simulation_report = read_toml(paths["simulation_report"])
    support_report = read_json(paths["support_report"])
    reference_inventory = read_json(paths["measured_reference_inventory"])
    if reference_inventory.get("audit_id") != "E-024-measured-reference-search":
        raise RuntimeError("measured-reference inventory has an unexpected audit ID")
    if reference_inventory.get("status") != "COMPLETE_TARGETED_INVENTORY":
        raise RuntimeError("measured-reference inventory is not complete")
    if any(
        item.get("provenance_qualified_measurement") is True
        for item in reference_inventory.get("candidate_findings", [])
    ):
        raise RuntimeError(
            "measured-reference inventory contains a qualifying measurement"
        )
    timing = validate_inputs(
        flow_monitor=paths["flow_monitor"],
        fluent_settings=paths["fluent_settings"],
        fluent_case=paths["fluent_case"],
        conversion_report=conversion_report,
        periodic_report=periodic_report,
        tracking_report=tracking_report,
        phantom_report=phantom_report,
        simulation_report=simulation_report,
        support_report=support_report,
        reconstruction=paths["reconstruction"],
    )
    axis_and_sign = timing.pop("axis_and_sign")
    run_linkage = timing.pop("run_linkage")
    reconstruction_hash = timing.pop("reconstruction_sha256")
    fluent_case_hash = timing.pop("fluent_case_sha256")

    steps, monitor_times, raw_flow = parse_fluent_monitor(paths["flow_monitor"])
    source_cycle = timing["source_cycle_s"]
    if not np.isclose(monitor_times[-1], source_cycle, rtol=0, atol=1e-12):
        raise RuntimeError("Fluent monitor duration and tracked source cycle disagree")
    weights, closed_flow = close_monitor_tail(
        monitor_times,
        raw_flow,
        blend_start_s=timing["blend_start_s"],
        blend_end_s=timing["blend_end_s"],
    )

    reconstruction = load_reconstruction_sections(
        paths["reconstruction"],
        section_index=args.section_index,
        section_window=args.section_window,
    )
    if not np.isclose(
        reconstruction["RR_h5_s"], timing["sequence_rr_s"], rtol=0, atol=1e-6
    ):
        raise RuntimeError("reconstruction and simulation RR values disagree")

    effective_period = timing["effective_motion_period_s"]
    amplitude_scale = source_cycle / effective_period
    source_centers = (np.arange(EXPECTED_BINS, dtype=np.float64) + 0.5) * (
        source_cycle / EXPECTED_BINS
    )
    sequence_centers = (np.arange(EXPECTED_BINS, dtype=np.float64) + 0.5) * (
        effective_period / EXPECTED_BINS
    )
    expected_flow = periodic_cubic_sample(
        monitor_times,
        closed_flow,
        source_centers,
        period_s=source_cycle,
    ) * amplitude_scale
    reconstructed_flow = reconstruction["corrected_flow_m3_per_s"]
    bin_spacing = effective_period / EXPECTED_BINS
    metrics = waveform_metrics(
        expected_flow, reconstructed_flow, bin_spacing_s=bin_spacing
    )
    non_tail_mask = source_centers < timing["blend_start_s"]
    non_tail_full = waveform_metrics(
        expected_flow[non_tail_mask],
        reconstructed_flow[non_tail_mask],
        bin_spacing_s=bin_spacing,
    )
    non_tail_metrics = {
        "included_bins_one_based": (
            np.flatnonzero(non_tail_mask).astype(int) + 1
        ).tolist(),
        "excluded_synthetic_tail_bins_one_based": (
            np.flatnonzero(~non_tail_mask).astype(int) + 1
        ).tolist(),
        "metric_pair_count": non_tail_full["metric_pair_count"],
        "pearson_r": non_tail_full["pearson_r"],
        "rmse_ml_per_s": non_tail_full["rmse_ml_per_s"],
        "range_normalized_rmse": non_tail_full["range_normalized_rmse"],
        "mae_ml_per_s": non_tail_full["mae_ml_per_s"],
        "bias_ml_per_s": non_tail_full["bias_ml_per_s"],
        "residual_sign_counts": non_tail_full["residual_sign_counts"],
        "interpretation": (
            "Sensitivity only; excluding synthetic-tail midpoint targets does not "
            "remove the systematic low-flow finding"
        ),
    }

    sensitivity_records = []
    for index, record in reconstruction["section_records"].items():
        current = waveform_metrics(
            expected_flow, record["flow_m3_per_s"], bin_spacing_s=bin_spacing
        )
        sensitivity_records.append(
            {
                "readout_index_zero_based": index,
                "support_voxels": record["support_voxels"],
                "direct_voxels": record["direct_voxels"],
                "regularized_voxels": record["regularized_voxels"],
                "pearson_r": current["pearson_r"],
                "rmse_ml_per_s": current["rmse_ml_per_s"],
                "bias_ml_per_s": current["bias_ml_per_s"],
                "stroke_volume_ratio": current[
                    "reconstructed_to_expected_stroke_volume_ratio"
                ],
                "peak_ratio": current["reconstructed_to_expected_peak_ratio"],
            }
        )
    sensitivity = {
        "first_index_zero_based": sensitivity_records[0][
            "readout_index_zero_based"
        ],
        "last_index_zero_based": sensitivity_records[-1][
            "readout_index_zero_based"
        ],
        "pearson_r_min": min(item["pearson_r"] for item in sensitivity_records),
        "pearson_r_max": max(item["pearson_r"] for item in sensitivity_records),
        "rmse_ml_per_s_min": min(
            item["rmse_ml_per_s"] for item in sensitivity_records
        ),
        "rmse_ml_per_s_max": max(
            item["rmse_ml_per_s"] for item in sensitivity_records
        ),
        "stroke_volume_ratio_min": min(
            item["stroke_volume_ratio"] for item in sensitivity_records
        ),
        "stroke_volume_ratio_max": max(
            item["stroke_volume_ratio"] for item in sensitivity_records
        ),
        "records": sensitivity_records,
    }

    dense_source = np.linspace(0.0, source_cycle, 721)
    dense_expected = periodic_cubic_sample(
        monitor_times, closed_flow, dense_source, period_s=source_cycle
    ) * amplitude_scale
    dense_sequence = dense_source / source_cycle * effective_period

    diagnostic_spacing_s = 0.0001
    diagnostic_source = np.linspace(
        0.0,
        source_cycle,
        int(round(source_cycle / diagnostic_spacing_s)) + 1,
    )
    diagnostic_expected = periodic_cubic_sample(
        monitor_times,
        closed_flow,
        diagnostic_source,
        period_s=source_cycle,
    ) * amplitude_scale
    diagnostic_sequence = diagnostic_source / source_cycle * effective_period
    negative_grid = diagnostic_expected < 0.0
    negative_source = diagnostic_source[negative_grid]
    interpolation_diagnostics = {
        "diagnostic_grid_spacing_source_s": diagnostic_spacing_s,
        "diagnostic_grid_sample_count": int(diagnostic_source.size),
        "minimum_retimed_flow_ml_per_s": float(diagnostic_expected.min() * 1e6),
        "maximum_retimed_flow_ml_per_s": float(diagnostic_expected.max() * 1e6),
        "minimum_source_time_s": float(
            diagnostic_source[int(np.argmin(diagnostic_expected))]
        ),
        "minimum_retimed_time_s": float(
            diagnostic_sequence[int(np.argmin(diagnostic_expected))]
        ),
        "negative_grid_sample_count": int(np.count_nonzero(negative_grid)),
        "negative_grid_sample_fraction": float(np.mean(negative_grid)),
        "negative_source_interval_s": (
            [float(negative_source[0]), float(negative_source[-1])]
            if negative_source.size
            else None
        ),
        "minimum_magnitude_as_fraction_of_positive_peak": float(
            abs(diagnostic_expected.min()) / diagnostic_expected.max()
        ),
        "dense_integral_stroke_volume_ml": float(
            np.trapz(diagnostic_expected, diagnostic_sequence) * 1e6
        ),
        "midpoint_vs_dense_stroke_volume_percent_difference": float(
            (
                metrics["expected_midpoint_stroke_volume_ml"]
                / (np.trapz(diagnostic_expected, diagnostic_sequence) * 1e6)
                - 1.0
            )
            * 100.0
        ),
        "interpretation": (
            "The negative values occur only between nonnegative monitor knots near "
            "periodic closure and are Catmull-Rom interpolation undershoot, not a "
            "resolved reverse-flow observation. The diagnostic grid quantifies but "
            "does not alter the production interpolation."
        ),
    }
    make_figure(
        figure_path,
        sequence_centers_s=sequence_centers,
        dense_sequence_s=dense_sequence,
        dense_expected_m3_s=dense_expected,
        expected_m3_s=expected_flow,
        reconstructed_m3_s=reconstructed_flow,
        metrics=metrics,
        synthetic_tail_start_sequence_s=(
            timing["blend_start_s"] / source_cycle * effective_period
        ),
    )

    raw_metrics = waveform_metrics(
        reconstruction["raw_flow_m3_per_s"],
        reconstructed_flow,
        bin_spacing_s=bin_spacing,
    )
    peak_index = int(np.argmax(reconstructed_flow))
    regularized_peak_fraction = float(
        reconstruction["regularized_flow_m3_per_s"][peak_index]
        / reconstructed_flow[peak_index]
    )
    binwise = []
    for index in range(EXPECTED_BINS):
        binwise.append(
            {
                "bin_one_based": index + 1,
                "source_center_s": float(source_centers[index]),
                "sequence_center_s": float(sequence_centers[index]),
                "cfd_expected_m3_per_s": float(expected_flow[index]),
                "cfd_expected_ml_per_s": float(expected_flow[index] * 1e6),
                "koma_reconstructed_m3_per_s": float(reconstructed_flow[index]),
                "koma_reconstructed_ml_per_s": float(
                    reconstructed_flow[index] * 1e6
                ),
                "residual_ml_per_s": float(
                    (reconstructed_flow[index] - expected_flow[index]) * 1e6
                ),
                "koma_direct_contribution_ml_per_s": float(
                    reconstruction["direct_flow_m3_per_s"][index] * 1e6
                ),
                "koma_regularized_contribution_ml_per_s": float(
                    reconstruction["regularized_flow_m3_per_s"][index] * 1e6
                ),
            }
        )

    source_records = {
        name: artifact_record(path)
        for name, path in paths.items()
        if name != "reconstruction"
    }
    source_records["reconstruction"] = {
        "path": str(paths["reconstruction"]),
        "bytes": paths["reconstruction"].stat().st_size,
        "sha256": reconstruction_hash,
    }
    source_records["fluent_case"]["sha256"] = fluent_case_hash
    source_records["implementation"] = artifact_record(Path(__file__).resolve())

    report: dict[str, Any] = {
        "schema_version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETE_DESCRIPTIVE_INTERNAL_COMPARISON",
        "evidence_class": "internal_consistency_not_independent_validation",
        "claim_boundary": {
            "supported": (
                "The accepted Koma reconstruction closely follows the CFD-driven "
                "inlet-flow waveform after applying the documented periodic closure "
                "and motion retiming."
            ),
            "not_supported": [
                "agreement with a measured pump or flowmeter waveform",
                "agreement with measured PC-MRI or PIV velocity values",
                "scanner-to-simulation spatial registration",
                "physical validation of the phantom experiment",
            ],
        },
        "method": {
            "analysis_timing": "post_hoc_descriptive_no_preregistered_acceptance_gate",
            "fluent_quantity": "surface-volumeflowrate on inlet",
            "fluent_units": (
                "m^3/s, inferred from FFF.set declaring the SI unit system and "
                "defining vfr_inlet as surface-volumeflowrate; the report-file "
                "header is unitless"
            ),
            "monitor_samples": int(steps.size),
            "monitor_spacing_s": float(np.diff(monitor_times)[0]),
            "source_cycle_s": source_cycle,
            "periodic_closure": "quintic tail blend to opening field",
            "periodic_blend_start_s": timing["blend_start_s"],
            "periodic_blend_end_s": timing["blend_end_s"],
            "temporal_interpolation": "periodic uniform Catmull-Rom",
            "source_bin_sampling": "15 equal-width source-cycle midpoints",
            "retiming": (
                "source 0.720-s path -> initial Float32 phantom period -> final "
                "seam-safe Koma motion period"
            ),
            "net_velocity_and_flow_scale": "source_cycle_s/effective_motion_period_s",
            "amplitude_scale": amplitude_scale,
            "integration": (
                "sum X/readout velocity over the fixed seed-derived support mask "
                "(voxel occupancy with per-plane enclosed-hole processing and no "
                "dilation) times the phase-by-slice pixel area"
            ),
        },
        "axis_and_sign_provenance": axis_and_sign,
        "run_linkage": run_linkage,
        "timing": {
            **timing,
            "source_to_initial_period_scale": source_cycle
            / timing["initial_phantom_period_s"],
            "initial_to_final_period_scale": timing["initial_phantom_period_s"]
            / effective_period,
            "net_source_to_final_velocity_scale": amplitude_scale,
            "bin_spacing_s": bin_spacing,
            "synthetic_tail_start_sequence_s": timing["blend_start_s"]
            / source_cycle
            * effective_period,
        },
        "temporal_semantics": {
            "time_origin": "artificial periodic Koma motion seam",
            "time_axis_is_scanner_trigger_delay": False,
            "expected_value_semantics": (
                "tail-closed and amplitude-scaled CFD inlet monitor sampled at "
                "source-cycle midpoints"
            ),
            "dense_curve_semantics": (
                "display sampling of the same Catmull-Rom interpolation used for "
                "tracking; source cadence remains 10 ms"
            ),
            "metric_pair_count": EXPECTED_BINS,
            "metrics_include_synthetic_tail_bins_one_based": (
                np.flatnonzero(~non_tail_mask).astype(int) + 1
            ).tolist(),
            "raw_sampled_peak_source_s": float(
                monitor_times[int(np.argmax(raw_flow))]
            ),
            "raw_sampled_peak_retimed_s": float(
                monitor_times[int(np.argmax(raw_flow))]
                / source_cycle
                * effective_period
            ),
            "discrete_target_peak_bin_one_based": metrics[
                "expected_peak_bin_one_based"
            ],
            "effective_motion_period_s": effective_period,
            "synthetic_tail_start_retimed_s": timing["blend_start_s"]
            / source_cycle
            * effective_period,
            "source_midpoint_formula": "(one_based_bin - 0.5) * 0.720 / 15",
            "retimed_midpoint_formula": (
                "(one_based_bin - 0.5) * effective_motion_period / 15"
            ),
        },
        "fluent_monitor": {
            "raw_min_m3_per_s": float(raw_flow.min()),
            "raw_max_m3_per_s": float(raw_flow.max()),
            "raw_sampled_peak_time_s": float(
                monitor_times[int(np.argmax(raw_flow))]
            ),
            "raw_sampled_peak_mapped_to_sequence_s": float(
                monitor_times[int(np.argmax(raw_flow))]
                / source_cycle
                * effective_period
            ),
            "closed_endpoint_exact": bool(closed_flow[0] == closed_flow[-1]),
            "maximum_tail_weight": float(weights.max()),
            "raw_flow_m3_per_s": raw_flow.tolist(),
            "closed_flow_m3_per_s": closed_flow.tolist(),
            "times_s": monitor_times.tolist(),
        },
        "interpolation_diagnostics": interpolation_diagnostics,
        "data_quality": {
            "intended_grain": "one inlet volume-flow value per Fluent time step",
            "sample_count": int(steps.size),
            "step_ids_contiguous_zero_based": bool(
                np.array_equal(steps, np.arange(steps.size))
            ),
            "duplicate_time_count": int(
                monitor_times.size - np.unique(monitor_times).size
            ),
            "nonfinite_time_count": int(np.count_nonzero(~np.isfinite(monitor_times))),
            "nonfinite_flow_count": int(np.count_nonzero(~np.isfinite(raw_flow))),
            "strictly_increasing_time": bool(np.all(np.diff(monitor_times) > 0)),
            "uniform_spacing": bool(
                np.allclose(
                    np.diff(monitor_times),
                    np.diff(monitor_times)[0],
                    rtol=1e-6,
                    atol=1e-12,
                )
            ),
            "unit_lineage": (
                "FFF.set declares case-file-units SI, defines vfr_inlet as "
                "surface-volumeflowrate on inlet, and binds it to the monitor file; "
                "the monitor header itself has no unit label"
            ),
            "source_independence": False,
            "source_independence_reason": (
                "the solver-generated monitor is from the prescribed inlet boundary "
                "of the same CFD-to-particle-motion pipeline"
            ),
            "measured_reference_search_scope": [
                item["path"] for item in reference_inventory["scope"]["included"]
            ],
            "measured_reference_search_result": reference_inventory["conclusion"],
            "measured_reference_search_confidence": reference_inventory["confidence"],
            "measured_reference_inventory_sha256": source_records[
                "measured_reference_inventory"
            ]["sha256"],
            "historical_reference_note": (
                "Workbench journals reference absent waveform UDFs and CFD-export "
                "CSV/image files; none is a present, provenance-qualified physical "
                "measurement"
            ),
        },
        "reconstruction_section": {
            "component": reconstruction["component"],
            "component_index_zero_based": reconstruction[
                "component_index_zero_based"
            ],
            "readout_index_zero_based": reconstruction[
                "section_index_zero_based"
            ],
            "support_voxels": reconstruction["support_voxels"],
            "direct_voxels": reconstruction["direct_voxels"],
            "regularized_voxels": reconstruction["regularized_voxels"],
            "direct_fraction": reconstruction["direct_voxels"]
            / reconstruction["support_voxels"],
            "voxel_area_m2": reconstruction["voxel_area_m2"],
            "voxelized_section_area_m2": reconstruction[
                "voxelized_section_area_m2"
            ],
            "voxel_size_mm_readout_phase_slice": reconstruction[
                "voxel_size_mm_readout_phase_slice"
            ],
            "Venc_m_per_s": reconstruction["Venc_m_per_s"],
            "regularized_contribution_fraction_at_reconstructed_peak": (
                regularized_peak_fraction
            ),
            "raw_to_corrected_flow_rmse_ml_per_s": raw_metrics["rmse_ml_per_s"],
        },
        "comparison_metrics": metrics,
        "metric_definitions": {
            "residual": "simulated Koma reconstruction minus retimed CFD target",
            "range_normalized_rmse": (
                "RMSE divided by max(target) minus min(target) over the included "
                "discrete midpoint pairs"
            ),
            "pearson_r": "Pearson correlation over paired discrete midpoint values",
            "origin_constrained_slope": (
                "least-squares reconstruction/target scale with zero intercept"
            ),
            "stroke_volume": (
                "midpoint Riemann sum of 15 flow values times final motion-period/15"
            ),
            "peak_ratio": (
                "maximum reconstructed discrete midpoint divided by maximum target "
                "discrete midpoint"
            ),
        },
        "non_synthetic_tail_sensitivity": non_tail_metrics,
        "section_sensitivity": sensitivity,
        "binwise_values": binwise,
        "limitations": [
            "The Fluent waveform is a solver-generated volume-flow monitor at the prescribed inlet boundary; it is not a pump/flowmeter measurement.",
            "The 0.640-0.720-s velocity-field tail is a controlled synthetic periodic closure.",
            "Periodic Catmull-Rom interpolation produces a very small negative undershoot near closure between nonnegative monitor knots; it is quantified as an interpolation artifact and left unchanged to match production behavior.",
            "The reconstruction is derived from the same CFD-driven motion, so correlation is not independent validation.",
            "Absolute flow uses a binary seed-derived voxel support and is sensitive to cross-section discretization.",
            "The support-constrained derivative contains spatially regularized edge voxels; direct and regularized contributions are reported separately.",
            "The midpoint comparison does not model the finite PC gradient/ADC temporal footprint as a separate reference average.",
            "No qualifying measured PC-MRI/4D-flow or PIV velocity dataset, nor its spatial registration and temporal phase mapping, was found in the audited workspace and targeted ANSYS project locations.",
            "The accepted Koma run uses VENC 5.0 m/s, whereas recovered scanner metadata reports VENC 3.0 m/s; this was a documented anti-aliasing protocol adaptation.",
        ],
        "independent_validation_input_required": {
            "waveform": (
                "machine-readable deidentified volumetric-flow samples from the same "
                "phantom run, including units and sign convention, sensor and "
                "cross-section location, timestamps/sampling interval, acquisition "
                "clock and trigger, phase-zero definition, cycle selection/averaging, "
                "calibration record, and uncertainty"
            ),
            "velocity": (
                "deidentified measured PC-MRI/4D-flow DICOM or NIfTI, or calibrated "
                "PIV data, from the same phantom run, including raw-versus-processed "
                "status, component axis/sign convention, units, voxel/grid geometry, "
                "timing and trigger, VENC, mask, background-phase/unwrap processing, "
                "and uncertainty"
            ),
            "registration": (
                "scanner/PIV-to-simulation spatial transform including orientation and "
                "handedness, matched cross-section/comparison ROI, and temporal phase "
                "mapping"
            ),
            "run_linkage": (
                "phantom geometry/version, fluid properties and temperature, pump "
                "program, measurement/scan run IDs, and timestamps demonstrating that "
                "waveform and velocity data belong to the matched experiment"
            ),
        },
        "inputs": source_records,
        "outputs": {
            "figure": artifact_record(figure_path),
            "markdown_summary": str(summary_path),
            "json_report": str(report_path),
        },
        "software": {
            "python": sys.version,
            "numpy": np.__version__,
            "h5py": h5py.__version__,
            "matplotlib": matplotlib.__version__,
            "platform": platform.platform(),
        },
    }
    summary_text = markdown_summary(report)
    summary_temporary = summary_path.with_suffix(".partial.md")
    summary_temporary.write_text(summary_text, encoding="utf-8")
    os.replace(summary_temporary, summary_path)
    report["outputs"]["markdown_summary"] = artifact_record(summary_path)

    report_temporary = report_path.with_suffix(".partial.json")
    report_temporary.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(report_temporary, report_path)

    print(f"Status: {report['status']}")
    print(f"Pearson r: {metrics['pearson_r']:.6f}")
    print(f"RMSE: {metrics['rmse_ml_per_s']:.3f} mL/s")
    print(
        "Range-normalized RMSE: "
        f"{100 * metrics['range_normalized_rmse']:.3f}%"
    )
    print(
        "Stroke-volume ratio: "
        f"{metrics['reconstructed_to_expected_stroke_volume_ratio']:.6f}"
    )
    print(f"Figure: {figure_path}")
    print(f"Summary: {summary_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
