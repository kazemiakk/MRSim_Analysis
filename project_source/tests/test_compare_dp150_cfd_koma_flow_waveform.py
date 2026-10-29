from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from scripts.compare_dp150_cfd_koma_flow_waveform import (
    close_monitor_tail,
    integrate_section,
    load_reconstruction_sections,
    parse_fluent_monitor,
    periodic_cubic_sample,
    sha256,
    validate_inputs,
    waveform_metrics,
)


def test_parse_fluent_monitor_reads_numeric_records(tmp_path: Path) -> None:
    path = tmp_path / "monitor.out"
    path.write_text(
        '\n'.join(
            [
                '"vfr_inlet-rfile"',
                '("Time Step" "vfr_inlet" "flow-time")',
                "0 0 0",
                "1 1.25e-4 0.01",
                "2 0 0.02",
            ]
        ),
        encoding="utf-8",
    )
    steps, times, flow = parse_fluent_monitor(path)
    np.testing.assert_array_equal(steps, [0, 1, 2])
    np.testing.assert_allclose(times, [0.0, 0.01, 0.02])
    np.testing.assert_allclose(flow, [0.0, 1.25e-4, 0.0])


def test_tail_closure_preserves_prefix_and_closes_endpoint() -> None:
    times = np.linspace(0.0, 0.72, 73)
    raw = 1.0 + times
    weights, closed = close_monitor_tail(
        times,
        raw,
        blend_start_s=0.64,
        blend_end_s=0.72,
    )
    prefix = times <= 0.64
    np.testing.assert_allclose(closed[prefix], raw[prefix])
    assert weights[0] == 0.0
    assert weights[-1] == 1.0
    assert closed[-1] == closed[0]


def test_periodic_cubic_matches_knots_and_constant_wrap() -> None:
    times = np.linspace(0.0, 1.0, 11)
    values = np.full(times.shape, 3.25)
    values[-1] = values[0]
    query = np.asarray([0.0, 0.15, 0.95, 1.0, 1.15])
    got = periodic_cubic_sample(times, values, query, period_s=1.0)
    np.testing.assert_allclose(got, 3.25)


def test_periodic_cubic_rejects_open_endpoint() -> None:
    times = np.linspace(0.0, 1.0, 11)
    values = np.arange(11, dtype=float)
    with pytest.raises(ValueError, match="endpoints are not closed"):
        periodic_cubic_sample(times, values, 0.2, period_s=1.0)


def test_integrate_section_uses_only_selected_voxels() -> None:
    velocity = np.asarray(
        [
            [[1.0, np.nan], [2.0, np.nan]],
            [[3.0, np.nan], [4.0, np.nan]],
        ]
    )
    mask = np.asarray([[True, False], [True, False]])
    got = integrate_section(velocity, mask, voxel_area_m2=2.0)
    np.testing.assert_allclose(got, [6.0, 14.0])


def test_waveform_metrics_recover_exact_scale() -> None:
    expected = np.asarray([1.0, 2.0, 4.0, 3.0])
    reconstructed = 0.9 * expected
    metrics = waveform_metrics(expected, reconstructed, bin_spacing_s=0.25)
    assert metrics["pearson_r"] == pytest.approx(1.0)
    assert metrics["origin_constrained_slope"] == pytest.approx(0.9)
    assert metrics["reconstructed_to_expected_peak_ratio"] == pytest.approx(0.9)
    assert metrics["reconstructed_to_expected_stroke_volume_ratio"] == pytest.approx(
        0.9
    )
    assert metrics["stroke_volume_percent_error"] == pytest.approx(-10.0)
    assert metrics["range_normalization_denominator_ml_per_s"] == pytest.approx(3e6)
    assert metrics["residual_sign_counts"] == {
        "negative": 4,
        "zero": 0,
        "positive": 0,
    }
    assert metrics["peak_bin_difference"] == 0


def test_validate_inputs_binds_axis_case_and_derivative_hashes(
    tmp_path: Path,
) -> None:
    case = tmp_path / "FFF-18.cas.h5"
    case.write_bytes(b"case")
    reconstruction = tmp_path / "reconstruction.h5"
    reconstruction.write_bytes(b"reconstruction")
    settings = tmp_path / "FFF.set"
    settings.write_text(
        '\n'.join(
            [
                '(case-file-units "si")',
                'name . "vfr_inlet"',
                'surface-volumeflowrate',
                'file-name ".\\\\vfr_inlet-rfile.out"',
            ]
        ),
        encoding="utf-8",
    )
    monitor = tmp_path / "vfr_inlet-rfile.out"
    monitor.write_text("0 0 0\n", encoding="utf-8")
    source_pvd_hash = "source-pvd"
    periodic_pvd_hash = "periodic-pvd"
    result = validate_inputs(
        flow_monitor=monitor,
        fluent_settings=settings,
        fluent_case=case,
        conversion_report={
            "source": {"case_file": str(case), "case_sha256": sha256(case)},
            "output": {"pvd_sha256": source_pvd_hash},
        },
        periodic_report={
            "source": {"pvd_sha256": source_pvd_hash},
            "output": {"pvd_sha256": periodic_pvd_hash},
            "closure_method": {
                "type": "quintic_tail_blend_to_opening_field",
                "blend_start_s": 0.64,
                "blend_end_s": 0.72,
            },
            "validation": {"passed": True, "endpoint_exactly_equal": True},
        },
        tracking_report={
            "inputs": {"periodic_pvd_sha256": periodic_pvd_hash},
            "run": {
                "time_interpolation": "cubic",
                "axis_order_zero_based": [2, 0, 1],
                "mri_axes_from_source": ["z", "x", "y"],
                "duration_s": 0.72,
            },
            "quality_control": {"acceptance": {"passed": True}},
        },
        phantom_report={
            "status": "PASS",
            "source_cycle_s": 0.72,
            "realized_sequence_rr_s": 0.7169,
        },
        simulation_report={
            "status": "PASS",
            "seam_safe_motion": True,
            "motion_timing": {
                "rescale_requested": True,
                "seam_safe_requested": True,
                "original_period_s": 0.7169,
                "effective_period_s": 1.012,
                "sequence_rr_s": 1.012,
            },
        },
        support_report={
            "status": "PASS_RECONSTRUCTION_DERIVATIVE",
            "outputs": {
                "reconstruction_h5": {
                    "path": str(reconstruction),
                    "sha256": sha256(reconstruction),
                }
            },
        },
        reconstruction=reconstruction,
    )
    assert result["axis_and_sign"]["comparison_axis"] == "MRI +X/readout = CFD +z"
    assert result["axis_and_sign"]["sign_inversion_applied"] is False
    assert all(result["run_linkage"].values())


def test_load_reconstruction_sections_uses_declared_hdf5_order_and_area(
    tmp_path: Path,
) -> None:
    path = tmp_path / "reconstruction.h5"
    corrected = np.zeros((3, 15, 2, 2, 1), dtype=np.float32)
    raw = np.zeros_like(corrected)
    corrected[0, :, :, :, 0] = 2.0
    raw[0, :, :, :, 0] = 1.0
    support = np.asarray([[[True], [False]], [[True], [False]]])
    direct = np.asarray([[[True], [False]], [[False], [False]]])
    regularized = support & ~direct
    with h5py.File(path, "w") as h5:
        h5.attrs["velocity_components"] = "X,Y,Z"
        h5.attrs["voxel_size_mm_readout_phase_slice"] = [1.25, 1.25, 2.5]
        h5.attrs["RR_s"] = 1.012
        h5.attrs["Venc_m_per_s"] = 5.0
        h5.create_dataset("velocity_corrected_m_per_s", data=corrected)
        h5.create_dataset("velocity_raw_m_per_s", data=raw)
        h5.create_dataset("support_mask", data=support)
        h5.create_dataset("direct_mask", data=direct)
        h5.create_dataset("regularized_mask", data=regularized)

    loaded = load_reconstruction_sections(path, section_index=0, section_window=0)
    expected_area = 1.25 * 2.5 * 1e-6
    np.testing.assert_allclose(
        loaded["corrected_flow_m3_per_s"], 2 * 2.0 * expected_area
    )
    np.testing.assert_allclose(
        loaded["raw_flow_m3_per_s"], 2 * 1.0 * expected_area
    )
    assert loaded["component_index_zero_based"] == 0
    assert loaded["support_voxels"] == 2
