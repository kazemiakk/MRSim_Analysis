"""Small numerical tests for E-025 orchestration, without production inputs."""

from __future__ import annotations

import json

import numpy as np
import pytest

import scripts.compare_dp150_hemodynamics as comparison


def test_paired_metrics_vector_distance_and_rms_normalization():
    reference = np.array([[1., 0., 0.], [0., 2., 0.]])
    estimate = reference + [1., 2., 2.]
    result = comparison.paired_velocity_metrics(reference, estimate, np.array([1., 3.]), np.ones(2, bool))
    assert result["vector_rmse_m_s"] == pytest.approx(3.)
    assert result["vector_nrmse_reference_rms"] == pytest.approx(3. / np.sqrt(13/4))
    assert result["X"]["bias"] == pytest.approx(1.)
    assert result["Y"]["bias"] == pytest.approx(2.)
    assert result["Z"]["bias"] == pytest.approx(2.)
    # Speed error and vector error are different observables.
    assert result["speed"]["rmse"] != pytest.approx(result["vector_rmse_m_s"])


def test_paired_metrics_excludes_missing_masked_and_zero_weight_vectors():
    reference = np.array([[1., 2., 3.], [np.nan, 2., 3.], [100., 100., 100.], [50., 50., 50.]])
    estimate = np.array([[2., 2., 3.], [1., 2., 3.], [0., 0., 0.], [0., 0., 0.]])
    result = comparison.paired_velocity_metrics(reference, estimate, np.array([2., 1., 1., 0.]),
                                                np.array([True, True, False, True]))
    assert result["vector_rmse_m_s"] == pytest.approx(1.)
    assert result["vector_nrmse_reference_rms"] == pytest.approx(1./np.sqrt(14))
    assert result["speed"]["n_valid"] == 1


def test_paired_metrics_empty_support_and_zero_reference_are_undefined():
    reference = np.zeros((2, 3))
    empty = comparison.paired_velocity_metrics(reference, reference, np.ones(2), np.zeros(2, bool))
    assert np.isnan(empty["vector_rmse_m_s"])
    assert np.isnan(empty["vector_nrmse_reference_rms"])
    zero = comparison.paired_velocity_metrics(reference, np.ones((2, 3)), np.ones(2), np.ones(2, bool))
    assert zero["vector_rmse_m_s"] == pytest.approx(np.sqrt(3))
    assert np.isnan(zero["vector_nrmse_reference_rms"])


def test_finite_summary_uses_weights_and_excludes_unsupported_values():
    result = comparison.finite_summary(
        np.array([1., 3., 100., np.nan, 1000.]),
        np.array([1., 3., 0., 2., 9.]), np.array([True, True, True, True, False]),
    )
    assert result == pytest.approx(dict(mean=2.5, max=3., p99=3., weight=4.))
    empty = comparison.finite_summary(np.array([np.nan]), np.array([1.]), np.array([True]))
    assert empty["weight"] == 0
    assert all(np.isnan(empty[key]) for key in ("mean", "max", "p99"))


def test_finite_summary_single_value_and_weighted_percentile():
    one = comparison.finite_summary(np.array([7.]), np.array([2.]), np.array([True]))
    assert one == pytest.approx(dict(mean=7., max=7., p99=7., weight=2.))
    # Ninety-nine percent of the weight lies at the lower value. The explicit
    # midpoint-CDF interpolation convention yields 9.9, not the unweighted p99.
    result = comparison.finite_summary(np.array([0., 10.]), np.array([99., 1.]), np.ones(2, bool))
    assert result["mean"] == pytest.approx(.1)
    assert result["p99"] == pytest.approx(9.9)


def test_normalized_smoothing_preserves_constant_flow_and_missing_holes():
    valid = np.zeros((9, 11, 13), bool)
    valid[2:7, 2:9, 2:11] = True
    valid[4, 5, 6] = False
    u = np.full((*valid.shape, 3), np.nan)
    u[valid] = [1., -2., .5]
    result = comparison.matched_smooth(u, valid, np.array([.001, .002, .004]), sigma_m=.001)
    np.testing.assert_allclose(result[valid], np.tile([1., -2., .5], (valid.sum(), 1)), atol=1e-12)
    assert np.isnan(result[~valid]).all()


def test_zero_smoothing_is_identity_and_components_are_not_mixed():
    rng = np.random.default_rng(144)
    u = rng.normal(size=(5, 6, 7, 3))
    valid = np.ones(u.shape[:-1], bool)
    identity = comparison.matched_smooth(u, valid, [1., 2., 3.], sigma_m=0)
    np.testing.assert_allclose(identity, u)
    swapped = comparison.matched_smooth(u[..., [2, 0, 1]], valid, [1., 2., 3.], sigma_m=1)
    original = comparison.matched_smooth(u, valid, [1., 2., 3.], sigma_m=1)
    np.testing.assert_allclose(swapped, original[..., [2, 0, 1]])


def test_safe_json_recursively_converts_numpy_and_nonfinite_to_null():
    source = {
        "array": np.array([[np.nan, np.inf, -np.inf, 1.25]]),
        "scalar": np.float32(2.5), "integer": np.int64(15),
        "bool": np.bool_(True), "tuple": (np.float64(np.nan), {2: np.int32(3)}),
    }
    safe = comparison.safe_json(source)
    assert safe == dict(array=[[None, None, None, 1.25]], scalar=2.5, integer=15,
                        bool=True, tuple=[None, {"2": 3}])
    json.dumps(safe, allow_nan=False)


def _legacy_regression_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(comparison, "PRODUCTION", tmp_path)
    folder = tmp_path / "final_image_analysis/cfd_koma_flow_waveform_e024"
    folder.mkdir(parents=True)
    corrected = np.full((15, 2, 3, 71, 3), np.nan)
    support = np.zeros((2, 3, 71), bool)
    support[0, 0, 70] = support[1, 2, 70] = True
    for b in range(15):
        corrected[b, 0, 0, 70] = [b+1, 100., 100.]
        corrected[b, 1, 2, 70] = [2*(b+1), 100., 100.]
    expected = 3 * np.arange(1, 16) * 1.25e-3 * 2.5e-3
    payload = dict(binwise_values=[dict(koma_reconstructed_m3_per_s=float(q)) for q in expected],
                   comparison_metrics={"test_fixture": True})
    (folder / "dp150_cfd_koma_flow_waveform_report.json").write_text(json.dumps(payload), encoding="utf-8")
    return corrected, support, expected


def test_legacy_regression_uses_axis_x_plane70_and_binary_support(tmp_path, monkeypatch):
    corrected, support, expected = _legacy_regression_fixture(tmp_path, monkeypatch)
    result = comparison.e024_regression(corrected, support, 1.012)
    assert result["passed"] is True
    assert result["max_abs_difference_m3_s"] < 1e-18
    assert result["reproduced_stroke_volume_ml"] == pytest.approx(expected.sum() * 1.012 / 15 * 1e6)
    assert result["legacy_values"] == {"test_fixture": True}


def test_legacy_regression_fails_closed_on_changed_velocity(tmp_path, monkeypatch):
    corrected, support, _ = _legacy_regression_fixture(tmp_path, monkeypatch)
    corrected[5, 0, 0, 70, 0] += 1
    with pytest.raises(ValueError, match="regression failed"):
        comparison.e024_regression(corrected, support, 1.012)


def test_combine_error_summaries_matches_pooled_raw_observations():
    reference = np.array([1., 2., 3., 4., 5.])
    estimate = np.array([1.2, 1.5, 4.0, 2.0, 7.0])
    weights = np.array([1., 4., 2., 3., 1.])
    first = comparison.weighted_error_metrics(reference[:2], estimate[:2], weights[:2])
    second = comparison.weighted_error_metrics(reference[2:], estimate[2:], weights[2:])
    combined = comparison.combine_error_summaries([first, second])
    pooled = comparison.weighted_error_metrics(reference, estimate, weights)
    for key in combined:
        assert combined[key] == pytest.approx(pooled[key])
    assert combined["rmse"] != pytest.approx(np.mean([first["rmse"], second["rmse"]]))


def test_combine_error_summaries_handles_empty_and_zero_reference():
    empty = comparison.weighted_error_metrics(np.array([np.nan]), np.array([1.]))
    no_values = comparison.combine_error_summaries([empty])
    assert no_values["weight_sum"] == 0
    assert np.isnan(no_values["rmse"])
    zero = comparison.weighted_error_metrics(np.zeros(2), np.array([1., 2.]))
    combined = comparison.combine_error_summaries([empty, zero])
    assert combined["weight_sum"] == 2
    assert combined["rmse"] == pytest.approx(np.sqrt(2.5))
    assert np.isnan(combined["nrmse_reference_rms"])


def test_combine_error_summaries_preserves_signed_bias_cancellation():
    positive = comparison.weighted_error_metrics(np.array([2.]), np.array([3.]))
    negative = comparison.weighted_error_metrics(np.array([2.]), np.array([1.]))
    combined = comparison.combine_error_summaries([positive, negative])
    assert combined["bias"] == pytest.approx(0)
    assert combined["mae"] == pytest.approx(1)
    assert combined["rmse"] == pytest.approx(1)
    assert combined["nrmse_reference_rms"] == pytest.approx(.5)
