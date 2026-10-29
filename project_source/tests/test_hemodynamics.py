"""Analytical controls for mask-aware SI hemodynamic post-processing."""

import numpy as np
import pytest

from mrsimtracks.hemodynamics import (
    differential_metrics,
    jet_metrics,
    midpoint_cycle_integrals,
    region_kinetic_energy,
    section_flow,
    velocity_gradient,
    wall_cycle_metrics,
    weighted_error_metrics,
)


def test_weighted_errors_use_paired_finite_values_not_zero_fill():
    got = weighted_error_metrics([1, 2, 3, 100], [2, 4, np.nan, 0],
                                 weights=[1, 3, 2, 8], valid=[1, 1, 1, 0])
    assert got["n_valid"] == 2
    assert got["paired_weight_fraction"] == pytest.approx(4 / 6)
    assert got["bias"] == pytest.approx(7 / 4)
    assert got["rmse"] == pytest.approx(np.sqrt(13 / 4))
    assert got["pearson_r"] == pytest.approx(1)


def test_zero_range_and_empty_errors_are_not_artificially_normalized():
    got = weighted_error_metrics([0, 0], [1, 1])
    assert np.isnan(got["nrmse_reference_range"])
    assert np.isnan(got["nrmse_reference_rms"])
    assert np.isnan(got["pearson_r"])
    empty = weighted_error_metrics([np.nan], [1])
    assert empty["n_valid"] == 0
    assert np.isnan(empty["rmse"])


@pytest.mark.parametrize("weights", [[-1, 1], [np.nan, 1]])
def test_bad_weights_fail_closed(weights):
    with pytest.raises(ValueError, match="Weights"):
        weighted_error_metrics([1, 2], [1, 2], weights)


def test_section_flow_sign_normal_and_partial_area():
    u = np.array([[2., 0, 0], [-1., 0, 0], [np.nan, 0, 0]])
    result = section_flow(u, [1e-6, 2e-6, 1e-6], normal=[2, 0, 0])
    assert result["net_m3_s"] == pytest.approx(0)
    assert result["forward_m3_s"] == pytest.approx(2e-6)
    assert result["reverse_m3_s"] == pytest.approx(2e-6)
    assert result["sampled_area_fraction"] == pytest.approx(0.75)
    assert np.isnan(section_flow(u, 1, valid=[False] * 3)["net_m3_s"])


def test_midpoint_cycle_integrates_forward_and_reverse_net_flow():
    result = midpoint_cycle_integrals([2e-6, 2e-6, -1e-6, -1e-6], 2)
    assert result["net_m3"] == pytest.approx(1e-6)
    assert result["forward_m3"] == pytest.approx(2e-6)
    assert result["reverse_m3"] == pytest.approx(1e-6)
    with pytest.raises(ValueError, match="complete"):
        midpoint_cycle_integrals([1, np.nan], 1)


def test_constant_velocity_energy_does_not_invent_density():
    u = [[3., 4, 0], [np.nan, 0, 0]]
    got = region_kinetic_energy(u, [2e-6, 2e-6])
    assert got["energy_per_density_m5_s2"] == pytest.approx(25e-6)
    assert got["sampled_volume_fraction"] == pytest.approx(0.5)
    assert "energy_j" not in got
    got = region_kinetic_energy(u, [2e-6, 2e-6], density_kg_m3=1000)
    assert got["energy_j"] == pytest.approx(0.025)


def _grid():
    dz, dy, dx = 0.003, 0.002, 0.001
    z, y, x = np.meshgrid(np.arange(-4, 5) * dz, np.arange(-5, 6) * dy,
                          np.arange(-6, 7) * dx, indexing="ij")
    return x, y, z, (dx, dy, dz)


def test_solid_body_rotation_curl_and_zero_strain():
    x, y, z, spacing = _grid()
    omega = 7.
    u = np.stack((-omega * y, omega * x, np.ones_like(z) * 0.2), axis=-1)
    grad, valid = velocity_gradient(u, spacing)
    result = differential_metrics(grad, u, viscosity_pa_s=0.004)
    expected = np.broadcast_to([0, 0, 2 * omega], result["vorticity_s_inv"][valid].shape)
    np.testing.assert_allclose(result["vorticity_s_inv"][valid], expected, atol=1e-12)
    np.testing.assert_allclose(result["divergence_s_inv"][valid], 0, atol=1e-12)
    np.testing.assert_allclose(result["dissipation_w_m3"][valid], 0, atol=1e-25)
    np.testing.assert_allclose(result["helicity_m_s_inv2"][valid], 2.8, atol=1e-12)


def test_poiseuille_gradients_and_dissipation_are_exact_for_quadratic():
    x, y, z, spacing = _grid()
    radius, peak = 0.014, 0.8
    u = np.zeros(x.shape + (3,))
    u[..., 0] = peak * (1 - (y**2 + z**2) / radius**2)
    fluid = y**2 + z**2 < radius**2
    grad, valid = velocity_gradient(u, spacing, valid=fluid)
    np.testing.assert_allclose(grad[..., 0, 1][valid], (-2 * peak * y / radius**2)[valid], atol=1e-10)
    np.testing.assert_allclose(grad[..., 0, 2][valid], (-2 * peak * z / radius**2)[valid], atol=1e-10)
    metrics = differential_metrics(grad, u)
    expected_per_mu = 4 * peak**2 * (y**2 + z**2) / radius**4
    np.testing.assert_allclose(metrics["dissipation_per_viscosity_s_inv2"][valid], expected_per_mu[valid], atol=1e-9)
    np.testing.assert_allclose(metrics["helicity_m_s_inv2"][valid], 0, atol=1e-12)


def test_poiseuille_flow_and_energy_against_analytical_integrals():
    # Equal-area radial annuli, sampled at midpoint r^2, integrate flow exactly.
    n, radius, peak, length = 1000, 0.01, 0.8, 0.1
    radius_sq = (np.arange(n) + 0.5) / n * radius**2
    u = np.zeros((n, 3))
    u[:, 0] = peak * (1 - radius_sq / radius**2)
    areas = np.full(n, np.pi * radius**2 / n)
    assert section_flow(u, areas)["net_m3_s"] == pytest.approx(np.pi * radius**2 * peak / 2)
    energy = region_kinetic_energy(u, areas * length)
    assert energy["energy_per_density_m5_s2"] == pytest.approx(
        np.pi * radius**2 * length * peak**2 / 6, rel=1e-6)


def test_gradient_mask_erodes_center_and_six_neighbors_without_wall_zero_fill():
    u = np.ones((7, 7, 7, 3))
    u[3, 3, 3] = np.nan
    grad, valid = velocity_gradient(u, [1, 2, 3])
    assert np.count_nonzero(valid) == 5**3 - 7
    assert np.all(np.isnan(grad[~valid]))
    np.testing.assert_array_equal(grad[valid], 0)
    mask = np.ones((7, 7, 7), bool)
    mask[2, 2, 2] = False
    u[:] = 1
    _, valid = velocity_gradient(u, [1, 2, 3], valid=mask)
    assert not valid[2, 2, 2] and not valid[2, 2, 3]


def test_jet_centroid_width_and_direction_are_flux_weighted():
    u = np.array([[1., 0, 0], [3., 0, 0], [-10, 0, 0]])
    positions = np.array([[0., -1, 0], [0., 1, 0], [0, 20, 0]]) * 0.01
    result = jet_metrics(u, positions, 1e-6)
    np.testing.assert_allclose(result["centroid_m"], [0, 0.005, 0])
    assert result["rms_radius_m"] == pytest.approx(np.sqrt(0.75) * 0.01)
    np.testing.assert_array_equal(result["direction_unit"], [1, 0, 0])
    assert result["angle_to_normal_deg"] == 0
    empty = jet_metrics(-np.abs(u), positions, 1e-6)
    assert np.isnan(empty["rms_radius_m"])


def test_jet_uses_explicit_common_absolute_threshold():
    u = np.array([[1., 0, 0], [3., 0, 0]])
    positions = np.array([[0., -0.01, 0], [0., 0.01, 0]])
    result = jet_metrics(u, positions, 1e-6, minimum_forward_speed_m_s=2)
    assert result["forward_flux_m3_s"] == pytest.approx(3e-6)
    np.testing.assert_allclose(result["centroid_m"], positions[1])
    assert result["rms_radius_m"] == 0
    missing = jet_metrics(np.full((2, 3), np.nan), positions, 1e-6)
    assert np.isnan(missing["forward_flux_m3_s"])


def test_wall_cycles_constant_reversing_zero_and_missing():
    shear = np.zeros((4, 4, 3))
    shear[:, 0, 0] = 2
    shear[:, 1, 0] = [2, 2, -2, -2]
    shear[:, 3, 0] = 2
    shear[1, 3, 0] = np.nan
    result = wall_cycle_metrics(shear, 1.0, complete_cycle=True)
    np.testing.assert_allclose(result["tawss"][:3], [2, 2, 0])
    np.testing.assert_allclose(result["osi"][:2], [0, 0.5])
    assert np.isnan(result["osi"][2])
    assert np.isnan(result["tawss"][3])
    assert not result["complete_cycle_valid"][3]
    with pytest.raises(ValueError, match="complete"):
        wall_cycle_metrics(shear, 1.0, complete_cycle=False)


def test_wall_cycle_invalid_mask_is_not_ignored():
    result = wall_cycle_metrics(np.ones((3, 2, 3)), 1.0, complete_cycle=True,
                                valid=[[True, True], [False, True], [True, True]])
    assert np.isnan(result["tawss"][0])
    assert result["osi"][1] == pytest.approx(0)


@pytest.mark.parametrize("normal", [[0, 0, 0], [np.nan, 1, 0], [1, 0]])
def test_invalid_normals_are_rejected(normal):
    with pytest.raises(ValueError, match="Normal"):
        section_flow([[1., 0, 0]], 1, normal)


@pytest.mark.parametrize("spacing", [[0, 1, 1], [np.nan, 1, 1], [1, 1]])
def test_invalid_spacing_is_rejected(spacing):
    with pytest.raises(ValueError, match="Spacing"):
        velocity_gradient(np.ones((3, 3, 3, 3)), spacing)
