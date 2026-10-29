import numpy as np
import pytest

from mrsimtracks.resolved_energy import cycle_energy_decomposition


def test_constant_velocity_is_resolved_not_turbulent():
    u = np.tile([3., 4., 0.], (4, 2, 1))
    r = cycle_energy_decomposition(u, [2e-6, 1e-6], density_kg_m3=1000)
    assert r['mean_resolved_ke_j'] == pytest.approx(.0375)
    assert r['mean_vector_ke_j'] == pytest.approx(.0375)
    assert r['coherent_variation_ke_j'] == 0
    assert not any(k.lower().startswith('tke') for k in r)


def test_pulsatile_reversal_has_zero_mean_but_not_zero_energy():
    u = np.zeros((4, 1, 3)); u[:, 0, 0] = [2, -2, 2, -2]
    r = cycle_energy_decomposition(u, 3e-6)
    assert r['mean_vector_ke_per_density_m5_s2'] == 0
    assert r['coherent_variation_ke_per_density_m5_s2'] == pytest.approx(6e-6)
    assert r['coherent_fraction'] == pytest.approx(1)
    assert r['relative_closure'] < 1e-14
    assert 'mean_resolved_ke_j' not in r


def test_weighted_time_identity_and_rotation_invariance():
    rng = np.random.default_rng(370); u = rng.normal(size=(7, 3, 4, 3))
    v = rng.uniform(size=(3, 4)); w = np.arange(1., 8.)
    a = cycle_energy_decomposition(u, v, time_weights=w)
    b = cycle_energy_decomposition(u[..., [2, 0, 1]], v, time_weights=w*8)
    expected = .5*np.average(np.sum(u*u, axis=-1), axis=0, weights=w)
    assert a['mean_resolved_ke_per_density_m5_s2'] == pytest.approx(np.sum(expected*v))
    assert a['relative_closure'] < 1e-14
    assert a['coherent_variation_ke_per_density_m5_s2'] == pytest.approx(b['coherent_variation_ke_per_density_m5_s2'])


def test_missing_selected_bin_fails_and_unselected_nan_remains_nan():
    u = np.ones((2, 2, 3)); u[:, 1] = np.nan
    r = cycle_energy_decomposition(u, [1, 1], [True, False])
    assert r['sampled_volume_m3'] == 1
    assert np.isnan(r['mean_velocity_m_s'][1]).all()
    with pytest.raises(ValueError, match='every bin'):
        cycle_energy_decomposition(u, [1, 1])


@pytest.mark.parametrize('weights', [[1, 0], [1, -1], [1, np.nan], [1]])
def test_invalid_time_weights(weights):
    with pytest.raises(ValueError, match='Time weights'):
        cycle_energy_decomposition(np.ones((2, 1, 3)), 1, time_weights=weights)


@pytest.mark.parametrize('volume', [-1, np.nan, 0])
def test_invalid_or_empty_volume(volume):
    with pytest.raises(ValueError):
        cycle_energy_decomposition(np.ones((2, 1, 3)), volume)


@pytest.mark.parametrize('density', [-1, 0, np.nan])
def test_invalid_density(density):
    with pytest.raises(ValueError, match='Density'):
        cycle_energy_decomposition(np.ones((2, 1, 3)), 1, density_kg_m3=density)
