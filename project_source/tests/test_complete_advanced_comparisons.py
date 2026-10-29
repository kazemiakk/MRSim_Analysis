import numpy as np
import pytest

from scripts.complete_advanced_comparisons import scalar_fields, scalar_region_integrals, pair_cores, flow_summaries


def grid():
    z, y, x = np.meshgrid(np.arange(5)*.003, np.arange(6)*.002, np.arange(7)*.001, indexing='ij')
    return x, y, z


def test_shear_scalar_units_and_interior_loss_integral():
    x, y, z = grid(); a = 4.; u = np.stack([a*y, 0*x, 0*x], axis=-1)
    f, valid = scalar_fields(u, [.001, .002, .003], np.ones(x.shape, bool))
    assert valid.sum() == 3*4*5
    assert np.allclose(f['vorticity_magnitude_s_inv'][valid], a)
    assert np.allclose(f['strain_frobenius_s_inv'][valid], a/np.sqrt(2))
    assert np.allclose(f['dissipation_per_viscosity_s_inv2'][valid], a*a)
    assert np.allclose(f['enstrophy_s_inv2'][valid], a*a/2)
    assert np.allclose(f['divergence_s_inv'][valid], 0)
    r = scalar_region_integrals(f, np.full(x.shape, 6e-9), valid)
    assert r['dissipation_per_viscosity_m3_s2'] == pytest.approx(16*60*6e-9)


def test_rotation_is_not_viscous_strain_or_turbulence():
    x, y, z = grid(); omega = 2.
    f, valid = scalar_fields(np.stack([-omega*y, omega*x, 0*x], axis=-1), [.001, .002, .003], np.ones(x.shape, bool))
    assert np.allclose(f['vorticity_magnitude_s_inv'][valid], 4)
    assert np.allclose(f['dissipation_per_viscosity_s_inv2'][valid], 0, atol=1e-25)
    assert np.allclose(f['enstrophy_s_inv2'][valid], 8)


def test_zero_vorticity_relative_helicity_is_missing_not_zero():
    x, y, z = grid(); u = np.stack([0*x+1, 0*x, 0*x], axis=-1)
    f, valid = scalar_fields(u, [.001, .002, .003], np.ones(x.shape, bool))
    assert np.isnan(f['relative_helicity'][valid]).all()


def test_missing_velocity_excludes_its_gradient_stencil():
    x, y, z = grid(); u = np.stack([y, 0*x, 0*x], axis=-1); u[2, 2, 2] = np.nan
    f, valid = scalar_fields(u, [.001, .002, .003], np.ones(x.shape, bool))
    assert not valid[2, 2, 2] and not valid[2, 2, 3]
    assert np.isnan(f['vorticity_magnitude_s_inv'][2, 2, 3])


def core(method, direction, centroid=(0, 0, 0)):
    return dict(bin=1, section='S', threshold_fraction=.5, method=method,
                centroid_m=centroid, direction_unit=direction, rms_radius_m=.01)


def test_flow_core_direction_and_missing_values():
    r = pair_cores([core('CFD', (1, 0, 0)), core('Koma', (0, 1, 0), (.001, 0, 0))])[0]
    assert r['direction_error_deg'] == pytest.approx(90)
    assert r['centroid_error_mm'] == pytest.approx(1)
    r = pair_cores([core('CFD', (None, None, None)), core('Koma', (1, 0, 0))])[0]
    assert not r['paired_core_available'] and np.isnan(r['direction_error_deg'])


def test_duplicate_and_unpaired_cores_fail_closed():
    with pytest.raises(ValueError, match='Duplicate'):
        pair_cores([core('CFD', (1, 0, 0))]*2)
    with pytest.raises(ValueError, match='Unpaired'):
        pair_cores([core('CFD', (1, 0, 0))])


def test_simultaneous_reverse_transport_is_not_net_reverse_time_partition():
    rows = [dict(section='S', bin=b, method=m, net_m3_s=1e-6,
                 forward_m3_s=2e-6, reverse_m3_s=1e-6, sampled_area_m2=1e-4)
            for m in ['CFD', 'Koma'] for b in [1, 2]]
    for r in flow_summaries(rows, .8):
        assert r['net_reverse_time_partition_ml'] == 0
        assert r['simultaneous_reverse_volume_ml'] == pytest.approx(.8)
        assert r['net_nominal_cycle_ml'] == pytest.approx(.8)
