import numpy as np
import pytest

from scripts.prepare_3d_flow_control import geometry, trapezoid_moments
from scripts.analyze_3d_flow_control import known_velocity_metrics, conditional_error
from scripts.finalize_3d_flow_control import repo_name, ROOT


def test_three_models_have_fixed_known_velocities_and_3d_geometry():
    data = geometry()
    assert len(data['static_landmarks']['positions']) == 5
    assert all(np.unique(data['static_landmarks']['positions'][:, c]).size > 1 for c in range(3))
    for key, speed in [('plug_positive', 1.5), ('plug_negative', -.75)]:
        model = data[key]
        assert model['positions'].shape == (19200, 3)
        np.testing.assert_array_equal(model['velocity'], [speed, 0, 0])
        assert [np.unique(model['positions'][:, c]).size for c in range(3)] == [400, 8, 6]
        assert np.all(model['rho'] == 1)
    np.testing.assert_array_equal(data['plug_positive']['positions'], data['plug_negative']['positions'])


def test_gradient_moments_of_symmetric_triangle():
    times = np.array([0., .5, 1., 1.5, 2., 3.])
    m0, m1 = trapezoid_moments(times, 2., 1., 0., 1., 0.)
    assert m0[1] == pytest.approx(.25)
    assert m1[1] == pytest.approx(1/12)
    assert m0[-1] == pytest.approx(2.)
    assert m1[-1] == pytest.approx(2.)
    np.testing.assert_array_equal(m0[-2:], [2, 2])


def test_gradient_moments_translate_in_time_and_scale_linearly():
    t = np.array([0., .5, 1., 2., 3., 4.])
    a, b = trapezoid_moments(t, 3., .5, 1., .5, 0.)
    c, d = trapezoid_moments(t+5., -6., .5, 1., .5, 5.)
    np.testing.assert_allclose(c, -2*a)
    np.testing.assert_allclose(d, -2*(b+5*a))


def test_bipolar_gradient_has_zero_area_but_known_first_moment():
    t = np.array([10.])
    a0, a1 = trapezoid_moments(t, 1., .1, .2, .1, 0.)
    b0, b1 = trapezoid_moments(t, -1., .1, .2, .1, .5)
    assert float((a0+b0)[0]) == pytest.approx(0., abs=1e-15)
    assert float((a1+b1)[0]) == pytest.approx(-.15)


def test_known_velocity_gate_reports_coverage_without_imputing_missing_phase():
    ref = np.ones((15, 2, 2, 2), np.complex64)
    images = {e:ref.copy() for e in ('REF', 'X', 'Y', 'Z')}
    images['X'] *= np.exp(1j*np.pi*1.5/5)
    images['Y'][0, 0, 0, 0] = 0
    core = np.ones((2, 2, 2), bool)
    result, fields, mask = known_velocity_metrics(images, np.array([1.5, 0, 0]), core, core)
    assert result['fixed_core_coverage_fraction'] == pytest.approx(7/8)
    assert result['metrics']['vector_rmse_m_s'] < 1e-6
    assert not mask[0, 0, 0]
    assert result['bins'][0]['core_coverage_fraction'] == pytest.approx(7/8)
    assert result['bins'][1]['core_coverage_fraction'] == 1
    direct = np.angle(images['X']*images['REF'].conj())*(5/np.pi)
    np.testing.assert_array_equal(fields[..., 0], direct)


def test_conditional_signal_law_uses_native_gradient_sign_and_absolute_clock():
    rng = np.random.default_rng(8)
    ref = rng.normal(size=(8, 400))+1j*rng.normal(size=(8, 400))
    times = np.linspace(0, 2000, 8)
    m0 = np.zeros((400, 3)); m0[:, 0] = 1e-10
    m1 = np.zeros((400, 3)); m1[:, 0] = -.1
    velocity = np.array([1.5, 0, 0])
    predicted = ref*np.exp(-2j*np.pi*((m1@velocity)[None]+times[:, None]*(m0@velocity)[None]))
    result = conditional_error(ref, predicted, times, m0, m1, velocity)
    assert result['relative_complex_L2'] < 1e-15
    wrong_sign = conditional_error(ref, predicted.conjugate(), times, m0, m1, velocity)
    assert wrong_sign['relative_complex_L2'] > .1


def test_manifest_path_normalization_accepts_repo_relative_and_absolute_paths():
    target = ROOT/'outputs/known_flow_3d_e051/analysis_v1/report.json'
    assert repo_name(target) == repo_name(target.relative_to(ROOT))
    with pytest.raises(ValueError):
        repo_name(ROOT.parent/'outside_e051_report.json')
