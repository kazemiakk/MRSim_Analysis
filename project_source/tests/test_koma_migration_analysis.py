import numpy as np
import pytest
from scripts.analyze_koma_migration import image_quantities, paired_delta
from scripts.validate_koma_migration import interpolate_manual


def test_velocity_mask_and_fixed_support():
    ref = np.ones((2, 2, 2, 2), complex)
    images = {'REF': ref, 'X': ref*np.exp(.1j*np.pi), 'Y': ref*np.exp(-.2j*np.pi), 'Z': ref.copy()}
    images['X'] = images['X'].copy(); images['X'][1, 0, 0, 0] = .01
    mag, v, fixed = image_quantities(images, 2.5)
    assert fixed.sum() == 7 and not fixed[0, 0, 0]
    np.testing.assert_allclose(v[0, 1, 1, 1], [.25, -.5, 0], atol=1e-12)
    assert np.isnan(v[1, 0, 0, 0]).all()


def test_zero_signal_is_unavailable():
    z = np.zeros((1, 2, 2, 2), complex)
    _, v, m = image_quantities({e: z for e in ['REF', 'X', 'Y', 'Z']}, 1)
    assert not m.any() and np.isnan(v).all()


def test_paired_wrapped_phase_and_gain():
    mag = np.ones((1, 2, 2, 2)); mask = np.ones((2, 2, 2), bool)
    v = np.full(mag.shape+(3,), .99)
    delta = paired_delta((mag, v), (mag*2, -v), mask, 1)
    assert delta['magnitude_relative_rmse'] == 1
    assert delta['mean_magnitude_ratio'] == 2
    assert delta['velocity_wrapped_component_rmse_m_s'] == pytest.approx(.02)


def test_empty_support_is_null():
    m = np.zeros((1, 2, 2, 2)); v = np.full(m.shape+(3,), np.nan)
    assert paired_delta((m, v), (m, v), np.zeros((2, 2, 2), bool), 1)['magnitude_relative_rmse'] is None


def test_independent_trilinear_complex_sampler():
    ax = np.arange(3.)
    zz, yy, xx = np.meshgrid(ax, ax, ax, indexing='ij')
    a = (zz+2*yy+3*xx)*(1+2j)
    p = np.array([[.2, 1.4, .8], [1.8, 0, 2]])
    sampled = interpolate_manual(a[None], [ax, ax, ax], p)
    np.testing.assert_allclose(sampled[0], (p@np.array([1, 2, 3]))*(1+2j), atol=1e-12)
