import numpy as np
import pytest

from scripts.plot_aorta_extended_comparison import axes_and_edges, project, waveform_metrics


def test_grid_edges_match_centered_inverse_dft():
    axes, edges = axes_and_edges((32, 32, 72), (.28, .128, .128))
    assert axes[0][36] == 0
    np.testing.assert_allclose(edges[0][[0, -1]], [-141.9444444444, 138.0555555556])
    np.testing.assert_allclose(edges[1][[0, -1]], [-66, 62])
    np.testing.assert_allclose(edges[2][[0, -1]], [-66, 62])


def test_projection_weight_and_mask_not_zero_filling():
    values = np.array([[[1., 5.]], [[3., 500.]]])
    mask = np.array([[[True, False]], [[True, False]]])
    fraction = np.array([[[.25, 1]], [[.75, 1]]])
    result = project(values, mask, fraction, 0)
    assert result[0, 0] == 2.5
    assert np.isnan(result[0, 1])


@pytest.mark.parametrize('axis', [0, 1, 2])
def test_all_projection_axes_preserve_constant(axis):
    mask = np.ones((2, 3, 4), bool)
    result = project(np.full(mask.shape, 2.), mask, np.ones(mask.shape), axis)
    np.testing.assert_allclose(result, 2.)
    assert result.shape == tuple(x for i, x in enumerate(mask.shape) if i != axis)


def test_signed_flow_cycle_metrics_and_zero_reference():
    cfd = -np.arange(1., 9.)
    result = waveform_metrics(cfd, 1.1*cfd, .8)
    assert result['CFD_net_cycle_ml'] == pytest.approx(-3.6)
    assert result['Koma_net_cycle_ml'] == pytest.approx(-3.96)
    assert result['cycle_difference_pct'] == pytest.approx(10)
    assert result['nrmse_rms_pct'] == pytest.approx(10)
    assert result['CFD_abs_peak_bin'] == 8
    assert result['correlation'] == pytest.approx(1)
    zero = waveform_metrics(np.zeros(8), np.zeros(8), .8)
    assert zero['cycle_difference_pct'] is None
    assert zero['nrmse_rms_pct'] is None
    assert zero['correlation'] is None


def test_waveform_requires_exactly_eight_finite_bins():
    with pytest.raises(ValueError):
        waveform_metrics(np.zeros(7), np.zeros(7), .8)
    with pytest.raises(ValueError):
        waveform_metrics([np.nan]*8, np.zeros(8), .8)
