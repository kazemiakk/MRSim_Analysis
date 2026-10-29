import matplotlib.pyplot as plt
import numpy as np
from scripts.plot_aorta_inlet_complex import coverage_figure,native_waveform


def test_native_figure_preserves_saved_native_units_and_eight_bins():
    report=dict(native_source_phase_centers_s=[.4,.5,.6,.7,.04,.13,.22,.31],
        CFD_full_native_waveform_ml_s=np.arange(72).tolist(),CFD_native_time_ml_s=np.arange(8).tolist(),
        Koma_native_time_ml_s=(np.arange(8)+1).tolist(),coverage_fraction=.9998,excluded_area_mm2=.2,
        known_source_time_velocity_scale=.99,
        physical_units_metrics=dict(correlation=.97,rmse_ml_s=5,nominal_bin_cycle_volume_ratio=1.02))
    before=plt.rcParams.copy();fig=native_waveform(report)
    assert plt.rcParams==before
    order=np.argsort(report['native_source_phase_centers_s']);lines=fig.axes[0].lines
    np.testing.assert_array_equal(lines[0].get_ydata(),report['CFD_full_native_waveform_ml_s'])
    np.testing.assert_array_equal(lines[1].get_ydata(),np.array(report['CFD_native_time_ml_s'])[order])
    np.testing.assert_array_equal(lines[2].get_ydata(),np.array(report['Koma_native_time_ml_s'])[order])
    assert len(lines[1].get_xdata())==len(lines[2].get_xdata())==8
    plt.close(fig)


def test_coverage_percentages_use_separate_title_lines_without_rc_leak():
    before=plt.rcParams.copy()
    fig=coverage_figure(np.array([[0,0],[1,0],[0,1],[1,1]]),np.array([[0,1,2],[1,3,2]]),
        np.array([True,False]),np.array([True,True]),
        dict(original_partial_area_fraction=.5,coverage_fraction=1))
    assert plt.rcParams==before
    assert fig.axes[0].get_title()=='Original complete-stencil guard\n50.000% retained'
    assert fig.axes[1].get_title()=='Complex-image point-signal gate\n100.000% retained'
    assert not fig.axes[0].texts and not fig.axes[1].texts
    plt.close(fig)
