import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pytest
from scripts.plot_aorta_inlet_readiness import waveform_figure,audit_figure,check_report


def fixture():
    return dict(experiment='E-031',native_cfd_waveform=[dict(source_time_s=i*.01,flow_ml_s=i+1) for i in range(72)],
        matched_bins=[dict(bin=i+1,nominal_time_s=(i+.5)*.09,cfd_adc_average_ml_s=8-i,koma_inlet_ml_s=None,
             magnitude_fraction_of_global_max_min=.001,magnitude_fraction_of_global_max_max=.03) for i in range(8)],
        timing=dict(native_source_period_s=.72,mri_period_s=.726),
        coverage_by_offset=[dict(support=name,inward_offset_mm=off,strict_stencil_area_fraction=off/200,
             any_stencil_neighbor_area_fraction=off/20) for name in ('core05','core15') for off in (0,4,8,12,16,20)])


def test_waveform_values_no_fake_koma_curve():
    before=plt.rcParams.copy()
    r=fixture();fig=waveform_figure(r)
    assert plt.rcParams==before
    assert len(fig.axes)==2
    assert all(len(ax.lines)==1 for ax in fig.axes)
    np.testing.assert_array_equal(fig.axes[0].lines[0].get_ydata(),np.arange(1,73))
    np.testing.assert_array_equal(fig.axes[1].lines[0].get_ydata(),np.arange(8,0,-1))
    assert any('unavailable, not zero' in t.get_text() for t in fig.axes[1].texts)
    plt.close(fig)


def test_audit_fraction_to_percent_and_fail_closed():
    before=plt.rcParams.copy()
    r=fixture();fig=audit_figure(r)
    assert plt.rcParams==before
    np.testing.assert_allclose(fig.axes[1].lines[0].get_ydata(),np.arange(0,21,4)/2)
    np.testing.assert_allclose(fig.axes[1].lines[1].get_ydata(),np.arange(0,101,20))
    assert fig.axes[0].get_ylim()[1]>15
    plt.close(fig)
    r['matched_bins'][0]['koma_inlet_ml_s']=0
    with pytest.raises(ValueError):check_report(r)
