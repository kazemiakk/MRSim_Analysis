import matplotlib.pyplot as plt
import numpy as np
from scripts.plot_aorta_replenishment_pilot import matched_figure


def test_matched_curves_use_saved_values_and_no_fitted_scaling():
    times=np.arange(8)*.09+.045
    audit=dict(CFD_partial_ml_s=list(range(8)),Koma_partial_ml_s=list(range(1,9)),
        paired_partial_area_mm2=650,fixed05_area_fraction=.66,partial_flow_rmse_ml_s=1,
        partial_flow_correlation=1,cycle_volume_ratio=1.2)
    before=plt.rcParams.copy();fig=matched_figure(audit,times)
    assert plt.rcParams==before
    np.testing.assert_array_equal(fig.axes[0].lines[0].get_ydata(),audit['CFD_partial_ml_s'])
    np.testing.assert_array_equal(fig.axes[0].lines[1].get_ydata(),audit['Koma_partial_ml_s'])
    np.testing.assert_array_equal(fig.axes[0].lines[0].get_xdata(),times)
    plt.close(fig)
