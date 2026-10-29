import numpy as np
import pytest
from scripts.integrate_aorta_inlet_complex import complex_velocity_samples,integrate_flow,synthetic_controls


def test_synthetic_partial_volume_missing_cancellation_and_wrap_controls():
    result=synthetic_controls()
    assert result['status']=='PASS_SYNTHETIC_CONTROLS'
    assert result['partial_volume_max_velocity_error_m_s']<1e-6


def test_no_signal_stays_nan_and_does_not_silently_integrate_to_zero():
    images={name:np.zeros((1,2,2,2),complex) for name in ('REF','X','Y','Z')}
    v,valid,ratio=complex_velocity_samples(images,(np.arange(2),)*3,[[.5,.5,.5]],2.5)
    assert not valid.any() and np.isnan(v).all() and np.all(ratio==0)
    with pytest.raises(ValueError,match='missing'):
        integrate_flow(v,[10],[1,0,0])


def test_flux_units_normal_sign_and_no_area_scaling():
    v=np.array([[[.1,.2,.3],[.2,.3,.4]]])
    np.testing.assert_allclose(integrate_flow(v,[3,7],[1,0,0]),[1.7])
    np.testing.assert_allclose(integrate_flow(v,[3,7],[-1,0,0]),[-1.7])
    with pytest.raises(ValueError,match='unit'):
        integrate_flow(v,[3,7],[2,0,0])


def test_bad_contracts_fail_closed():
    images={name:np.ones((1,2,2,2),complex) for name in ('REF','X','Y','Z')}
    with pytest.raises(ValueError,match='threshold'):
        complex_velocity_samples(images,(np.arange(2),)*3,[[.5,.5,.5]],2.5,0)
    images['Y'][0,0,0,0]=np.nan
    with pytest.raises(ValueError,match='finite'):
        complex_velocity_samples(images,(np.arange(2),)*3,[[.5,.5,.5]],2.5)
