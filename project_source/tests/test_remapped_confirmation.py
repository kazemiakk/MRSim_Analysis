import numpy as np
import pytest
from scripts.analyze_remapped_confirmation import fixed_mask, velocities, weighted_metrics, flow_metrics


def test_fixed_mask_requires_every_bin_and_encoding():
    a=np.ones((2,1,1,3),complex)
    images={e:a.copy() for e in ('REF','X','Y','Z')}
    images['Y'][1,0,0,1]=.01
    images['Z'][0,0,0,2]=0
    assert fixed_mask(images,.05).tolist()==[[[True,False,False]]]


def test_fixed_mask_rejects_missing_encoding():
    with pytest.raises(ValueError):fixed_mask({'REF':np.ones((1,1,1,1))},.05)


def test_phase_velocity_is_signed_and_not_filled():
    a=np.ones((1,1,1,2),complex)
    images=dict(REF=a,X=a*np.exp(1j*np.pi/4),Y=a*np.exp(-1j*np.pi/2),Z=a)
    result=velocities(images,4)
    np.testing.assert_allclose(result[0,0,0,0],[1,-2,0],atol=1e-12)


def test_vector_error_has_vector_not_component_denominator():
    a=np.array([[[[[1.,0,0],[2.,0,0]]]]])
    b=a+np.array([[[[[.1,.2,0],[.3,0,0]]]]])
    fraction=np.array([[[1.,3.]]]); mask=np.ones((1,1,2),bool)
    result=weighted_metrics(a,b,fraction,mask)
    assert result['vector_rmse_m_s']==pytest.approx(np.sqrt(.08))
    assert result['reference_vector_rms_m_s']==pytest.approx(np.sqrt(13/4))
    assert result['component_rmse_m_s'][1]==pytest.approx(.1)


def test_empty_support_is_unavailable_not_zero_error():
    a=np.zeros((1,1,1,1,3))
    assert weighted_metrics(a,a,np.ones((1,1,1)),np.zeros((1,1,1),bool)) is None
    assert flow_metrics([],[],1) is None


def test_flow_cycle_uses_acquired_equal_bins_and_no_fit():
    result=flow_metrics([1,2,3,2],[2,4,6,4],.8)
    assert result['cycle_CFD_ml']==pytest.approx(1.6)
    assert result['cycle_volume_error_fraction']==pytest.approx(1)
    assert result['rmse_ml_s']==pytest.approx(np.sqrt(4.5))
    assert result['correlation']==pytest.approx(1)


def test_nonfinite_excluded_reference_is_not_zero_filled():
    a=np.array([[[[[1.,0,0],[np.nan,np.nan,np.nan]]]]])
    b=np.ones_like(a)
    result=weighted_metrics(a,b,np.ones((1,1,2)),np.ones((1,1,2),bool))
    assert result['voxels']==1
    assert result['vector_rmse_m_s']==pytest.approx(np.sqrt(2))
