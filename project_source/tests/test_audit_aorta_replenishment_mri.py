import numpy as np
import pytest
from scripts.audit_aorta_replenishment_mri import flow_metrics,reconstruction_checks
from scripts.validate_aorta_replenishment_mri import manual_trilinear


def test_missing_or_constant_flow_is_not_a_zero_waveform_or_nan_metric():
    assert all(value is None for value in flow_metrics(None,None).values())
    assert flow_metrics([1,1],[2,2])['partial_flow_correlation'] is None
    result=flow_metrics([1,2,3],[2,4,6])
    assert result['cycle_volume_ratio']==2
    assert result['partial_flow_correlation']==1
    assert result['partial_flow_rmse_ml_s']==pytest.approx(np.sqrt(14/3))


def test_reconstruction_layout_mask_and_norm_identities():
    magnitude=np.ones((8,32,32,72),dtype=np.float32);magnitude[...,0]=0
    mask=(magnitude>0).astype(np.uint8)
    velocity=np.zeros((*magnitude.shape,3),dtype=np.float32)
    velocity[...,1]=mask*.1
    speed=np.linalg.norm(velocity,axis=-1)
    assert all(reconstruction_checks(magnitude,velocity,mask,speed,2.5).values())
    wrong=mask.copy();wrong[...,1]=0
    assert not reconstruction_checks(magnitude,velocity,wrong,speed,2.5)['unchanged_05_threshold']
    with pytest.raises(ValueError,match='layout'):
        reconstruction_checks(magnitude[0],velocity,mask,speed,2.5)


def test_manual_stencil_preserves_linear_fields_and_fails_outside():
    axes=(np.arange(3,dtype=float),)*3
    x,y,z=np.meshgrid(*axes,indexing='ij')
    points=np.array([[.5,.2,1.3],[1.5,1.2,.3]])
    fields=np.stack([x+2*y+3*z,x-y],axis=-1)
    sampled=manual_trilinear(fields,axes,points)
    np.testing.assert_allclose(sampled[:,0],points@[1,2,3])
    np.testing.assert_allclose(sampled[:,1],points@[1,-1,0])
    with pytest.raises(ValueError,match='leaves'):
        manual_trilinear(fields,axes,np.array([[3,1,1]]))
