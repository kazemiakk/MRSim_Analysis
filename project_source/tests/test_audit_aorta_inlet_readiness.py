import numpy as np
import pytest
from scripts.audit_aorta_inlet_readiness import face_quadrature,nodal_face_flow_ml_s,coverage


def test_face_area_constant_and_linear_flow():
    p=np.array([[0,0,0],[10,0,0],[10,10,0],[0,10,0],[5,5,0]],dtype=float)
    t,a,c=face_quadrature(p,[1,0,0],[0,1,0])
    assert a.sum()==100
    vel=np.zeros((5,3,2));vel[:,2,0]=2;vel[:,2,1]=p[:,0]
    q=nodal_face_flow_ml_s(vel,t,a,[0,0,1])
    np.testing.assert_allclose(q,[200,500])
    np.testing.assert_allclose(nodal_face_flow_ml_s(vel,t,a,[0,0,-1]),-q)
    vel[0,0,0]=np.nan
    with pytest.raises(ValueError):nodal_face_flow_ml_s(vel,t,a,[0,0,1])


def test_mask_support_is_not_partial_stencil_or_finite_velocity():
    axes=(np.arange(2),)*3
    mask=np.ones((2,2,2));points=np.array([[.5,.5,.5],[2,2,2]])
    r=coverage(axes,mask,points,np.array([3,1]))
    assert r['strict_stencil_area_fraction']==.75
    mask[0,0,0]=0
    r=coverage(axes,mask,points,np.array([3,1]))
    assert r['strict_stencil_area_fraction']==0
    assert r['any_stencil_neighbor_area_fraction']==.75
