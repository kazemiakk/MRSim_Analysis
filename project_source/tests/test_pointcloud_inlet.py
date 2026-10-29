import numpy as np
import pytest
from mrsimtracks.pointcloud_inlet import PointCloudInletReseeder


def cap(velocity=1):
    vertices=np.array([[0,0,0],[1,0,0],[0,1,0]],dtype=float)
    v=np.zeros((3,3,4));v[:,2]=velocity
    return PointCloudInletReseeder(vertices,[[0,1,2]],v,[0,0,1],frame_dt=.01,start_frame=0,dt=.002,rng=np.random.default_rng(7))


def test_cap_inflow_flux_and_uniform_triangle_depth():
    r=cap();vn,flux=r.inflow(0)
    np.testing.assert_allclose(flux,[.5])
    p=r.reseed(10000,0)
    assert np.all(p[:,0]>=0) and np.all(p[:,1]>=0) and np.all(p[:,:2].sum(1)<=1+1e-6)
    assert p[:,2].min()>=1e-6 and p[:,2].max()<=.002001
    np.testing.assert_allclose(p.mean(0)[:2],[1/3,1/3],atol=.01)
    assert r.injected==10000


def test_no_area_weight_fallback_at_zero_flux():
    r=cap(0)
    assert r.reseed(0,0).shape==(0,3)
    with pytest.raises(ValueError,match='No positive inlet flux'):r.reseed(1,0)


def test_reproducible_and_periodic_sampling():
    a=cap();b=cap()
    np.testing.assert_array_equal(a.reseed(100,0),b.reseed(100,.04))
