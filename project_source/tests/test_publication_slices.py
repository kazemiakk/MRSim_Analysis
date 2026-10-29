import numpy as np
import pytest
from scripts.plot_publication_slices import coordinates,plane,vector_maps,rounded_limit,select_bins,geometry_planes,output_check


def test_center_conventions():
    assert coordinates((2,4,6),(1,2,3))[0].tolist()==[-3000,-2000,-1000,0,1000,2000]
    assert coordinates((2,4,6),(1,2,3),True)[2].tolist()==[-1500,1500]


def test_true_planes_preserve_samples_and_axis_order():
    a=np.arange(2*3*4).reshape(2,3,4)
    np.testing.assert_array_equal(plane(a,'XY',1),a[1])
    np.testing.assert_array_equal(plane(a,'XZ',2),a[:,2])
    np.testing.assert_array_equal(plane(a,'YZ',3),a[:,:,3])


def test_vector_error_is_not_difference_of_speeds():
    a=np.array([[[[1.,0,0],[-1.,0,0]]]])
    c=-a
    maps,m=vector_maps(a,c,np.ones(a.shape[:-1],bool))
    assert m.all();np.testing.assert_array_equal(maps[2],2*np.ones(a.shape[:-1]))
    np.testing.assert_array_equal(maps[0],maps[1])


def test_missing_and_masked_samples_stay_missing():
    a=np.ones((1,2,3,3));c=a.copy();c[0,0,0]=np.nan
    mask=np.ones(a.shape[:-1],bool);mask[0,1,1]=False
    maps,m=vector_maps(a,c,mask)
    assert m.sum()==4
    assert all(np.isnan(v[0,0,0]) and np.isnan(v[0,1,1]) for v in maps)


def test_scale_never_clips_and_rejects_nonfinite():
    for v in [0,1e-9,.0013,.214,4.91033,123.4]:assert rounded_limit(v)>=v
    with pytest.raises(ValueError):rounded_limit(np.nan)


def test_phase_selection_excludes_synthetic_tail():
    rows=[dict(section='s',bin=i,net_m3_s=q) for i,q in [(1,1),(2,5),(14,10),(15,0)]]
    assert select_bins(rows,'s',True)==dict(peak=1,low=0)


def test_geometry_selection_is_fixed_and_output_protected(tmp_path):
    g=np.zeros((3,4,5),bool);g[1,2,:]=True
    assert geometry_planes(g)==[('XY',1),('XZ',2)]
    with pytest.raises(FileExistsError):output_check(tmp_path)
