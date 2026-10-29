import numpy as np
import pytest
from scripts.plot_aorta_branch_views import same_depth_projection,branch_planes,all_branch_plane,main
from scripts.aorta_hemodynamic_reference import AortaGrid


def test_shared_projection_depth_not_independent_maxima():
    a=np.array([[[2,1],[4,8]],[[5,3],[1,6]]],float)
    c=np.array([[[9,7],[2,3]],[[1,2],[8,9]]],float)
    e=abs(c-a);v,d,m=same_depth_projection([a,c,e],np.ones_like(a,bool))
    np.testing.assert_array_equal(d,np.argmax(a,axis=0).T)
    np.testing.assert_array_equal(v[1],np.array([[1,2],[2,3]]))
    assert m.all()


def test_empty_projection_remains_nan():
    a=np.ones((3,4,5));m=np.ones_like(a,bool);m[:,:,2]=False
    values,_,valid=same_depth_projection([a,a,a],m)
    assert not valid[2].any()
    assert all(np.isnan(v[2]).all() for v in values)


def test_inverse_native_coordinate_mapping():
    g=AortaGrid();p=np.array([[.01,.02,.03],[-.03,.04,.05]])
    source=g.to_source(p)
    np.testing.assert_allclose((source-np.array(g.center))[:,g.order],p)


def test_branch_planes_geometry_only_and_reproducible():
    rng=np.random.default_rng(4)
    wall=np.concatenate([rng.normal([x,y,98],[1,1,2],size=(100,3)) for x,y in [(-10,-20),(10,0),(30,20)]])
    a=branch_planes(wall,np.arange(-40,41,4));b=branch_planes(wall,np.arange(-40,41,4))
    assert a==b
    assert [p['region'] for p in a]==['B1','B2','B3']
    assert [p['source_y_mm'] for p in a]==[-20,0,20]


def test_preserve_completed_target(tmp_path):
    with pytest.raises(FileExistsError):main(tmp_path)


def test_best_plane_uses_all_geometric_regions_or_fails():
    rng=np.random.default_rng(3)
    wall=np.concatenate([rng.normal([x,y,98],[1,.5,1],size=(100,3)) for x,y in [(-20,-1),(0,0),(20,1)]])
    result=all_branch_plane(wall,np.arange(-12,13,4))
    assert result['source_y_mm']==0 and min(result['fractions'])>0
    wall=np.concatenate([rng.normal([x,y,98],[1,.5,1],size=(100,3)) for x,y in [(-20,-20),(0,0),(20,20)]])
    with pytest.raises(ValueError,match='No voxel slab'):all_branch_plane(wall,np.arange(-40,41,4))
