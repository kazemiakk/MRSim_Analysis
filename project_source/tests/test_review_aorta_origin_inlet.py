import numpy as np
import pytest
from scripts.review_aorta_origin_inlet import planar_candidate


def test_detect_face_with_bulk_points():
    rng=np.random.default_rng(17)
    face=rng.uniform(-15,15,(500,3));face[:,2]=-.025*face[:,0]+.021*face[:,1]
    bulk=rng.uniform(-15,15,(700,3));bulk[:,2]=rng.uniform(1,20,700)
    points=np.r_[face,bulk]
    normal,offset,mask=planar_candidate(points)
    expected=np.array([.025,-.021,1]);expected/=np.linalg.norm(expected)
    np.testing.assert_allclose(normal,expected,atol=1e-12)
    assert abs(offset)<1e-12
    assert mask[:500].all() and not mask[500:].any()
    n2,d2,m2=planar_candidate(points)
    np.testing.assert_array_equal(n2,normal)
    np.testing.assert_array_equal(m2,mask)
    assert d2==offset


@pytest.mark.parametrize('points',[np.zeros((3,3)),np.full((100,3),np.nan),np.zeros((100,2)),np.zeros((200,3))])
def test_invalid_or_degenerate_fail(points):
    with pytest.raises(ValueError):
        planar_candidate(points)
