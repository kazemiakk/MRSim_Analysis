import numpy as np
import pytest

from scripts.review_aorta_rim_plane import fit_rim_plane, verify_selection
from scripts.build_aorta_inlet_picker import SOURCE_SHA, WALL_SHA, source_to_mri_mm


def test_known_plane_and_sign():
    result = fit_rim_plane([[0,0,2], [3,0,2], [0,4,2]])
    np.testing.assert_allclose(result['normal_ABC'], [0,0,1])
    np.testing.assert_allclose(result['origin_mm'], [1,4/3,2])
    assert result['triangle_area_mm2'] == 6
    assert result['max_point_plane_residual_mm'] == 0
    assert fit_rim_plane([[0,0,2],[0,4,2],[3,0,2]])['normal_ABC'] == [0,0,-1]


@pytest.mark.parametrize('points', [
    [[0,0,0],[0,0,0],[1,1,1]], [[0,0,0],[1,0,0],[2,0,0]],
    [[0,0,0],[1,0,0],[2,.0001,0]], [[0,0,0],[1,0,0],[0,np.nan,0]], [[0,0,0]],
])
def test_invalid_points_fail(points):
    with pytest.raises(ValueError):
        fit_rim_plane(points)


def test_basis_and_translation():
    points=np.array([[1,2,3],[4,1,8],[1,7,6]], dtype=float)
    a=fit_rim_plane(points)
    b=fit_rim_plane(points+[30,-50,70])
    np.testing.assert_allclose(a['normal_ABC'], b['normal_ABC'])
    basis=np.array([a['basis_u'], a['basis_v'], a['normal_ABC']])
    np.testing.assert_allclose(basis@basis.T,np.eye(3),atol=1e-14)
    assert a['triangle_area_mm2'] == b['triangle_area_mm2']


def test_source_selection_validation():
    samples=np.array([[.001,.002,.003]], dtype=np.float32)
    point=samples[0].astype(float)
    record=dict(purpose='user_marked_inlet_location', source_coordinates_sha256=SOURCE_SHA,
                wall_samples_sha256=WALL_SHA, wall_associated_sample_index=0, original_coordinate_row=0,
                source_xyz_m=point.tolist(), source_xyz_mm=(point*1000).tolist(),
                mri_xyz_mm=source_to_mri_mm(point).tolist())
    np.testing.assert_array_equal(verify_selection(record,samples,samples,np.array([0])),point*1000)
    for change in [dict(purpose='browser_test_not_user_inlet'), dict(original_coordinate_row=1),
                   dict(source_coordinates_sha256='bad'),dict(source_xyz_m=[0,0,0]),
                   dict(mri_xyz_mm=[0,0,0]),dict(source_xyz_mm=[0,0,0])]:
        with pytest.raises(ValueError):
            verify_selection({**record,**change},samples,samples,np.array([0]))
