import numpy as np
import pytest

from scripts.build_aorta_inlet_picker import AXIS_ORDER, CENTER_M, make_selection, source_to_mri_mm
from scripts.serve_aorta_inlet_picker import validate_payload


def test_transform_preserves_documented_axes_and_center():
    np.testing.assert_allclose(source_to_mri_mm(CENTER_M), [0, 0, 0])
    np.testing.assert_allclose(source_to_mri_mm(CENTER_M+[.001, .002, .003]), [3, 1, 2])
    original = np.array([[.01, -.03, -.12], [.05, .02, .08]])
    reconstructed = np.empty_like(original)
    reconstructed[:, AXIS_ORDER] = source_to_mri_mm(original)/1000
    np.testing.assert_allclose(reconstructed+CENTER_M, original, atol=1e-16)


def test_selection_uses_trusted_source_values_not_display_rounding():
    points = np.array([[.01, .02, .03], [.04, .05, .06]], dtype=np.float32)
    data = make_selection(1, points, np.array([17, 39]),
        {'source_coordinates_sha256': 'source', 'wall_samples_sha256': 'wall'}, note='Inlet rim')
    np.testing.assert_array_equal(data['source_xyz_m'], points[1].astype(float))
    np.testing.assert_allclose(data['mri_xyz_mm'], source_to_mri_mm(points[1]))
    assert data['original_coordinate_row'] == 39
    assert data['plane_normal'] is None
    assert data['inlet_cap_verified'] is False
    assert data['purpose'] == 'user_marked_inlet_location'


def test_browser_annotations_are_explicitly_not_user_choices():
    data = make_selection(0, np.zeros((1, 3)), np.array([0]),
        {'source_coordinates_sha256': 'source', 'wall_samples_sha256': 'wall'}, test_mode=True)
    assert data['purpose'] == 'browser_test_not_user_inlet'


@pytest.mark.parametrize('index', [-1, 1, True, .5, None])
def test_invalid_sample_index_rejected(index):
    with pytest.raises(ValueError):
        make_selection(index, np.zeros((1, 3)), np.array([0]), {})


@pytest.mark.parametrize('payload', [None, {}, {'sample_index': True},
    {'sample_index': 0, 'filename': '../../original.npy'},
    {'sample_index': 0, 'source_xyz_m': [0, 0, 0]},
    {'sample_index': 0, 'note': 'a'*501}, {'sample_index': 0, 'note': None}])
def test_untrusted_save_payload_rejected(payload):
    with pytest.raises(ValueError):
        validate_payload(payload)


def test_valid_minimal_save_payload():
    assert validate_payload({'sample_index': 3}) == (3, '')
    assert validate_payload({'sample_index': 5, 'note': 'Near the inlet'}) == (5, 'Near the inlet')
