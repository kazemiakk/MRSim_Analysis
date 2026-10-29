import numpy as np
import pytest

from scripts.analyze_phantom_precision import sort_profiles, partial_flow, coverage, region_rows


def test_profile_sort_follows_labels_not_acquisition_order():
    labels = np.array([[1, 0, 1], [0, 0, 0], [0, 0, 1], [1, 0, 0]])
    values = np.arange(12).reshape(4, 3).astype(complex)
    out = sort_profiles(values, labels, (2, 1, 2, 3))
    for i, (y, z, b) in enumerate(labels):
        np.testing.assert_array_equal(out[b, z, y], values[i])


@pytest.mark.parametrize('labels', [np.array([[0, 0, 0], [0, 0, 0]]), np.array([[0, 0, 0], [2, 0, 0]])])
def test_duplicate_or_outside_labels_rejected(labels):
    with pytest.raises(ValueError):
        sort_profiles(np.ones(4, complex), labels, (1, 1, 2, 2))


def test_missing_profile_rejected():
    with pytest.raises(ValueError):
        sort_profiles(np.ones(2, complex), np.array([[0, 0, 0]]), (1, 1, 2, 2))


def test_empty_flow_is_unavailable_not_zero():
    result = partial_flow(np.ones((2, 1, 2)), np.ones((2, 1, 2)), np.ones((1, 2)), np.zeros((1, 2), bool), 1)
    assert result['CFD_ml_s'] is None and result['Koma_ml_s'] is None
    assert result['metrics'] is None and result['coverage_fraction'] == 0


def test_flow_units_weighting_and_matching_support():
    a = np.array([[[1., 4.]], [[2., 6.]]])
    b = a+1
    result = partial_flow(a, b, np.array([[2., 1.]]), np.array([[True, False]]), 1)
    assert result['coverage_fraction'] == pytest.approx(2/3)
    assert result['CFD_ml_s'] == [2., 4.]
    assert result['Koma_ml_s'] == [4., 6.]
    assert result['metrics']['rmse_ml_s'] == pytest.approx(2)


def test_coverage_uses_fraction_weight_not_pixel_count():
    assert coverage(np.array([.25, .75]), np.array([True, False])) == pytest.approx(.25)
    assert coverage(np.zeros(2), np.ones(2, bool)) is None


def test_pair_uses_intersection_and_same_denominator():
    reference = np.zeros((2, 1, 1, 3, 3))
    fields = {'saved_f32':reference+1, 'candidate_f64':reference+2}
    masks = {'saved_f32':np.array([[[True, True, False]]]), 'candidate_f64':np.array([[[False, True, True]]])}
    rows = region_rows(reference, fields, masks, np.ones((1, 1, 3)), np.array([-.03, 0, .03]), .05)
    paired = [r for r in rows if r['region'] == 'whole' and r['support'] == 'paired']
    assert all(r['coverage_fraction'] == pytest.approx(1/3) for r in paired)
    assert all(r['metrics']['voxels'] == 1 for r in paired)
    assert paired[0]['metrics']['vector_rmse_m_s'] == pytest.approx(np.sqrt(3))
    assert paired[1]['metrics']['vector_rmse_m_s'] == pytest.approx(2*np.sqrt(3))
