from __future__ import annotations

import numpy as np
import pytest

from scripts.plot_dp150_koma_clean import (
    center_plane_indices,
    coordinate_slice,
    masked_plane,
    support_rms_projection,
    validate_two_bin_input,
)


def test_validate_two_bin_input_rejects_multi_bin_reconstruction() -> None:
    reconstruction = {"magnitude": np.zeros((15, 2, 3, 4))}

    with pytest.raises(
        ValueError,
        match=r"requires exactly two reconstructed cardiac bins; found 15",
    ):
        validate_two_bin_input(reconstruction)


def test_validate_two_bin_input_accepts_two_bins() -> None:
    reconstruction = {"magnitude": np.zeros((2, 2, 3, 4))}

    assert validate_two_bin_input(reconstruction) == 2


def test_center_plane_indices_select_maximum_support_planes() -> None:
    support = np.zeros((5, 7, 9), dtype=bool)
    support[2, :, 2:7] = True
    support[:, 3, 3:6] = True

    slice_index, phase_index = center_plane_indices(support)

    assert slice_index == 2
    assert phase_index == 3


def test_coordinate_slice_is_inclusive_at_selected_centers() -> None:
    coordinates = np.array([-3.0, -1.0, 1.0, 3.0])

    selected = coordinate_slice(coordinates, -1.0, 1.0)

    assert selected == slice(1, 3)


def test_masked_plane_hides_values_outside_support() -> None:
    values = np.arange(6, dtype=float).reshape(2, 3)
    mask = np.array([[False, True, False], [True, False, True]])

    result = masked_plane(values, mask)

    assert np.isnan(result[~mask]).all()
    assert np.array_equal(result[mask], values[mask])


def test_support_rms_projection_excludes_background_values() -> None:
    values = np.array([[[3.0]], [[4.0]], [[100.0]]])
    support = np.array([[[True]], [[True]], [[False]]])

    result = support_rms_projection(values, support, axis=0)

    assert np.isclose(result[0, 0], np.sqrt((9.0 + 16.0) / 2.0))
