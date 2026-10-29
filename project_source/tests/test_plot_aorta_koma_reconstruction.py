"""Tests for artifact-cleaned aorta KomaMRI projection panels."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "plot_aorta_koma_reconstruction.py"
SPEC = spec_from_file_location("plot_aorta_koma_reconstruction", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_display_mask_keeps_largest_3d_component():
    volume = np.zeros((8, 8, 8), dtype=np.float32)
    volume[1:5, 1:5, 1:5] = 10.0
    volume[7, 7, 7] = 9.0

    mask, report = MODULE.build_display_mask(volume, 0.5)

    assert mask[2, 2, 2]
    assert not mask[7, 7, 7]
    assert report["threshold_component_count"] == 2
    assert report["retained_largest_component_voxels"] == 64
    assert report["removed_threshold_voxels"] == 1


def test_panel_axes_and_max_support_slice_selection():
    volume = np.zeros((4, 5, 6), dtype=np.float32)
    mask = np.zeros_like(volume, dtype=bool)
    mask[2, 1:4, 1:5] = True
    volume[mask] = 1.0

    panels, selection = MODULE.panel_data(
        volume,
        mask,
        np.array([12.0, 10.0, 8.0]),
        "max-support",
    )

    assert [panel[1].shape for panel in panels] == [
        (5, 6),
        (4, 6),
        (4, 5),
        (5, 6),
        (4, 6),
        (4, 5),
    ]
    assert selection["indices_python_slice_phase_readout"] == [2, 1, 1]
    assert selection["coordinates_mm"] == {
        "XY_plane_source_Y": 1.0,
        "XZ_plane_source_X": -2.0,
        "YZ_plane_source_Z": -3.0,
    }
