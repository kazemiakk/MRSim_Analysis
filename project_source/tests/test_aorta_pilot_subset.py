"""Tests for the aorta KomaMRI pilot-subset audit."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "audit_aorta_pilot_subset.py"
SPEC = spec_from_file_location("audit_aorta_pilot_subset", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUDIT = module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def test_voxel_density_uses_mri_axis_order_and_excludes_outside_points():
    points = np.array(
        [[0.25, 0.25, 0.25], [-0.25, -0.25, -0.25], [2.0, 2.0, 2.0]],
        dtype=np.float32,
    )

    density, inside = AUDIT.voxel_density(
        points,
        np.zeros(3),
        np.array([2, 0, 1]),
        np.ones(3),
        np.array([2, 2, 2]),
    )

    assert inside == 2
    assert density.sum() == 2
    assert np.count_nonzero(density) == 2
