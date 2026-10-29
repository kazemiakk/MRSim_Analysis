"""Tests for the aorta KomaMRI temporal-response audit."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "audit_aorta_koma_temporal_response.py"
SPEC = spec_from_file_location("audit_aorta_koma_temporal_response", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUDIT = module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def test_tracking_phases_group_into_two_broad_windows():
    times = (np.arange(8) + 0.5) * 0.72 / 8
    values = np.arange(1, 9, dtype=float)

    grouped, members = AUDIT.group_tracking_profile(times, values, 0.72, 2)

    np.testing.assert_allclose(grouped, [2.5, 6.5])
    assert members == [[0, 1, 2, 3], [4, 5, 6, 7]]


def test_temporal_relative_range_is_scale_invariant():
    profile = np.array([1.0, 2.0, 3.0])

    assert np.isclose(
        AUDIT.temporal_relative_range(profile),
        AUDIT.temporal_relative_range(10 * profile),
    )


def test_two_sample_correlation_is_not_reported():
    assert AUDIT.safe_correlation(np.array([1, 2]), np.array([2, 3])) is None


def test_cardiac_bin_assignment_wraps_cycle_endpoint():
    indices = AUDIT.cardiac_bin_indices(
        np.array([0.0, 0.18, 0.36, 0.54, 0.72]), 0.72, 4
    )

    np.testing.assert_array_equal(indices, [0, 1, 2, 3, 0])
