"""Tests for discrete aorta cardiac-bin preview generation."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np


ROOT = Path(__file__).parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILD = load_script("build_aorta_tracking_bin_preview")
ANIMATE = load_script("animate_aorta_cardiac_bins")


def test_phase_centers_map_to_distinct_nearby_source_steps():
    indices, requested, sampled = BUILD.phase_step_indices(8, 0.72, 0.002, 360)

    assert len(np.unique(indices)) == 8
    assert np.all(np.diff(indices) > 0)
    np.testing.assert_allclose(
        requested, (np.arange(8) + 0.5) * 0.72 / 8
    )
    assert np.max(np.abs(sampled - requested)) <= 0.0010000001


def test_voxelization_excludes_invalid_velocity_but_keeps_density():
    points = np.array(
        [
            [-0.25, -0.25, -0.25],
            [-0.20, -0.20, -0.20],
            [0.25, 0.25, 0.25],
        ],
        dtype=np.float32,
    )
    velocity = np.array(
        [[1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [9.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    valid = np.array([True, True, False])

    density, voxel_velocity, speed, mask, report = BUILD.voxelize_phase(
        points,
        velocity,
        valid,
        np.ones(3),
        np.array([2, 2, 2]),
        1,
    )

    assert density.sum() == 3
    assert report["particles_with_valid_centered_velocity"] == 2
    assert report["valid_velocity_particles_inside_fov"] == 2
    assert mask.sum() == 1
    assert np.isclose(voxel_velocity[0][mask][0], 2.0)
    assert np.isclose(speed[mask][0], 2.0)


def test_largest_component_filter_removes_detached_island():
    candidate = np.zeros((6, 6, 6), dtype=bool)
    candidate[1:4, 1:4, 1:4] = True
    candidate[5, 5, 5] = True

    cleaned, report = ANIMATE.largest_component(candidate)

    assert report["raw_component_count"] == 2
    assert cleaned.sum() == 27
    assert not cleaned[5, 5, 5]


def test_weighted_projection_uses_magnitude_weights():
    magnitude = np.array([[[1.0]], [[3.0]]], dtype=np.float32)
    values = np.array([[[2.0]], [[6.0]]], dtype=np.float32)
    mask = np.ones_like(magnitude, dtype=bool)

    projection = ANIMATE.weighted_projection(values, magnitude, mask)

    np.testing.assert_allclose(projection, [[5.0]])
