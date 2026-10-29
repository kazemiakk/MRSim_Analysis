from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.analyze_dp150_scanner_reconstruction import resolve_particle_label
from scripts.plot_dp150_koma_publication import (
    find_throat,
    masked_max_projection,
    prepare_longitudinal_panels,
    voxelize_particle_support,
    weighted_projection,
)


def test_particle_label_is_inferred_from_production_run_path() -> None:
    label, slug = resolve_particle_label(
        Path("simulation_motion_1m_scanner_matched_venc5/pc_reconstruction.h5")
    )

    assert label == "1M"
    assert slug == "1m"


def test_explicit_particle_label_preserves_si_suffix_style() -> None:
    assert resolve_particle_label(Path("reconstruction.h5"), "100K") == (
        "100k",
        "100k",
    )


def test_voxelize_particle_support_uses_slice_phase_readout_order() -> None:
    fov = np.array([4.0, 2.0, 2.0])
    matrix = np.array([4, 2, 2])
    points = np.array(
        [
            [-1.5, -0.5, -0.5],
            [1.5, 0.5, 0.5],
        ]
    )

    support, metadata = voxelize_particle_support(points, fov, matrix)

    assert support.shape == (2, 2, 4)
    assert support[0, 0, 0]
    assert support[1, 1, 3]
    assert metadata["particles_outside_fov"] == 0


def test_masked_max_projection_is_nan_outside_support() -> None:
    values = np.arange(12, dtype=float).reshape(2, 2, 3)
    mask = np.zeros_like(values, dtype=bool)
    mask[:, 0, 1] = True

    projection = masked_max_projection(values, mask, axis=0)

    assert np.isnan(projection[0, 0])
    assert projection[0, 1] == values[:, 0, 1].max()
    assert np.isnan(projection[1, 2])


def test_weighted_projection_uses_only_valid_signal_support() -> None:
    values = np.array([[[1.0]], [[3.0]], [[100.0]]])
    magnitude = np.array([[[1.0]], [[3.0]], [[1000.0]]])
    mask = np.array([[[True]], [[True]], [[False]]])

    projection = weighted_projection(values, magnitude, mask, axis=0)

    assert np.isclose(projection[0, 0], 2.5)


def test_find_throat_ignores_empty_and_endpoint_planes() -> None:
    support = np.zeros((3, 3, 20), dtype=bool)
    support[:, :, 2:18] = True
    support[:, :, 9] = False
    support[1, 1, 9] = True

    throat, area = find_throat(support)

    assert throat == 9
    assert area[throat] == 1


def test_longitudinal_panels_remove_stored_mask_leakage() -> None:
    support = np.zeros((2, 3, 4), dtype=bool)
    support[:, 1, 1:3] = True
    magnitude = np.ones((2, *support.shape), dtype=np.float32)
    reconstruction = {
        "magnitude": magnitude,
        "velocity": np.ones((3, 2, *support.shape), dtype=np.float32),
        "speed": np.ones((2, *support.shape), dtype=np.float32),
        "mask": np.ones((2, *support.shape), dtype=bool),
    }

    panels, signal_mask, _ = prepare_longitudinal_panels(
        reconstruction, support
    )

    assert np.array_equal(signal_mask, np.broadcast_to(support, signal_mask.shape))
    projected_support = support.any(axis=0)
    for cardiac_bin in range(2):
        assert np.isnan(panels[cardiac_bin][1][~projected_support]).all()
        assert np.isnan(panels[cardiac_bin][2][~projected_support]).all()
