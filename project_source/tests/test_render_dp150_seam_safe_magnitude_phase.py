from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
from PIL import Image
import pytest

from scripts.render_dp150_seam_safe_magnitude_phase import (
    build_ffmpeg_command,
    load_corrected_reconstruction,
    one_voxel_boundary,
    render_six_panels,
    temporal_phase_qc,
)


def test_ffmpeg_command_avoids_optional_drawtext_filter(tmp_path: Path) -> None:
    plan = SimpleNamespace(
        fps=Fraction(3750, 253),
        frame_period_s=Fraction(253, 3750),
        encoded_frames=30,
        label="physical 67.4667 ms/bin",
    )
    command = build_ffmpeg_command(
        "ffmpeg", tmp_path / "frames.ffconcat", tmp_path / "partial.mp4", plan, "test"
    )

    assert "drawtext" not in " ".join(command)
    filter_index = command.index("-vf")
    assert command[filter_index + 1] == "format=yuv420p"
    assert any("physical 67.4667 ms/bin" in item for item in command)


def test_one_voxel_boundary_partitions_support() -> None:
    support = np.ones((5, 5, 5), dtype=bool)
    boundary, interior = one_voxel_boundary(support)
    assert not np.any(boundary & interior)
    np.testing.assert_array_equal(boundary | interior, support)
    assert interior.sum() == 27


def test_temporal_qc_uses_circular_phase_difference() -> None:
    support = np.ones((3, 3, 3), dtype=bool)
    phase = np.zeros((3, 15, 3, 3, 3), dtype=np.float32)
    phase[:, 14] = np.pi - 0.02
    phase[:, 0] = -np.pi + 0.02
    result = temporal_phase_qc(phase, support)
    seam = result["components"]["X"]["edges"][-1]
    assert seam["boundary_q95_rad"] == pytest.approx(0.04, abs=1e-6)


def test_loader_requires_fixed_partition(tmp_path: Path) -> None:
    path = tmp_path / "test.h5"
    shape = (15, 2, 2, 2)
    with h5py.File(path, "w") as target:
        target["reference_magnitude"] = np.ones(shape, np.float32)
        target["wrapped_phase_corrected_rad"] = np.zeros((3, *shape), np.float32)
        target["velocity_corrected_m_per_s"] = np.zeros((3, *shape), np.float32)
        target["speed_corrected_m_per_s"] = np.zeros(shape, np.float32)
        target["support_mask"] = np.ones(shape[1:], np.uint8)
        target["direct_mask"] = np.ones(shape[1:], np.uint8)
        target["regularized_mask"] = np.ones(shape[1:], np.uint8)
        target.attrs["RR_s"] = 1.012
        target.attrs["Venc_m_per_s"] = 5.0
        target.attrs["evidence_class"] = "test"
    with pytest.raises(RuntimeError, match="overlap"):
        load_corrected_reconstruction(path)


def test_loader_accepts_complete_fixed_support(tmp_path: Path) -> None:
    path = tmp_path / "valid.h5"
    shape = (15, 2, 3, 4)
    support = np.ones(shape[1:], np.uint8)
    with h5py.File(path, "w") as target:
        target["reference_magnitude"] = np.ones(shape, np.float32)
        target["wrapped_phase_corrected_rad"] = np.zeros((3, *shape), np.float32)
        target["velocity_corrected_m_per_s"] = np.zeros((3, *shape), np.float32)
        target["speed_corrected_m_per_s"] = np.zeros(shape, np.float32)
        target["support_mask"] = support
        target["direct_mask"] = support
        target["regularized_mask"] = np.zeros_like(support)
        target.attrs["RR_s"] = 1.012
        target.attrs["Venc_m_per_s"] = 5.0
        target.attrs["evidence_class"] = "test"

    loaded = load_corrected_reconstruction(path)

    assert loaded["phase"].shape == (3, *shape)
    assert loaded["support"].sum() == np.prod(shape[1:])


def test_phase_render_is_true_grayscale_and_exact_size(tmp_path: Path) -> None:
    definitions = [
        {
            "title": f"panel {index}",
            "xlabel": "x",
            "ylabel": "y",
            "extent": [-1, 1, -1, 1],
        }
        for index in range(6)
    ]
    panels = [np.linspace(-1, 1, 20).reshape(4, 5) for _ in range(6)]
    panels[0] = panels[0].copy()
    panels[0][0, 0] = np.nan
    path = tmp_path / "phase.png"

    render_six_panels(
        panels, definitions, path, "test", "phase", 1.0, dpi=40
    )

    with Image.open(path) as image:
        assert image.size == (720, 360)
        rgb = np.asarray(image.convert("RGB"))
    np.testing.assert_array_equal(rgb[..., 0], rgb[..., 1])
    np.testing.assert_array_equal(rgb[..., 1], rgb[..., 2])
