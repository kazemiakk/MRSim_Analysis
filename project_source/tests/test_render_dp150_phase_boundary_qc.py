from __future__ import annotations

from fractions import Fraction

import numpy as np
from PIL import Image
import pytest

from scripts.render_dp150_phase_boundary_qc import (
    Panel,
    PlaybackPlan,
    circular_phase_projection,
    displayed_p999,
    physical_and_slow_plans,
    playback_indices,
    render_frame,
    select_fixed_sections,
    signed_phase_colormap,
)


def test_circular_projection_preserves_phase_wrap() -> None:
    phase = np.deg2rad(np.array([179.0, -179.0]))[:, None, None]
    projected = circular_phase_projection(
        phase, np.ones_like(phase), np.ones_like(phase, dtype=bool), axis=0
    )

    assert np.isclose(abs(projected[0, 0]), np.pi, atol=np.deg2rad(1.1))


def test_circular_projection_uses_weights_and_marks_no_data() -> None:
    phase = np.array([0.0, np.pi / 2])[:, None]
    weights = np.array([3.0, 1.0])[:, None]
    valid = np.ones_like(phase, dtype=bool)

    projected = circular_phase_projection(phase, weights, valid, axis=0)
    missing = circular_phase_projection(phase, weights, np.zeros_like(valid), axis=0)

    assert np.isclose(projected[0], np.angle(3.0 + 1.0j))
    assert np.isnan(missing[0])


def test_fixed_sections_follow_established_dp150_offsets() -> None:
    support = np.ones((4, 5, 220), dtype=bool)
    support[:, :, 98] = False
    support[0, 0, 98] = True
    readout = -250.0 + (np.arange(220) + 0.5) * 1.25

    sections = select_fixed_sections(support, readout)

    assert [item.index for item in sections] == [70, 98, 122]
    assert [item.offset_from_throat_mm for item in sections] == [-35.0, 0.0, 30.0]


def test_playback_plans_use_physical_rr_and_exact_420_ms_slow_hold() -> None:
    physical, slow = physical_and_slow_plans(
        rr_s=1.01199996471405, bins=15, cycles=2, slow_hold_ms=420
    )

    assert physical.fps == Fraction(3750, 253)
    assert physical.frame_period_s == Fraction(253, 3750)
    assert slow.fps == Fraction(50, 21)
    assert slow.frame_period_s == Fraction(21, 50)
    assert playback_indices(physical) == list(range(15)) * 2
    assert physical.encoded_frames == 30


def test_display_limit_is_p99_9_of_actual_panel_population() -> None:
    first = np.linspace(0.0, 1.0, 1001)
    second = np.linspace(0.0, 2.0, 1001)
    panel_values = {
        "projection_xy": [first],
        "projection_xz": [first],
        "projection_yz": [first],
        "upstream": [first],
        "stenosis_throat": [second],
        "immediate_post_stenotic": [first],
    }
    limit, statistics = displayed_p999(
        panel_values
    )

    assert np.isclose(limit, np.quantile(second, 0.999))
    assert statistics["displayed_values"] == 6006
    assert statistics["limit_rule"].startswith("maximum p99.9")
    assert set(statistics["per_panel_class"]) == set(panel_values)


def test_display_limit_rejects_missing_panel_class() -> None:
    with pytest.raises(ValueError, match="exactly the six panel classes"):
        displayed_p999({"projection_xy": [np.array([0.0])]})


def test_render_frame_is_exact_aspect_and_true_grayscale(tmp_path) -> None:
    values = np.linspace(-1, 1, 20).reshape(4, 5)
    support = np.ones_like(values, dtype=bool)
    support[0, 0] = False
    no_data = np.zeros_like(support)
    no_data[1, 1] = True
    displayed = values.copy()
    displayed[1, 1] = np.nan
    panel = Panel(
        "test",
        displayed,
        support,
        no_data,
        (-2.5, 2.5, -2.0, 2.0),
        "x",
        "y",
        "test",
    )
    output = tmp_path / "frame.png"

    render_frame(
        [panel] * 6,
        output,
        "X",
        cardiac_bin=1,
        bins=15,
        phase_center_ms=33.7,
        limit=1.0,
        dpi=40,
    )

    with Image.open(output) as image:
        assert image.size == (18 * 40, 9 * 40)
        rgb = np.asarray(image.convert("RGB"))
    assert np.array_equal(rgb[..., 0], rgb[..., 1])
    assert np.array_equal(rgb[..., 1], rgb[..., 2])


def test_invalid_is_black_and_valid_negative_floor_is_not_black() -> None:
    colormap = signed_phase_colormap()

    invalid = np.asarray(colormap(np.ma.masked))[0:3]
    valid_floor = np.asarray(colormap(0.0))[0:3]

    assert np.array_equal(invalid, np.zeros(3))
    assert np.all(valid_floor > 0)
    assert np.allclose(valid_floor, valid_floor[0])


def test_playback_index_helper_rejects_no_implicit_interpolation() -> None:
    plan = PlaybackPlan("test", "TEST", Fraction(24, 1), 3, 4)

    assert playback_indices(plan) == [0, 1, 2, 3] * 3
