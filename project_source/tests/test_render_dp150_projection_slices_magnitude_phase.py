from __future__ import annotations

from PIL import Image
import numpy as np

from scripts.render_dp150_projection_slices_magnitude_phase import (
    plane_definitions,
    render_six_panels,
)


def test_render_six_panels_respects_requested_dpi(tmp_path) -> None:
    definitions = plane_definitions((2, 3, 4), np.array([4.0, 3.0, 2.0]))
    panels = [
        np.zeros((3, 4)),
        np.zeros((2, 4)),
        np.zeros((2, 3)),
        np.zeros((3, 4)),
        np.zeros((2, 4)),
        np.zeros((2, 3)),
    ]
    output = tmp_path / "six_panel.png"

    render_six_panels(
        panels,
        definitions,
        output,
        "test",
        "magnitude",
        display_limit=1.0,
        dpi=72,
    )

    with Image.open(output) as image:
        assert image.size == (18 * 72, 9 * 72)
