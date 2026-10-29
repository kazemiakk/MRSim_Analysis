from __future__ import annotations

import hashlib
import json

import numpy as np
from PIL import Image
import pytest

from scripts import render_dp150_phase_boundary_qc as e016
from scripts.render_dp150_phase_stable_mask_clean import (
    EXCLUDED_GRAY,
    INHERITED_E016_LIMITS_RAD,
    load_e016_limit_source,
    refine_stable_mask_with_signal,
    render_frame,
    stable_display_statistics,
    stable_mask_coverage,
    stable_reconstruction_mask,
)


def test_stable_mask_is_support_intersection_of_all_bins() -> None:
    support = np.array([[[True, True, False, True]]])
    masks = np.ones((15, 1, 1, 4), dtype=bool)
    masks[7, 0, 0, 1] = False
    masks[:, 0, 0, 2] = True

    stable = stable_reconstruction_mask(support, masks)

    assert stable.tolist() == [[[True, False, False, True]]]


def test_stable_mask_rejects_incomplete_cardiac_axis() -> None:
    with pytest.raises(ValueError, match="all 15"):
        stable_reconstruction_mask(
            np.ones((2, 2, 2), dtype=bool),
            np.ones((14, 2, 2, 2), dtype=bool),
        )


def test_signal_refinement_is_one_component_independent_intersection() -> None:
    initial = np.ones((2, 2, 2), dtype=bool)
    x_valid = np.ones_like(initial)
    z_valid = np.ones_like(initial)
    x_valid[0, 0, 0] = False
    z_valid[1, 1, 1] = False

    stable, removed = refine_stable_mask_with_signal(
        initial, [x_valid, np.ones_like(initial), z_valid]
    )

    assert removed == 2
    assert stable.sum() == 6
    assert not stable[0, 0, 0]
    assert not stable[1, 1, 1]


def test_coverage_reports_static_projection_and_section_exclusions() -> None:
    support = np.ones((2, 3, 4), dtype=bool)
    stable = support.copy()
    stable[0, 0, :] = False
    masks = np.broadcast_to(stable, (15, *stable.shape)).copy()
    sections = (
        e016.Section("Upstream", 0, -1.0, -1.0),
        e016.Section("Stenosis throat", 1, 0.0, 0.0),
        e016.Section("Immediate post-stenotic", 2, 1.0, 1.0),
    )

    coverage = stable_mask_coverage(support, stable, masks, sections)

    assert coverage["full_exact_support_voxels"] == 24
    assert coverage["stable_voxels"] == 20
    assert coverage["fixed_across_all_bins_and_components"] is True
    assert [item["stable_voxels"] for item in coverage["sections"]] == [5, 5, 5]
    assert coverage["projection_lines_of_sight"]["YZ"]["excluded_lines_of_sight"] == 1
    assert coverage["projection_lines_of_sight"]["XY"]["excluded_lines_of_sight"] == 0


def test_panels_hatch_projection_only_without_a_stable_line_of_sight() -> None:
    shape = (3, 4, 5)
    support = np.ones(shape, dtype=bool)
    stable = support.copy()
    stable[:, 0, 0] = False
    phase = np.zeros(shape, dtype=float)
    weights = np.ones(shape, dtype=float)
    sections = (
        e016.Section("Upstream", 0, -2.0, -2.0),
        e016.Section("Stenosis throat", 2, 0.0, 0.0),
        e016.Section("Immediate post-stenotic", 4, 2.0, 2.0),
    )

    panels = e016.build_panels(
        phase,
        weights,
        stable,
        support,
        sections,
        np.arange(5, dtype=float),
        np.arange(4, dtype=float),
        np.arange(3, dtype=float),
    )

    assert panels[0].no_data[0, 0]
    assert not panels[1].no_data.any()
    assert not panels[2].no_data.any()
    assert panels[3].no_data.sum() == 3
    assert np.isnan(panels[3].values[panels[3].no_data]).all()


def test_stable_statistics_keep_inherited_limit_and_report_own_p999() -> None:
    first = np.linspace(0.0, 1.0, 1001)
    throat = np.linspace(0.0, 2.0, 1001)
    values = {
        "projection_xy": [first],
        "projection_xz": [first],
        "projection_yz": [first],
        "upstream": [first],
        "stenosis_throat": [throat],
        "immediate_post_stenotic": [first],
    }

    stats = stable_display_statistics(values, inherited_limit=1.5)

    assert stats["display_limit_rad"] == 1.5
    assert stats["rescaled_for_e017"] is False
    assert np.isclose(
        stats["stable_domain_maximum_six_class_p99_9_rad"],
        np.quantile(throat, 0.999),
    )
    assert stats["clipped_at_inherited_e016_limit"] == 250


def test_e016_limit_source_is_hash_bound_and_canonical(tmp_path) -> None:
    report = {
        "experiment_id": "E-016",
        "inputs": {
            "reconstruction": {"sha256": e016.CANONICAL_RECONSTRUCTION_SHA256},
            "seed_derived_support": {"sha256": e016.CANONICAL_SUPPORT_SHA256},
            "raw_complex_images": {
                name: {"sha256": value}
                for name, value in e016.CANONICAL_RAW_SHA256.items()
            },
        },
        "display": {
            "component_limits": {
                name: {"p99_9_component_limit_rad": value}
                for name, value in INHERITED_E016_LIMITS_RAD.items()
            }
        },
    }
    path = tmp_path / "e016.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    limits, record = load_e016_limit_source(path, digest)

    assert limits == INHERITED_E016_LIMITS_RAD
    assert record["sha256"] == digest
    with pytest.raises(ValueError, match="SHA-256"):
        load_e016_limit_source(path, "0" * 64)


def test_render_frame_uses_grayscale_hatched_exclusions(tmp_path) -> None:
    values = np.linspace(-1.0, 1.0, 64).reshape(8, 8)
    support = np.ones_like(values, dtype=bool)
    support[:2, :2] = False
    no_data = np.zeros_like(support)
    no_data[2:6, 2:6] = True
    displayed = values.copy()
    displayed[no_data] = np.nan
    panel = e016.Panel(
        "stable mask test",
        displayed,
        support,
        no_data,
        (-4.0, 4.0, -4.0, 4.0),
        "x",
        "y",
        "upstream",
    )
    output = tmp_path / "stable.png"

    render_frame(
        [panel] * 6,
        output,
        "X",
        cardiac_bin=1,
        bins=15,
        phase_center_ms=33.7,
        limit=1.0,
        dpi=60,
    )

    with Image.open(output) as image:
        assert image.size == (18 * 60, 9 * 60)
        rgb = np.asarray(image.convert("RGB"))
    assert np.array_equal(rgb[..., 0], rgb[..., 1])
    assert np.array_equal(rgb[..., 1], rgb[..., 2])
    excluded_value = int(EXCLUDED_GRAY[1:3], 16)
    assert np.count_nonzero(np.all(rgb == excluded_value, axis=-1)) > 100
    assert np.count_nonzero(np.all(rgb == 0, axis=-1)) > 100
