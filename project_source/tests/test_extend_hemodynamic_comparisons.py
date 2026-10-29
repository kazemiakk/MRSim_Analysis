import numpy as np
import pytest

from scripts.aorta_hemodynamic_reference import AortaGrid
from scripts.extend_hemodynamic_comparisons import require_same_reference, pilot_sections


def phantom(grid):
    return dict(fixed_source_center_m=grid.center, axis_order_one_based=np.array(grid.order)+1,
                realized_rr_s=.726, source_cycle_s=.72)


def test_reference_reuse_requires_same_axes_origin_timing():
    grid = AortaGrid(); p = phantom(grid); w = np.ones((8, 72))/72
    assert all(require_same_reference(grid, p, .726, w, w).values())
    for key, value in [('fixed_source_center_m', [0, 0, 0]), ('axis_order_one_based', [1, 2, 3]),
                       ('realized_rr_s', .72), ('source_cycle_s', .8)]:
        with pytest.raises(ValueError, match='Reference-reuse'):
            require_same_reference(grid, p | {key: value}, .726, w, w)
    with pytest.raises(ValueError, match='Reference-reuse'):
        require_same_reference(grid, p, .726, w, np.roll(np.eye(8, 72), 1, axis=1))


def test_empty_sections_remain_unavailable_not_zero_flow():
    grid = AortaGrid(shape=(4, 4, 4)); u = np.ones((2, 4, 4, 4, 3))
    r = pilot_sections(u, u, np.ones((4, 4, 4)), np.zeros((4, 4, 4), bool), grid, .72)
    assert r['summary']
    assert all(x['status'] == 'UNAVAILABLE_EMPTY_SUPPORT' for x in r['summary'])
    assert all(np.isnan(x['net_ml_s']) for x in r['series'])


def test_partial_section_sign_and_cycle_volume():
    grid = AortaGrid(shape=(4, 4, 4)); u = np.zeros((2, 4, 4, 4, 3)); u[..., 0] = 1
    r = pilot_sections(u, 2*u, np.ones((4, 4, 4)), np.ones((4, 4, 4), bool), grid, .72)
    assert all(x['net_cycle_volume_ratio'] == pytest.approx(2) for x in r['summary'])
    assert all(x['CFD_cycle']['net_m3'] == pytest.approx(.72*np.prod(grid.fov[1:])) for x in r['summary'])
