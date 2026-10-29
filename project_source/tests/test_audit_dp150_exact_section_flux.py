import numpy as np
import pyvista as pv
import pytest

from scripts.audit_dp150_exact_section_flux import triangle_flux, exact_mesh_section


def test_constant_triangle_flux_sign_and_winding():
    points = np.array([[0., 0, 0], [2, 0, 0], [0, 3, 0]])
    result = triangle_flux(points, [[0, 1, 2]], [-2., -2, -2])
    assert result["area_m2"] == pytest.approx(3)
    assert result["net_m3_s"] == pytest.approx(-6)
    assert triangle_flux(points, [[2, 1, 0]], [-2., -2, -2])["net_m3_s"] == pytest.approx(-6)


def test_linear_triangle_flux_is_exact():
    # u=x+2y+1 on x>=0,y>=0,x+y<=1: area=1/2, mean u=2.
    points = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0]])
    values = points[:, 0] + 2 * points[:, 1] + 1
    assert triangle_flux(points, [[0, 1, 2]], values)["net_m3_s"] == pytest.approx(1)


def test_cube_slice_recovers_linear_section_integral():
    mesh = pv.ImageData(dimensions=(2, 2, 2)).cast_to_unstructured_grid().triangulate()
    u = mesh.points[:, 0] + 2 * mesh.points[:, 1] + 3 * mesh.points[:, 2]
    result = exact_mesh_section(mesh, u, .25)
    assert result["area_m2"] == pytest.approx(1)
    assert result["net_m3_s"] == pytest.approx(.5 + 1 + .75)


def test_missing_velocity_and_degenerate_triangle_fail():
    points = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0]])
    with pytest.raises(ValueError, match="finite normal velocity"):
        triangle_flux(points, [[0, 1, 2]], [0, np.nan, 1])
    with pytest.raises(ValueError, match="Degenerate"):
        triangle_flux(points, [[0, 0, 1]], [0, 1, 1])
