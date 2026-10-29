from __future__ import annotations

import numpy as np
import pyvista as pv
import pytest

from scripts.dp150_hemodynamic_reference import (
    MRIGrid, bin_sample_times, build_voxel_lookup, build_section_lookup, geometry_sha256,
    load_lookup, native_tet_moments, save_lookup, transformed_nodal_velocity,
)
from mrsimtracks.io import _interp_time


def cube_mesh():
    return pv.ImageData(dimensions=(2, 2, 2), spacing=(1, 1, 1)).cast_to_unstructured_grid().triangulate()


def cube_grid():
    return MRIGrid((1, 1, 1), (1.0, 1.0, 1.0), (0.5, 0.5, 0.5), (0, 1, 2))


def test_production_coordinates_and_vector_order():
    grid = MRIGrid((32, 64, 400), (0.5, 0.08, 0.08), (0.0, 0.0, 0.12699999287724495))
    np.testing.assert_allclose(grid.spacing_rps_m, [0.00125, 0.00125, 0.0025])
    r, p, s = grid.axis_centers_m()
    np.testing.assert_allclose([r[0], p[0], s[0]], [-0.249375, -0.039375, -0.03875])
    points = np.array([[r[70], 0, 0], [r[98], 0, 0], [r[122], 0, 0]])
    source = grid.mri_to_source(points)
    np.testing.assert_allclose(source[:, 2], [-0.03487500712275505, 0.00012499287724495, 0.03012499287724495], atol=1e-15)
    np.testing.assert_allclose(grid.source_to_mri(source), points)
    np.testing.assert_array_equal(grid.velocity_to_mri([[1, 2, 3]]), [[3, 1, 2]])


def test_cube_quadrature_separates_second_moment_from_squared_mean():
    mesh = cube_mesh()
    lookup = build_voxel_lookup(mesh, cube_grid(), subsamples=2, chunk_voxels=1)
    velocity = np.asarray(mesh.points) * [1, 2, 3]
    result = lookup.average(velocity, chunk_size=3)
    np.testing.assert_allclose(result["volume_fraction"], 1)
    np.testing.assert_allclose(result["velocity_mri"][:, 0, 0, 0], [0.5, 1, 1.5])
    np.testing.assert_allclose(result["mean_speed_squared"], 14 * 0.3125)
    assert result["mean_speed_squared"].item() > np.sum(result["velocity_mri"] ** 2)


def test_tetra_fraction_is_estimate_and_outside_stays_nan():
    mesh = pv.UnstructuredGrid(np.array([4, 0, 1, 2, 3]), np.array([10], dtype=np.uint8),
                              np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float))
    grid = MRIGrid((1, 1, 2), (2., 1., 1.), (1., .5, .5), (0, 1, 2))
    lookup = build_voxel_lookup(mesh, grid, subsamples=2)
    result = lookup.average(np.ones((4, 3)))
    np.testing.assert_allclose(result["volume_fraction"], [[[0.125, 0]]])
    np.testing.assert_allclose(result["velocity_mri"][:, 0, 0, 0], 1)
    assert np.isnan(result["velocity_mri"][:, 0, 0, 1]).all()


def test_native_linear_tetra_squared_speed_integral_is_exact():
    mesh = cube_mesh()
    result = native_tet_moments(mesh, np.asarray(mesh.points) * [1, 2, 3], grid=cube_grid())
    assert result["total_volume_m3"] == pytest.approx(1)
    assert result["integral_speed_squared_m5_s2"] == pytest.approx(14 / 3)
    np.testing.assert_allclose(result["integral_velocity_m4_s"], [.5, 1., 1.5])
    assert result["cell_centers_mri_m"].shape == (mesh.n_cells, 3)


def test_midpoint_and_three_sample_time_contract():
    times = bin_sample_times(15, 1.012)
    np.testing.assert_allclose(times[:, 0] / 1.012 * .72, (np.arange(15) + .5) * .72 / 15)
    three = bin_sample_times(15, 1.012, samples_per_bin=3)
    np.testing.assert_allclose(three.mean(axis=1), times[:, 0])
    assert np.all(three[:, 0] > np.arange(15) * 1.012 / 15)
    assert np.all(three[:, -1] < (np.arange(15) + 1) * 1.012 / 15)
    with pytest.raises(ValueError):
        bin_sample_times(15, 1.012, samples_per_bin=2)


def test_temporal_interpolation_reuses_production_and_retimes_velocity():
    class Flow:
        tmax = .72
        active_key = "Velocity"
        def __init__(self):
            self.active_mesh = cube_mesh()
            self.times = np.linspace(0, self.tmax, 73)
            self.fields = [np.tile([t, 2*t, 3*t], (8, 1)).astype(np.float32) for t in self.times]
            self.fields[-1] = self.fields[0].copy()
        def set_active_time(self, t):
            self.active_mesh.point_data[self.active_key] = _interp_time(self.times, self.tmax, 72, lambda i: self.fields[i], t, "cubic")
    flow = Flow()
    grid = MRIGrid((1,1,1), (1.,1.,1.), (.5,.5,.5))
    effective = 1.0120000839
    t_source = .185
    actual = transformed_nodal_velocity(flow, grid, t_source / .72 * effective, effective)
    expected = _interp_time(flow.times, .72, 72, lambda i: flow.fields[i], t_source, "cubic")[:, [2,0,1]] * (.72/effective)
    np.testing.assert_allclose(actual, expected, rtol=1e-7)
    repeated = transformed_nodal_velocity(flow, grid, t_source / .72 * effective + effective, effective)
    np.testing.assert_allclose(repeated, actual, rtol=1e-7)


def test_cache_roundtrip_fail_closed_and_no_overwrite(tmp_path):
    mesh, grid = cube_mesh(), cube_grid()
    lookup = build_voxel_lookup(mesh, grid)
    path = tmp_path / "lookup.npz"
    save_lookup(path, lookup)
    restored = load_lookup(path, expected_grid=grid, expected_mesh_sha256=geometry_sha256(mesh), expected_subsamples=2)
    np.testing.assert_array_equal(restored.fraction_grid(), lookup.fraction_grid())
    np.testing.assert_array_equal(restored.barycentric, lookup.barycentric)
    with pytest.raises(FileExistsError):
        save_lookup(path, lookup)
    with pytest.raises(ValueError, match="provenance"):
        load_lookup(path, expected_grid=grid, expected_mesh_sha256="wrong", expected_subsamples=2)
    with pytest.raises(ValueError, match="provenance"):
        load_lookup(path, expected_grid=grid, expected_mesh_sha256=geometry_sha256(mesh), expected_subsamples=3)
    assert not list(tmp_path.glob("*.partial"))


def test_lookup_rejects_wrong_field_and_invalid_inputs():
    lookup = build_voxel_lookup(cube_mesh(), cube_grid())
    with pytest.raises(ValueError):
        lookup.average(np.zeros((2, 3)))
    with pytest.raises(ValueError):
        lookup.average(np.full((8, 3), np.nan))
    with pytest.raises(ValueError):
        build_voxel_lookup(cube_mesh(), cube_grid(), subsamples=0)
    with pytest.raises(ValueError):
        MRIGrid((1,1,0), (1.,1.,1.), (0.,0.,0.))


def test_plane_samples_do_not_average_along_readout(tmp_path):
    mesh, grid = cube_mesh(), cube_grid()
    plane = build_section_lookup(mesh, grid, transverse_subsamples=2)
    assert plane.samples_per_voxel == 4
    values = np.column_stack([mesh.points[:, 0], np.zeros((mesh.n_points, 2))])
    result = plane.average(values)
    # Linear velocity at readout center x=.5: <u²>=.25, while the volume
    # quadrature would give .3125 because it includes x=.25 and .75.
    np.testing.assert_allclose(result["mean_speed_squared"], .25)
    np.testing.assert_allclose(result["volume_fraction"], 1)
    path = tmp_path / "plane.npz"
    save_lookup(path, plane)
    restored = load_lookup(path, expected_grid=grid, expected_mesh_sha256=geometry_sha256(mesh), expected_subsamples=(1, 2, 2))
    assert restored.samples_per_voxel == 4
    with pytest.raises(ValueError, match="provenance"):
        load_lookup(path, expected_grid=grid, expected_mesh_sha256=geometry_sha256(mesh), expected_subsamples=2)


def test_native_moments_condition_wedge_without_changing_nodal_fields():
    mesh = pv.UnstructuredGrid(np.array([6, 0, 1, 2, 3, 4, 5]), np.array([13], dtype=np.uint8),
        np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1], [1,0,1], [0,1,1]], dtype=float))
    result = native_tet_moments(mesh, np.tile([1., 2., 3.], (6,1)))
    assert result["total_volume_m3"] == pytest.approx(.5)
    assert result["integral_speed_squared_m5_s2"] == pytest.approx(7.)
