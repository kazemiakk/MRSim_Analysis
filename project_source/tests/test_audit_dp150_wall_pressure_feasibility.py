import h5py
import numpy as np
import pytest

from scripts.audit_dp150_wall_pressure_feasibility import (
    geometry_audit,
    retiming_constraints,
    source_field_inventory,
    support_audit,
)


def test_support_boundary_and_direct_gradient_stencils():
    support = np.zeros((7, 7, 7), bool)
    support[1:6, 1:6, 1:6] = True
    direct = support.copy()
    direct[1, 3, 3] = False
    result = support_audit(support, direct)
    assert result["overall"]["support_voxels"] == 125
    assert result["boundary_six_connected_one_voxel"]["support_voxels"] == 98
    assert result["boundary_six_connected_one_voxel"]["regularized_voxels"] == 1
    assert result["all_direct_center_and_six_neighbor_gradient_stencils"] == 26
    assert result["boundary_voxels_with_complete_in_lumen_central_stencil"] == 0


def test_direct_outside_support_fails():
    support = np.ones((3, 3, 3), bool)
    direct = support.copy()
    support[0, 0, 0] = False
    with pytest.raises(ValueError, match="contained"):
        support_audit(support, direct)


def test_geometry_ratios_preserve_anisotropy():
    points = np.array([[.01, 0, .1], [-.01, 0, .1], [0, .01, .1],
                       [0, -.01, .1], [.005, 0, 0], [-.005, 0, 0]])
    result = geometry_audit(points, [1.25, 1.25, 2.5])
    assert result["wall_vertex_min_radius_mm"] == 5
    assert result["throat_diameter_samples_phase_slice"] == [8, 4]
    assert result["throat_radius_samples_phase_slice"] == [4, 2]


def test_retiming_does_not_apply_invalid_pressure_scale():
    result = retiming_constraints(.72 / 1.012)
    assert result["inertial_acceleration_scale_a_squared"] != result["viscous_acceleration_scale_same_viscosity_a"]
    assert not result["original_pressure_is_matched_reference"]
    assert not result["pressure_scaling_applied"]


def test_field_inventory_distinguishes_absent_and_present(tmp_path):
    path = tmp_path / "frame.h5"
    with h5py.File(path, "w") as h5:
        h5.create_dataset("results/1/phase-1/cells/SV_P/1", data=[1., 2., 3.])
    result = source_field_inventory(path)["fields"]
    assert result["cell_pressure"]["present"]
    assert result["cell_pressure"]["shape"] == [3]
    assert not result["face_wall_shear"]["present"]
