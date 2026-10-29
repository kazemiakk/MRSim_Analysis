"""Focused safety tests for the source-only aorta pipeline entry points."""

import pickle
from pathlib import Path

import numpy as np
import pytest

import run_aorta_pointcloud as runner
from scripts import audit_aorta_uniform_seeding as audit


def _write_canonical_case(
    root: Path,
    *,
    coordinate_rows: int = 4,
    velocity_rows: int = 4,
) -> tuple[Path, np.ndarray, np.ndarray]:
    case = root / "processed" / "dp2"
    case.mkdir(parents=True)
    coordinates = np.arange(coordinate_rows * 3, dtype=np.float32).reshape(
        coordinate_rows, 3
    )
    velocities = np.zeros((velocity_rows, 3, 2), dtype=np.float32)
    np.save(case / "point_coordinates.npy", coordinates)
    np.save(case / "velocities.npy", velocities)
    return case, coordinates, velocities


def test_runner_loads_canonical_matched_arrays(tmp_path):
    case, expected_coordinates, expected_velocities = _write_canonical_case(
        tmp_path
    )

    sources = runner.resolve_pointcloud_sources(tmp_path, "dp2")
    coordinates, velocities = runner.load_pointcloud_arrays(sources)

    assert sources.case_directory == case.resolve()
    assert sources.coordinates == (case / "point_coordinates.npy").resolve()
    assert sources.velocities == (case / "velocities.npy").resolve()
    assert sources.coordinates_provenance == "processed_simulation_directory"
    np.testing.assert_array_equal(coordinates, expected_coordinates)
    np.testing.assert_array_equal(velocities, expected_velocities)


def test_runner_accepts_explicit_coordinate_override(tmp_path):
    _write_canonical_case(tmp_path)
    override = tmp_path / "reviewed_coordinates.npy"
    np.save(override, np.ones((4, 3), dtype=np.float32))

    sources = runner.resolve_pointcloud_sources(
        tmp_path, "dp2", coordinates=override
    )

    assert sources.coordinates == override.resolve()
    assert sources.coordinates_provenance == "explicit_coordinates_override"


def test_runner_rejects_coordinate_velocity_row_mismatch(tmp_path):
    _write_canonical_case(tmp_path, coordinate_rows=4, velocity_rows=3)
    sources = runner.resolve_pointcloud_sources(tmp_path, "dp2")

    with pytest.raises(ValueError, match="row-count mismatch"):
        runner.load_pointcloud_arrays(sources)


def test_runner_rejects_report_output_collision(tmp_path):
    destination = tmp_path / "tracks.h5"

    with pytest.raises(ValueError, match="same path"):
        runner.validate_destination_paths(
            destination, destination, overwrite=False
        )


def test_atomic_tracking_output_publishes_only_on_success(tmp_path):
    destination = tmp_path / "tracks.h5"
    destination.write_bytes(b"previous-good-output")
    unrelated = tmp_path / "unrelated.partial"
    unrelated.write_bytes(b"keep")

    with runner.atomic_tracking_output(destination, overwrite=True) as partial:
        partial.write_bytes(b"new-complete-output")
        assert destination.read_bytes() == b"previous-good-output"

    assert destination.read_bytes() == b"new-complete-output"
    assert not runner.partial_tracking_path(destination).exists()

    with pytest.raises(RuntimeError, match="simulated failure"):
        with runner.atomic_tracking_output(
            destination, overwrite=True
        ) as partial:
            partial.write_bytes(b"incomplete-output")
            raise RuntimeError("simulated failure")

    assert destination.read_bytes() == b"new-complete-output"
    assert not runner.partial_tracking_path(destination).exists()
    assert unrelated.read_bytes() == b"keep"


def test_audit_resolves_canonical_coordinates_without_geometry(tmp_path):
    case, expected_coordinates, expected_velocities = _write_canonical_case(
        tmp_path
    )

    sources = audit.resolve_audit_sources(tmp_path, "dp2")
    coordinates, velocities = audit.load_audit_arrays(sources)

    assert sources.case_directory == case.resolve()
    assert sources.coordinates_format == "npy"
    assert not (case / "geometry.pkl").exists()
    np.testing.assert_array_equal(coordinates, expected_coordinates)
    np.testing.assert_array_equal(velocities, expected_velocities)


def test_audit_preserves_explicit_case_geometry_fallback(tmp_path, capsys):
    case = tmp_path / "legacy" / "dp2"
    case.mkdir(parents=True)
    coordinates = np.arange(12, dtype=np.float32).reshape(4, 3)
    with (case / "geometry.pkl").open("wb") as stream:
        pickle.dump({"mesh_coords": coordinates}, stream)
    np.save(case / "velocities.npy", np.zeros((4, 3, 2), dtype=np.float32))

    sources = audit.resolve_audit_sources(
        tmp_path, "ignored", case=case
    )
    loaded_coordinates, _ = audit.load_audit_arrays(sources)

    assert sources.coordinates_format == "geometry_pickle"
    assert "legacy geometry pickle" in capsys.readouterr().out
    np.testing.assert_array_equal(loaded_coordinates, coordinates)
