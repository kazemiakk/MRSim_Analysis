r"""Convert the raw dp150 Fluent CFF case into a time-resolved PVD flow.

The converter intentionally includes frame 00000, so the output spans the full
recorded interval from 0.00 through 0.72 s. Fluent stores velocity at cell
centres; VTK's cell-to-point filter supplies the point field required by
MRSimTracks. Curved-wall vertices are then set to the no-slip value while the
two planar open caps retain their interpolated velocities.

Run in an ephemeral, non-project environment so the repository checkout cannot
shadow the installed MRSimTracks release:

    uv run --isolated --no-project --with "mrsimtracks==0.2.1" `
        python -I scripts\convert_dp150_fluent_cff.py
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

import h5py
import numpy as np
import pyvista as pv
import vtk


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASE = (
    ROOT.parent / "external_inputs" / "dp150_fluent" / "FFF-18.cas.h5"
)
DEFAULT_DATA_DIR = DEFAULT_CASE.parent
DEFAULT_OUTPUT = ROOT / "outputs" / "dp150_volume_pipeline"
ACTIVE_KEY = "Velocity"
EXPECTED_VERSION = "0.2.1"
DATA_VARIABLES = "settings/Data Variables"
CELL_ROOT = "results/1/phase-1/cells"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, default=DEFAULT_CASE)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--first-frame", type=int, default=0)
    parser.add_argument("--last-frame", type=int, default=72)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing VTU frames instead of validating and reusing them.",
    )
    return parser.parse_args()


def package_provenance() -> dict[str, str]:
    version = importlib.metadata.version("mrsimtracks")
    import mrsimtracks

    module = Path(mrsimtracks.__file__).resolve()
    if version != EXPECTED_VERSION:
        raise RuntimeError(
            f"MRSimTracks {EXPECTED_VERSION} is required, found distribution {version}"
        )
    if "site-packages" not in {part.lower() for part in module.parts}:
        raise RuntimeError(
            "The repository checkout is shadowing the installed MRSimTracks wheel. "
            "Run this script with Python -I."
        )
    return {"distribution_version": version, "module_path": str(module)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_case(case_path: Path) -> pv.UnstructuredGrid:
    reader = vtk.vtkFLUENTCFFReader()
    reader.SetFileName(str(case_path))
    reader.Update()
    output = reader.GetOutputDataObject(0)
    if output is None or output.GetNumberOfBlocks() != 1:
        raise RuntimeError("Expected one Fluent volume block")
    grid = pv.wrap(output.GetBlock(0)).copy(deep=True)
    if not isinstance(grid, pv.UnstructuredGrid):
        raise TypeError(f"Expected UnstructuredGrid, found {type(grid).__name__}")
    if grid.n_points == 0 or grid.n_cells == 0:
        raise ValueError("Fluent case contains an empty volume mesh")
    return grid


def extract_boundaries(
    grid: pv.UnstructuredGrid, output_dir: Path
) -> tuple[np.ndarray, dict[str, object]]:
    surface = grid.extract_surface(
        algorithm="dataset_surface", pass_pointid=True, pass_cellid=True
    )
    original_ids = np.asarray(surface.point_data["vtkOriginalPointIds"], dtype=np.int64)
    z_min, z_max = float(grid.bounds[4]), float(grid.bounds[5])
    long_extent = z_max - z_min
    cap_tol = max(1e-8, 1e-6 * long_extent)
    centers_z = np.asarray(surface.cell_centers().points[:, 2])
    inlet_cells = np.flatnonzero(np.abs(centers_z - z_min) <= cap_tol)
    outlet_cells = np.flatnonzero(np.abs(centers_z - z_max) <= cap_tol)
    wall_cells = np.flatnonzero(
        (np.abs(centers_z - z_min) > cap_tol)
        & (np.abs(centers_z - z_max) > cap_tol)
    )
    if inlet_cells.size == 0 or outlet_cells.size == 0 or wall_cells.size == 0:
        raise RuntimeError(
            "Could not separate inlet, outlet, and wall surfaces from the axial bounds"
        )

    def save_surface(name: str, cell_ids: np.ndarray, region_id: int) -> pv.PolyData:
        part = (
            surface.extract_cells(cell_ids)
            .extract_surface(algorithm="dataset_surface")
            .triangulate()
            .clean()
        )
        part.cell_data["region_id"] = np.full(part.n_cells, region_id, np.int32)
        part.save(output_dir / f"{name}.vtp", binary=True)
        return part

    inlet = save_surface("inlet", inlet_cells, 0)
    outlet = save_surface("outlet", outlet_cells, 1)
    wall = save_surface("wall", wall_cells, 0)

    cap_points = (np.abs(surface.points[:, 2] - z_min) <= cap_tol) | (
        np.abs(surface.points[:, 2] - z_max) <= cap_tol
    )
    wall_point_ids = np.unique(original_ids[~cap_points])
    if np.any(wall_point_ids < 0) or np.any(wall_point_ids >= grid.n_points):
        raise RuntimeError("Surface-to-volume point IDs are invalid")

    metadata = {
        "classification_axis": "z",
        "cap_tolerance_m": cap_tol,
        "z_min_m": z_min,
        "z_max_m": z_max,
        "inlet": {"points": inlet.n_points, "triangles": inlet.n_cells},
        "outlet": {"points": outlet.n_points, "triangles": outlet.n_cells},
        "wall": {"points": wall.n_points, "triangles": wall.n_cells},
        "no_slip_volume_point_count": int(wall_point_ids.size),
    }
    return wall_point_ids, metadata


def read_frame(data_path: Path, expected_cells: int) -> tuple[float, int, np.ndarray]:
    with h5py.File(data_path, "r") as h5:
        text = h5[DATA_VARIABLES][0].decode("utf-8", errors="strict")
        time_match = re.search(r"\(flow-time\s+([^)]+)\)", text)
        step_match = re.search(r"\(time-step\s+([^)]+)\)", text)
        if not time_match or not step_match:
            raise ValueError(f"Missing flow-time or time-step in {data_path}")
        flow_time = float(time_match.group(1))
        step = int(float(step_match.group(1)))
        components = [
            np.asarray(h5[f"{CELL_ROOT}/SV_{component}/1"], dtype=np.float32)
            for component in ("U", "V", "W")
        ]
    velocity = np.column_stack(components)
    if velocity.shape != (expected_cells, 3):
        raise ValueError(
            f"Velocity shape {velocity.shape} does not match {expected_cells} cells"
        )
    if not np.isfinite(velocity).all():
        raise ValueError(f"Non-finite velocity found in {data_path}")
    return flow_time, step, velocity


def point_velocity_frame(
    geometry: pv.UnstructuredGrid,
    cell_velocity: np.ndarray,
    wall_point_ids: np.ndarray,
) -> pv.UnstructuredGrid:
    frame = geometry.copy(deep=False)
    frame.cell_data.clear()
    frame.point_data.clear()
    frame.cell_data[ACTIVE_KEY] = cell_velocity
    point_frame = frame.cell_data_to_point_data(pass_cell_data=False)
    velocity = np.asarray(point_frame.point_data[ACTIVE_KEY], dtype=np.float32)
    velocity[wall_point_ids] = 0.0
    point_frame.point_data[ACTIVE_KEY] = velocity
    return point_frame


def write_pvd(pvd_path: Path, entries: list[tuple[float, Path]]) -> None:
    root = ET.Element(
        "VTKFile", type="Collection", version="0.1", byte_order="LittleEndian"
    )
    collection = ET.SubElement(root, "Collection")
    for time_s, frame_path in entries:
        ET.SubElement(
            collection,
            "DataSet",
            timestep=f"{time_s:.12g}",
            group="",
            part="0",
            file=frame_path.as_posix(),
        )
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(pvd_path, encoding="utf-8", xml_declaration=True)


def speed_summary(velocity: np.ndarray) -> dict[str, float]:
    speed = np.linalg.norm(velocity.astype(np.float64), axis=1)
    return {
        "mean_m_per_s": float(speed.mean()),
        "q99_m_per_s": float(np.quantile(speed, 0.99)),
        "max_m_per_s": float(speed.max()),
    }


def closure_summary(first: np.ndarray, last: np.ndarray) -> dict[str, object]:
    a = first.astype(np.float64).reshape(-1)
    b = last.astype(np.float64).reshape(-1)
    delta = b - a
    rms_first = float(np.sqrt(np.mean(a * a)))
    rms_last = float(np.sqrt(np.mean(b * b)))
    rmse = float(np.sqrt(np.mean(delta * delta)))
    denom = 0.5 * (rms_first + rms_last)
    correlation = float(np.corrcoef(a, b)[0, 1])
    return {
        "first_component_rms_m_per_s": rms_first,
        "last_component_rms_m_per_s": rms_last,
        "component_rmse_m_per_s": rmse,
        "relative_rmse_to_mean_endpoint_rms": rmse / denom if denom else None,
        "flattened_vector_correlation": correlation,
        "periodic_closure_passed": False,
        "interpretation": (
            "The recorded first and last fields are not cycle-matched. This PVD is "
            "approved only for a single non-looping visualization pilot."
        ),
    }


def main() -> None:
    args = parse_args()
    provenance = package_provenance()
    case_path = args.case.resolve()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    flow_dir = output_dir / "flow"
    frame_dir = flow_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    if not case_path.is_file():
        raise FileNotFoundError(case_path)
    if args.first_frame < 0 or args.last_frame < args.first_frame:
        raise ValueError("Invalid frame range")

    print(f"Reading Fluent geometry: {case_path}", flush=True)
    geometry = read_case(case_path)
    print(
        f"Geometry: {geometry.n_points:,} points, {geometry.n_cells:,} cells, "
        f"bounds={geometry.bounds}",
        flush=True,
    )
    wall_point_ids, boundary_report = extract_boundaries(geometry, flow_dir)

    entries: list[tuple[float, Path]] = []
    frame_reports: list[dict[str, object]] = []
    first_velocity: np.ndarray | None = None
    last_velocity: np.ndarray | None = None
    for frame_index in range(args.first_frame, args.last_frame + 1):
        data_path = data_dir / f"FFF-18-{frame_index:05d}.dat.h5"
        if not data_path.is_file():
            raise FileNotFoundError(data_path)
        flow_time, stored_step, cell_velocity = read_frame(data_path, geometry.n_cells)
        if stored_step != frame_index:
            raise ValueError(
                f"Frame {frame_index} contains Fluent time-step {stored_step}"
            )
        frame = point_velocity_frame(geometry, cell_velocity, wall_point_ids)
        point_velocity = np.asarray(frame.point_data[ACTIVE_KEY])
        if first_velocity is None:
            first_velocity = point_velocity.copy()
        last_velocity = point_velocity.copy()

        frame_name = f"dp150_{frame_index:05d}.vtu"
        frame_path = frame_dir / frame_name
        if args.overwrite or not frame_path.exists():
            frame.save(frame_path, binary=True)
        else:
            existing = pv.read(frame_path)
            if (
                existing.n_points != geometry.n_points
                or existing.n_cells != geometry.n_cells
                or ACTIVE_KEY not in existing.point_data
            ):
                raise RuntimeError(f"Existing output is incompatible: {frame_path}")
        entries.append((flow_time, Path("frames") / frame_name))
        report = {
            "index": frame_index,
            "fluent_time_step": stored_step,
            "source_time_s": flow_time,
            "source_file": str(data_path),
            "output_file": str(frame_path),
            "output_bytes": frame_path.stat().st_size,
            "point_speed": speed_summary(point_velocity),
        }
        frame_reports.append(report)
        print(
            f"[{frame_index:02d}/{args.last_frame:02d}] t={flow_time:.3f} s  "
            f"mean={report['point_speed']['mean_m_per_s']:.4f} m/s  "
            f"max={report['point_speed']['max_m_per_s']:.4f} m/s",
            flush=True,
        )

    source_times = np.asarray([entry[0] for entry in entries], dtype=float)
    if len(source_times) < 2 or np.any(np.diff(source_times) <= 0):
        raise ValueError("Frame times must be strictly increasing")
    source_dt = np.diff(source_times)
    median_source_dt = float(np.median(source_dt))
    if not np.allclose(source_dt, median_source_dt, rtol=1e-4, atol=1e-7):
        raise ValueError(f"Frame times are not nominally uniformly spaced: {source_dt}")
    # Fluent serializes flow-time with small floating-point jitter. Use the
    # nearest nanosecond as the PVD clock so cubic interpolation sees a truly
    # uniform grid, while preserving the source values in the report.
    canonical_dt = round(median_source_dt, 6)
    canonical_start = round(float(source_times[0]), 6)
    times = canonical_start + np.arange(len(source_times), dtype=float) * canonical_dt
    entries = [
        (float(time_s), relative_path)
        for time_s, (_, relative_path) in zip(times, entries, strict=True)
    ]
    for frame_report, pvd_time in zip(frame_reports, times, strict=True):
        frame_report["pvd_time_s"] = float(pvd_time)
    assert first_velocity is not None and last_velocity is not None

    pvd_path = flow_dir / "dp150_flow.pvd"
    write_pvd(pvd_path, entries)
    cell_counts = Counter(int(value) for value in np.asarray(geometry.celltypes))
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
        },
        "mrsimtracks": provenance,
        "converter": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256(Path(__file__).resolve()),
            "cell_to_point_method": "vtkCellDataToPointData",
            "curved_wall_no_slip_enforced": True,
        },
        "source": {
            "case_file": str(case_path),
            "case_bytes": case_path.stat().st_size,
            "case_sha256": sha256(case_path),
            "data_directory": str(data_dir),
            "frame_range_inclusive": [args.first_frame, args.last_frame],
        },
        "geometry": {
            "points": geometry.n_points,
            "cells": geometry.n_cells,
            "cell_types": {
                pv.CellType(key).name: value
                for key, value in sorted(cell_counts.items())
            },
            "bounds_m": list(geometry.bounds),
            "extents_m": [
                geometry.bounds[1] - geometry.bounds[0],
                geometry.bounds[3] - geometry.bounds[2],
                geometry.bounds[5] - geometry.bounds[4],
            ],
        },
        "boundaries": boundary_report,
        "temporal": {
            "frame_count": len(entries),
            "first_time_s": float(times[0]),
            "last_time_s": float(times[-1]),
            "spacing_s": canonical_dt,
            "duration_s": float(times[-1] - times[0]),
            "source_first_time_s": float(source_times[0]),
            "source_last_time_s": float(source_times[-1]),
            "source_spacing_median_s": median_source_dt,
            "source_spacing_max_deviation_s": float(
                np.max(np.abs(source_dt - median_source_dt))
            ),
            "single_cycle_pilot_only": True,
        },
        "endpoint_closure": closure_summary(first_velocity, last_velocity),
        "frames": frame_reports,
        "output": {
            "pvd_file": str(pvd_path),
            "pvd_sha256": sha256(pvd_path),
            "total_vtu_bytes": sum(item[1].stat().st_size for item in [
                (time_s, flow_dir / relative) for time_s, relative in entries
            ]),
        },
    }
    report_path = output_dir / "conversion_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote PVD: {pvd_path}", flush=True)
    print(f"Wrote conversion report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
