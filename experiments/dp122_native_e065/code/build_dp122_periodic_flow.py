# E-065 copy of MRSimTracks scripts/build_dp150_periodic_flow.py; only dp150_* file names -> dp122_* (4 replacements). Run with explicit roots.
r"""Create an explicitly labeled periodic dp150 flow for repeated MRI motion.

The raw Fluent record does not close at 0.72 s. This script preserves every
frame through ``--blend-start`` and applies a quintic tail blend to the opening
field thereafter. The closing field is therefore exactly equal to the opening
field, while the original converted PVD remains untouched.

This is a controlled simulation input, not recovered CFD information. The
report records the complete correction and the unavailable 2 ms source-data
limitation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pyvista as pv


DEFAULT_SOURCE = Path("outputs/dp150_volume_pipeline")
DEFAULT_OUTPUT = Path("outputs/dp150_mri_pipeline")
ACTIVE_KEY = "Velocity"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--blend-start", type=float, default=0.64)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pvd_entries(path: Path) -> list[tuple[float, Path]]:
    root = ET.parse(path).getroot()
    entries = [
        (float(item.get("timestep")), (path.parent / item.get("file")).resolve())
        for item in root.iter("DataSet")
    ]
    entries.sort(key=lambda item: item[0])
    if len(entries) < 2:
        raise ValueError(f"Expected at least two PVD entries in {path}")
    return entries


def smoothstep5(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value**3 * (10.0 - 15.0 * value + 6.0 * value**2)


def write_pvd(path: Path, times: list[float], files: list[Path]) -> None:
    vtk = ET.Element(
        "VTKFile",
        type="Collection",
        version="0.1",
        byte_order="LittleEndian",
    )
    collection = ET.SubElement(vtk, "Collection")
    for time_s, file_path in zip(times, files, strict=True):
        ET.SubElement(
            collection,
            "DataSet",
            timestep=format(time_s, ".12g"),
            group="",
            part="0",
            file=file_path.relative_to(path.parent).as_posix(),
        )
    tree = ET.ElementTree(vtk)
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    source_pvd = source_root / "flow" / "dp122_flow.pvd"
    conversion_report_path = source_root / "conversion_report.json"
    if not source_pvd.exists() or not conversion_report_path.exists():
        raise FileNotFoundError("Run convert_dp150_fluent_cff.py before periodic closure")
    conversion = json.loads(conversion_report_path.read_text(encoding="utf-8"))
    if conversion["endpoint_closure"]["periodic_closure_passed"] is not False:
        raise RuntimeError("Source conversion report does not document the closure failure")

    entries = pvd_entries(source_pvd)
    times = np.asarray([item[0] for item in entries], dtype=np.float64)
    duration = float(times[-1] - times[0])
    spacing = np.diff(times)
    if not np.allclose(spacing, spacing[0], rtol=0, atol=1e-12):
        raise ValueError("Source PVD times are not uniformly spaced")
    if not (times[0] <= args.blend_start < times[-1]):
        raise ValueError("blend-start must lie inside the source interval")

    flow_dir = output_root / "flow_periodic"
    frame_dir = flow_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    output_pvd = flow_dir / "dp122_periodic.pvd"
    report_path = output_root / "periodic_flow_report.json"
    existing = list(frame_dir.glob("dp122_periodic_*.vtu"))
    if (existing or output_pvd.exists() or report_path.exists()) and not args.overwrite:
        raise FileExistsError("Periodic outputs exist; pass --overwrite to replace them")

    first_mesh = pv.read(entries[0][1])
    if ACTIVE_KEY not in first_mesh.point_data:
        raise KeyError(f"{ACTIVE_KEY!r} is missing from {entries[0][1]}")
    first = np.ascontiguousarray(first_mesh.point_data[ACTIVE_KEY], dtype=np.float32)
    output_files: list[Path] = []
    corrections: list[dict[str, float | int]] = []
    endpoint_field = None
    penultimate_field = None
    second_field = None

    for index, (time_s, source_file) in enumerate(entries):
        mesh = pv.read(source_file)
        raw = np.ascontiguousarray(mesh.point_data[ACTIVE_KEY], dtype=np.float32)
        if raw.shape != first.shape:
            raise ValueError(f"Velocity shape changed in {source_file}")
        fraction = (time_s - args.blend_start) / (duration - args.blend_start)
        alpha = smoothstep5(fraction) if time_s >= args.blend_start else 0.0
        corrected = np.ascontiguousarray((1.0 - alpha) * raw + alpha * first, dtype=np.float32)
        delta = np.linalg.norm(corrected - raw, axis=1)
        speed = np.linalg.norm(corrected, axis=1)
        mesh.point_data[ACTIVE_KEY] = corrected
        target = frame_dir / f"dp122_periodic_{index:05d}.vtu"
        mesh.save(target, binary=True)
        output_files.append(target)
        corrections.append(
            {
                "index": index,
                "time_s": float(time_s),
                "blend_weight": alpha,
                "correction_rms_m_per_s": float(np.sqrt(np.mean(delta**2))),
                "correction_q99_m_per_s": float(np.quantile(delta, 0.99)),
                "correction_max_m_per_s": float(delta.max(initial=0.0)),
                "corrected_speed_mean_m_per_s": float(speed.mean()),
                "corrected_speed_max_m_per_s": float(speed.max(initial=0.0)),
                "output_bytes": target.stat().st_size,
            }
        )
        if index == 1:
            second_field = corrected.copy()
        if index == len(entries) - 2:
            penultimate_field = corrected.copy()
        if index == len(entries) - 1:
            endpoint_field = corrected.copy()
        print(
            f"Periodic frame {index + 1:02d}/{len(entries)} "
            f"t={time_s:.2f}s alpha={alpha:.4f}",
            flush=True,
        )

    write_pvd(output_pvd, times.tolist(), output_files)
    for name in ("inlet.vtp", "outlet.vtp", "wall.vtp"):
        surface = pv.read(source_root / "flow" / name)
        surface.save(flow_dir / name, binary=True)

    if endpoint_field is None or second_field is None or penultimate_field is None:
        raise RuntimeError("Unable to calculate endpoint diagnostics")
    endpoint_delta = endpoint_field - first
    closure_rmse = float(np.sqrt(np.mean(endpoint_delta**2)))
    dt = float(spacing[0])
    derivative_open = (second_field - first) / dt
    derivative_close = (endpoint_field - penultimate_field) / dt
    derivative_rmse = float(np.sqrt(np.mean((derivative_open - derivative_close) ** 2)))
    max_speed = max(item["corrected_speed_max_m_per_s"] for item in corrections)
    report = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {"python": sys.version, "platform": platform.platform()},
        "source": {
            "pvd": str(source_pvd),
            "pvd_sha256": sha256(source_pvd),
            "conversion_report": str(conversion_report_path),
            "frame_count": len(entries),
            "saved_spacing_s": dt,
            "finer_cfd_samples_available": False,
            "limitation": (
                "Only 10 ms Fluent fields are present. Temporal interpolation used during "
                "tracking is not equivalent to a CFD solution saved at 2 ms or 0.5 ms."
            ),
        },
        "closure_method": {
            "type": "quintic_tail_blend_to_opening_field",
            "blend_start_s": args.blend_start,
            "blend_end_s": float(times[-1]),
            "preserved_source_through_s": args.blend_start,
            "scientific_scope": (
                "Controlled periodic MRI simulation input. The blended tail is synthetic "
                "and must not be reported as an unmodified CFD result."
            ),
        },
        "validation": {
            "endpoint_component_rmse_m_per_s": closure_rmse,
            "endpoint_exactly_equal": bool(np.array_equal(endpoint_field, first)),
            "endpoint_derivative_rmse_m_per_s2": derivative_rmse,
            "maximum_corrected_point_speed_m_per_s": max_speed,
            "passed": bool(np.array_equal(endpoint_field, first)),
        },
        "frames": corrections,
        "output": {
            "pvd": str(output_pvd),
            "pvd_sha256": sha256(output_pvd),
            "frame_bytes": int(sum(path.stat().st_size for path in output_files)),
        },
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Periodic closure: {'PASS' if report['validation']['passed'] else 'FAIL'}")
    print(f"PVD: {output_pvd}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
