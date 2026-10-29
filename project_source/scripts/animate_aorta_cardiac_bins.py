r"""Animate discrete aorta cardiac bins from Koma-like HDF5 volumes.

The default input is a tracking-derived preview, not a production KomaMRI
reconstruction. Pass ``--input`` explicitly for reconstructed production data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import label as connected_components


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT
    / "outputs"
    / "aorta_dp2_mri_pipeline"
    / "koma"
    / "multibin_8x4mm"
    / "tracking_preview"
    / "aorta_tracking_8bin_preview.h5"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=(
            "Input HDF5 volume. The default is a tracking-derived preview; "
            "pass an explicit path for KomaMRI production data."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--label", default="aorta_8bin_cardiac_cine")
    parser.add_argument(
        "--display-mask-fraction",
        type=float,
        default=0.15,
        help=(
            "Per-bin magnitude threshold. When a stored mask exists, the "
            "display mask is its intersection with this threshold."
        ),
    )
    parser.add_argument("--frame-duration-ms", type=int, default=500)
    parser.add_argument(
        "--mp4-cycles",
        type=int,
        default=2,
        help="Number of complete discrete-bin cycles included in the MP4.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def largest_component(mask: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    regions, count = connected_components(
        mask, structure=np.ones((3, 3, 3), dtype=np.uint8)
    )
    sizes = np.bincount(regions.ravel())[1:]
    if not sizes.size:
        raise RuntimeError("Cardiac-bin display mask is empty")
    largest = int(np.argmax(sizes)) + 1
    result = regions == largest
    return result, {
        "raw_component_count": int(count),
        "raw_mask_voxels": int(mask.sum()),
        "display_mask_voxels": int(result.sum()),
        "retained_mask_fraction": float(result.sum() / mask.sum()),
    }


def weighted_projection(
    values: np.ndarray, magnitude: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    weights = np.where(mask, magnitude, 0.0)
    denominator = weights.sum(axis=0)
    return np.divide(
        (values * weights).sum(axis=0),
        denominator,
        out=np.zeros_like(denominator, dtype=np.float32),
        where=denominator > 0,
    )


def masked_colormap(name: str):
    colormap = plt.get_cmap(name).copy()
    colormap.set_bad("white")
    return colormap


def render_frame(
    cardiac_bin: int,
    magnitude_projection: np.ndarray,
    speed_projection: np.ndarray,
    velocity_projection: np.ndarray,
    support_projection: np.ndarray,
    extent: list[float],
    limits: dict[str, float],
    phase_time_s: float,
    sampled_source_time_s: float | None,
    data_kind: str,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(
        1, 3, figsize=(16, 5.8), constrained_layout=True, facecolor="white"
    )
    masked_magnitude = np.ma.masked_where(
        ~support_projection, magnitude_projection
    )
    masked_speed = np.ma.masked_where(~support_projection, speed_projection)
    masked_velocity = np.ma.masked_where(
        ~support_projection, velocity_projection
    )
    images = [
        axes[0].imshow(
            masked_magnitude,
            cmap=masked_colormap("Greys"),
            origin="lower",
            extent=extent,
            vmin=0,
            vmax=limits["magnitude"],
            interpolation="nearest",
            aspect="equal",
        ),
        axes[1].imshow(
            masked_speed,
            cmap=masked_colormap("turbo"),
            origin="lower",
            extent=extent,
            vmin=0,
            vmax=limits["speed"],
            interpolation="nearest",
            aspect="equal",
        ),
        axes[2].imshow(
            masked_velocity,
            cmap=masked_colormap("RdBu_r"),
            origin="lower",
            extent=extent,
            vmin=-limits["velocity"],
            vmax=limits["velocity"],
            interpolation="nearest",
            aspect="equal",
        ),
    ]
    titles = (
        (
            "Particle density"
            if "tracking_derived" in data_kind
            else "Reference magnitude"
        ),
        "Mean speed",
        "MRI-readout velocity",
    )
    colorbar_labels = ("a.u.", "m/s", "m/s")
    x = np.linspace(extent[0], extent[1], support_projection.shape[1])
    y = np.linspace(extent[2], extent[3], support_projection.shape[0])
    for axis, image, title, colorbar_label in zip(
        axes, images, titles, colorbar_labels, strict=True
    ):
        axis.contour(
            x,
            y,
            support_projection.astype(float),
            levels=[0.5],
            colors=["#64748b"],
            linewidths=0.55,
        )
        axis.set_title(title, fontsize=12, color="#172033")
        axis.set_xlabel("MRI readout / source Z (mm)")
        axis.set_ylabel("MRI phase / source X (mm)")
        axis.tick_params(labelsize=9, colors="#334155")
        axis.grid(False)
        figure.colorbar(
            image, ax=axis, shrink=0.72, pad=0.025, label=colorbar_label
        )

    source_note = (
        f" | source sample {sampled_source_time_s:.3f} s"
        if sampled_source_time_s is not None
        else ""
    )
    bin_count = limits["bin_count"]
    figure.suptitle(
        f"Aorta cardiac cine | bin {cardiac_bin + 1}/{bin_count} | "
        f"phase center {phase_time_s:.3f} s{source_note}",
        fontsize=16,
        color="#172033",
    )
    scope = (
        "Tracking-derived preview - not KomaMRI"
        if "tracking_derived" in data_kind
        else "KomaMRI reconstruction"
    )
    figure.text(
        0.5,
        0.008,
        f"{scope} | discrete cardiac bins; no temporal interpolation",
        ha="center",
        fontsize=9,
        color="#475569",
    )
    figure.savefig(output_path, dpi=120, facecolor="white")
    plt.close(figure)


def save_gif(frame_paths: list[Path], output: Path, duration_ms: int) -> None:
    frames = []
    for path in frame_paths:
        with Image.open(path) as image:
            frames.append(image.convert("RGB"))
    paletted = [
        frame.convert("P", palette=Image.Palette.ADAPTIVE, colors=192)
        for frame in frames
    ]
    paletted[0].save(
        output,
        save_all=True,
        append_images=paletted[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )


def save_contact_sheet(frame_paths: list[Path], output: Path) -> None:
    frames = []
    for path in frame_paths:
        with Image.open(path) as image:
            frame = image.convert("RGB")
            width = 960
            frames.append(
                frame.resize(
                    (width, round(frame.height * width / frame.width)),
                    Image.Resampling.LANCZOS,
                )
            )
    columns = 2
    rows = int(np.ceil(len(frames) / columns))
    height = max(frame.height for frame in frames)
    sheet = Image.new("RGB", (columns * 960, rows * height), "white")
    for index, frame in enumerate(frames):
        sheet.paste(frame, ((index % columns) * 960, (index // columns) * height))
    sheet.save(output)


def save_mp4(
    frame_paths: list[Path],
    output: Path,
    duration_ms: int,
    cycles: int,
) -> dict[str, object]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return {"created": False, "reason": "ffmpeg was not found"}
    concat_path = output.with_suffix(".frames.txt")
    lines = []
    duration_s = duration_ms / 1000
    for _ in range(cycles):
        for frame in frame_paths:
            normalized = frame.resolve().as_posix().replace("'", "'\\''")
            lines.extend((f"file '{normalized}'", f"duration {duration_s:.6f}"))
    normalized = frame_paths[-1].resolve().as_posix().replace("'", "'\\''")
    lines.append(f"file '{normalized}'")
    concat_path.write_text("\n".join(lines) + "\n", encoding="ascii")
    expected_duration_s = len(frame_paths) * duration_s * cycles
    command = [
        ffmpeg,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_path),
        "-t",
        f"{expected_duration_s:.6f}",
        "-vf",
        "fps=24,format=yuv420p",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-movflags",
        "+faststart",
        str(output),
    ]
    completed = subprocess.run(
        command, check=True, capture_output=True, text=True
    )
    return {
        "created": True,
        "cycles": cycles,
        "duration_s": expected_duration_s,
        "ffmpeg_command": command,
        "ffmpeg_stderr_tail": completed.stderr.splitlines()[-4:],
        "path": str(output),
        "bytes": output.stat().st_size,
        "sha256": sha256_file(output),
        **probe_video(output),
    }


def probe_video(path: Path) -> dict[str, object]:
    """Return decoder-backed MP4 metadata when ffprobe is available."""

    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return {"probe_available": False}
    completed = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            (
                "stream=width,height,avg_frame_rate,nb_frames,duration:"
                "format=duration,size"
            ),
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    stream = payload["streams"][0]
    container = payload.get("format", {})
    frame_count = stream.get("nb_frames")
    duration = container.get("duration", stream.get("duration"))
    return {
        "probe_available": True,
        "dimensions_px": [int(stream["width"]), int(stream["height"])],
        "average_frame_rate": stream.get("avg_frame_rate"),
        "frames": (
            int(frame_count) if frame_count not in (None, "N/A") else None
        ),
        "probed_duration_s": float(duration),
        "probed_bytes": int(container.get("size", path.stat().st_size)),
    }


def main() -> None:
    args = parse_args()
    if not 0 < args.display_mask_fraction <= 1:
        raise ValueError("--display-mask-fraction must be in (0, 1]")
    if args.frame_duration_ms < 20 or args.mp4_cycles < 1:
        raise ValueError("Frame duration and MP4 cycles must be positive")
    output_dir = (args.output_dir or args.input.parent / "cine").resolve()
    frame_dir = output_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.input, "r") as source:
        magnitude = np.asarray(source["reference_magnitude"], dtype=np.float32)
        velocity = np.asarray(source["velocity"], dtype=np.float32)
        speed = np.asarray(source["speed"], dtype=np.float32)
        stored_mask = (
            np.asarray(source["mask"], dtype=bool) if "mask" in source else None
        )
        sampled_source_times = (
            np.asarray(source["sampled_source_time_s"], dtype=float)
            if "sampled_source_time_s" in source
            else None
        )
        fov_m = np.asarray(source.attrs["FOV_m"], dtype=float)
        matrix = np.asarray(source.attrs["matrix"], dtype=int)
        rr_s = float(source.attrs["RR_s"])
        data_kind = str(source.attrs.get("data_kind", "koma_mri_reconstruction"))

    if magnitude.ndim != 4 or velocity.shape != (3, *magnitude.shape):
        raise ValueError("Expected magnitude (bin,z,y,x) and velocity (3,bin,z,y,x)")
    if speed.shape != magnitude.shape:
        raise ValueError("Speed must match magnitude")
    bin_count = magnitude.shape[0]
    if tuple(magnitude.shape[1:]) != tuple(matrix[::-1]):
        raise ValueError("HDF5 volume shape and matrix attribute disagree")

    masks = np.zeros_like(magnitude, dtype=bool)
    mask_reports = []
    for cardiac_bin in range(bin_count):
        magnitude_threshold = (
            args.display_mask_fraction * float(magnitude[cardiac_bin].max())
        )
        candidate = magnitude[cardiac_bin] >= magnitude_threshold
        if stored_mask is not None:
            candidate &= stored_mask[cardiac_bin]
        masks[cardiac_bin], report = largest_component(candidate)
        report["cardiac_bin"] = cardiac_bin + 1
        report["magnitude_threshold"] = magnitude_threshold
        mask_reports.append(report)

    weights = np.where(masks, magnitude, 0.0)
    magnitude_projections = weights.max(axis=1)
    speed_projections = np.stack(
        [
            weighted_projection(speed[index], magnitude[index], masks[index])
            for index in range(bin_count)
        ]
    )
    velocity_projections = np.stack(
        [
            weighted_projection(
                velocity[0, index], magnitude[index], masks[index]
            )
            for index in range(bin_count)
        ]
    )
    support_projections = masks.any(axis=1)
    limits = {
        "magnitude": float(
            np.quantile(magnitude_projections[magnitude_projections > 0], 0.995)
        ),
        "speed": float(np.quantile(speed[masks], 0.99)),
        "velocity": float(np.quantile(np.abs(velocity[:, masks]), 0.99)),
        "bin_count": bin_count,
    }
    limits["speed"] = max(limits["speed"], 1e-6)
    limits["velocity"] = max(limits["velocity"], 1e-6)
    extent = [
        -fov_m[0] * 500,
        fov_m[0] * 500,
        -fov_m[1] * 500,
        fov_m[1] * 500,
    ]

    frame_paths = []
    phase_times = []
    for cardiac_bin in range(bin_count):
        phase_time = (cardiac_bin + 0.5) * rr_s / bin_count
        phase_times.append(phase_time)
        frame_path = frame_dir / f"frame_{cardiac_bin + 1:03d}.png"
        render_frame(
            cardiac_bin,
            magnitude_projections[cardiac_bin],
            speed_projections[cardiac_bin],
            velocity_projections[cardiac_bin],
            support_projections[cardiac_bin],
            extent,
            limits,
            phase_time,
            (
                float(sampled_source_times[cardiac_bin])
                if sampled_source_times is not None
                else None
            ),
            data_kind,
            frame_path,
        )
        frame_paths.append(frame_path)

    gif_path = output_dir / f"{args.label}.gif"
    mp4_path = output_dir / f"{args.label}.mp4"
    contact_sheet_path = output_dir / f"{args.label}_contact_sheet.png"
    save_gif(frame_paths, gif_path, args.frame_duration_ms)
    save_contact_sheet(frame_paths, contact_sheet_path)
    mp4_report = save_mp4(
        frame_paths,
        mp4_path,
        args.frame_duration_ms,
        args.mp4_cycles,
    )

    with Image.open(gif_path) as gif:
        gif_metadata = {
            "path": str(gif_path),
            "dimensions_px": list(gif.size),
            "frames": int(gif.n_frames),
            "duration_ms": int(gif.info["duration"]),
            "loop": int(gif.info["loop"]),
            "bytes": gif_path.stat().st_size,
            "sha256": sha256_file(gif_path),
        }
    report = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_DISCRETE_CARDIAC_BIN_CINE",
        "input": str(args.input.resolve()),
        "input_sha256": sha256_file(args.input),
        "data_kind": data_kind,
        "scientific_scope": (
            "Discrete tracking-derived cardiac-bin visualization; not a "
            "KomaMRI simulation and no temporal frames are interpolated."
            if "tracking_derived" in data_kind
            else "Discrete KomaMRI reconstructed cardiac-bin visualization; "
            "no temporal frames are interpolated and this is not independent "
            "PC-MRI accuracy validation."
        ),
        "cardiac_bins": bin_count,
        "RR_s": rr_s,
        "phase_center_times_s": phase_times,
        "sampled_source_times_s": (
            sampled_source_times.tolist()
            if sampled_source_times is not None
            else None
        ),
        "FOV_m": fov_m.tolist(),
        "matrix": matrix.tolist(),
        "limits": limits,
        "mask_source": (
            "stored mask intersected with per-bin magnitude threshold, then "
            "largest connected component"
            if stored_mask is not None
            else "per-bin magnitude threshold, then largest connected component"
        ),
        "display_mask_fraction_of_phase_max": args.display_mask_fraction,
        "mask_reports": mask_reports,
        "temporal_interpolation": False,
        "gif": gif_metadata,
        "mp4": mp4_report,
        "contact_sheet": {
            "path": str(contact_sheet_path),
            "bytes": contact_sheet_path.stat().st_size,
            "sha256": sha256_file(contact_sheet_path),
        },
        "frame_paths": [str(path) for path in frame_paths],
    }
    report_path = output_dir / f"{args.label}_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"{report['status']}\nGIF: {gif_path}\nMP4: {mp4_path}\nReport: {report_path}"
    )


if __name__ == "__main__":
    main()
