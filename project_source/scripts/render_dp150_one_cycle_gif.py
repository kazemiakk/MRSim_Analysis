r"""Render the accepted production dp150 track as a slow one-cycle GIF.

The CFD long axis (z) is horizontal and x is vertical. Only a deterministic
display subset is rendered, while the title and report retain the full tracked
particle count. Trail segments are broken across every boundary-reset window,
and cap overshoot is clipped to the cap plane for presentation.

    uv run python scripts\render_dp150_one_cycle_gif.py --overwrite
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pyvista as pv
from matplotlib import colormaps
from matplotlib.colors import ListedColormap
from PIL import Image, ImageDraw, ImageFont


DEFAULT_ROOT = Path("outputs/dp150_mri_pipeline")
EXPECTED_VERSION = "0.2.1"
BACKGROUND = (255, 255, 255)
TEXT = (23, 37, 84)
TEXT_DIM = (91, 101, 125)
PROGRESS = (37, 99, 235)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--tracks", type=Path, default=None)
    parser.add_argument("--display-count", type=int, default=4_800)
    parser.add_argument("--tracer-count", type=int, default=400)
    parser.add_argument("--frame-stride-steps", type=int, default=10)
    parser.add_argument("--trail-frames", type=int, default=5)
    parser.add_argument("--duration-ms", type=int, default=80)
    parser.add_argument("--window-size", nargs=2, type=int, default=(1920, 640))
    parser.add_argument("--point-size", type=float, default=3.4)
    parser.add_argument("--gif-colors", type=int, default=192)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fonts() -> tuple[ImageFont.FreeTypeFont | ImageFont.ImageFont, ...]:
    regular = Path(r"C:\Windows\Fonts\segoeui.ttf")
    semibold = Path(r"C:\Windows\Fonts\seguisb.ttf")
    try:
        return (
            ImageFont.truetype(str(semibold), 26),
            ImageFont.truetype(str(regular), 22),
            ImageFont.truetype(str(regular), 17),
            ImageFont.truetype(str(semibold), 16),
        )
    except OSError:
        default = ImageFont.load_default()
        return default, default, default, default


def load_display_trajectory(
    path: Path,
    *,
    display_count: int,
    stride: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    with h5py.File(path, "r") as h5:
        position = h5["position"]
        reset = h5["reset"]
        initial = h5["initial_position"]
        n_steps, n_particles, n_xyz = position.shape
        if n_xyz != 3 or reset.shape != (n_steps, n_particles):
            raise ValueError("Unexpected position/reset shape")
        if display_count < 1 or display_count > n_particles:
            raise ValueError(f"display-count must be in [1, {n_particles}]")
        if stride < 1 or n_steps % stride:
            raise ValueError("frame-stride-steps must divide the stored step count")
        if str(h5.attrs.get("mrsimtracks_version", "")) != EXPECTED_VERSION:
            raise RuntimeError("The motion HDF5 is not labeled MRSimTracks 0.2.1")
        if not bool(h5.attrs.get("single_recorded_cycle", False)):
            raise RuntimeError("The input must contain exactly one recorded cycle")
        coordinate_space = str(h5.attrs.get("coordinate_space", ""))
        if coordinate_space != "MRI_centered_largest_to_smallest":
            raise RuntimeError(f"Unsupported coordinate space: {coordinate_space!r}")

        particle_ids = np.sort(rng.choice(n_particles, display_count, replace=False))
        end_indices = np.arange(stride - 1, n_steps, stride, dtype=np.int64)
        positions = np.empty((end_indices.size + 1, display_count, 3), dtype=np.float32)
        breaks = np.zeros((end_indices.size + 1, display_count), dtype=bool)
        times = np.empty(end_indices.size + 1, dtype=np.float64)
        positions[0] = initial[particle_ids]
        times[0] = 0.0
        dt = float(h5.attrs["dt"])
        previous_end = -1
        for frame, end in enumerate(end_indices, start=1):
            positions[frame] = position[end, particle_ids]
            breaks[frame] = np.asarray(
                reset[previous_end + 1 : end + 1, particle_ids], dtype=bool
            ).any(axis=0)
            times[frame] = (end + 1) * dt
            previous_end = int(end)

        metadata = {
            "n_steps": n_steps,
            "n_particles": n_particles,
            "stored_dt_s": dt,
            "cycle_duration_s": float(h5.attrs["cycle_duration_s"]),
            "mrsimtracks_version": str(h5.attrs["mrsimtracks_version"]),
            "method": str(h5.attrs["method"]),
            "time_interpolation": str(h5.attrs["time_interpolation"]),
            "single_recorded_cycle": bool(h5.attrs["single_recorded_cycle"]),
            "periodic_endpoint_closure_passed": bool(
                h5.attrs["periodic_endpoint_closure_passed"]
            ),
            "coordinate_space": coordinate_space,
            "source_axis_order_zero_based": np.asarray(
                h5.attrs["source_axis_order_zero_based"], dtype=int
            ),
            "source_geometry_center_m": np.asarray(
                h5.attrs["source_geometry_center_m"], dtype=float
            ),
        }

    speeds = np.zeros(positions.shape[:2], dtype=np.float32)
    frame_dt = np.diff(times)
    speeds[1:] = np.linalg.norm(np.diff(positions, axis=0), axis=2) / frame_dt[:, None]
    speeds[breaks] = np.nan
    if len(speeds) > 1:
        speeds[0] = speeds[1]
    return positions, breaks, speeds, times, {**metadata, "particle_ids": particle_ids}


def tracer_ids(
    positions: np.ndarray,
    breaks: np.ndarray,
    count: int,
) -> np.ndarray:
    if count < 1:
        return np.empty(0, dtype=np.int64)
    distance = np.linalg.norm(np.diff(positions, axis=0), axis=2)
    distance[breaks[1:]] = 0.0
    total = distance.sum(axis=0)
    order = np.argsort(total)[::-1]
    return np.sort(order[: min(count, positions.shape[1])])


def make_trail_polydata(count: int) -> pv.PolyData:
    points = np.zeros((2 * count, 3), dtype=np.float32)
    lines = np.column_stack(
        (
            np.full(count, 2, dtype=np.int64),
            np.arange(0, 2 * count, 2, dtype=np.int64),
            np.arange(1, 2 * count, 2, dtype=np.int64),
        )
    ).ravel()
    poly = pv.PolyData(points)
    poly.lines = lines
    return poly


def update_trail(
    poly: pv.PolyData,
    positions: np.ndarray,
    breaks: np.ndarray,
    selected: np.ndarray,
    frame: int,
    age: int,
    long_min: float,
    long_max: float,
    offscreen: np.ndarray,
) -> None:
    points = np.repeat(offscreen[None, :], 2 * selected.size, axis=0)
    end = frame - age + 1
    start = end - 1
    if start >= 0:
        valid = ~breaks[end, selected]
        if valid.any():
            start_points = positions[start, selected[valid]].copy()
            end_points = positions[end, selected[valid]].copy()
            start_points[:, 0] = np.clip(start_points[:, 0], long_min, long_max)
            end_points[:, 0] = np.clip(end_points[:, 0], long_min, long_max)
            ids = np.flatnonzero(valid)
            points[2 * ids] = start_points
            points[2 * ids + 1] = end_points
    poly.points = np.ascontiguousarray(points, dtype=np.float32)


def set_camera(plotter: pv.Plotter, wall: pv.PolyData, width: int, height: int) -> None:
    center = np.asarray(wall.center, dtype=float)
    extent_long = float(wall.bounds[1] - wall.bounds[0])
    extent_vertical = float(wall.bounds[3] - wall.bounds[2])
    aspect = width / height
    plotter.camera.position = (center[0], center[1], center[2] + max(extent_long, 1.0))
    plotter.camera.focal_point = tuple(center)
    plotter.camera.up = (0.0, 1.0, 0.0)
    plotter.enable_parallel_projection()
    plotter.camera.parallel_scale = max(
        extent_long / (2.0 * aspect * 0.92),
        extent_vertical / (2.0 * 0.72),
    )
    plotter.reset_camera_clipping_range()


def transform_mesh_to_mri(
    mesh: pv.DataSet,
    axis_order: np.ndarray,
    source_center: np.ndarray,
) -> pv.DataSet:
    transformed = mesh.copy(deep=True)
    transformed.points = np.ascontiguousarray(
        np.asarray(mesh.points)[:, axis_order] - source_center[axis_order],
        dtype=np.float32,
    )
    return transformed


def add_overlay(
    image: Image.Image,
    *,
    time_s: float,
    duration_s: float,
    speed_max: float,
    cmap: ListedColormap,
    progress: float,
    tracked_count: int,
    displayed_count: int,
) -> Image.Image:
    frame = image.convert("RGB")
    draw = ImageDraw.Draw(frame)
    title_font, time_font, small_font, small_bold = fonts()
    width, height = frame.size

    draw.text((28, 20), f"t = {time_s:0.3f} s / {duration_s:0.3f} s", font=time_font, fill=TEXT)
    title = "Particle tracking over one CFD cycle | phantom dp150"
    box = draw.textbbox((0, 0), title, font=title_font)
    draw.text((width - 28 - (box[2] - box[0]), 14), title, font=title_font, fill=TEXT_DIM)
    detail = f"{tracked_count:,} tracked | {displayed_count:,} displayed"
    detail_box = draw.textbbox((0, 0), detail, font=small_font)
    draw.text(
        (width - 28 - (detail_box[2] - detail_box[0]), 51),
        detail,
        font=small_font,
        fill=(120, 128, 145),
    )

    bar_width, bar_height = 390, 14
    bar_x, bar_y = width - bar_width - 55, height - 79
    gradient_values = np.linspace(0.0, 1.0, 256)
    gradient_rgb = np.asarray(cmap(gradient_values)[:, :3] * 255, dtype=np.uint8)
    gradient = np.repeat(gradient_rgb[None, :, :], bar_height, axis=0)
    gradient_image = Image.fromarray(gradient, mode="RGB").resize(
        (bar_width, bar_height), Image.Resampling.BILINEAR
    )
    frame.paste(gradient_image, (bar_x, bar_y))
    draw = ImageDraw.Draw(frame)
    draw.rectangle((bar_x, bar_y, bar_x + bar_width, bar_y + bar_height), outline=(98, 108, 128), width=1)
    label = "Particle speed (m/s)"
    label_box = draw.textbbox((0, 0), label, font=small_bold)
    draw.text(
        (bar_x + (bar_width - (label_box[2] - label_box[0])) // 2, bar_y - 25),
        label,
        font=small_bold,
        fill=TEXT_DIM,
    )
    draw.text((bar_x, bar_y + 18), "0.00", font=small_font, fill=TEXT_DIM)
    max_label = f"{speed_max:.2f}"
    max_box = draw.textbbox((0, 0), max_label, font=small_font)
    draw.text(
        (bar_x + bar_width - (max_box[2] - max_box[0]), bar_y + 18),
        max_label,
        font=small_font,
        fill=TEXT_DIM,
    )

    origin = (72, height - 73)
    draw.line((origin[0], origin[1], origin[0], origin[1] - 31), fill=(190, 52, 45), width=3)
    draw.polygon(
        [(origin[0], origin[1] - 37), (origin[0] - 5, origin[1] - 28), (origin[0] + 5, origin[1] - 28)],
        fill=(190, 52, 45),
    )
    draw.line((origin[0], origin[1], origin[0] + 36, origin[1]), fill=(37, 99, 235), width=3)
    draw.polygon(
        [(origin[0] + 42, origin[1]), (origin[0] + 33, origin[1] - 5), (origin[0] + 33, origin[1] + 5)],
        fill=(37, 99, 235),
    )
    draw.text((origin[0] - 4, origin[1] - 57), "x", font=small_bold, fill=(190, 52, 45))
    draw.text((origin[0] + 47, origin[1] - 10), "z", font=small_bold, fill=(37, 99, 235))

    track_y = height - 12
    draw.line((28, track_y, width - 28, track_y), fill=(222, 227, 234), width=4)
    draw.line(
        (28, track_y, 28 + int((width - 56) * progress), track_y),
        fill=PROGRESS,
        width=4,
    )
    return frame


def save_gif(path: Path, frames: list[Image.Image], duration_ms: int, colors: int) -> None:
    if not frames:
        raise ValueError("No animation frames were rendered")
    palette = frames[0].quantize(
        colors=colors,
        method=Image.Quantize.MEDIANCUT,
        dither=Image.Dither.NONE,
    )
    quantized = [
        frame.quantize(palette=palette, dither=Image.Dither.FLOYDSTEINBERG)
        for frame in frames
    ]
    durations = [duration_ms] * len(quantized)
    durations[-1] = max(800, 5 * duration_ms)
    quantized[0].save(
        path,
        save_all=True,
        append_images=quantized[1:],
        duration=durations,
        disposal=2,
        optimize=True,
        comment=b"One controlled dp150 cycle; MRSimTracks 0.2.1 production trajectory",
    )


def contact_sheet(frames: list[Image.Image], indices: list[int]) -> Image.Image:
    selected = [frames[index] for index in indices]
    thumb_width = 960
    thumbs = [
        frame.resize((thumb_width, round(frame.height * thumb_width / frame.width)), Image.Resampling.LANCZOS)
        for frame in selected
    ]
    sheet = Image.new("RGB", (thumb_width, sum(frame.height for frame in thumbs)), "white")
    y = 0
    for frame in thumbs:
        sheet.paste(frame, (0, y))
        y += frame.height
    return sheet


def main() -> None:
    args = parse_args()
    if args.duration_ms < 20 or args.gif_colors < 32 or args.gif_colors > 256:
        raise ValueError("duration-ms must be >= 20 and gif-colors must be in [32, 256]")
    if args.trail_frames < 1 or args.tracer_count < 0:
        raise ValueError("trail-frames must be >= 1 and tracer-count must be nonnegative")
    width, height = args.window_size
    if width < 800 or height < 300:
        raise ValueError("window-size is too small for the scientific overlay")

    root = args.root.resolve()
    tracks_path = (args.tracks or (root / "dp150_motion_1m_mri.h5")).resolve()
    wall_path = root / "flow_periodic" / "wall.vtp"
    inlet_path = root / "flow_periodic" / "inlet.vtp"
    outlet_path = root / "flow_periodic" / "outlet.vtp"
    tracking_report_path = root / "motion_1m_report.json"
    output_path = root / "dp150_motion_1m_one_cycle.gif"
    preview_path = root / "dp150_motion_1m_preview.png"
    sheet_path = root / "dp150_motion_1m_contact_sheet.png"
    report_path = root / "motion_1m_animation_report.json"
    required = [tracks_path, wall_path, inlet_path, outlet_path, tracking_report_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required inputs: {missing}")
    for path in (output_path, preview_path, sheet_path, report_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite to replace outputs")

    tracking_report = json.loads(tracking_report_path.read_text(encoding="utf-8"))
    if not tracking_report["quality_control"]["acceptance"]["passed"]:
        raise RuntimeError("The tracking report did not pass acceptance checks")
    expected_h5_hash = tracking_report["outputs"]["motion_hdf5_sha256"]
    actual_h5_hash = sha256(tracks_path)
    if expected_h5_hash != actual_h5_hash:
        raise RuntimeError("Pilot HDF5 hash does not match the accepted tracking report")

    rng = np.random.default_rng(args.seed)
    positions, breaks, speeds, times, metadata = load_display_trajectory(
        tracks_path,
        display_count=args.display_count,
        stride=args.frame_stride_steps,
        rng=rng,
    )
    selected_tracers = tracer_ids(positions, breaks, args.tracer_count)
    finite_speed = speeds[np.isfinite(speeds)]
    speed_max = float(np.quantile(finite_speed, 0.99)) if finite_speed.size else 1.0
    cmap = ListedColormap(colormaps["turbo"](np.linspace(0.04, 0.92, 256)))

    axis_order = metadata["source_axis_order_zero_based"]
    source_center = metadata["source_geometry_center_m"]
    wall = transform_mesh_to_mri(
        pv.read(wall_path)
        .extract_surface(algorithm="dataset_surface")
        .triangulate()
        .clean(),
        axis_order,
        source_center,
    )
    inlet = transform_mesh_to_mri(
        pv.read(inlet_path)
        .extract_surface(algorithm="dataset_surface")
        .triangulate()
        .clean(),
        axis_order,
        source_center,
    )
    outlet = transform_mesh_to_mri(
        pv.read(outlet_path)
        .extract_surface(algorithm="dataset_surface")
        .triangulate()
        .clean(),
        axis_order,
        source_center,
    )
    caps = inlet.merge(outlet)
    long_min = min(float(inlet.bounds[0]), float(outlet.bounds[0]))
    long_max = max(float(inlet.bounds[1]), float(outlet.bounds[1]))
    render_positions = positions.copy()
    render_positions[:, :, 0] = np.clip(
        render_positions[:, :, 0], long_min, long_max
    )

    plotter = pv.Plotter(off_screen=True, window_size=(width, height))
    plotter.set_background("white")
    plotter.enable_anti_aliasing("ssaa")
    plotter.add_mesh(
        wall,
        color="#B9C4D0",
        opacity=0.20,
        smooth_shading=True,
        show_scalar_bar=False,
        ambient=0.55,
        diffuse=0.35,
        specular=0.22,
        specular_power=20,
    )
    plotter.add_mesh(
        caps,
        color="#CBD5E1",
        opacity=0.13,
        smooth_shading=True,
        show_scalar_bar=False,
        ambient=0.7,
    )
    plotter.add_silhouette(
        wall.merge(caps), color="#697586", line_width=1.15, opacity=0.72
    )

    offscreen = np.asarray(wall.center, dtype=np.float32)
    offscreen[2] = float(wall.bounds[5]) + 10.0 * (
        float(wall.bounds[5]) - float(wall.bounds[4])
    )
    trails: list[pv.PolyData] = []
    for age in range(1, args.trail_frames + 1):
        poly = make_trail_polydata(selected_tracers.size)
        trails.append(poly)
        opacity = 0.07 + 0.27 * (args.trail_frames - age + 1) / args.trail_frames
        plotter.add_mesh(
            poly,
            color="#2563EB",
            opacity=opacity,
            line_width=1.25,
            render_lines_as_tubes=True,
            show_scalar_bar=False,
        )

    particle_cloud = pv.PolyData(render_positions[0])
    particle_cloud["speed"] = np.nan_to_num(speeds[0], nan=0.0, posinf=speed_max)
    plotter.add_mesh(
        particle_cloud,
        scalars="speed",
        cmap=cmap,
        clim=(0.0, speed_max),
        point_size=args.point_size * 1.9,
        render_points_as_spheres=True,
        opacity=0.12,
        show_scalar_bar=False,
    )
    plotter.add_mesh(
        particle_cloud,
        scalars="speed",
        cmap=cmap,
        clim=(0.0, speed_max),
        point_size=args.point_size,
        render_points_as_spheres=True,
        opacity=0.96,
        show_scalar_bar=False,
    )
    set_camera(plotter, wall, width, height)

    print(
        f"Rendering {len(times)} frames from {metadata['n_particles']:,} tracked "
        f"particles ({args.display_count:,} displayed) ...",
        flush=True,
    )
    frames: list[Image.Image] = []
    render_start = time.perf_counter()
    for frame_index, time_s in enumerate(times):
        for age, poly in enumerate(trails, start=1):
            update_trail(
                poly,
                render_positions,
                breaks,
                selected_tracers,
                frame_index,
                age,
                long_min,
                long_max,
                offscreen,
            )
        particle_cloud.points = np.ascontiguousarray(render_positions[frame_index])
        particle_cloud["speed"] = np.nan_to_num(
            speeds[frame_index], nan=0.0, posinf=speed_max, neginf=0.0
        )
        plotter.render()
        screenshot = Image.fromarray(plotter.screenshot(return_img=True)).convert("RGB")
        frame = add_overlay(
            screenshot,
            time_s=float(time_s),
            duration_s=float(metadata["cycle_duration_s"]),
            speed_max=speed_max,
            cmap=cmap,
            progress=frame_index / (len(times) - 1),
            tracked_count=int(metadata["n_particles"]),
            displayed_count=args.display_count,
        )
        frames.append(frame)
        if frame_index == 0 or (frame_index + 1) % 24 == 0 or frame_index == len(times) - 1:
            print(f"  frame {frame_index + 1}/{len(times)}", flush=True)
    plotter.close()
    render_seconds = time.perf_counter() - render_start

    peak_time = 0.18
    preview_index = int(np.argmin(np.abs(times - peak_time)))
    frames[preview_index].save(preview_path, quality=95)
    sheet_indices = [0, preview_index, len(frames) // 2, len(frames) - 1]
    contact_sheet(frames, sheet_indices).save(sheet_path, quality=94)
    print("Encoding GIF ...", flush=True)
    encode_start = time.perf_counter()
    save_gif(output_path, frames, args.duration_ms, args.gif_colors)
    encode_seconds = time.perf_counter() - encode_start

    with Image.open(output_path) as gif:
        verified_frames = int(getattr(gif, "n_frames", 1))
        verified_size = list(gif.size)
        loop_value = gif.info.get("loop")
        gif.seek(0)
        first = np.asarray(gif.convert("RGB"), dtype=np.int16)
        gif.seek(verified_frames - 1)
        last = np.asarray(gif.convert("RGB"), dtype=np.int16)
    first_last_mean_absolute_difference = float(np.abs(last - first).mean())
    nonwhite_fraction = float(np.any(first < 248, axis=2).mean())
    verification = {
        "frame_count": verified_frames,
        "expected_frame_count": len(frames),
        "dimensions_px": verified_size,
        "animated": verified_frames > 1,
        "loop_metadata": loop_value,
        "non_looping_one_cycle": loop_value is None,
        "first_last_mean_absolute_pixel_difference": first_last_mean_absolute_difference,
        "first_frame_nonwhite_fraction": nonwhite_fraction,
        "passed": (
            verified_frames == len(frames)
            and verified_size == [width, height]
            and loop_value is None
            and first_last_mean_absolute_difference > 0.1
            and nonwhite_fraction > 0.005
        ),
    }
    report = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "render_seconds": render_seconds,
            "encode_seconds": encode_seconds,
        },
        "input": {
            "hdf5": str(tracks_path),
            "hdf5_sha256": actual_h5_hash,
            "tracking_report": str(tracking_report_path),
            "tracking_report_sha256": sha256(tracking_report_path),
            "wall": str(wall_path),
        },
        "render": {
            "orientation": {"horizontal": "source z", "vertical": "source x", "view_direction": "-source y"},
            "tracked_particles": int(metadata["n_particles"]),
            "displayed_particles": args.display_count,
            "displayed_tracers": int(selected_tracers.size),
            "frame_stride_steps": args.frame_stride_steps,
            "rendered_frame_interval_s": args.frame_stride_steps * metadata["stored_dt_s"],
            "frame_duration_ms": args.duration_ms,
            "cycle_playback_seconds": (len(frames) - 1) * args.duration_ms / 1000.0,
            "final_hold_ms": max(800, 5 * args.duration_ms),
            "speed_clim_m_per_s": [0.0, speed_max],
            "reset_aware_trails": True,
            "cap_overshoot_clipped_for_presentation": True,
            "single_recorded_cycle": metadata["single_recorded_cycle"],
            "periodic_endpoint_closure_passed": metadata[
                "periodic_endpoint_closure_passed"
            ],
        },
        "verification": verification,
        "outputs": {
            "gif": str(output_path),
            "gif_bytes": output_path.stat().st_size,
            "gif_sha256": sha256(output_path),
            "preview": str(preview_path),
            "contact_sheet": str(sheet_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not verification["passed"]:
        raise RuntimeError(f"GIF verification failed: {verification}")
    print(f"Saved GIF: {output_path}", flush=True)
    print(f"Saved preview: {preview_path}", flush=True)
    print(f"Saved report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
