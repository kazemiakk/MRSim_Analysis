r"""Render MRSimTracks phantom particle tracks from PI-GNN processed geometry.

Example:
    uv run python scripts\render_phantom_tracks.py
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np
import pyvista as pv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from visualize_tracks import (
    capture_frame,
    crop_white_image,
    crop_white_frames,
    require_pillow,
    save_animation,
)


DEFAULT_WORKSPACE = Path(
    os.environ.get("MRSIM_WORKSPACE_ROOT", ROOT.parent)
).expanduser()
DEFAULT_ANALYSIS_ROOT = Path(
    os.environ.get("MRSIM_ANALYSIS_ROOT", DEFAULT_WORKSPACE / "MRSim_Analysis")
).expanduser()
DEFAULT_PIGNN_ROOT = Path(
    os.environ.get("MRSIM_PIGNN_ROOT", DEFAULT_WORKSPACE / "PI-GNN(SR)")
).expanduser()
DEFAULT_DP = "dp150"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dp-name", default=DEFAULT_DP)
    parser.add_argument(
        "--tracks",
        type=Path,
        default=None,
        help="MRSimTracks HDF5 file. Defaults to MRSim_Analysis phantom tracks.",
    )
    parser.add_argument(
        "--geometry",
        type=Path,
        default=None,
        help="PI-GNN phantom geometry.pkl. Defaults to processed phantom geometry.",
    )
    parser.add_argument(
        "--out-webp",
        type=Path,
        default=Path("outputs/phantom_dp150_mrsimtracks.webp"),
    )
    parser.add_argument(
        "--out-preview",
        type=Path,
        default=Path("outputs/phantom_dp150_mrsimtracks_preview.png"),
    )
    parser.add_argument(
        "--style",
        choices=("aorta", "paper"),
        default="aorta",
        help="Aorta style matches aorta_dp2_full_cycle_motion.gif; paper style uses white background.",
    )
    parser.add_argument(
        "--geometry-style",
        choices=("points", "surface"),
        default="points",
        help="Point-cloud geometry is faster and matches the aorta full-cycle GIF.",
    )
    parser.add_argument("--frame-stride", type=int, default=40)
    parser.add_argument("--particle-stride", type=int, default=1)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--animation-scale", type=float, default=0.55)
    parser.add_argument("--webp-quality", type=int, default=80)
    parser.add_argument("--max-geometry-points", type=int, default=500_000)
    parser.add_argument("--surface-alpha", type=float, default=0.005)
    parser.add_argument("--hide-paths", action="store_true")
    parser.add_argument(
        "--trail-frames",
        type=int,
        default=0,
        help="Draw short recent particle trails over this many rendered frames.",
    )
    parser.add_argument("--trail-width", type=float, default=2.0)
    parser.add_argument("--trail-opacity", type=float, default=0.50)
    parser.add_argument(
        "--glyph-particles",
        action="store_true",
        help="Render particle meshes as true glyph spheres. Slower than the default point-sphere actor.",
    )
    parser.add_argument("--point-size", type=float, default=7.0)
    parser.add_argument("--camera-shift-z", type=float, default=-0.127)
    parser.add_argument("--camera-zoom", type=float, default=2.5)
    parser.add_argument("--crop-pad", type=int, default=18)
    parser.add_argument("--no-crop", action="store_true")
    return parser.parse_args()


def default_tracks_path(dp_name: str) -> Path:
    return DEFAULT_ANALYSIS_ROOT / f"phantom_{dp_name}_tracks_10s.h5"


def default_geometry_path(dp_name: str) -> Path:
    return (
        DEFAULT_PIGNN_ROOT
        / "data"
        / "phantom"
        / "processed"
        / dp_name
        / "geometry.pkl"
    )


def load_tracks(
    path: Path,
    *,
    frame_stride: int,
    particle_stride: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    if frame_stride < 1:
        raise ValueError("frame_stride must be >= 1")
    if particle_stride < 1:
        raise ValueError("particle_stride must be >= 1")

    with h5py.File(path, "r") as h5:
        position_data = h5["position"]
        reset_data = h5["reset"]
        n_steps = int(position_data.shape[0])
        n_windows = n_steps // frame_stride
        if n_windows < 1:
            raise ValueError("frame_stride exceeds the stored timestep count")

        stop = n_windows * frame_stride
        time_indices = np.arange(frame_stride - 1, stop, frame_stride, dtype=np.int64)
        positions = position_data[time_indices, ::particle_stride, :]
        reset_window = reset_data[:stop, ::particle_stride]
        reset = reset_window.reshape(
            n_windows, frame_stride, reset_window.shape[1]
        ).any(axis=1)
        dt = float(h5.attrs["dt"]) * frame_stride

    return np.asarray(positions), np.asarray(reset), dt


def speed_from_positions(positions: np.ndarray, reset: np.ndarray, dt: float) -> np.ndarray:
    speed = np.zeros(positions.shape[:2], dtype=float)
    if len(positions) > 1:
        speed[1:] = np.linalg.norm(np.diff(positions, axis=0), axis=2) / dt
        speed[reset] = np.nan
    return speed


def build_track_segments(positions: np.ndarray, reset: np.ndarray) -> pv.PolyData | None:
    n_steps, n_particles, _ = positions.shape
    if n_steps < 2:
        return None

    points = positions.reshape(-1, 3)
    lines: list[int] = []
    for step in range(1, n_steps):
        previous_offset = (step - 1) * n_particles
        current_offset = step * n_particles
        for particle in np.flatnonzero(~reset[step]):
            lines.extend([2, previous_offset + int(particle), current_offset + int(particle)])

    if not lines:
        return None

    tracks = pv.PolyData(points)
    tracks.lines = np.asarray(lines, dtype=np.int64)
    return tracks


def build_recent_trails(
    positions: np.ndarray,
    reset: np.ndarray,
    step: int,
    *,
    trail_frames: int,
) -> pv.PolyData | None:
    if trail_frames < 2 or step < 1:
        return None

    start = max(0, step - trail_frames + 1)
    window = positions[start : step + 1]
    window_reset = reset[start : step + 1]
    n_steps, n_particles, _ = window.shape
    if n_steps < 2:
        return None

    points = window.reshape(-1, 3)
    lines: list[int] = []
    for local_step in range(1, n_steps):
        previous_offset = (local_step - 1) * n_particles
        current_offset = local_step * n_particles
        for particle in np.flatnonzero(~window_reset[local_step]):
            lines.extend([2, previous_offset + int(particle), current_offset + int(particle)])

    if not lines:
        return None

    trails = pv.PolyData(points)
    trails.lines = np.asarray(lines, dtype=np.int64)
    return trails


def load_geometry(
    path: Path,
    *,
    max_points: int,
    alpha: float,
    style: str,
) -> pv.PolyData:
    with path.open("rb") as fh:
        geometry = pickle.load(fh)

    coords = np.asarray(geometry["mesh_coords"])
    stride = max(1, int(np.ceil(len(coords) / max_points)))
    point_cloud = pv.PolyData(coords[::stride])
    if style == "points":
        print(
            f"Using phantom point cloud with {len(point_cloud.points):,} points "
            f"(1/{stride} of geometry).",
            flush=True,
        )
        return point_cloud

    print(
        f"Reconstructing phantom surface from {len(point_cloud.points):,} points...",
        flush=True,
    )
    return point_cloud.delaunay_3d(alpha=alpha).extract_surface(algorithm=None)


def configure_camera(
    plotter: pv.Plotter,
    *,
    shift_z: float,
    zoom: float,
) -> None:
    plotter.view_zx()
    plotter.enable_parallel_projection()
    plotter.reset_camera()

    fp = plotter.camera.focal_point
    pos = plotter.camera.position
    plotter.camera.focal_point = (fp[0], fp[1], fp[2] + shift_z)
    plotter.camera.position = (pos[0], pos[1], pos[2] + shift_z)
    plotter.camera.zoom(zoom)


def add_scene(
    plotter: pv.Plotter,
    *,
    geometry: pv.PolyData,
    geometry_style: str,
    style: str,
    tracks: pv.PolyData | None,
    particle_mesh: pv.PolyData,
    speed_max: float,
    glyph_particles: bool,
    point_size: float,
) -> None:
    dark = style == "aorta"
    if dark:
        plotter.set_background("#10151c", top="#263342")
    else:
        plotter.set_background("white")
    plotter.enable_anti_aliasing("ssaa")
    if geometry_style == "points":
        plotter.add_mesh(
            geometry,
            color="#d6dde5" if dark else "#9ca3af",
            opacity=0.13 if dark else 0.18,
            point_size=2,
            render_points_as_spheres=False,
            label="Phantom point cloud",
        )
    else:
        plotter.add_mesh(
            geometry,
            color="#d6dde5" if dark else "silver",
            opacity=0.18 if dark else 0.2,
            specular=0.5,
            show_edges=False,
            label="Phantom wall",
        )
    if tracks is not None:
        plotter.add_mesh(tracks, color="#1f77b4", opacity=0.75, line_width=2)
    plotter.add_mesh(
        particle_mesh,
        scalars="speed",
        cmap="turbo",
        clim=(0.0, speed_max),
        name="particles",
        point_size=point_size,
        render_points_as_spheres=not glyph_particles,
        label="Particles",
        scalar_bar_args={
            "title": "Particle speed",
            "color": "white" if dark else "black",
            "vertical": True,
            "position_x": 0.05,
            "position_y": 0.27,
        },
    )


def main() -> None:
    args = parse_args()
    Image = require_pillow()

    tracks_path = args.tracks or default_tracks_path(args.dp_name)
    geometry_path = args.geometry or default_geometry_path(args.dp_name)
    if not tracks_path.is_file():
        raise FileNotFoundError(f"tracks file not found: {tracks_path}")
    if not geometry_path.is_file():
        raise FileNotFoundError(
            f"geometry file not found: {geometry_path}. Pass --geometry or set "
            "MRSIM_PIGNN_ROOT to the PI-GNN checkout that contains geometry.pkl."
        )

    args.out_webp.parent.mkdir(parents=True, exist_ok=True)
    args.out_preview.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading tracks: {tracks_path}", flush=True)
    positions, reset, dt = load_tracks(
        tracks_path,
        frame_stride=args.frame_stride,
        particle_stride=args.particle_stride,
    )
    speed = speed_from_positions(positions, reset, dt)
    finite = speed[np.isfinite(speed)]
    speed_max = float(np.percentile(finite, 99)) if finite.size else 1.0

    print(
        f"Rendering {positions.shape[0]} frames, {positions.shape[1]} particles, "
        f"dt={dt:.4f}s, speed_max={speed_max:.3g}",
        flush=True,
    )
    geometry = load_geometry(
        geometry_path,
        max_points=args.max_geometry_points,
        alpha=args.surface_alpha,
        style=args.geometry_style,
    )
    tracks = None if args.hide_paths else build_track_segments(positions, reset)
    sphere = pv.Sphere(radius=0.0006, theta_resolution=16, phi_resolution=16)

    particles = pv.PolyData(positions[0])
    particles["speed"] = speed[0]
    particle_mesh = (
        particles.glyph(geom=sphere, scale=False, orient=False)
        if args.glyph_particles
        else particles
    )

    plotter = pv.Plotter(window_size=(1400, 900), off_screen=True)
    add_scene(
        plotter,
        geometry=geometry,
        geometry_style=args.geometry_style,
        style=args.style,
        tracks=tracks,
        particle_mesh=particle_mesh,
        speed_max=speed_max,
        glyph_particles=args.glyph_particles,
        point_size=args.point_size,
    )
    time_label = plotter.add_text(
        f"Step {args.frame_stride}/{positions.shape[0] * args.frame_stride}   t={dt:.4f} s",
        position="upper_left",
        font_size=12,
        color="white" if args.style == "aorta" else "black",
    )
    if args.style == "aorta":
        plotter.add_legend(bcolor="#18212c", border=True)
    plotter.add_axes()
    configure_camera(plotter, shift_z=args.camera_shift_z, zoom=args.camera_zoom)

    preview_step = 0
    if args.trail_frames > 1:
        preview_step = min(len(positions) - 1, max(args.trail_frames, len(positions) // 4))
        particles.points = positions[preview_step]
        particles["speed"] = speed[preview_step]
        particles.Modified()
        preview_trails = build_recent_trails(
            positions,
            reset,
            preview_step,
            trail_frames=args.trail_frames,
        )
        if preview_trails is not None:
            plotter.add_mesh(
                preview_trails,
                color="#4db6ff",
                opacity=args.trail_opacity,
                line_width=args.trail_width,
                name="recent_trails",
                show_scalar_bar=False,
            )
        time_label.SetText(
            2,
            f"Step {(preview_step + 1) * args.frame_stride}/{positions.shape[0] * args.frame_stride}   "
            f"t={(preview_step + 1) * dt:.4f} s",
        )

    preview = capture_frame(plotter)
    if args.style == "paper" and not args.no_crop:
        preview = crop_white_image(preview, pad=args.crop_pad, threshold=248)
    preview.save(args.out_preview)
    print(f"Saved preview: {args.out_preview}", flush=True)

    frames = []
    total_source_steps = positions.shape[0] * args.frame_stride
    for frame_i, step_i in enumerate(range(len(positions)), start=1):
        particles.points = positions[step_i]
        particles["speed"] = speed[step_i]
        particles.Modified()
        if args.glyph_particles:
            glyphs = particles.glyph(geom=sphere, scale=False, orient=False)
            plotter.add_mesh(
                glyphs,
                scalars="speed",
                cmap="turbo",
                clim=(0.0, speed_max),
                name="particles",
                show_scalar_bar=False,
            )
        else:
            plotter.render()
        trails = build_recent_trails(
            positions,
            reset,
            step_i,
            trail_frames=args.trail_frames,
        )
        if trails is not None:
            plotter.add_mesh(
                trails,
                color="#4db6ff",
                opacity=args.trail_opacity,
                line_width=args.trail_width,
                name="recent_trails",
                show_scalar_bar=False,
            )
        elif "recent_trails" in plotter.actors:
            plotter.remove_actor("recent_trails")
        source_step = (step_i + 1) * args.frame_stride
        time_label.SetText(
            2,
            f"Step {source_step}/{total_source_steps}   t={(step_i + 1) * dt:.4f} s",
        )
        frames.append(capture_frame(plotter).convert("RGB"))
        if frame_i == 1 or frame_i % 25 == 0 or frame_i == len(positions):
            print(f"Captured frame {frame_i}/{len(positions)}", flush=True)

    plotter.close()

    if args.style == "paper" and not args.no_crop:
        frames = crop_white_frames(frames, pad=args.crop_pad, threshold=248)
    save_animation(
        args.out_webp,
        frames,
        duration_ms=max(1, int(round(1000.0 / args.fps))),
        scale=args.animation_scale,
        crop=False,
        crop_pad=args.crop_pad,
        crop_threshold=248,
        gif_colors=96,
        webp_quality=args.webp_quality,
    )
    print(f"Saved animation: {args.out_webp}", flush=True)
    print(f"Preview size: {Image.open(args.out_preview).size}", flush=True)
    print(f"Animation frames: {len(frames)}", flush=True)


if __name__ == "__main__":
    main()
