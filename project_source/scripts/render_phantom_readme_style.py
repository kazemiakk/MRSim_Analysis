r"""Render phantom tracks in the same style as GitHub README assets.

Example:
    uv run python scripts\render_phantom_readme_style.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pyvista as pv
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from visualize_tracks import crop_white_frames, save_animation


TRACKS = Path("outputs/phantom_dp150_rot90cw_tracks_25.h5")
POINTS = Path("outputs/phantom_dp150_rot90cw_point_coordinates.npy")
OUT_WEBP = Path("outputs/phantom_dp150_github_readme_style_slow_tracers.webp")
OUT_GIF = Path("outputs/phantom_dp150_github_readme_style_slow_tracers.gif")
PREVIEW = Path("outputs/phantom_dp150_github_readme_style_slow_tracers_preview.png")

FRAME_COUNT = 180
DURATION_MS = 150
TRACER_COUNT = 120
TRAIL_STEPS = 9


def load_tracks(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    with h5py.File(path, "r") as h5:
        return np.asarray(h5["position"]), np.asarray(h5["reset"]), float(h5.attrs["dt"])


def speed_from_positions(positions: np.ndarray, reset: np.ndarray, dt: float) -> np.ndarray:
    speed = np.zeros(positions.shape[:2], dtype=float)
    speed[1:] = np.linalg.norm(np.diff(positions, axis=0), axis=2) / dt
    speed[reset] = np.nan
    return speed


def select_tracers(positions: np.ndarray, reset: np.ndarray, count: int) -> np.ndarray:
    valid_step = ~(reset[1:] | reset[:-1])
    step_distance = np.linalg.norm(np.diff(positions, axis=0), axis=2)
    step_distance[~valid_step] = 0.0
    total_distance = step_distance.sum(axis=0)
    valid_frames = (~reset).sum(axis=0)
    candidates = np.flatnonzero(valid_frames >= int(0.85 * len(positions)))
    if candidates.size == 0:
        candidates = np.arange(positions.shape[1])
    order = candidates[np.argsort(total_distance[candidates])[::-1]]
    return np.sort(order[: min(count, order.size)])


def build_trails(
    positions: np.ndarray,
    reset: np.ndarray,
    tracer_ids: np.ndarray,
    step: int,
    *,
    history: int,
) -> pv.PolyData:
    start = max(0, step - history)
    points: list[np.ndarray] = []
    lines: list[int] = []
    point_offset = 0

    for particle_i in tracer_ids:
        segment = positions[start : step + 1, particle_i]
        valid = ~reset[start : step + 1, particle_i]
        run_start: int | None = None
        for local_i, is_valid in enumerate(valid):
            if is_valid and run_start is None:
                run_start = local_i
            if run_start is None:
                continue
            is_last = local_i == len(valid) - 1
            if (not is_valid) or is_last:
                run_end = local_i + 1 if is_valid and is_last else local_i
                if run_end - run_start >= 2:
                    run = segment[run_start:run_end]
                    points.append(run)
                    ids = np.arange(point_offset, point_offset + len(run), dtype=np.int64)
                    lines.extend([len(run), *ids.tolist()])
                    point_offset += len(run)
                run_start = None if not is_valid else local_i

    if not points:
        return pv.PolyData()
    poly = pv.PolyData(np.vstack(points))
    poly.lines = np.asarray(lines, dtype=np.int64)
    return poly


def build_surface(points_path: Path) -> pv.PolyData:
    coords = np.load(points_path, mmap_mode="r")
    stride = max(1, int(np.ceil(len(coords) / 320_000)))
    cloud = pv.PolyData(np.asarray(coords[::stride]))
    return cloud.delaunay_3d(alpha=0.005).extract_surface()


def set_camera(plotter: pv.Plotter) -> None:
    plotter.view_xz()
    plotter.enable_parallel_projection()
    plotter.reset_camera()
    plotter.camera.zoom(1.18)


def main() -> None:
    OUT_WEBP.parent.mkdir(parents=True, exist_ok=True)
    positions, reset, dt = load_tracks(TRACKS)
    speed = speed_from_positions(positions, reset, dt)
    finite = speed[np.isfinite(speed)]
    speed_max = float(np.percentile(finite, 98.5)) if finite.size else 1.0
    surface = build_surface(POINTS)
    tracer_ids = select_tracers(positions, reset, TRACER_COUNT)
    frame_indices = np.linspace(0, len(positions) - 1, FRAME_COUNT, dtype=int)

    frames: list[Image.Image] = []
    for frame_i, step in enumerate(frame_indices, start=1):
        if frame_i == 1 or frame_i % 25 == 0 or frame_i == len(frame_indices):
            print(f"README-style tracer frame {frame_i}/{len(frame_indices)}", flush=True)
        plotter = pv.Plotter(off_screen=True, window_size=(1400, 900))
        plotter.set_background("white")
        plotter.enable_anti_aliasing("ssaa")
        plotter.add_mesh(
            surface,
            color="#d9d9d9",
            opacity=0.18,
            smooth_shading=True,
            show_scalar_bar=False,
        )

        trails = build_trails(
            positions,
            reset,
            tracer_ids,
            int(step),
            history=TRAIL_STEPS,
        )
        if trails.n_points:
            plotter.add_mesh(
                trails,
                color="#2563eb",
                line_width=1.05,
                opacity=0.22,
                render_lines_as_tubes=True,
                show_scalar_bar=False,
            )

        cloud = pv.PolyData(positions[step, tracer_ids])
        cloud["speed"] = np.nan_to_num(
            speed[step, tracer_ids], nan=0.0, posinf=speed_max, neginf=0.0
        )
        plotter.add_mesh(
            cloud,
            scalars="speed",
            cmap="jet",
            clim=(0.0, speed_max),
            point_size=4.6,
            render_points_as_spheres=True,
            opacity=0.92,
            show_scalar_bar=False,
        )
        set_camera(plotter)
        frame = Image.fromarray(plotter.screenshot(return_img=True)).convert("RGB")
        frames.append(frame)
        if frame_i == len(frame_indices) // 3:
            PREVIEW.parent.mkdir(parents=True, exist_ok=True)
            frame.save(PREVIEW)
        plotter.close()

    frames = crop_white_frames(frames, pad=90, threshold=248)
    save_animation(
        OUT_WEBP,
        frames,
        duration_ms=DURATION_MS,
        scale=1.0,
        crop=False,
        crop_pad=18,
        crop_threshold=248,
        gif_colors=96,
        webp_quality=88,
    )
    save_animation(
        OUT_GIF,
        frames,
        duration_ms=DURATION_MS,
        scale=1.0,
        crop=False,
        crop_pad=18,
        crop_threshold=248,
        gif_colors=128,
        webp_quality=88,
    )
    frames[min(len(frames) - 1, len(frames) // 3)].save(PREVIEW)
    print(f"Saved {OUT_WEBP}")
    print(f"Saved {OUT_GIF}")
    print(f"Saved {PREVIEW}")


if __name__ == "__main__":
    main()
