r"""Create Charles-style projection and center-slice panels from Koma output."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


DEFAULT_INPUT = Path(
    "outputs/dp150_mri_pipeline/koma/"
    "simulation_static_2m_ref/pc_reconstruction.h5"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Output/title label; inferred as static or motion when omitted.",
    )
    return parser.parse_args()


def robust_limit(volume: np.ndarray) -> float:
    positive = volume[volume > 0]
    return float(np.quantile(positive, 0.999)) if positive.size else 1.0


def panel_data(volume: np.ndarray) -> list[tuple[str, np.ndarray, str, str, list[float]]]:
    # Python HDF5 order is (slice, phase, readout). MRI readout is the long axis.
    nz, ny, nx = volume.shape
    fov = (512.0, 128.0, 128.0)
    x_extent = [-fov[0] / 2, fov[0] / 2]
    y_extent = [-fov[1] / 2, fov[1] / 2]
    z_extent = [-fov[2] / 2, fov[2] / 2]
    return [
        (
            "Projection - XY",
            volume.max(axis=0),
            "Readout / source z (mm)",
            "Phase / source x (mm)",
            [*x_extent, *y_extent],
        ),
        (
            "Projection - XZ",
            volume.max(axis=1),
            "Readout / source z (mm)",
            "Slice / source y (mm)",
            [*x_extent, *z_extent],
        ),
        (
            "Projection - YZ",
            volume.max(axis=2),
            "Phase / source x (mm)",
            "Slice / source y (mm)",
            [*y_extent, *z_extent],
        ),
        (
            "Center slice - XY",
            volume[nz // 2],
            "Readout / source z (mm)",
            "Phase / source x (mm)",
            [*x_extent, *y_extent],
        ),
        (
            "Center slice - XZ",
            volume[:, ny // 2, :],
            "Readout / source z (mm)",
            "Slice / source y (mm)",
            [*x_extent, *z_extent],
        ),
        (
            "Center slice - YZ",
            volume[:, :, nx // 2],
            "Phase / source x (mm)",
            "Slice / source y (mm)",
            [*y_extent, *z_extent],
        ),
    ]


def render_bin(volume: np.ndarray, phase: int, path: Path, label: str) -> None:
    limit = robust_limit(volume)
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), constrained_layout=True)
    for axis, (title, image, xlabel, ylabel, extent) in zip(
        axes.ravel(), panel_data(volume), strict=True
    ):
        axis.imshow(
            image,
            cmap="gray",
            origin="lower",
            extent=extent,
            vmin=0,
            vmax=limit,
            interpolation="nearest",
            aspect="equal",
        )
        axis.set_title(title, fontsize=12)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.grid(False)
    fig.suptitle(
        f"dp150 {label} KomaMRI reconstruction | cardiac bin {phase}",
        fontsize=15,
    )
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def render_pc_summary(
    magnitude: np.ndarray,
    velocity: np.ndarray,
    speed: np.ndarray,
    mask: np.ndarray,
    path: Path,
    label: str,
) -> None:
    if velocity.shape != (3, *magnitude.shape):
        raise ValueError(
            f"Expected velocity shape (3, bin, slice, phase, readout), got {velocity.shape}"
        )
    if speed.shape != magnitude.shape or mask.shape != magnitude.shape:
        raise ValueError("Speed/mask shapes do not match reference magnitude")

    valid_velocity = np.abs(velocity[0][mask])
    valid_speed = speed[mask]
    velocity_limit = (
        float(np.quantile(valid_velocity, 0.995)) if valid_velocity.size else 1.0
    )
    speed_limit = float(np.quantile(valid_speed, 0.995)) if valid_speed.size else 1.0
    velocity_limit = max(velocity_limit, 1e-6)
    speed_limit = max(speed_limit, 1e-6)
    extent = [-256.0, 256.0, -64.0, 64.0]

    fig, axes = plt.subplots(
        magnitude.shape[0],
        3,
        figsize=(18, 4.2 * magnitude.shape[0]),
        constrained_layout=True,
        squeeze=False,
    )
    for phase in range(magnitude.shape[0]):
        mag = magnitude[phase]
        weights = np.where(mask[phase], mag, 0.0)
        denominator = weights.sum(axis=0)
        readout_velocity = np.divide(
            (velocity[0, phase] * weights).sum(axis=0),
            denominator,
            out=np.zeros_like(denominator),
            where=denominator > 0,
        )
        projected_speed = np.divide(
            (speed[phase] * weights).sum(axis=0),
            denominator,
            out=np.zeros_like(denominator),
            where=denominator > 0,
        )

        magnitude_image = axes[phase, 0].imshow(
            mag.max(axis=0),
            cmap="gray",
            origin="lower",
            extent=extent,
            vmin=0,
            vmax=robust_limit(mag),
            interpolation="nearest",
            aspect="equal",
        )
        velocity_image = axes[phase, 1].imshow(
            readout_velocity,
            cmap="RdBu_r",
            origin="lower",
            extent=extent,
            vmin=-velocity_limit,
            vmax=velocity_limit,
            interpolation="nearest",
            aspect="equal",
        )
        speed_image = axes[phase, 2].imshow(
            projected_speed,
            cmap="turbo",
            origin="lower",
            extent=extent,
            vmin=0,
            vmax=speed_limit,
            interpolation="nearest",
            aspect="equal",
        )
        axes[phase, 0].set_title(f"Reference magnitude | bin {phase + 1}")
        axes[phase, 1].set_title(f"Readout velocity | bin {phase + 1}")
        axes[phase, 2].set_title(f"Speed | bin {phase + 1}")
        for axis in axes[phase]:
            axis.set_xlabel("Readout / source z (mm)")
            axis.set_ylabel("Phase / source x (mm)")
            axis.grid(False)
        fig.colorbar(magnitude_image, ax=axes[phase, 0], shrink=0.72)
        fig.colorbar(
            velocity_image,
            ax=axes[phase, 1],
            shrink=0.72,
            label="m/s",
        )
        fig.colorbar(speed_image, ax=axes[phase, 2], shrink=0.72, label="m/s")

    fig.suptitle(f"dp150 {label} phase-contrast MRI projections", fontsize=16)
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_dir = (args.output_dir or input_path.parent).resolve()
    inferred_label = "motion" if "motion" in input_path.parent.name.lower() else "static"
    label = (args.label or inferred_label).strip().lower().replace(" ", "_")
    output_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(input_path, "r") as h5:
        magnitude = np.asarray(h5["reference_magnitude"], dtype=np.float32)
        if magnitude.ndim != 4:
            raise ValueError(f"Expected (bin, slice, phase, readout), got {magnitude.shape}")
        rr = float(h5.attrs["RR_s"])
        venc = float(h5.attrs["Venc_m_per_s"])
        velocity = (
            np.asarray(h5["velocity"], dtype=np.float32)
            if "velocity" in h5
            else None
        )
        speed = np.asarray(h5["speed"], dtype=np.float32) if "speed" in h5 else None
        mask = np.asarray(h5["mask"], dtype=bool) if "mask" in h5 else None
    if not np.isfinite(magnitude).all() or magnitude.max(initial=0) <= 0:
        raise RuntimeError("Reconstructed magnitude is blank or non-finite")

    outputs: list[dict[str, object]] = []
    for index in range(magnitude.shape[0]):
        path = output_dir / f"{label}_bin{index + 1}_projection_slices.png"
        render_bin(magnitude[index], index + 1, path, label)
        with Image.open(path) as image:
            outputs.append(
                {
                    "cardiac_bin": index + 1,
                    "path": str(path),
                    "dimensions_px": list(image.size),
                    "bytes": path.stat().st_size,
                }
            )

    pc_summary = None
    if velocity is not None and speed is not None and mask is not None:
        pc_summary = output_dir / f"{label}_pc_velocity_projections.png"
        render_pc_summary(magnitude, velocity, speed, mask, pc_summary, label)
        with Image.open(pc_summary) as image:
            outputs.append(
                {
                    "type": "phase_contrast_velocity_summary",
                    "path": str(pc_summary),
                    "dimensions_px": list(image.size),
                    "bytes": pc_summary.stat().st_size,
                }
            )

    report = {
        "status": "PASS",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path),
        "hdf5_shape_python": list(magnitude.shape),
        "dimension_order_python": ["cardiac_bin", "slice", "phase", "readout"],
        "RR_s": rr,
        "Venc_m_per_s": venc,
        "label": label,
        "outputs": outputs,
    }
    report_path = output_dir / f"{label}_projection_slices_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Projection/slice QC: PASS\nReport: {report_path}")


if __name__ == "__main__":
    main()
