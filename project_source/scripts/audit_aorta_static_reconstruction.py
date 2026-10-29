r"""Quantify static aorta reconstruction signal outside the known spin support."""

from __future__ import annotations

import argparse
import json
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import binary_dilation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=Path, required=True)
    parser.add_argument("--phantom-report", type=Path, required=True)
    parser.add_argument("--reconstruction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dilation-voxels", type=int, default=2)
    return parser.parse_args()


def occupancy_mask(points, center, axis_order, fov, matrix):
    centered = points[:, axis_order] - center[axis_order]
    scaled = (centered + fov / 2) / fov
    index = np.floor(scaled * matrix).astype(np.int64)
    valid = np.all((index >= 0) & (index < matrix), axis=1)
    index = index[valid]
    mask = np.zeros(tuple(matrix[::-1]), dtype=bool)
    mask[index[:, 2], index[:, 1], index[:, 0]] = True
    return mask, int(valid.sum())


def main() -> None:
    args = parse_args()
    with args.phantom_report.open("rb") as stream:
        phantom_report = tomllib.load(stream)
    center = np.asarray(phantom_report["source_coordinate_center_m"], dtype=float)
    axis_order = (
        np.asarray(
            phantom_report["source_axis_order_largest_to_smallest_one_based"],
            dtype=np.int64,
        )
        - 1
    )
    with h5py.File(args.points, "r") as h5:
        points = np.asarray(h5["points"], dtype=np.float32)
    with h5py.File(args.reconstruction, "r") as h5:
        magnitude = np.asarray(h5["reference_magnitude"], dtype=np.float32)
        fov = np.asarray(h5.attrs["FOV_m"], dtype=float)
        matrix = np.asarray(h5.attrs["matrix"], dtype=np.int64)
    mask, valid_points = occupancy_mask(points, center, axis_order, fov, matrix)
    support = binary_dilation(mask, iterations=args.dilation_voxels)
    if magnitude.shape[1:] != support.shape:
        raise ValueError(
            f"reconstruction {magnitude.shape[1:]} and support {support.shape} disagree"
        )

    bins = []
    for index, volume in enumerate(magnitude):
        signal = volume[support]
        background = volume[~support]
        total_energy = float(np.sum(volume.astype(np.float64) ** 2))
        background_energy = float(np.sum(background.astype(np.float64) ** 2))
        bins.append(
            {
                "cardiac_bin": index + 1,
                "support_voxel_count": int(support.sum()),
                "background_voxel_count": int((~support).sum()),
                "background_energy_fraction": background_energy / total_energy,
                "support_median": float(np.median(signal)),
                "support_p99": float(np.quantile(signal, 0.99)),
                "support_max": float(signal.max()),
                "background_p99": float(np.quantile(background, 0.99)),
                "background_max": float(background.max()),
                "background_max_to_support_p99": float(
                    background.max() / np.quantile(signal, 0.99)
                ),
            }
        )
    first = magnitude[0][support]
    second = magnitude[1][support]
    correlation = float(np.corrcoef(first, second)[0, 1])
    report = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_DESCRIPTIVE_ARTIFACT_AUDIT",
        "scope": (
            "Background signal is measured outside a two-voxel dilation of "
            "the known static spin occupancy. Thresholds are not claimed as "
            "externally validated acceptance criteria."
        ),
        "points": str(args.points.resolve()),
        "reconstruction": str(args.reconstruction.resolve()),
        "fov_m": fov.tolist(),
        "matrix": matrix.tolist(),
        "axis_order_zero_based": axis_order.tolist(),
        "dilation_voxels": args.dilation_voxels,
        "valid_seed_points": valid_points,
        "seed_points_total": len(points),
        "bin_support_correlation": correlation,
        "bins": bins,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
