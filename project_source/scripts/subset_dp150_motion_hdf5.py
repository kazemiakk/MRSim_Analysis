"""Create a reproducible tracked-particle subset for a dp150 KomaMRI pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--count", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20_260_831)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path, block_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def distribution_metrics(source: np.ndarray, subset: np.ndarray) -> dict[str, object]:
    lower = source.min(axis=0)
    upper = source.max(axis=0)
    subset_lower = subset.min(axis=0)
    subset_upper = subset.max(axis=0)
    span_ratio = (subset_upper - subset_lower) / (upper - lower)

    bins = (48, 8, 8)
    ranges = list(zip(lower, upper, strict=True))
    source_hist, _ = np.histogramdd(source, bins=bins, range=ranges)
    subset_hist, _ = np.histogramdd(subset, bins=bins, range=ranges)
    source_probability = source_hist / source_hist.sum()
    subset_probability = subset_hist / subset_hist.sum()
    occupied = source_hist >= 20
    expected = source_hist * (len(subset) / len(source))
    stable = expected >= 5
    relative_error = (
        (subset_hist[stable] - expected[stable]) / expected[stable]
    )

    axial_source, axial_edges = np.histogram(source[:, 0], bins=100)
    axial_subset, _ = np.histogram(subset[:, 0], bins=axial_edges)
    axial_source_probability = axial_source / axial_source.sum()
    axial_subset_probability = axial_subset / axial_subset.sum()

    return {
        "source_extent_min_m": lower.tolist(),
        "source_extent_max_m": upper.tolist(),
        "subset_extent_min_m": subset_lower.tolist(),
        "subset_extent_max_m": subset_upper.tolist(),
        "extent_span_ratio": span_ratio.tolist(),
        "coarse_spatial_bins": list(bins),
        "coarse_spatial_total_variation": float(
            0.5 * np.abs(subset_probability - source_probability).sum()
        ),
        "occupied_bin_coverage": float(
            np.count_nonzero((subset_hist > 0) & occupied)
            / np.count_nonzero(occupied)
        ),
        "relative_bin_error_rms_expected_count_ge_5": float(
            np.sqrt(np.mean(relative_error**2))
        ),
        "relative_bin_error_p95_absolute_expected_count_ge_5": float(
            np.quantile(np.abs(relative_error), 0.95)
        ),
        "axial_total_variation_100_bins": float(
            0.5
            * np.abs(
                axial_subset_probability - axial_source_probability
            ).sum()
        ),
        "axial_maximum_absolute_probability_error": float(
            np.max(
                np.abs(
                    axial_subset_probability - axial_source_probability
                )
            )
        ),
    }


def main() -> None:
    args = parse_args()
    source_path = args.input.resolve()
    output_path = args.output.resolve()
    report_path = (
        args.report.resolve()
        if args.report is not None
        else output_path.with_suffix(".report.json")
    )
    partial_path = output_path.with_suffix(output_path.suffix + ".partial")

    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if args.count < 1:
        raise ValueError("--count must be positive")
    for path in (output_path, report_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if partial_path.exists():
        partial_path.unlink()

    started = time.perf_counter()
    source_hash = sha256(source_path)
    print(f"Source SHA-256: {source_hash}", flush=True)

    with h5py.File(source_path, "r") as source:
        required = {
            "initial_position",
            "position",
            "position_time_s",
            "reset",
            "source_seed_index",
        }
        missing = sorted(required.difference(source.keys()))
        if missing:
            raise KeyError(f"Missing source datasets: {missing}")

        source_count = int(source.attrs["n_particles"])
        steps = int(source.attrs["n_steps"])
        if args.count > source_count:
            raise ValueError(
                f"Cannot select {args.count:,} particles from {source_count:,}"
            )
        if source["position"].shape != (steps, source_count, 3):
            raise ValueError("Unexpected position dataset shape")
        if source["reset"].shape != (steps, source_count):
            raise ValueError("Unexpected reset dataset shape")

        rng = np.random.default_rng(args.seed)
        selected = np.sort(
            rng.choice(source_count, args.count, replace=False)
        ).astype(np.int64, copy=False)
        selected_sha = hashlib.sha256(
            selected.astype("<i8", copy=False).tobytes()
        ).hexdigest()

        initial_source = np.asarray(
            source["initial_position"], dtype=np.float32
        )
        initial_subset = np.ascontiguousarray(
            initial_source[selected], dtype=np.float32
        )
        metrics = distribution_metrics(initial_source, initial_subset)
        source_seed_index = np.asarray(
            source["source_seed_index"], dtype=np.int64
        )[selected]
        position_time = np.asarray(
            source["position_time_s"], dtype=np.float64
        )

        free_bytes = shutil.disk_usage(output_path.parent).free
        estimated_bytes = int(source_path.stat().st_size * args.count / source_count)
        if free_bytes < estimated_bytes * 1.25:
            raise OSError(
                "Insufficient free space for the subset HDF5 plus atomic-write margin"
            )

        selected_reset_events = 0
        source_reset_events = 0
        finite = bool(np.isfinite(initial_subset).all())
        with h5py.File(partial_path, "w") as target:
            for key, value in source.attrs.items():
                target.attrs[key] = value
            target.attrs["n_particles"] = args.count
            target.attrs["subset_source_particle_count"] = source_count
            target.attrs["subset_selection_seed"] = args.seed
            target.attrs["subset_selection_method"] = (
                "numpy.default_rng.choice_without_replacement_sorted"
            )
            target.attrs["subset_particle_index_sha256"] = selected_sha
            target.attrs["subset_source_hdf5_sha256"] = source_hash
            target.attrs["subset_source_hdf5"] = str(source_path)

            target.create_dataset(
                "initial_position", data=initial_subset, dtype=np.float32
            )
            target.create_dataset(
                "source_seed_index", data=source_seed_index, dtype=np.int64
            )
            target.create_dataset(
                "position_time_s", data=position_time, dtype=np.float64
            )
            position_target = target.create_dataset(
                "position",
                shape=(steps, args.count, 3),
                dtype=np.float32,
                chunks=(1, min(65_536, args.count), 3),
            )
            reset_target = target.create_dataset(
                "reset",
                shape=(steps, args.count),
                dtype=np.bool_,
                chunks=(1, min(65_536, args.count)),
            )

            for step in range(steps):
                source_position = np.asarray(
                    source["position"][step], dtype=np.float32
                )
                selected_position = np.ascontiguousarray(
                    source_position[selected], dtype=np.float32
                )
                source_reset = np.asarray(source["reset"][step], dtype=np.bool_)
                selected_reset = np.ascontiguousarray(source_reset[selected])
                finite = finite and bool(np.isfinite(selected_position).all())
                source_reset_events += int(np.count_nonzero(source_reset))
                selected_reset_events += int(np.count_nonzero(selected_reset))
                position_target[step] = selected_position
                reset_target[step] = selected_reset
                if step == 0 or (step + 1) % 120 == 0 or step + 1 == steps:
                    print(f"Copied trajectory step {step + 1:,}/{steps:,}", flush=True)

    os.replace(partial_path, output_path)
    output_hash = sha256(output_path)
    source_reset_fraction = source_reset_events / (steps * source_count)
    subset_reset_fraction = selected_reset_events / (steps * args.count)
    reset_fraction_difference = abs(subset_reset_fraction - source_reset_fraction)
    checks = {
        "finite": finite,
        "particle_count": args.count == 100_000,
        "minimum_extent_span_ratio_ge_0p98": min(
            metrics["extent_span_ratio"]
        )
        >= 0.98,
        "occupied_bin_coverage_ge_0p99": metrics["occupied_bin_coverage"]
        >= 0.99,
        "axial_total_variation_le_0p025": metrics[
            "axial_total_variation_100_bins"
        ]
        <= 0.025,
        "coarse_spatial_total_variation_le_0p10": metrics[
            "coarse_spatial_total_variation"
        ]
        <= 0.10,
        "reset_fraction_difference_le_5e_5": reset_fraction_difference <= 5e-5,
    }
    report = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "source": {
            "file": str(source_path),
            "bytes": source_path.stat().st_size,
            "sha256": source_hash,
            "particle_count": source_count,
            "trajectory_steps": steps,
        },
        "subset": {
            "file": str(output_path),
            "bytes": output_path.stat().st_size,
            "sha256": output_hash,
            "particle_count": args.count,
            "selection_seed": args.seed,
            "selection_method": (
                "numpy.default_rng.choice_without_replacement_sorted"
            ),
            "selected_identity_index_minimum_zero_based": int(selected[0]),
            "selected_identity_index_maximum_zero_based": int(selected[-1]),
            "selected_identity_indices_sha256_little_endian_int64": selected_sha,
        },
        "uniformity": metrics,
        "reset_history": {
            "source_event_count": source_reset_events,
            "subset_event_count": selected_reset_events,
            "source_particle_step_fraction": source_reset_fraction,
            "subset_particle_step_fraction": subset_reset_fraction,
            "absolute_fraction_difference": reset_fraction_difference,
        },
        "acceptance": {"passed": all(checks.values()), "checks": checks},
        "runtime_seconds": time.perf_counter() - started,
        "note": (
            "This is a deterministic identity subset of the validated 1M "
            "trajectory, not a separately retracked 100k CFD realization."
        ),
    }
    temporary_report = report_path.with_suffix(report_path.suffix + ".partial")
    temporary_report.write_text(
        json.dumps(report, indent=2, default=json_value) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_report, report_path)
    if report["status"] != "PASS":
        raise RuntimeError(f"Subset QC failed; see {report_path}")
    print(f"Subset validation PASS: {report_path}", flush=True)


if __name__ == "__main__":
    main()
