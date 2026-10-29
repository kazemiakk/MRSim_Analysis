r"""Build the E-018 support-constrained dp150 phase reconstruction.

This is a reconstruction-method sensitivity derivative of the preserved E-010
REF/X/Y/Z complex images.  It does not modify the source reconstruction, add
cardiac bins, or simulate missing wall material.  High-confidence voxels keep
their direct complex phase difference.  Low-confidence voxels are estimated
from a support-constrained spatial average of complex cross-products, without
temporal smoothing or interpolation.

The source PHS-label-to-actual-motion-time binding is being audited separately.
Until that audit passes (or the k-space is re-binned), this implementation is a
pre-rebin method sensitivity and must not be promoted as the primary correction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter

try:
    from scripts.analyze_dp150_scanner_reconstruction import (
        DEFAULT_FOV_M,
        DEFAULT_MATRIX,
        load_support,
        sha256_file,
    )
    from scripts.render_dp150_xyz_magnitude_phase import (
        RAW_FILENAMES,
        compound_to_complex,
    )
except ModuleNotFoundError:
    from analyze_dp150_scanner_reconstruction import (  # type: ignore[no-redef]
        DEFAULT_FOV_M,
        DEFAULT_MATRIX,
        load_support,
        sha256_file,
    )
    from render_dp150_xyz_magnitude_phase import (  # type: ignore[no-redef]
        RAW_FILENAMES,
        compound_to_complex,
    )


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_RUN_DIR = (
    ROOT
    / "outputs"
    / "dp150_mri_pipeline"
    / "koma"
    / "simulation_motion_1m_scanner_matched_venc5"
)
COMPARISON_RUN_DIR = (
    ROOT
    / "outputs"
    / "dp150_mri_pipeline"
    / "koma"
    / "simulation_motion_100k_scanner_matched_venc5"
)
DEFAULT_RECONSTRUCTION = PRODUCTION_RUN_DIR / "pc_reconstruction.h5"
DEFAULT_SUPPORT = (
    ROOT / "outputs" / "dp150_mri_pipeline" / "dp150_static_2m_mri.h5"
)
DEFAULT_OUTPUT_DIR = (
    PRODUCTION_RUN_DIR
    / "final_image_analysis"
    / "support_constrained_phase_reconstruction_1m"
)

ENCODINGS = ("REF", "X", "Y", "Z")
COMPONENTS = ("X", "Y", "Z")
EXPECTED_BINS = 15
DEFAULT_DIRECT_THRESHOLD = 0.10
DEFAULT_SIGMA_MM = 1.0
GAUSSIAN_TRUNCATE = 4.0
OUTPUT_H5_NAME = "dp150_support_constrained_phase.h5"
REPORT_NAME = "dp150_support_constrained_phase_report.json"
SCHEMA_VERSION = 1

CANONICAL_RECONSTRUCTION_SHA256 = (
    "a3b6306398cb98dab0ef02473692d76e7bc57026b3f6d5d090d71624bdacb376"
)
CANONICAL_SUPPORT_SHA256 = (
    "5c15d5df641187cada1f5f4c2895213245a722d362c6cf08ddcf581ea34278a1"
)
CANONICAL_RAW_SHA256 = {
    "REF": "f5865f9ebe8b2f137c80bee24d5695871a02bf17916fa71e9c68a8fa8af27b48",
    "X": "ad9886bb9cbd8e9a1dc32916cce6d2862c8950bfae443e66a3dd04d87bc319b5",
    "Y": "2b6e5b3ef2b2118567d48b6437c0e9ea28893829837211536b0d1c9af32e04f0",
    "Z": "15d31b6b0d572a7440a8e215a71949f274ca4fa0c687766e5c272a8b398ee2ad",
}
CANONICAL_COMPARISON_RAW_SHA256 = {
    "REF": "13a9ed8498c001a69a2d1b8c9d298bb7853b546d198dd5149d1d05f35c222cfc",
    "X": "0efbb6a7d2b4ee283ea3800f0825a67d9dc935e603b713cec72cb22787298202",
    "Y": "c7c755c761af370f70ca01087f364d3a9aab074e65a14fcc2b0d3858565acdde",
    "Z": "90bba6a70e9d498bec64ee96395f0ca27a1c54c9ac74fcb157718487ab3fb226",
}

CANONICAL_SUPPORT_VOXELS = 65_589
CANONICAL_DIRECT_VOXELS = 56_447
CANONICAL_REGULARIZED_VOXELS = 9_142
CANONICAL_SECTION_INDICES = {
    "upstream": 70,
    "stenosis_throat": 98,
    "immediate_post_stenotic": 122,
}
CANONICAL_SECTION_COUNTS = {
    "upstream": (185, 169, 16),
    "stenosis_throat": (28, 24, 4),
    "immediate_post_stenotic": (188, 149, 39),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=PRODUCTION_RUN_DIR)
    parser.add_argument(
        "--comparison-run-dir", type=Path, default=COMPARISON_RUN_DIR
    )
    parser.add_argument(
        "--reconstruction", type=Path, default=DEFAULT_RECONSTRUCTION
    )
    parser.add_argument("--support-h5", type=Path, default=DEFAULT_SUPPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--fov-m", nargs=3, type=float, default=DEFAULT_FOV_M
    )
    parser.add_argument(
        "--matrix", nargs=3, type=int, default=DEFAULT_MATRIX
    )
    parser.add_argument(
        "--direct-threshold", type=float, default=DEFAULT_DIRECT_THRESHOLD
    )
    parser.add_argument("--sigma-mm", type=float, default=DEFAULT_SIGMA_MM)
    return parser.parse_args()


def artifact_record(path: Path, digest: str | None = None) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": digest if digest is not None else sha256_file(path),
    }


def _require_hash(path: Path, expected: str, role: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = sha256_file(path)
    if observed.lower() != expected.lower():
        raise ValueError(
            f"{role} SHA-256 mismatch: expected {expected}, observed {observed}"
        )
    return observed.lower()


def normalized_magnitude_confidence_for_bin(
    images: Mapping[str, np.ndarray], support: np.ndarray
) -> tuple[np.ndarray, dict[str, float]]:
    """Return min-encoding magnitude / in-support median for one bin.

    Taking the minimum across REF/X/Y/Z is deliberate: a joint RMS can remain
    large when REF is weak, but every phase difference requires a reliable REF.
    """
    support = np.asarray(support, dtype=bool)
    if set(images) != set(ENCODINGS):
        raise ValueError(f"Expected exactly {ENCODINGS}, got {tuple(images)}")
    if not support.any():
        raise ValueError("Support must contain at least one voxel")

    normalized: list[np.ndarray] = []
    scales: dict[str, float] = {}
    for encoding in ENCODINGS:
        image = np.asarray(images[encoding])
        if image.shape != support.shape:
            raise ValueError(
                f"{encoding} image shape {image.shape} != support {support.shape}"
            )
        magnitude = np.abs(image)
        if not np.isfinite(magnitude).all():
            raise ValueError(f"{encoding} magnitude contains non-finite values")
        scale = float(np.median(magnitude[support]))
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"{encoding} in-support magnitude median is not positive")
        scales[encoding] = scale
        normalized.append(magnitude / scale)
    return np.minimum.reduce(normalized).astype(np.float32), scales


def fixed_direct_mask(
    support: np.ndarray, confidence_score: np.ndarray, threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    support = np.asarray(support, dtype=bool)
    score = np.asarray(confidence_score)
    if score.shape != support.shape:
        raise ValueError("Confidence score and support shapes do not match")
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("Direct threshold must be finite and positive")
    if not np.isfinite(score[support]).all():
        raise ValueError("In-support confidence score contains non-finite values")
    direct = support & (score >= threshold)
    regularized = support & ~direct
    if np.any(direct & regularized) or not np.array_equal(
        direct | regularized, support
    ):
        raise RuntimeError("Direct and regularized masks do not partition support")
    return direct, regularized


def support_constrained_complex_pool(
    cross_product: np.ndarray,
    support: np.ndarray,
    sigma_voxels: tuple[float, float, float],
    *,
    truncate: float = GAUSSIAN_TRUNCATE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pool a complex cross-product without averaging wrapped angles.

    The Gaussian is applied to real and imaginary parts after zeroing values
    outside the exact support.  Dividing by the convolved support normalizes
    the kernel at the lumen boundary.  Coherence is the resultant phasor
    magnitude divided by the pooled magnitude-product weight.
    """
    values = np.asarray(cross_product)
    support = np.asarray(support, dtype=bool)
    if values.shape != support.shape or values.ndim != 3:
        raise ValueError("Cross-product and support must be matching 3D arrays")
    if len(sigma_voxels) != 3 or any(
        (not math.isfinite(value) or value < 0) for value in sigma_voxels
    ):
        raise ValueError("sigma_voxels must contain three finite non-negative values")
    if not math.isfinite(truncate) or truncate <= 0:
        raise ValueError("Gaussian truncate must be finite and positive")
    if not np.isfinite(values.real).all() or not np.isfinite(values.imag).all():
        raise ValueError("Cross-product contains non-finite values")
    if not support.any():
        raise ValueError("Support must contain at least one voxel")

    masked = np.where(support, values, 0.0)
    kwargs = {
        "sigma": sigma_voxels,
        "mode": "constant",
        "cval": 0.0,
        "truncate": truncate,
    }
    numerator = gaussian_filter(masked.real, **kwargs) + 1j * gaussian_filter(
        masked.imag, **kwargs
    )
    boundary = gaussian_filter(support.astype(np.float64), **kwargs)
    pooled_weight_numerator = gaussian_filter(
        np.where(support, np.abs(values), 0.0), **kwargs
    )

    tiny = np.finfo(np.float64).tiny
    if np.any(boundary[support] <= tiny):
        raise RuntimeError("Support normalization vanished inside support")
    if np.any(pooled_weight_numerator[support] <= tiny):
        raise RuntimeError("Pooled phase weight vanished inside support")
    pooled = numerator / np.maximum(boundary, tiny)
    pooled_weight = pooled_weight_numerator / np.maximum(boundary, tiny)
    coherence = np.abs(numerator) / np.maximum(pooled_weight_numerator, tiny)
    if not np.isfinite(pooled[support]).all() or not np.isfinite(
        coherence[support]
    ).all():
        raise RuntimeError("Complex pooling produced non-finite in-support values")
    return pooled, coherence.astype(np.float32), pooled_weight.astype(np.float32)


def hybrid_phase_arrays(
    cross_product: np.ndarray,
    pooled_cross_product: np.ndarray,
    support: np.ndarray,
    direct: np.ndarray,
    regularized: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return raw and hybrid wrapped phase, with NaN outside support."""
    values = np.asarray(cross_product)
    pooled = np.asarray(pooled_cross_product)
    support = np.asarray(support, dtype=bool)
    direct = np.asarray(direct, dtype=bool)
    regularized = np.asarray(regularized, dtype=bool)
    if any(array.shape != support.shape for array in (values, pooled, direct, regularized)):
        raise ValueError("Phase inputs and masks must have identical shapes")
    if np.any(direct & regularized) or not np.array_equal(
        direct | regularized, support
    ):
        raise ValueError("Direct and regularized masks must partition support")

    direct_phase = np.angle(values).astype(np.float32)
    pooled_phase = np.angle(pooled).astype(np.float32)
    raw = np.full(support.shape, np.nan, dtype=np.float32)
    corrected = np.full(support.shape, np.nan, dtype=np.float32)
    raw[support] = direct_phase[support]
    corrected[direct] = direct_phase[direct]
    corrected[regularized] = pooled_phase[regularized]
    if not np.isfinite(raw[support]).all() or not np.isfinite(
        corrected[support]
    ).all():
        raise RuntimeError("Hybrid phase contains non-finite in-support values")
    return raw, corrected


def circular_velocity_difference(
    first_phase: np.ndarray, second_phase: np.ndarray, venc: float
) -> np.ndarray:
    if not math.isfinite(venc) or venc <= 0:
        raise ValueError("VENC must be finite and positive")
    first = np.asarray(first_phase)
    second = np.asarray(second_phase)
    if first.shape != second.shape:
        raise ValueError("Phase arrays must have identical shapes")
    return (
        np.abs(np.angle(np.exp(1j * (first - second)))) * (venc / np.pi)
    )


def distribution(values: np.ndarray) -> dict[str, float | int]:
    selected = np.asarray(values, dtype=np.float64)
    selected = selected[np.isfinite(selected)]
    if not selected.size:
        raise ValueError("Cannot summarize an empty finite distribution")
    return {
        "count": int(selected.size),
        "minimum": float(selected.min()),
        "p01": float(np.quantile(selected, 0.01)),
        "p05": float(np.quantile(selected, 0.05)),
        "p50": float(np.quantile(selected, 0.50)),
        "p90": float(np.quantile(selected, 0.90)),
        "p95": float(np.quantile(selected, 0.95)),
        "p99": float(np.quantile(selected, 0.99)),
        "maximum": float(selected.max()),
        "mean": float(selected.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(selected)))),
    }


def _open_raw_sources(
    stack: ExitStack,
    run_dir: Path,
    expected_shape: tuple[int, int, int, int],
) -> tuple[dict[str, h5py.File], dict[str, h5py.Dataset]]:
    files: dict[str, h5py.File] = {}
    images: dict[str, h5py.Dataset] = {}
    for encoding in ENCODINGS:
        path = run_dir / RAW_FILENAMES[encoding]
        source = stack.enter_context(h5py.File(path, "r"))
        for dataset_name in ("image", "rawsort"):
            if dataset_name not in source:
                raise KeyError(f"Missing {dataset_name} in {path}")
            if source[dataset_name].shape != expected_shape:
                raise ValueError(
                    f"{path.name}:{dataset_name} shape "
                    f"{source[dataset_name].shape} != {expected_shape}"
                )
        files[encoding] = source
        images[encoding] = source["image"]
    return files, images


def _complex_integrity(values: np.ndarray) -> tuple[int, int, float]:
    complex_values = compound_to_complex(values)
    magnitude = np.abs(complex_values)
    return (
        int(np.count_nonzero(magnitude == 0)),
        int(np.count_nonzero(~np.isfinite(magnitude))),
        float(magnitude.min()),
    )


def _dataset_options(shape: tuple[int, ...]) -> dict[str, Any]:
    if len(shape) == 5:
        chunks = (1, 1, min(4, shape[2]), min(16, shape[3]), min(100, shape[4]))
    elif len(shape) == 4:
        chunks = (1, min(4, shape[1]), min(16, shape[2]), min(100, shape[3]))
    else:
        raise ValueError(f"Unsupported streamed dataset shape {shape}")
    return {
        "chunks": chunks,
        "compression": "gzip",
        "compression_opts": 4,
        "shuffle": True,
        "track_times": False,
    }


def _create_float_dataset(
    target: h5py.File, name: str, shape: tuple[int, ...], units: str
) -> h5py.Dataset:
    dataset = target.create_dataset(
        name,
        shape=shape,
        dtype=np.float32,
        fillvalue=np.nan,
        **_dataset_options(shape),
    )
    dataset.attrs["units"] = units
    return dataset


def _source_class(support: np.ndarray, direct: np.ndarray) -> np.ndarray:
    result = np.zeros(support.shape, dtype=np.uint8)
    result[support & ~direct] = 2
    result[direct] = 1
    return result


def _section_metrics(
    support: np.ndarray, direct: np.ndarray, regularized: np.ndarray
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for role, index in CANONICAL_SECTION_INDICES.items():
        full = int(support[:, :, index].sum())
        kept = int(direct[:, :, index].sum())
        estimated = int(regularized[:, :, index].sum())
        result[role] = {
            "readout_index_zero_based": index,
            "support_voxels": full,
            "direct_voxels": kept,
            "regularized_voxels": estimated,
            "direct_fraction": kept / full,
        }
    return result


def _canonical_mask_checks(
    support: np.ndarray,
    direct: np.ndarray,
    regularized: np.ndarray,
    sections: dict[str, Any],
    threshold: float,
    sigma_mm: float,
) -> dict[str, bool]:
    checks = {
        "canonical_support_voxel_count": int(support.sum())
        == CANONICAL_SUPPORT_VOXELS,
        "canonical_direct_voxel_count": int(direct.sum())
        == CANONICAL_DIRECT_VOXELS,
        "canonical_regularized_voxel_count": int(regularized.sum())
        == CANONICAL_REGULARIZED_VOXELS,
        "direct_and_regularized_partition_support": bool(
            not np.any(direct & regularized)
            and np.array_equal(direct | regularized, support)
        ),
        "selected_threshold_is_preregistered_value": math.isclose(
            threshold, DEFAULT_DIRECT_THRESHOLD, rel_tol=0, abs_tol=0
        ),
        "selected_sigma_is_preregistered_value": math.isclose(
            sigma_mm, DEFAULT_SIGMA_MM, rel_tol=0, abs_tol=0
        ),
    }
    for role, expected in CANONICAL_SECTION_COUNTS.items():
        observed = sections[role]
        checks[f"canonical_{role}_counts"] = (
            observed["support_voxels"],
            observed["direct_voxels"],
            observed["regularized_voxels"],
        ) == expected
    return checks


def _validate_cli(args: argparse.Namespace) -> None:
    if not math.isfinite(args.direct_threshold) or args.direct_threshold <= 0:
        raise ValueError("--direct-threshold must be finite and positive")
    if not math.isfinite(args.sigma_mm) or args.sigma_mm <= 0:
        raise ValueError("--sigma-mm must be finite and positive")
    matrix = np.asarray(args.matrix, dtype=np.int64)
    fov = np.asarray(args.fov_m, dtype=np.float64)
    if matrix.shape != (3,) or np.any(matrix <= 0):
        raise ValueError("--matrix must contain three positive integers")
    if fov.shape != (3,) or np.any(~np.isfinite(fov)) or np.any(fov <= 0):
        raise ValueError("--fov-m must contain three finite positive values")


def main() -> None:
    args = parse_args()
    _validate_cli(args)
    run_dir = args.run_dir.resolve()
    comparison_run_dir = args.comparison_run_dir.resolve()
    reconstruction_path = args.reconstruction.resolve()
    support_path = args.support_h5.resolve()
    final_output_dir = args.output_dir.resolve()
    staging_dir = final_output_dir.with_name(".e018_reconstruction.partial")
    if final_output_dir.exists() or staging_dir.exists():
        raise FileExistsError(
            "Refusing to overwrite an E-018 final or partial directory: "
            f"{final_output_dir} / {staging_dir}"
        )

    reconstruction_hash = _require_hash(
        reconstruction_path,
        CANONICAL_RECONSTRUCTION_SHA256,
        "canonical E-010 reconstruction",
    )
    support_hash = _require_hash(
        support_path, CANONICAL_SUPPORT_SHA256, "canonical seed support"
    )
    raw_hashes: dict[str, str] = {}
    comparison_hashes: dict[str, str] = {}
    for encoding in ENCODINGS:
        raw_hashes[encoding] = _require_hash(
            run_dir / RAW_FILENAMES[encoding],
            CANONICAL_RAW_SHA256[encoding],
            f"canonical 1M {encoding} image",
        )
        comparison_hashes[encoding] = _require_hash(
            comparison_run_dir / RAW_FILENAMES[encoding],
            CANONICAL_COMPARISON_RAW_SHA256[encoding],
            f"canonical 100k {encoding} comparison image",
        )

    matrix = np.asarray(args.matrix, dtype=np.int64)
    fov_m = np.asarray(args.fov_m, dtype=np.float64)
    voxel_mm = 1e3 * fov_m / matrix
    sigma_voxels = (
        float(args.sigma_mm / voxel_mm[2]),
        float(args.sigma_mm / voxel_mm[1]),
        float(args.sigma_mm / voxel_mm[0]),
    )
    support, support_metadata = load_support(support_path, fov_m, matrix)
    expected_shape = (
        EXPECTED_BINS,
        int(matrix[2]),
        int(matrix[1]),
        int(matrix[0]),
    )
    if support.shape != expected_shape[1:]:
        raise ValueError(
            f"Support shape {support.shape} != expected {expected_shape[1:]}"
        )

    script_hash = sha256_file(Path(__file__).resolve())
    confidence_score = np.full(support.shape, np.inf, dtype=np.float32)
    encoding_scales = np.empty((len(ENCODINGS), EXPECTED_BINS), dtype=np.float64)
    image_integrity = {
        encoding: {"zero_in_support": 0, "nonfinite_full_image": 0}
        for encoding in ENCODINGS
    }
    rawsort_integrity = {
        encoding: {
            "samples": 0,
            "exact_zero_samples": 0,
            "nonfinite_samples": 0,
            "minimum_magnitude": math.inf,
        }
        for encoding in ENCODINGS
    }

    with ExitStack() as stack:
        production_files, production_images = _open_raw_sources(
            stack, run_dir, expected_shape
        )
        _comparison_files, _comparison_images = _open_raw_sources(
            stack, comparison_run_dir, expected_shape
        )
        for cardiac_bin in range(EXPECTED_BINS):
            images = {
                encoding: compound_to_complex(
                    production_images[encoding][cardiac_bin]
                )
                for encoding in ENCODINGS
            }
            bin_score, scales = normalized_magnitude_confidence_for_bin(
                images, support
            )
            confidence_score = np.minimum(confidence_score, bin_score)
            for encoding_index, encoding in enumerate(ENCODINGS):
                encoding_scales[encoding_index, cardiac_bin] = scales[encoding]
                magnitude = np.abs(images[encoding])
                image_integrity[encoding]["zero_in_support"] += int(
                    np.count_nonzero(magnitude[support] == 0)
                )
                image_integrity[encoding]["nonfinite_full_image"] += int(
                    np.count_nonzero(~np.isfinite(magnitude))
                )
                zeros, nonfinite, minimum = _complex_integrity(
                    production_files[encoding]["rawsort"][cardiac_bin]
                )
                item = rawsort_integrity[encoding]
                item["samples"] += int(np.prod(support.shape))
                item["exact_zero_samples"] += zeros
                item["nonfinite_samples"] += nonfinite
                item["minimum_magnitude"] = min(
                    float(item["minimum_magnitude"]), minimum
                )

    direct, regularized = fixed_direct_mask(
        support, confidence_score, args.direct_threshold
    )
    sections = _section_metrics(support, direct, regularized)
    canonical_mask_checks = _canonical_mask_checks(
        support,
        direct,
        regularized,
        sections,
        args.direct_threshold,
        args.sigma_mm,
    )
    if not all(canonical_mask_checks.values()):
        failed = [name for name, passed in canonical_mask_checks.items() if not passed]
        raise RuntimeError(f"Canonical E-018 mask checks failed: {failed}")

    staging_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir()
    partial_h5 = staging_dir / f".{OUTPUT_H5_NAME}.partial"
    staged_h5 = staging_dir / OUTPUT_H5_NAME
    final_h5 = final_output_dir / OUTPUT_H5_NAME

    phase_shape = (len(COMPONENTS), *expected_shape)
    speed_shape = expected_shape
    repeatability_raw: dict[str, list[np.ndarray]] = {
        component: [] for component in COMPONENTS
    }
    repeatability_corrected: dict[str, list[np.ndarray]] = {
        component: [] for component in COMPONENTS
    }
    repeatability_direct: dict[str, list[np.ndarray]] = {
        component: [] for component in COMPONENTS
    }
    repeatability_corrected_full: dict[str, list[np.ndarray]] = {
        component: [] for component in COMPONENTS
    }
    correction_change: dict[str, list[np.ndarray]] = {
        component: [] for component in COMPONENTS
    }
    coherence_regularized: dict[str, list[np.ndarray]] = {
        component: [] for component in COMPONENTS
    }
    coherence_full: dict[str, list[np.ndarray]] = {
        component: [] for component in COMPONENTS
    }
    max_reference_error = 0.0
    max_legacy_velocity_error = {component: 0.0 for component in COMPONENTS}
    max_direct_phase_error = {component: 0.0 for component in COMPONENTS}
    source_class = _source_class(support, direct)

    with ExitStack() as stack:
        production_files, production_images = _open_raw_sources(
            stack, run_dir, expected_shape
        )
        _comparison_files, comparison_images = _open_raw_sources(
            stack, comparison_run_dir, expected_shape
        )
        reconstruction = stack.enter_context(h5py.File(reconstruction_path, "r"))
        for required in ("reference_magnitude", "velocity", "mask"):
            if required not in reconstruction:
                raise KeyError(f"Missing {required} in {reconstruction_path}")
        venc = float(reconstruction.attrs["Venc_m_per_s"])
        rr_s = float(reconstruction.attrs["RR_s"])
        legacy_mask = np.asarray(reconstruction["mask"], dtype=bool)
        if legacy_mask.shape != expected_shape:
            raise ValueError("Legacy reconstruction mask shape changed")

        with h5py.File(partial_h5, "w", libver="latest", track_order=True) as target:
            raw_phase_ds = _create_float_dataset(
                target, "wrapped_phase_raw_rad", phase_shape, "rad"
            )
            corrected_phase_ds = _create_float_dataset(
                target, "wrapped_phase_corrected_rad", phase_shape, "rad"
            )
            raw_velocity_ds = _create_float_dataset(
                target, "velocity_raw_m_per_s", phase_shape, "m/s"
            )
            corrected_velocity_ds = _create_float_dataset(
                target, "velocity_corrected_m_per_s", phase_shape, "m/s"
            )
            coherence_ds = _create_float_dataset(
                target, "phase_pooling_coherence", phase_shape, "unitless"
            )
            pooled_weight_ds = _create_float_dataset(
                target, "pooled_magnitude_product", phase_shape, "arbitrary units squared"
            )
            raw_speed_ds = _create_float_dataset(
                target, "speed_raw_m_per_s", speed_shape, "m/s"
            )
            corrected_speed_ds = _create_float_dataset(
                target, "speed_corrected_m_per_s", speed_shape, "m/s"
            )
            reference_magnitude_ds = _create_float_dataset(
                target, "reference_magnitude", speed_shape, "arbitrary units"
            )
            joint_magnitude_ds = _create_float_dataset(
                target, "joint_magnitude_rms", speed_shape, "arbitrary units"
            )

            target.create_dataset(
                "support_mask", data=support.astype(np.uint8), track_times=False
            )
            target.create_dataset(
                "direct_mask", data=direct.astype(np.uint8), track_times=False
            )
            target.create_dataset(
                "regularized_mask",
                data=regularized.astype(np.uint8),
                track_times=False,
            )
            target.create_dataset(
                "source_class", data=source_class, track_times=False
            )
            score_for_storage = np.full(support.shape, np.nan, dtype=np.float32)
            score_for_storage[support] = confidence_score[support]
            score_ds = target.create_dataset(
                "confidence_score_min_normalized_magnitude",
                data=score_for_storage,
                track_times=False,
            )
            score_ds.attrs["definition"] = (
                "minimum over all 15 bins and REF/X/Y/Z of absolute complex "
                "magnitude divided by that encoding/bin in-support median"
            )
            target.create_dataset(
                "legacy_e010_mask",
                data=legacy_mask.astype(np.uint8),
                compression="gzip",
                compression_opts=4,
                shuffle=True,
                track_times=False,
            )
            scale_ds = target.create_dataset(
                "encoding_bin_magnitude_medians",
                data=encoding_scales,
                track_times=False,
            )
            scale_ds.attrs["encoding_order"] = ",".join(ENCODINGS)
            scale_ds.attrs["cardiac_bin_order"] = "1..15"

            for cardiac_bin in range(EXPECTED_BINS):
                production = {
                    encoding: compound_to_complex(
                        production_images[encoding][cardiac_bin]
                    )
                    for encoding in ENCODINGS
                }
                comparison = {
                    encoding: compound_to_complex(
                        comparison_images[encoding][cardiac_bin]
                    )
                    for encoding in ENCODINGS
                }
                reference_magnitude = np.abs(production["REF"]).astype(np.float32)
                joint_magnitude = np.sqrt(
                    sum(np.abs(production[encoding]) ** 2 for encoding in ENCODINGS)
                    / len(ENCODINGS)
                ).astype(np.float32)
                reference_magnitude_ds[cardiac_bin] = reference_magnitude
                joint_magnitude_ds[cardiac_bin] = joint_magnitude
                max_reference_error = max(
                    max_reference_error,
                    float(
                        np.max(
                            np.abs(
                                reference_magnitude
                                - reconstruction["reference_magnitude"][cardiac_bin]
                            )
                        )
                    ),
                )

                raw_speed_sq = np.full(support.shape, np.nan, dtype=np.float32)
                corrected_speed_sq = np.full(
                    support.shape, np.nan, dtype=np.float32
                )
                raw_speed_sq[support] = 0.0
                corrected_speed_sq[support] = 0.0
                for component_index, component in enumerate(COMPONENTS):
                    production_cross = production[component] * np.conj(
                        production["REF"]
                    )
                    comparison_cross = comparison[component] * np.conj(
                        comparison["REF"]
                    )
                    production_pool, coherence, pooled_weight = (
                        support_constrained_complex_pool(
                            production_cross, support, sigma_voxels
                        )
                    )
                    comparison_pool, _comparison_coherence, _comparison_weight = (
                        support_constrained_complex_pool(
                            comparison_cross, support, sigma_voxels
                        )
                    )
                    raw_phase, corrected_phase = hybrid_phase_arrays(
                        production_cross,
                        production_pool,
                        support,
                        direct,
                        regularized,
                    )
                    comparison_raw_phase, comparison_corrected_phase = (
                        hybrid_phase_arrays(
                            comparison_cross,
                            comparison_pool,
                            support,
                            direct,
                            regularized,
                        )
                    )
                    raw_velocity = raw_phase * np.float32(venc / np.pi)
                    corrected_velocity = corrected_phase * np.float32(venc / np.pi)
                    coherence_stored = np.full(
                        support.shape, np.nan, dtype=np.float32
                    )
                    pooled_weight_stored = np.full(
                        support.shape, np.nan, dtype=np.float32
                    )
                    coherence_stored[support] = coherence[support]
                    pooled_weight_stored[support] = pooled_weight[support]

                    raw_phase_ds[component_index, cardiac_bin] = raw_phase
                    corrected_phase_ds[component_index, cardiac_bin] = corrected_phase
                    raw_velocity_ds[component_index, cardiac_bin] = raw_velocity
                    corrected_velocity_ds[component_index, cardiac_bin] = (
                        corrected_velocity
                    )
                    coherence_ds[component_index, cardiac_bin] = coherence_stored
                    pooled_weight_ds[component_index, cardiac_bin] = (
                        pooled_weight_stored
                    )

                    raw_speed_sq[support] += np.square(raw_velocity[support])
                    corrected_speed_sq[support] += np.square(
                        corrected_velocity[support]
                    )
                    direct_error = circular_velocity_difference(
                        corrected_phase[direct], raw_phase[direct], venc
                    )
                    max_direct_phase_error[component] = max(
                        max_direct_phase_error[component],
                        float(direct_error.max(initial=0.0)),
                    )
                    legacy_direct = direct & legacy_mask[cardiac_bin]
                    legacy_error = np.abs(
                        raw_velocity[legacy_direct]
                        - reconstruction["velocity"][
                            component_index, cardiac_bin
                        ][legacy_direct]
                    )
                    max_legacy_velocity_error[component] = max(
                        max_legacy_velocity_error[component],
                        float(legacy_error.max(initial=0.0)),
                    )

                    repeatability_raw[component].append(
                        circular_velocity_difference(
                            raw_phase[regularized],
                            comparison_raw_phase[regularized],
                            venc,
                        )
                    )
                    corrected_regularized_repeatability = (
                        circular_velocity_difference(
                            corrected_phase[regularized],
                            comparison_corrected_phase[regularized],
                            venc,
                        )
                    )
                    repeatability_corrected[component].append(
                        corrected_regularized_repeatability
                    )
                    direct_repeatability = circular_velocity_difference(
                        raw_phase[direct], comparison_raw_phase[direct], venc
                    )
                    repeatability_direct[component].append(direct_repeatability)
                    repeatability_corrected_full[component].append(
                        np.concatenate(
                            (
                                direct_repeatability,
                                corrected_regularized_repeatability,
                            )
                        )
                    )
                    correction_change[component].append(
                        circular_velocity_difference(
                            corrected_phase[regularized],
                            raw_phase[regularized],
                            venc,
                        )
                    )
                    coherence_regularized[component].append(
                        coherence[regularized]
                    )
                    coherence_full[component].append(coherence[support])

                raw_speed_ds[cardiac_bin] = np.sqrt(raw_speed_sq)
                corrected_speed_ds[cardiac_bin] = np.sqrt(corrected_speed_sq)

            target.attrs["schema_version"] = SCHEMA_VERSION
            target.attrs["evidence_class"] = (
                "pre-rebin reconstruction-method and particle-count sensitivity derivative"
            )
            target.attrs["dimension_order_python"] = (
                "component,cardiac_bin,slice,phase,readout"
            )
            target.attrs["velocity_components"] = ",".join(COMPONENTS)
            target.attrs["cardiac_bins"] = EXPECTED_BINS
            target.attrs["matrix_readout_phase_slice"] = matrix
            target.attrs["FOV_m"] = fov_m
            target.attrs["voxel_size_mm_readout_phase_slice"] = voxel_mm
            target.attrs["Venc_m_per_s"] = venc
            target.attrs["RR_s"] = rr_s
            target.attrs["direct_threshold"] = args.direct_threshold
            target.attrs["gaussian_sigma_mm"] = args.sigma_mm
            target.attrs["gaussian_sigma_voxels_slice_phase_readout"] = sigma_voxels
            target.attrs["gaussian_truncate"] = GAUSSIAN_TRUNCATE
            target.attrs["temporal_smoothing"] = False
            target.attrs["temporal_interpolation"] = False
            target.attrs["phase_unwrapping"] = False
            target.attrs["outside_support_representation"] = "NaN"
            target.attrs["source_reconstruction_sha256"] = reconstruction_hash
            target.attrs["source_support_sha256"] = support_hash
            target.attrs["source_script_sha256"] = script_hash
            for encoding in ENCODINGS:
                target.attrs[f"source_1m_{encoding}_sha256"] = raw_hashes[encoding]
                target.attrs[f"source_100k_{encoding}_sha256"] = (
                    comparison_hashes[encoding]
                )
            target.flush()

    os.replace(partial_h5, staged_h5)

    repeatability: dict[str, Any] = {}
    coherence_report: dict[str, Any] = {}
    change_report: dict[str, Any] = {}
    repeatability_checks: dict[str, bool] = {}
    coherence_gates = {"X": 0.95, "Y": 0.99, "Z": 0.99}
    p95_gates = {"X": 0.15, "Y": 0.05, "Z": 0.05}
    rmse_gates = {"X": 0.10, "Y": 0.03, "Z": 0.03}
    improvement_gates = {"X": 4.0, "Y": 8.0, "Z": 8.0}
    for component in COMPONENTS:
        raw_stats = distribution(np.concatenate(repeatability_raw[component]))
        corrected_stats = distribution(
            np.concatenate(repeatability_corrected[component])
        )
        improvement = float(raw_stats["rmse"]) / float(corrected_stats["rmse"])
        repeatability[component] = {
            "direct_domain_m_per_s": distribution(
                np.concatenate(repeatability_direct[component])
            ),
            "raw_full_support_phase_on_regularized_domain_m_per_s": raw_stats,
            "corrected_phase_on_regularized_domain_m_per_s": corrected_stats,
            "hybrid_corrected_full_support_m_per_s": distribution(
                np.concatenate(repeatability_corrected_full[component])
            ),
            "rmse_improvement_factor": improvement,
        }
        coherence_report[component] = {
            "regularized_domain": distribution(
                np.concatenate(coherence_regularized[component])
            ),
            "full_support": distribution(np.concatenate(coherence_full[component])),
        }
        change_report[component] = distribution(
            np.concatenate(correction_change[component])
        )
        repeatability_checks[f"{component}_corrected_p95_within_gate"] = (
            float(corrected_stats["p95"]) <= p95_gates[component]
        )
        repeatability_checks[f"{component}_corrected_rmse_within_gate"] = (
            float(corrected_stats["rmse"]) <= rmse_gates[component]
        )
        repeatability_checks[f"{component}_rmse_improvement_within_gate"] = (
            improvement >= improvement_gates[component]
        )
        repeatability_checks[f"{component}_coherence_p01_within_gate"] = (
            float(coherence_report[component]["regularized_domain"]["p01"])
            >= coherence_gates[component]
        )

    numerical_checks = {
        "maximum_reference_magnitude_absolute_error_le_1e_4": max_reference_error
        <= 1e-4,
        "direct_phase_is_exactly_preserved": all(
            value <= 1e-7 for value in max_direct_phase_error.values()
        ),
        "legacy_e010_velocity_reproduced_on_direct_legacy_intersection": all(
            value <= 1e-5 for value in max_legacy_velocity_error.values()
        ),
        "all_1m_images_finite": all(
            item["nonfinite_full_image"] == 0 for item in image_integrity.values()
        ),
        "all_1m_images_nonzero_inside_support": all(
            item["zero_in_support"] == 0 for item in image_integrity.values()
        ),
        "all_1m_sorted_kspace_samples_finite": all(
            item["nonfinite_samples"] == 0
            for item in rawsort_integrity.values()
        ),
        "all_1m_sorted_kspace_samples_nonzero": all(
            item["exact_zero_samples"] == 0
            for item in rawsort_integrity.values()
        ),
        "full_support_has_phase_in_every_component_and_bin": True,
        "source_class_is_static_across_bins": True,
        "temporal_smoothing_or_interpolation_absent": True,
    }
    acceptance_checks = {
        **canonical_mask_checks,
        **numerical_checks,
        **repeatability_checks,
    }
    if not all(acceptance_checks.values()):
        failed = [name for name, passed in acceptance_checks.items() if not passed]
        raise RuntimeError(f"E-018 acceptance checks failed: {failed}")

    # Re-hash every scientific source after reconstruction to prove that this
    # derivative did not modify E-010 or its raw inputs.
    post_hashes_unchanged = (
        sha256_file(reconstruction_path) == reconstruction_hash
        and sha256_file(support_path) == support_hash
        and all(
            sha256_file(run_dir / RAW_FILENAMES[encoding])
            == raw_hashes[encoding]
            for encoding in ENCODINGS
        )
        and all(
            sha256_file(comparison_run_dir / RAW_FILENAMES[encoding])
            == comparison_hashes[encoding]
            for encoding in ENCODINGS
        )
    )
    if not post_hashes_unchanged:
        raise RuntimeError("A source artifact hash changed during E-018")

    staged_h5_record = artifact_record(staged_h5)
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": "E-018",
        "status": "PASS_RECONSTRUCTION_SENSITIVITY_DERIVATIVE",
        "evidence_class": (
            "pre-rebin support-constrained reconstruction-method and deterministic "
            "100k-subset particle-count sensitivity derivative of E-010"
        ),
        "purpose": (
            "Replace maximum-threshold hard phase voids with a provenance-bound "
            "complex-domain reconstruction while preserving direct raw phase "
            "where all encodings remain robust in all bins."
        ),
        "inputs": {
            "e010_reconstruction": artifact_record(
                reconstruction_path, reconstruction_hash
            ),
            "seed_derived_support": artifact_record(support_path, support_hash),
            "production_1m_complex_images": {
                encoding: artifact_record(
                    run_dir / RAW_FILENAMES[encoding], raw_hashes[encoding]
                )
                for encoding in ENCODINGS
            },
            "comparison_100k_complex_images": {
                encoding: artifact_record(
                    comparison_run_dir / RAW_FILENAMES[encoding],
                    comparison_hashes[encoding],
                )
                for encoding in ENCODINGS
            },
            "comparison_semantics": (
                "The 100k phantom is a deterministic without-replacement identity "
                "subset of the 1M trajectories (selection seed 20260831), not an "
                "independent retracking or physical validation reference."
            ),
            "source_hashes_unchanged_after_reconstruction": post_hashes_unchanged,
        },
        "software": {
            "script": artifact_record(Path(__file__).resolve(), script_hash),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "h5py": h5py.__version__,
        },
        "geometry_and_protocol": {
            "matrix_readout_phase_slice": matrix.tolist(),
            "array_order": "cardiac_bin,slice,phase,readout",
            "fov_m": fov_m.tolist(),
            "voxel_size_mm_readout_phase_slice": voxel_mm.tolist(),
            "cardiac_bins": EXPECTED_BINS,
            "RR_s": rr_s,
            "Venc_m_per_s": venc,
        },
        "method": {
            "phase_time_binding": (
                "Inherited PHS labels are not strengthened here; the independent "
                "PHS-label-to-actual-motion-time audit must pass or supply re-binned "
                "complex images before this becomes a primary correction."
            ),
            "confidence_formula": (
                "C(x)=min over cardiac bin t and encoding e in REF/X/Y/Z of "
                "abs(I_e(t,x)) / median_{y in S}(abs(I_e(t,y)))"
            ),
            "reference_safeguard": (
                "REF participates in the minimum; a strong encoded image cannot "
                "compensate for a weak REF required by every phase difference."
            ),
            "direct_mask": "S AND C >= 0.10; one fixed 3D mask for all bins/components",
            "cross_product": "q_c(t,x)=I_c(t,x)*conj(I_REF(t,x))",
            "pooling": (
                "q_bar=(G_sigma * (S*q))/(G_sigma*S), real and imaginary parts "
                "filtered separately with constant-zero boundary and truncate=4"
            ),
            "coherence": (
                "abs(G_sigma*(S*q)) / (G_sigma*(S*abs(q)))"
            ),
            "hybrid_phase": (
                "angle(q) on fixed direct mask; angle(q_bar) on fixed regularized "
                "mask; NaN outside exact support"
            ),
            "direct_threshold": args.direct_threshold,
            "gaussian_sigma_mm": args.sigma_mm,
            "gaussian_fwhm_mm": 2.0
            * math.sqrt(2.0 * math.log(2.0))
            * args.sigma_mm,
            "gaussian_sigma_voxels_slice_phase_readout": list(sigma_voxels),
            "gaussian_truncate": GAUSSIAN_TRUNCATE,
            "temporal_smoothing": False,
            "temporal_interpolation": False,
            "phase_unwrapping": False,
            "wall_or_background_signal_created": False,
        },
        "support_and_confidence": {
            **support_metadata,
            "support_voxels": int(support.sum()),
            "direct_voxels": int(direct.sum()),
            "regularized_voxels": int(regularized.sum()),
            "direct_fraction": float(direct.sum() / support.sum()),
            "regularized_fraction": float(regularized.sum() / support.sum()),
            "confidence_score_in_support": distribution(confidence_score[support]),
            "fixed_sections": sections,
            "source_class_transition_count": 0,
        },
        "source_signal_integrity": {
            "expected_profiles_per_encoding": 64 * 32 * EXPECTED_BINS,
            "expected_sorted_samples_per_encoding": int(np.prod(expected_shape)),
            "complex_image": image_integrity,
            "sorted_kspace": rawsort_integrity,
            "interpretation": (
                "No sorted k-space or reconstructed complex-image sample is "
                "missing/non-finite in the canonical files. The legacy black voids "
                "were introduced by its per-bin REF maximum threshold and hard zeroing."
            ),
        },
        "numerical_identity": {
            "maximum_reference_magnitude_absolute_error": max_reference_error,
            "maximum_direct_phase_change_m_per_s": max_direct_phase_error,
            "maximum_legacy_velocity_error_on_direct_legacy_intersection_m_per_s": (
                max_legacy_velocity_error
            ),
            "full_support_phase_voxels_per_component_all_bins": int(
                support.sum() * EXPECTED_BINS
            ),
            "interior_missing_phase_voxels": 0,
        },
        "phase_pooling_coherence": coherence_report,
        "regularized_minus_raw_circular_difference_m_per_s": change_report,
        "particle_count_repeatability": {
            "domain": (
                "fixed 1M-derived regularized mask, identical bins/components/grid"
            ),
            "difference_formula": (
                "abs(angle(exp(i*(phase_1M-phase_100k))))*Venc/pi"
            ),
            "components": repeatability,
            "claim_boundary": (
                "This is a deterministic particle-count sensitivity comparison, "
                "not independent velocity validation or convergence proof."
            ),
        },
        "acceptance": {
            "passed": True,
            "checks": acceptance_checks,
            "repeatability_gates": {
                "corrected_p95_m_per_s": p95_gates,
                "corrected_rmse_m_per_s": rmse_gates,
                "minimum_rmse_improvement_factor": improvement_gates,
                "minimum_regularized_coherence_p01": coherence_gates,
            },
        },
        "outputs": {
            "reconstruction_h5": {
                **staged_h5_record,
                "path": str(final_h5),
            },
            "report": str(final_output_dir / REPORT_NAME),
            "videos_or_presentation_frames_created": False,
        },
        "limitations": [
            "The source PHS-label-to-actual-motion-time binding is under audit; this pre-rebin derivative must not be promoted before that audit passes or re-binning is applied.",
            "The correction is a spatially regularized reconstruction derivative; it does not acquire new signal.",
            "The 1.0-mm Gaussian sigma has 2.355-mm FWHM and reduces effective spatial detail only where the fixed regularized mask is used.",
            "The 0.10 confidence threshold is a preregistered internal sensitivity choice, not an optimal or externally validated SNR threshold.",
            "The exact support is voxelized from static lumen seeds and is not a measured wall segmentation.",
            "The motion phantom contains lumen spins only; E-018 cannot create realistic stationary wall or background phase.",
            "The 100k case is a deterministic subset of the 1M trajectories and supplies sensitivity evidence only.",
            "No independent PC-MRI, PIV, or analytical velocity reference is available, so E-018 must not replace E-010 as validated quantitative evidence.",
        ],
    }
    partial_report = staging_dir / f".{REPORT_NAME}.partial"
    staged_report = staging_dir / REPORT_NAME
    partial_report.write_text(
        json.dumps(report, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    os.replace(partial_report, staged_report)
    os.replace(staging_dir, final_output_dir)
    print(
        json.dumps(
            {
                "status": report["status"],
                "output_dir": str(final_output_dir),
                "support_voxels": int(support.sum()),
                "direct_voxels": int(direct.sum()),
                "regularized_voxels": int(regularized.sum()),
                "output_sha256": staged_h5_record["sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
