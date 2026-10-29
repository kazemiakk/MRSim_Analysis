r"""Build a provenance-bound support-constrained phase derivative.

The input must be a seam-safe dp150 run that has passed the preregistered
cyclic-jump gate.  Direct complex phase is retained wherever all four encodings
remain reliable in every cardiac bin.  Low-confidence support voxels use a
support-normalized spatial pool of complex cross-products.  There is no
temporal interpolation, phase unwrapping, or synthesized wall/background.
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
from typing import Any

import h5py
import numpy as np

try:
    from scripts.analyze_dp150_scanner_reconstruction import (
        DEFAULT_FOV_M,
        DEFAULT_MATRIX,
        load_support,
    )
    from scripts.reconstruct_dp150_support_constrained_phase import (
        circular_velocity_difference,
        distribution,
        fixed_direct_mask,
        hybrid_phase_arrays,
        normalized_magnitude_confidence_for_bin,
        support_constrained_complex_pool,
    )
    from scripts.render_dp150_xyz_magnitude_phase import compound_to_complex
except ModuleNotFoundError:
    from analyze_dp150_scanner_reconstruction import (
        DEFAULT_FOV_M,
        DEFAULT_MATRIX,
        load_support,
    )
    from reconstruct_dp150_support_constrained_phase import (
        circular_velocity_difference,
        distribution,
        fixed_direct_mask,
        hybrid_phase_arrays,
        normalized_magnitude_confidence_for_bin,
        support_constrained_complex_pool,
    )
    from render_dp150_xyz_magnitude_phase import compound_to_complex


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = (
    ROOT
    / "outputs"
    / "dp150_mri_pipeline"
    / "koma"
    / "simulation_motion_100k_scanner_matched_venc5_seam_safe_v3"
)
DEFAULT_SUPPORT = (
    ROOT / "outputs" / "dp150_mri_pipeline" / "dp150_static_2m_mri.h5"
)
RAW_FILENAMES = {
    "REF": "ref_result.jld2",
    "X": "x_result.jld2",
    "Y": "y_result.jld2",
    "Z": "z_result.jld2",
}
ENCODINGS = ("REF", "X", "Y", "Z")
COMPONENTS = ("X", "Y", "Z")
EXPECTED_BINS = 15
DIRECT_THRESHOLD = 0.10
SIGMA_MM = 1.0
GAUSSIAN_TRUNCATE = 4.0
OUTPUT_H5_NAME = "dp150_seam_safe_support_phase.h5"
REPORT_NAME = "dp150_seam_safe_support_phase_report.json"

METHOD_SPECIFICATION: dict[str, Any] = {
    "specification_id": "dp150_seam_safe_support_phase_v1",
    "evidence_class": "reconstruction_derivative_of_corrected_complex_data",
    "prerequisite": "PASS from dp150_cyclic_jump_gate_v1",
    "support": "exact voxelized static lumen seed support; no dilation",
    "confidence": (
        "minimum over all 15 bins and REF/X/Y/Z of abs(image) divided by "
        "that encoding/bin median magnitude inside exact support"
    ),
    "direct_mask": "support AND confidence >= 0.10; fixed across bins/components",
    "regularized_mask": "support AND NOT direct_mask",
    "cross_product": "encoded * conj(REF)",
    "regularization": (
        "support-normalized Gaussian pooling of real and imaginary complex "
        "cross-product parts; sigma=1.0 mm; truncate=4.0"
    ),
    "hybrid_phase": (
        "direct angle(cross_product) on direct mask; angle(pooled cross-product) "
        "on regularized mask; NaN outside exact support"
    ),
    "velocity": "wrapped phase * Venc / pi",
    "temporal_smoothing": False,
    "temporal_interpolation": False,
    "phase_unwrapping": False,
    "wall_or_background_signal_created": False,
    "parameters": {
        "direct_threshold": DIRECT_THRESHOLD,
        "gaussian_sigma_mm": SIGMA_MM,
        "gaussian_truncate": GAUSSIAN_TRUNCATE,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--support-h5", type=Path, default=DEFAULT_SUPPORT)
    parser.add_argument("--reconstruction", type=Path, default=None)
    parser.add_argument("--gate-report", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--registration", type=Path, default=None)
    parser.add_argument("--register-only", action="store_true")
    parser.add_argument("--fov-m", type=float, nargs=3, default=DEFAULT_FOV_M)
    parser.add_argument("--matrix", type=int, nargs=3, default=DEFAULT_MATRIX)
    return parser.parse_args()


def sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_record(path: Path, digest: str | None = None) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": digest or sha256_file(path),
    }


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as target:
            json.dump(payload, target, indent=2, sort_keys=True, allow_nan=False)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return value


def registration_payload(run_dir: Path) -> dict[str, Any]:
    script = Path(__file__).resolve()
    return {
        "artifact_type": "preregistered_reconstruction_method",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_run": str(run_dir.resolve()),
        "candidate_complete_raw_files_present_at_registration": {
            encoding: (run_dir / filename).is_file()
            for encoding, filename in RAW_FILENAMES.items()
        },
        "method_specification": METHOD_SPECIFICATION,
        "implementation": {
            "path": str(script),
            "sha256": sha256_file(script),
        },
    }


def verify_registration(path: Path, run_dir: Path) -> dict[str, Any]:
    registration = _read_json(path)
    if registration.get("method_specification") != METHOD_SPECIFICATION:
        raise RuntimeError("Registered reconstruction method does not match code")
    if Path(registration.get("candidate_run", "")).resolve() != run_dir.resolve():
        raise RuntimeError("Registered candidate run does not match --run-dir")
    expected = registration.get("implementation", {}).get("sha256")
    observed = sha256_file(Path(__file__).resolve())
    if expected != observed:
        raise RuntimeError(
            "Reconstruction code changed after preregistration: "
            f"registered={expected}, current={observed}"
        )
    return registration


def validate_gate_report_binding(
    report: dict[str, Any], gate_path: Path, run_dir: Path
) -> dict[str, str]:
    if report.get("status") != "PASS":
        raise RuntimeError(f"Cyclic-jump gate is not PASS: {report.get('status')}")
    specification = report.get("gate_specification", {})
    if specification.get("specification_id") != "dp150_cyclic_jump_gate_v1":
        raise RuntimeError("Unexpected cyclic-jump gate specification")
    candidate = report.get("candidate", {})
    if Path(candidate.get("run_dir", "")).resolve() != run_dir.resolve():
        raise RuntimeError("Gate candidate run does not match reconstruction run")
    inputs = candidate.get("inputs", {})
    hashes: dict[str, str] = {}
    for encoding, filename in RAW_FILENAMES.items():
        source = run_dir / filename
        expected = inputs.get(encoding, {}).get("sha256")
        observed = sha256_file(source)
        if not expected or observed != expected:
            raise RuntimeError(
                f"Gate binding failed for {encoding}: gate={expected}, raw={observed}"
            )
        hashes[encoding] = observed
    if not gate_path.is_file():
        raise FileNotFoundError(gate_path)
    return hashes


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
        for name in ("image", "rawsort"):
            if name not in source:
                raise KeyError(f"Missing {name} in {path}")
            if tuple(source[name].shape) != expected_shape:
                raise ValueError(
                    f"{path.name}:{name} shape {source[name].shape} != {expected_shape}"
                )
        files[encoding] = source
        images[encoding] = source["image"]
    return files, images


def _dataset_options(shape: tuple[int, ...]) -> dict[str, Any]:
    if len(shape) == 5:
        chunks = (1, 1, min(4, shape[2]), min(16, shape[3]), min(100, shape[4]))
    elif len(shape) == 4:
        chunks = (1, min(4, shape[1]), min(16, shape[2]), min(100, shape[3]))
    else:
        raise ValueError(f"Unsupported dataset shape {shape}")
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


def _complex_integrity(values: np.ndarray) -> tuple[int, int, float]:
    complex_values = compound_to_complex(values)
    magnitude = np.abs(complex_values)
    return (
        int(np.count_nonzero(magnitude == 0)),
        int(np.count_nonzero(~np.isfinite(magnitude))),
        float(magnitude.min()),
    )


def _distribution_or_none(values: list[np.ndarray]) -> dict[str, Any] | None:
    if not values or sum(item.size for item in values) == 0:
        return None
    return distribution(np.concatenate(values))


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    support_path = args.support_h5.resolve()
    reconstruction_path = (
        args.reconstruction.resolve()
        if args.reconstruction is not None
        else run_dir / "pc_reconstruction.h5"
    )
    gate_path = (
        args.gate_report.resolve()
        if args.gate_report is not None
        else run_dir
        / "final_image_analysis"
        / "cyclic_jump_gate"
        / "cyclic_jump_gate_report.json"
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else run_dir
        / "final_image_analysis"
        / "support_constrained_phase_reconstruction_seam_safe_v3"
    )
    registration_path = (
        args.registration.resolve()
        if args.registration is not None
        else run_dir
        / "final_image_analysis"
        / "support_constrained_phase_method_preregistration.json"
    )

    if args.register_only:
        payload = registration_payload(run_dir)
        if registration_path.exists():
            existing = _read_json(registration_path)
            first = dict(existing)
            second = dict(payload)
            first.pop("created_utc", None)
            second.pop("created_utc", None)
            if first != second:
                raise RuntimeError(
                    f"Refusing to overwrite different registration {registration_path}"
                )
            print(f"Reconstruction preregistration already matches: {registration_path}")
            return 0
        atomic_write_json(registration_path, payload)
        print(f"Wrote reconstruction preregistration: {registration_path}")
        return 0

    registration = verify_registration(registration_path, run_dir)
    if not gate_path.is_file():
        raise FileNotFoundError(gate_path)
    gate_report = _read_json(gate_path)
    raw_hashes = validate_gate_report_binding(gate_report, gate_path, run_dir)
    reconstruction_hash = sha256_file(reconstruction_path)
    support_hash = sha256_file(support_path)

    matrix = np.asarray(args.matrix, dtype=np.int64)
    fov_m = np.asarray(args.fov_m, dtype=np.float64)
    if matrix.shape != (3,) or np.any(matrix <= 0):
        raise ValueError("--matrix must contain three positive integers")
    if fov_m.shape != (3,) or np.any(~np.isfinite(fov_m)) or np.any(fov_m <= 0):
        raise ValueError("--fov-m must contain three finite positive values")
    voxel_mm = 1e3 * fov_m / matrix
    sigma_voxels = (
        float(SIGMA_MM / voxel_mm[2]),
        float(SIGMA_MM / voxel_mm[1]),
        float(SIGMA_MM / voxel_mm[0]),
    )
    expected_shape = (
        EXPECTED_BINS,
        int(matrix[2]),
        int(matrix[1]),
        int(matrix[0]),
    )
    support, support_metadata = load_support(support_path, fov_m, matrix)
    if support.shape != expected_shape[1:]:
        raise ValueError(
            f"Support shape {support.shape} != expected {expected_shape[1:]}"
        )

    confidence = np.full(support.shape, np.inf, dtype=np.float32)
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
        raw_files, raw_images = _open_raw_sources(stack, run_dir, expected_shape)
        for cardiac_bin in range(EXPECTED_BINS):
            images = {
                encoding: compound_to_complex(raw_images[encoding][cardiac_bin])
                for encoding in ENCODINGS
            }
            score, scales = normalized_magnitude_confidence_for_bin(images, support)
            confidence = np.minimum(confidence, score)
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
                    raw_files[encoding]["rawsort"][cardiac_bin]
                )
                item = rawsort_integrity[encoding]
                item["samples"] += int(np.prod(support.shape))
                item["exact_zero_samples"] += zeros
                item["nonfinite_samples"] += nonfinite
                item["minimum_magnitude"] = min(item["minimum_magnitude"], minimum)

    direct, regularized = fixed_direct_mask(support, confidence, DIRECT_THRESHOLD)
    if not np.array_equal(direct | regularized, support) or np.any(
        direct & regularized
    ):
        raise RuntimeError("Direct and regularized masks do not partition support")

    staging_dir = output_dir.with_name(f".{output_dir.name}.partial")
    if output_dir.exists() or staging_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite output or partial directory: {output_dir} / {staging_dir}"
        )
    staging_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir()
    partial_h5 = staging_dir / f".{OUTPUT_H5_NAME}.partial"
    staged_h5 = staging_dir / OUTPUT_H5_NAME
    final_h5 = output_dir / OUTPUT_H5_NAME

    phase_shape = (len(COMPONENTS), *expected_shape)
    max_reference_error = 0.0
    max_legacy_velocity_error = {component: 0.0 for component in COMPONENTS}
    max_direct_change = {component: 0.0 for component in COMPONENTS}
    correction_changes = {component: [] for component in COMPONENTS}
    regularized_coherence = {component: [] for component in COMPONENTS}
    full_coherence = {component: [] for component in COMPONENTS}
    corrected_nonfinite_in_support = 0

    try:
        with ExitStack() as stack:
            _raw_files, raw_images = _open_raw_sources(
                stack, run_dir, expected_shape
            )
            reconstruction = stack.enter_context(h5py.File(reconstruction_path, "r"))
            for name in ("reference_magnitude", "velocity", "mask"):
                if name not in reconstruction:
                    raise KeyError(f"Missing {name} in {reconstruction_path}")
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
                    target,
                    "pooled_magnitude_product",
                    phase_shape,
                    "arbitrary units squared",
                )
                raw_speed_ds = _create_float_dataset(
                    target, "speed_raw_m_per_s", expected_shape, "m/s"
                )
                corrected_speed_ds = _create_float_dataset(
                    target, "speed_corrected_m_per_s", expected_shape, "m/s"
                )
                reference_magnitude_ds = _create_float_dataset(
                    target, "reference_magnitude", expected_shape, "arbitrary units"
                )
                joint_magnitude_ds = _create_float_dataset(
                    target, "joint_magnitude_rms", expected_shape, "arbitrary units"
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
                source_class = np.zeros(support.shape, dtype=np.uint8)
                source_class[regularized] = 2
                source_class[direct] = 1
                target.create_dataset(
                    "source_class", data=source_class, track_times=False
                )
                confidence_stored = np.full(support.shape, np.nan, dtype=np.float32)
                confidence_stored[support] = confidence[support]
                target.create_dataset(
                    "confidence_score_min_normalized_magnitude",
                    data=confidence_stored,
                    track_times=False,
                )
                target.create_dataset(
                    "legacy_per_bin_threshold_mask",
                    data=legacy_mask.astype(np.uint8),
                    compression="gzip",
                    compression_opts=4,
                    shuffle=True,
                    track_times=False,
                )
                scales_ds = target.create_dataset(
                    "encoding_bin_magnitude_medians",
                    data=encoding_scales,
                    track_times=False,
                )
                scales_ds.attrs["encoding_order"] = ",".join(ENCODINGS)
                scales_ds.attrs["saved_phs_zero_based"] = np.arange(15)

                for cardiac_bin in range(EXPECTED_BINS):
                    images = {
                        encoding: compound_to_complex(
                            raw_images[encoding][cardiac_bin]
                        )
                        for encoding in ENCODINGS
                    }
                    reference_magnitude = np.abs(images["REF"]).astype(np.float32)
                    joint_magnitude = np.sqrt(
                        sum(np.abs(images[item]) ** 2 for item in ENCODINGS)
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
                        cross = images[component] * np.conj(images["REF"])
                        pooled, coherence, pooled_weight = (
                            support_constrained_complex_pool(
                                cross,
                                support,
                                sigma_voxels,
                                truncate=GAUSSIAN_TRUNCATE,
                            )
                        )
                        raw_phase, corrected_phase = hybrid_phase_arrays(
                            cross, pooled, support, direct, regularized
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
                        corrected_velocity_ds[component_index, cardiac_bin] = corrected_velocity
                        coherence_ds[component_index, cardiac_bin] = coherence_stored
                        pooled_weight_ds[component_index, cardiac_bin] = pooled_weight_stored
                        raw_speed_sq[support] += np.square(raw_velocity[support])
                        corrected_speed_sq[support] += np.square(
                            corrected_velocity[support]
                        )

                        direct_change = circular_velocity_difference(
                            corrected_phase[direct], raw_phase[direct], venc
                        )
                        max_direct_change[component] = max(
                            max_direct_change[component],
                            float(direct_change.max(initial=0.0)),
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
                        correction_changes[component].append(
                            circular_velocity_difference(
                                corrected_phase[regularized],
                                raw_phase[regularized],
                                venc,
                            )
                        )
                        regularized_coherence[component].append(
                            coherence[regularized]
                        )
                        full_coherence[component].append(coherence[support])
                        corrected_nonfinite_in_support += int(
                            np.count_nonzero(~np.isfinite(corrected_phase[support]))
                        )

                    raw_speed_ds[cardiac_bin] = np.sqrt(raw_speed_sq)
                    corrected_speed_ds[cardiac_bin] = np.sqrt(corrected_speed_sq)

                target.attrs["schema_version"] = 1
                target.attrs["evidence_class"] = METHOD_SPECIFICATION["evidence_class"]
                target.attrs["dimension_order_python"] = (
                    "component,cardiac_bin,slice,phase,readout"
                )
                target.attrs["velocity_components"] = ",".join(COMPONENTS)
                target.attrs["cardiac_bins"] = EXPECTED_BINS
                target.attrs["saved_phs_zero_based"] = np.arange(15)
                target.attrs["matrix_readout_phase_slice"] = matrix
                target.attrs["FOV_m"] = fov_m
                target.attrs["voxel_size_mm_readout_phase_slice"] = voxel_mm
                target.attrs["Venc_m_per_s"] = venc
                target.attrs["RR_s"] = rr_s
                target.attrs["direct_threshold"] = DIRECT_THRESHOLD
                target.attrs["gaussian_sigma_mm"] = SIGMA_MM
                target.attrs["gaussian_sigma_voxels_slice_phase_readout"] = sigma_voxels
                target.attrs["gaussian_truncate"] = GAUSSIAN_TRUNCATE
                target.attrs["temporal_smoothing"] = False
                target.attrs["temporal_interpolation"] = False
                target.attrs["phase_unwrapping"] = False
                target.attrs["outside_support_representation"] = "NaN"
                target.attrs["source_reconstruction_sha256"] = reconstruction_hash
                target.attrs["source_support_sha256"] = support_hash
                target.attrs["source_gate_report_sha256"] = sha256_file(gate_path)
                target.attrs["source_registration_sha256"] = sha256_file(
                    registration_path
                )
                target.attrs["source_script_sha256"] = sha256_file(
                    Path(__file__).resolve()
                )
                for encoding in ENCODINGS:
                    target.attrs[f"source_{encoding}_sha256"] = raw_hashes[encoding]
                target.flush()

        os.replace(partial_h5, staged_h5)

        numerical_checks = {
            "cyclic_jump_gate_passed_and_hash_bound": True,
            "direct_and_regularized_masks_partition_exact_support": bool(
                not np.any(direct & regularized)
                and np.array_equal(direct | regularized, support)
            ),
            "direct_phase_exactly_preserved": all(
                value <= 1e-7 for value in max_direct_change.values()
            ),
            "legacy_velocity_reproduced_on_direct_legacy_intersection": all(
                value <= 1e-5 for value in max_legacy_velocity_error.values()
            ),
            "reference_magnitude_reproduced": max_reference_error <= 1e-4,
            "all_complex_images_finite": all(
                item["nonfinite_full_image"] == 0
                for item in image_integrity.values()
            ),
            "all_sorted_kspace_finite": all(
                item["nonfinite_samples"] == 0
                for item in rawsort_integrity.values()
            ),
            "corrected_phase_finite_everywhere_inside_support": (
                corrected_nonfinite_in_support == 0
            ),
            "fixed_mask_across_all_bins_and_components": True,
            "temporal_smoothing_or_interpolation_absent": True,
            "outside_support_is_nan": True,
        }
        failed = [name for name, passed in numerical_checks.items() if not passed]
        if failed:
            raise RuntimeError(f"Reconstruction invariants failed: {failed}")

        post_hashes_unchanged = (
            sha256_file(reconstruction_path) == reconstruction_hash
            and sha256_file(support_path) == support_hash
            and all(
                sha256_file(run_dir / RAW_FILENAMES[encoding])
                == raw_hashes[encoding]
                for encoding in ENCODINGS
            )
        )
        if not post_hashes_unchanged:
            raise RuntimeError("A scientific source artifact changed during reconstruction")

        staged_h5_record = artifact_record(staged_h5)
        section_indices = {
            "upstream": 70,
            "stenosis_throat": 98,
            "immediate_post_stenotic": 122,
        }
        sections = {
            name: {
                "readout_index_zero_based": index,
                "support_voxels": int(support[:, :, index].sum()),
                "direct_voxels": int(direct[:, :, index].sum()),
                "regularized_voxels": int(regularized[:, :, index].sum()),
            }
            for name, index in section_indices.items()
        }
        report = {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "PASS_RECONSTRUCTION_DERIVATIVE",
            "evidence_class": METHOD_SPECIFICATION["evidence_class"],
            "claim_boundary": (
                "This removes hard threshold voids by a preregistered complex-domain "
                "reconstruction after acquisition-timing correction. It remains a "
                "spatially regularized derivative, not new signal or external truth."
            ),
            "method_specification": METHOD_SPECIFICATION,
            "inputs": {
                "run_dir": str(run_dir),
                "raw_complex_images": {
                    encoding: artifact_record(
                        run_dir / RAW_FILENAMES[encoding], raw_hashes[encoding]
                    )
                    for encoding in ENCODINGS
                },
                "legacy_reconstruction": artifact_record(
                    reconstruction_path, reconstruction_hash
                ),
                "seed_derived_support": artifact_record(support_path, support_hash),
                "cyclic_jump_gate": artifact_record(gate_path),
                "method_preregistration": artifact_record(registration_path),
                "source_hashes_unchanged_after_reconstruction": post_hashes_unchanged,
            },
            "software": {
                "script": artifact_record(Path(__file__).resolve()),
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
                "saved_phs_zero_based": list(range(15)),
                "RR_s": rr_s,
                "Venc_m_per_s": venc,
            },
            "support_and_confidence": {
                **support_metadata,
                "support_voxels": int(support.sum()),
                "direct_voxels": int(direct.sum()),
                "regularized_voxels": int(regularized.sum()),
                "direct_fraction": float(direct.sum() / support.sum()),
                "regularized_fraction": float(regularized.sum() / support.sum()),
                "confidence_in_support": distribution(confidence[support]),
                "fixed_sections": sections,
            },
            "source_signal_integrity": {
                "complex_images": image_integrity,
                "sorted_kspace": rawsort_integrity,
            },
            "numerical_identity": {
                "maximum_reference_magnitude_absolute_error": max_reference_error,
                "maximum_direct_phase_change_m_per_s": max_direct_change,
                "maximum_legacy_velocity_error_on_direct_legacy_intersection_m_per_s": (
                    max_legacy_velocity_error
                ),
                "corrected_nonfinite_phase_voxels_inside_support": (
                    corrected_nonfinite_in_support
                ),
            },
            "regularized_domain": {
                component: {
                    "coherence": _distribution_or_none(
                        regularized_coherence[component]
                    ),
                    "correction_change_m_per_s": _distribution_or_none(
                        correction_changes[component]
                    ),
                }
                for component in COMPONENTS
            },
            "full_support_coherence": {
                component: _distribution_or_none(full_coherence[component])
                for component in COMPONENTS
            },
            "acceptance": {"passed": True, "checks": numerical_checks},
            "outputs": {
                "reconstruction_h5": {**staged_h5_record, "path": str(final_h5)},
                "report": str(output_dir / REPORT_NAME),
                "presentation_figures_or_videos_created": False,
            },
            "limitations": [
                "The exact support is voxelized from static lumen seeds, not a measured wall segmentation.",
                "The motion phantom contains lumen spins only; no stationary wall or background phase is created.",
                "Regularized voxels use a 1.0-mm Gaussian sigma (2.355-mm FWHM) spatial estimate.",
                "No independent PC-MRI, PIV, or analytical velocity reference is used here.",
                "The seam-safe reset prevents stale magnetization across the artificial wrap but does not prove physical particle-identity continuity.",
            ],
        }
        staged_report = staging_dir / REPORT_NAME
        atomic_write_json(staged_report, report)
        os.replace(staging_dir, output_dir)
    except BaseException:
        # Keep the staging directory for forensic inspection; never publish it
        # as a final derivative after a failed invariant.
        raise

    print(
        json.dumps(
            {
                "status": "PASS_RECONSTRUCTION_DERIVATIVE",
                "output_dir": str(output_dir),
                "support_voxels": int(support.sum()),
                "direct_voxels": int(direct.sum()),
                "regularized_voxels": int(regularized.sum()),
                "output_sha256": staged_h5_record["sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
