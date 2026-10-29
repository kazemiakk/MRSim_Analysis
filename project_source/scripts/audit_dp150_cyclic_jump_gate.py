r"""Audit cyclic REF-magnitude and PC-phase jumps in a corrected dp150 run.

This is an internal consistency gate.  It does not establish agreement with CFD,
scanner data, or a physical wall model.  The gate definition is intentionally
kept in one serialisable specification so it can be preregistered before the
candidate reconstruction is inspected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
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
    from scripts.render_dp150_xyz_magnitude_phase import compound_to_complex
except ModuleNotFoundError:
    from analyze_dp150_scanner_reconstruction import (
        DEFAULT_FOV_M,
        DEFAULT_MATRIX,
        load_support,
    )
    from render_dp150_xyz_magnitude_phase import compound_to_complex


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE_RUN = (
    ROOT
    / "outputs"
    / "dp150_mri_pipeline"
    / "koma"
    / "simulation_motion_100k_scanner_matched_venc5"
)
DEFAULT_CANDIDATE_RUN = (
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
ENCODINGS = ("X", "Y", "Z")
BASELINE_CHRONOLOGICAL_SAVED_PHS = tuple(range(9, 15)) + tuple(range(9))
CANDIDATE_CHRONOLOGICAL_SAVED_PHS = tuple(range(15))

GATE_SPECIFICATION: dict[str, Any] = {
    "specification_id": "dp150_cyclic_jump_gate_v1",
    "evidence_class": "internal_cyclic_consistency_not_external_truth",
    "cardiac_bins": 15,
    "support_definition": {
        "static_support": "voxelized dp150_static_2m_mri.h5 positions",
        "erosion": (
            "one binary iteration using the center plus six face-connected "
            "neighbors; array borders are outside"
        ),
        "reliable_reference_mask": (
            "logical intersection across all 15 masks in the current baseline "
            "100k pc_reconstruction.h5"
        ),
        "gate_support": "eroded static support AND reliable reference mask",
        "minimum_voxels": 5000,
    },
    "chronology": {
        "baseline_saved_phs_zero_based": list(BASELINE_CHRONOLOGICAL_SAVED_PHS),
        "candidate_saved_phs_zero_based": list(CANDIDATE_CHRONOLOGICAL_SAVED_PHS),
        "seam_edge_chronological_zero_based": 14,
        "nonseam_edges_chronological_zero_based": list(range(14)),
    },
    "magnitude_edge": (
        "Q95 over gate support of 2*abs(m[k+1]-m[k])/"
        "(m[k+1]+m[k]+epsilon), with cyclic k+1"
    ),
    "epsilon": (
        "1e-6 times the median baseline REF magnitude over all gate-support "
        "voxels and chronological bins; the same epsilon is used for both runs"
    ),
    "phase_edge": (
        "for each X/Y/Z, phi=angle(encoded*conj(REF)); Q95 over gate "
        "support of abs(angle(exp(1j*(phi[k+1]-phi[k])))), with cyclic k+1"
    ),
    "ratios": {
        "R_M": "magnitude seam edge / maximum magnitude nonseam edge",
        "O_M": "magnitude seam edge / median magnitude nonseam edge",
        "R_P": (
            "maximum over X/Y/Z of component phase seam edge / maximum "
            "component phase nonseam edge"
        ),
        "O_P": (
            "maximum over X/Y/Z of component phase seam edge / median "
            "component phase nonseam edge"
        ),
    },
    "pass_limits_inclusive": {
        "candidate_R_M_max": 1.10,
        "candidate_R_P_max": 1.10,
        "candidate_O_M_max": 1.50,
        "candidate_O_P_max": 1.50,
        "candidate_O_M_minus_baseline_O_M_max": 0.10,
        "candidate_O_P_minus_baseline_O_P_max": 0.10,
    },
    "quantile_method": "numpy.quantile default linear method at q=0.95",
    "zero_denominator_ratio_rule": (
        "1.0 when numerator and denominator are both zero; infinity when only "
        "the denominator is zero"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", type=Path, default=DEFAULT_BASELINE_RUN)
    parser.add_argument("--candidate-run", type=Path, default=DEFAULT_CANDIDATE_RUN)
    parser.add_argument("--support-h5", type=Path, default=DEFAULT_SUPPORT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--register-only",
        action="store_true",
        help="Write the frozen gate specification without reading candidate results.",
    )
    return parser.parse_args()


def sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


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


def one_voxel_face_erosion(support: np.ndarray) -> np.ndarray:
    support = np.asarray(support, dtype=bool)
    if support.ndim != 3:
        raise ValueError(f"Expected 3D support, got {support.shape}")
    eroded = np.zeros_like(support)
    if min(support.shape) < 3:
        return eroded
    eroded[1:-1, 1:-1, 1:-1] = (
        support[1:-1, 1:-1, 1:-1]
        & support[:-2, 1:-1, 1:-1]
        & support[2:, 1:-1, 1:-1]
        & support[1:-1, :-2, 1:-1]
        & support[1:-1, 2:, 1:-1]
        & support[1:-1, 1:-1, :-2]
        & support[1:-1, 1:-1, 2:]
    )
    return eroded


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator > 0:
        return float(numerator / denominator)
    if numerator == 0:
        return 1.0
    return math.inf


def _finite_q95(values: np.ndarray) -> float:
    selected = np.asarray(values, dtype=np.float64)
    if selected.size == 0 or not np.isfinite(selected).all():
        raise ValueError("Edge population must be nonempty and finite")
    return float(np.quantile(selected, 0.95))


def _read_scalar(source: h5py.File, name: str) -> Any | None:
    if name not in source:
        return None
    value = source[name][()]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if isinstance(value, np.generic):
        value = value.item()
    return value


def load_selected_complex_run(
    run_dir: Path,
    gate_support: np.ndarray,
    chronological_saved_phs: tuple[int, ...],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if sorted(chronological_saved_phs) != list(range(15)):
        raise ValueError("Chronology must be a permutation of saved PHS 0..14")
    selected: dict[str, np.ndarray] = {}
    records: dict[str, Any] = {}
    support_count = int(gate_support.sum())
    expected_shape = (15, *gate_support.shape)

    with ExitStack() as stack:
        sources: dict[str, h5py.File] = {}
        for encoding, filename in RAW_FILENAMES.items():
            path = run_dir / filename
            if not path.is_file():
                raise FileNotFoundError(path)
            source = stack.enter_context(h5py.File(path, "r"))
            if "image" not in source:
                raise KeyError(f"{path} has no image dataset")
            if tuple(source["image"].shape) != expected_shape:
                raise ValueError(
                    f"{path} image shape {source['image'].shape} != {expected_shape}"
                )
            sources[encoding] = source
            records[encoding] = {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "motion_timing_signature": _read_scalar(
                    source, "motion_timing_signature"
                ),
            }

        for encoding, source in sources.items():
            values = np.empty((15, support_count), dtype=np.complex64)
            for chronological_index, saved_phs in enumerate(
                chronological_saved_phs
            ):
                volume = compound_to_complex(source["image"][saved_phs])
                selected_volume = np.asarray(volume[gate_support], np.complex64)
                if not (
                    np.isfinite(selected_volume.real).all()
                    and np.isfinite(selected_volume.imag).all()
                ):
                    raise ValueError(
                        f"{encoding} saved PHS {saved_phs} contains non-finite values"
                    )
                values[chronological_index] = selected_volume
            selected[encoding] = values

    signatures = {
        record["motion_timing_signature"] for record in records.values()
    }
    if len(signatures) != 1:
        raise RuntimeError(f"Encoding timing signatures disagree: {signatures}")
    return selected, records


def compute_edge_metrics(
    selected: dict[str, np.ndarray],
    chronological_saved_phs: tuple[int, ...],
    epsilon: float,
) -> list[dict[str, Any]]:
    if set(selected) != set(RAW_FILENAMES):
        raise ValueError(f"Expected {set(RAW_FILENAMES)}, got {set(selected)}")
    shape = selected["REF"].shape
    if len(shape) != 2 or shape[0] != 15 or shape[1] == 0:
        raise ValueError(f"Expected selected arrays shaped (15,N), got {shape}")
    if any(values.shape != shape for values in selected.values()):
        raise ValueError("Selected encoding shapes disagree")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")

    magnitude = np.abs(selected["REF"]).astype(np.float64)
    phase = {
        encoding: np.angle(
            selected[encoding] * np.conj(selected["REF"])
        ).astype(np.float64)
        for encoding in ENCODINGS
    }
    edges: list[dict[str, Any]] = []
    for edge_index in range(15):
        following = (edge_index + 1) % 15
        magnitude_delta = (
            2.0 * np.abs(magnitude[following] - magnitude[edge_index])
            / (magnitude[following] + magnitude[edge_index] + epsilon)
        )
        phase_q95_rad: dict[str, float] = {}
        phase_q95_deg: dict[str, float] = {}
        for encoding in ENCODINGS:
            circular_delta = np.abs(
                np.angle(
                    np.exp(
                        1j * (phase[encoding][following] - phase[encoding][edge_index])
                    )
                )
            )
            value = _finite_q95(circular_delta)
            phase_q95_rad[encoding] = value
            phase_q95_deg[encoding] = math.degrees(value)
        edges.append(
            {
                "edge_index_chronological_zero_based": edge_index,
                "from_chronological_bin_zero_based": edge_index,
                "to_chronological_bin_zero_based": following,
                "from_saved_phs_zero_based": chronological_saved_phs[edge_index],
                "to_saved_phs_zero_based": chronological_saved_phs[following],
                "is_cycle_seam": edge_index == 14,
                "magnitude_normalized_difference_q95": _finite_q95(
                    magnitude_delta
                ),
                "phase_circular_difference_q95_rad": phase_q95_rad,
                "phase_circular_difference_q95_deg": phase_q95_deg,
            }
        )
    return edges


def summarize_edges(edges: list[dict[str, Any]]) -> dict[str, Any]:
    if len(edges) != 15 or [edge["is_cycle_seam"] for edge in edges] != [
        False
    ] * 14 + [True]:
        raise ValueError("Expected 14 nonseam edges followed by one seam edge")

    magnitude = np.asarray(
        [edge["magnitude_normalized_difference_q95"] for edge in edges],
        dtype=np.float64,
    )
    magnitude_max = float(np.max(magnitude[:14]))
    magnitude_median = float(np.median(magnitude[:14]))
    magnitude_summary = {
        "seam": float(magnitude[14]),
        "nonseam_max": magnitude_max,
        "nonseam_median": magnitude_median,
        "R_seam_over_nonseam_max": safe_ratio(magnitude[14], magnitude_max),
        "O_seam_over_nonseam_median": safe_ratio(
            magnitude[14], magnitude_median
        ),
    }

    phase_components: dict[str, Any] = {}
    phase_r: list[float] = []
    phase_o: list[float] = []
    for encoding in ENCODINGS:
        values = np.asarray(
            [edge["phase_circular_difference_q95_rad"][encoding] for edge in edges],
            dtype=np.float64,
        )
        nonseam_max = float(np.max(values[:14]))
        nonseam_median = float(np.median(values[:14]))
        ratio_max = safe_ratio(values[14], nonseam_max)
        ratio_median = safe_ratio(values[14], nonseam_median)
        phase_r.append(ratio_max)
        phase_o.append(ratio_median)
        phase_components[encoding] = {
            "seam_rad": float(values[14]),
            "seam_deg": math.degrees(float(values[14])),
            "nonseam_max_rad": nonseam_max,
            "nonseam_median_rad": nonseam_median,
            "R_seam_over_nonseam_max": ratio_max,
            "O_seam_over_nonseam_median": ratio_median,
        }
    return {
        "magnitude": magnitude_summary,
        "phase_by_component": phase_components,
        "combined_ratios": {
            "R_M": magnitude_summary["R_seam_over_nonseam_max"],
            "O_M": magnitude_summary["O_seam_over_nonseam_median"],
            "R_P": float(max(phase_r)),
            "O_P": float(max(phase_o)),
        },
    }


def evaluate_gate(
    baseline_summary: dict[str, Any], candidate_summary: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    baseline = baseline_summary["combined_ratios"]
    candidate = candidate_summary["combined_ratios"]
    limits = GATE_SPECIFICATION["pass_limits_inclusive"]
    definitions = {
        "candidate_R_M_le_1_10": (
            candidate["R_M"], limits["candidate_R_M_max"]
        ),
        "candidate_R_P_le_1_10": (
            candidate["R_P"], limits["candidate_R_P_max"]
        ),
        "candidate_O_M_le_1_50": (
            candidate["O_M"], limits["candidate_O_M_max"]
        ),
        "candidate_O_P_le_1_50": (
            candidate["O_P"], limits["candidate_O_P_max"]
        ),
        "candidate_O_M_minus_baseline_le_0_10": (
            candidate["O_M"] - baseline["O_M"],
            limits["candidate_O_M_minus_baseline_O_M_max"],
        ),
        "candidate_O_P_minus_baseline_le_0_10": (
            candidate["O_P"] - baseline["O_P"],
            limits["candidate_O_P_minus_baseline_O_P_max"],
        ),
    }
    checks = {
        name: {
            "measured": float(measured),
            "inclusive_upper_limit": float(limit),
            "pass": bool(math.isfinite(measured) and measured <= limit),
        }
        for name, (measured, limit) in definitions.items()
    }
    return ("PASS" if all(item["pass"] for item in checks.values()) else "FAIL"), checks


def registration_payload(candidate_run: Path) -> dict[str, Any]:
    script_path = Path(__file__).resolve()
    return {
        "artifact_type": "preregistered_internal_consistency_gate",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_run": str(candidate_run.resolve()),
        "candidate_complete_raw_files_present_at_registration": {
            encoding: (candidate_run / filename).is_file()
            for encoding, filename in RAW_FILENAMES.items()
        },
        "gate_specification": GATE_SPECIFICATION,
        "implementation": {
            "path": str(script_path),
            "sha256": sha256_file(script_path),
        },
    }


def _verify_registration(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing preregistration {path}; run with --register-only first"
        )
    with path.open("r", encoding="utf-8") as source:
        registration = json.load(source)
    if registration.get("gate_specification") != GATE_SPECIFICATION:
        raise RuntimeError("Preregistered gate specification does not match code")
    expected_hash = registration.get("implementation", {}).get("sha256")
    actual_hash = sha256_file(Path(__file__).resolve())
    if expected_hash != actual_hash:
        raise RuntimeError(
            "Gate implementation changed after preregistration: "
            f"registered={expected_hash}, current={actual_hash}"
        )
    return registration


def main() -> int:
    args = parse_args()
    baseline_run = args.baseline_run.resolve()
    candidate_run = args.candidate_run.resolve()
    support_path = args.support_h5.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else candidate_run / "final_image_analysis" / "cyclic_jump_gate"
    )
    registration_path = output_dir / "cyclic_jump_gate_preregistration.json"
    report_path = output_dir / "cyclic_jump_gate_report.json"

    if args.register_only:
        payload = registration_payload(candidate_run)
        if registration_path.exists():
            with registration_path.open("r", encoding="utf-8") as source:
                existing = json.load(source)
            comparable_existing = dict(existing)
            comparable_payload = dict(payload)
            comparable_existing.pop("created_utc", None)
            comparable_payload.pop("created_utc", None)
            if comparable_existing != comparable_payload:
                raise RuntimeError(
                    f"Refusing to overwrite different preregistration {registration_path}"
                )
            print(f"Preregistration already matches: {registration_path}")
            return 0
        atomic_write_json(registration_path, payload)
        print(f"Wrote frozen gate preregistration: {registration_path}")
        return 0

    registration = _verify_registration(registration_path)
    fov_m = np.asarray(DEFAULT_FOV_M, dtype=np.float64)
    matrix = np.asarray(DEFAULT_MATRIX, dtype=np.int64)
    static_support, support_metadata = load_support(support_path, fov_m, matrix)

    baseline_reconstruction = baseline_run / "pc_reconstruction.h5"
    with h5py.File(baseline_reconstruction, "r") as source:
        if "mask" not in source:
            raise KeyError(f"{baseline_reconstruction} has no mask dataset")
        masks = np.asarray(source["mask"], dtype=bool)
    if masks.shape != (15, *static_support.shape):
        raise ValueError(
            f"Baseline masks {masks.shape} do not match support {static_support.shape}"
        )
    reliable_reference = np.all(masks, axis=0)
    eroded_support = one_voxel_face_erosion(static_support)
    gate_support = eroded_support & reliable_reference
    support_count = int(gate_support.sum())
    minimum_count = int(
        GATE_SPECIFICATION["support_definition"]["minimum_voxels"]
    )
    if support_count < minimum_count:
        raise RuntimeError(
            f"Gate support has {support_count} voxels; minimum is {minimum_count}"
        )

    baseline_selected, baseline_inputs = load_selected_complex_run(
        baseline_run, gate_support, BASELINE_CHRONOLOGICAL_SAVED_PHS
    )
    baseline_median = float(np.median(np.abs(baseline_selected["REF"])))
    epsilon = 1e-6 * baseline_median
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise RuntimeError(f"Invalid baseline-derived epsilon {epsilon}")

    candidate_selected, candidate_inputs = load_selected_complex_run(
        candidate_run, gate_support, CANDIDATE_CHRONOLOGICAL_SAVED_PHS
    )
    baseline_edges = compute_edge_metrics(
        baseline_selected, BASELINE_CHRONOLOGICAL_SAVED_PHS, epsilon
    )
    candidate_edges = compute_edge_metrics(
        candidate_selected, CANDIDATE_CHRONOLOGICAL_SAVED_PHS, epsilon
    )
    baseline_summary = summarize_edges(baseline_edges)
    candidate_summary = summarize_edges(candidate_edges)
    status, checks = evaluate_gate(baseline_summary, candidate_summary)

    report = {
        "artifact_type": "dp150_cyclic_jump_gate_report",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "evidence_class": GATE_SPECIFICATION["evidence_class"],
        "interpretation_limit": (
            "PASS means the candidate cycle boundary is not an outlier under "
            "the preregistered internal metrics. It does not validate CFD truth, "
            "scanner agreement, particle identity continuity, or wall realism."
        ),
        "preregistration": {
            "path": str(registration_path.resolve()),
            "sha256": sha256_file(registration_path),
            "created_utc": registration["created_utc"],
        },
        "gate_specification": GATE_SPECIFICATION,
        "support": {
            "source": str(support_path),
            "source_sha256": sha256_file(support_path),
            "baseline_reconstruction": str(baseline_reconstruction.resolve()),
            "baseline_reconstruction_sha256": sha256_file(
                baseline_reconstruction
            ),
            "static_voxels": int(static_support.sum()),
            "eroded_static_voxels": int(eroded_support.sum()),
            "all_bin_reliable_reference_voxels": int(
                reliable_reference.sum()
            ),
            "gate_support_voxels": support_count,
            "minimum_gate_support_voxels": minimum_count,
            "support_metadata": support_metadata,
        },
        "epsilon": {
            "baseline_ref_median_over_gate_support_and_bins": baseline_median,
            "scale": 1e-6,
            "value": epsilon,
        },
        "baseline": {
            "run_dir": str(baseline_run),
            "inputs": baseline_inputs,
            "chronological_saved_phs_zero_based": list(
                BASELINE_CHRONOLOGICAL_SAVED_PHS
            ),
            "edges": baseline_edges,
            "summary": baseline_summary,
        },
        "candidate": {
            "run_dir": str(candidate_run),
            "inputs": candidate_inputs,
            "chronological_saved_phs_zero_based": list(
                CANDIDATE_CHRONOLOGICAL_SAVED_PHS
            ),
            "edges": candidate_edges,
            "summary": candidate_summary,
        },
        "checks": checks,
    }
    atomic_write_json(report_path, report)
    print(f"Cyclic jump gate: {status}")
    print(f"Gate support voxels: {support_count}")
    print(json.dumps(candidate_summary["combined_ratios"], indent=2))
    print(f"Report: {report_path}")
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
