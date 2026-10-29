from __future__ import annotations

import numpy as np

from scripts.audit_dp150_cyclic_jump_gate import (
    CANDIDATE_CHRONOLOGICAL_SAVED_PHS,
    compute_edge_metrics,
    evaluate_gate,
    one_voxel_face_erosion,
    safe_ratio,
    summarize_edges,
)


def test_one_voxel_face_erosion_uses_six_neighbors_and_outside_border() -> None:
    solid = np.ones((5, 5, 5), dtype=bool)
    eroded = one_voxel_face_erosion(solid)
    expected = np.zeros_like(solid)
    expected[1:-1, 1:-1, 1:-1] = True
    np.testing.assert_array_equal(eroded, expected)

    solid[2, 2, 1] = False
    eroded = one_voxel_face_erosion(solid)
    assert not eroded[2, 2, 2]
    assert eroded[1, 1, 2]


def test_edge_metrics_use_cyclic_phase_difference_and_report_all_edges() -> None:
    ref = np.ones((15, 20), dtype=np.complex64)
    phase = np.zeros((15, 20), dtype=np.float32)
    phase[:, :] = np.arange(15, dtype=np.float32)[:, None] * 0.01
    # The last-to-first edge wraps across +/-pi and should remain a small
    # circular difference instead of appearing close to 2*pi.
    phase[14] = np.pi - 0.02
    phase[0] = -np.pi + 0.02
    selected = {"REF": ref}
    for encoding, scale in zip(("X", "Y", "Z"), (1.0, 0.5, 0.25), strict=True):
        selected[encoding] = np.exp(1j * scale * phase).astype(np.complex64)

    edges = compute_edge_metrics(
        selected, CANDIDATE_CHRONOLOGICAL_SAVED_PHS, epsilon=1e-6
    )

    assert len(edges) == 15
    assert edges[-1]["is_cycle_seam"]
    assert edges[-1]["from_saved_phs_zero_based"] == 14
    assert edges[-1]["to_saved_phs_zero_based"] == 0
    assert np.isclose(
        edges[-1]["phase_circular_difference_q95_rad"]["X"], 0.04, atol=1e-6
    )


def _synthetic_edges(
    magnitude_nonseam: float,
    magnitude_seam: float,
    phase_nonseam: tuple[float, float, float],
    phase_seam: tuple[float, float, float],
) -> list[dict[str, object]]:
    edges: list[dict[str, object]] = []
    for index in range(15):
        phase_values = phase_seam if index == 14 else phase_nonseam
        edges.append(
            {
                "is_cycle_seam": index == 14,
                "magnitude_normalized_difference_q95": (
                    magnitude_seam if index == 14 else magnitude_nonseam
                ),
                "phase_circular_difference_q95_rad": dict(
                    zip(("X", "Y", "Z"), phase_values, strict=True)
                ),
            }
        )
    return edges


def test_summary_uses_worst_component_ratio() -> None:
    summary = summarize_edges(
        _synthetic_edges(
            magnitude_nonseam=0.2,
            magnitude_seam=0.22,
            phase_nonseam=(0.2, 0.4, 0.5),
            phase_seam=(0.21, 0.42, 0.40),
        )
    )
    ratios = summary["combined_ratios"]
    assert np.isclose(ratios["R_M"], 1.1)
    assert np.isclose(ratios["O_M"], 1.1)
    assert np.isclose(ratios["R_P"], 1.05)
    assert np.isclose(ratios["O_P"], 1.05)


def test_gate_is_inclusive_and_checks_baseline_outlier_delta() -> None:
    baseline = summarize_edges(
        _synthetic_edges(
            0.2, 0.198, (0.2, 0.2, 0.2), (0.198, 0.198, 0.198)
        )
    )
    candidate = summarize_edges(
        _synthetic_edges(
            0.2,
            0.22,
            (0.2, 0.2, 0.2),
            (0.22, 0.20, 0.20),
        )
    )
    status, checks = evaluate_gate(baseline, candidate)
    assert status == "FAIL"
    assert checks["candidate_R_M_le_1_10"]["pass"]
    assert checks["candidate_R_P_le_1_10"]["pass"]
    assert not checks["candidate_O_M_minus_baseline_le_0_10"]["pass"]


def test_zero_over_zero_ratio_is_neutral() -> None:
    assert safe_ratio(0.0, 0.0) == 1.0
