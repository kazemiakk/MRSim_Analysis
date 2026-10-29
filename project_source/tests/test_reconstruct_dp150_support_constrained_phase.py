from __future__ import annotations

import numpy as np
import pytest

from scripts.reconstruct_dp150_support_constrained_phase import (
    ENCODINGS,
    circular_velocity_difference,
    fixed_direct_mask,
    hybrid_phase_arrays,
    normalized_magnitude_confidence_for_bin,
    support_constrained_complex_pool,
)


def _constant_images(shape: tuple[int, int, int], value: float = 10.0):
    return {
        encoding: np.full(shape, value + 0j, dtype=np.complex64)
        for encoding in ENCODINGS
    }


def test_confidence_requires_reference_and_every_encoding() -> None:
    shape = (1, 1, 5)
    support = np.ones(shape, dtype=bool)
    images = _constant_images(shape)
    images["REF"][0, 0, 0] = 0.5
    images["Z"][0, 0, 1] = 0.25

    score, scales = normalized_magnitude_confidence_for_bin(images, support)

    assert scales == {encoding: 10.0 for encoding in ENCODINGS}
    assert score[0, 0, 0] == pytest.approx(0.05)
    assert score[0, 0, 1] == pytest.approx(0.025)
    assert np.all(score[0, 0, 2:] == 1.0)


def test_confidence_is_invariant_to_encoding_specific_global_scale() -> None:
    shape = (1, 2, 3)
    support = np.ones(shape, dtype=bool)
    base = np.arange(1, 7, dtype=np.float32).reshape(shape)
    images = {
        "REF": base.astype(np.complex64),
        "X": (2.0 * base).astype(np.complex64),
        "Y": (5.0 * base).astype(np.complex64),
        "Z": (11.0 * base).astype(np.complex64),
    }
    first, _ = normalized_magnitude_confidence_for_bin(images, support)
    scaled = {
        encoding: image * factor
        for (encoding, image), factor in zip(
            images.items(), (3.0, 7.0, 13.0, 17.0), strict=True
        )
    }
    second, _ = normalized_magnitude_confidence_for_bin(scaled, support)
    np.testing.assert_allclose(first, second, rtol=2e-7, atol=1e-7)


def test_fixed_mask_uses_minimum_over_bins_and_partitions_support() -> None:
    support = np.array([[[True, True, True, False]]])
    per_bin = np.ones((3, *support.shape), dtype=np.float32)
    per_bin[1, 0, 0, 1] = 0.09
    confidence = per_bin.min(axis=0)

    direct, regularized = fixed_direct_mask(support, confidence, 0.10)

    np.testing.assert_array_equal(
        direct, np.array([[[True, False, True, False]]])
    )
    np.testing.assert_array_equal(
        regularized, np.array([[[False, True, False, False]]])
    )
    np.testing.assert_array_equal(direct | regularized, support)
    assert not np.any(direct & regularized)


def test_support_pool_preserves_constant_phase_at_boundary() -> None:
    shape = (3, 5, 7)
    support = np.zeros(shape, dtype=bool)
    support[1, 1:4, 1:6] = True
    expected_phase = 0.73
    cross_product = np.full(shape, 50.0 * np.exp(-2.1j), dtype=np.complex128)
    cross_product[support] = 2.5 * np.exp(1j * expected_phase)

    pooled, coherence, pooled_weight = support_constrained_complex_pool(
        cross_product, support, (0.4, 0.8, 0.8)
    )

    np.testing.assert_allclose(
        np.angle(pooled[support]), expected_phase, rtol=0, atol=2e-7
    )
    np.testing.assert_allclose(coherence[support], 1.0, rtol=0, atol=2e-7)
    np.testing.assert_allclose(
        pooled_weight[support], 2.5, rtol=0, atol=2e-6
    )


def test_support_pool_averages_wrapped_phase_circularly() -> None:
    support = np.ones((1, 1, 3), dtype=bool)
    phase = np.deg2rad(np.array([179.0, 180.0, -179.0]))
    cross_product = np.exp(1j * phase)[None, None, :]

    pooled, coherence, _weight = support_constrained_complex_pool(
        cross_product, support, (0.0, 0.0, 1.0)
    )

    center = float(np.angle(pooled[0, 0, 1]))
    assert abs(abs(center) - np.pi) < np.deg2rad(0.1)
    assert coherence[0, 0, 1] > 0.999


def test_hybrid_phase_preserves_direct_and_marks_outside_nan() -> None:
    support = np.array([[[True, True, False]]])
    direct = np.array([[[True, False, False]]])
    regularized = np.array([[[False, True, False]]])
    raw_phase = np.array([[[0.4, -2.8, 1.5]]])
    pooled_phase = np.array([[[1.2, 0.7, -0.2]]])
    raw_cross = np.exp(1j * raw_phase)
    pooled_cross = np.exp(1j * pooled_phase)

    raw, corrected = hybrid_phase_arrays(
        raw_cross, pooled_cross, support, direct, regularized
    )

    assert raw[0, 0, 0] == pytest.approx(0.4)
    assert raw[0, 0, 1] == pytest.approx(-2.8)
    assert corrected[0, 0, 0] == raw[0, 0, 0]
    assert corrected[0, 0, 1] == pytest.approx(0.7)
    assert np.isnan(raw[0, 0, 2])
    assert np.isnan(corrected[0, 0, 2])


def test_circular_velocity_difference_respects_phase_wrap() -> None:
    first = np.deg2rad(np.array([179.0]))
    second = np.deg2rad(np.array([-179.0]))
    result = circular_velocity_difference(first, second, venc=5.0)
    assert result[0] == pytest.approx(2.0 * 5.0 / 180.0)


def test_pool_rejects_zero_weight_support() -> None:
    support = np.ones((1, 2, 2), dtype=bool)
    with pytest.raises(RuntimeError, match="weight vanished"):
        support_constrained_complex_pool(
            np.zeros(support.shape, dtype=np.complex64),
            support,
            (0.4, 0.8, 0.8),
        )
