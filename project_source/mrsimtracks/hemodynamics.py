"""Mask-aware hemodynamic calculations in SI units.

These routines do not infer material properties, fill missing velocity with
zero, or establish independent physical validation. Integrals over incomplete
support are explicitly accompanied by sampled-area/volume coverage. A caller
must keep the physical reference and estimator on the same support.
"""

from __future__ import annotations

import numpy as np


def _vectors(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim < 1 or values.shape[-1] != 3:
        raise ValueError("Vector arrays must have final dimension 3.")
    return values


def _weights(weights: np.ndarray | float, shape: tuple[int, ...]) -> np.ndarray:
    weights = np.broadcast_to(np.asarray(weights, dtype=float), shape)
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("Weights must be finite and nonnegative.")
    return weights


def _mask(valid: np.ndarray | None, shape: tuple[int, ...]) -> np.ndarray:
    if valid is None:
        return np.ones(shape, dtype=bool)
    return np.broadcast_to(np.asarray(valid, dtype=bool), shape)


def _normal(normal: np.ndarray) -> np.ndarray:
    normal = np.asarray(normal, dtype=float)
    if normal.shape != (3,) or not np.all(np.isfinite(normal)):
        raise ValueError("Normal must be a finite three-vector.")
    length = np.linalg.norm(normal)
    if length == 0:
        raise ValueError("Normal must be nonzero.")
    return normal / length


def weighted_error_metrics(
    reference: np.ndarray,
    estimate: np.ndarray,
    weights: np.ndarray | float | None = None,
    valid: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Scalar-value errors on finite paired samples with positive weights.

    Bias is estimate minus reference. NRMSE denominators are explicitly named;
    zero reference range/RMS produces NaN, not an arbitrary epsilon-normalized
    score. Components may be evaluated individually; flattening vector arrays
    instead computes a pooled component statistic, not a vector-distance error.
    """
    reference = np.asarray(reference, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    if reference.shape != estimate.shape:
        raise ValueError("Reference and estimate shapes must match.")
    weight = _weights(1.0 if weights is None else weights, reference.shape)
    requested = _mask(valid, reference.shape) & (weight > 0)
    paired = requested & np.isfinite(reference) & np.isfinite(estimate)
    selected_weight = float(np.sum(weight[paired]))
    requested_weight = float(np.sum(weight[requested]))
    result = {
        "n_requested": int(np.count_nonzero(requested)),
        "n_valid": int(np.count_nonzero(paired)),
        "weight_sum": selected_weight,
        "requested_weight_sum": requested_weight,
        "paired_weight_fraction": selected_weight / requested_weight
        if requested_weight > 0 else np.nan,
    }
    names = ("bias", "mae", "rmse", "nrmse_reference_range",
             "nrmse_reference_rms", "pearson_r", "reference_mean",
             "estimate_mean", "reference_range", "reference_rms")
    if selected_weight == 0:
        result.update(dict.fromkeys(names, np.nan))
        return result
    w = weight[paired] / selected_weight
    ref, est = reference[paired], estimate[paired]
    diff = est - ref
    ref_mean, est_mean = float(np.sum(w * ref)), float(np.sum(w * est))
    ref_range = float(np.max(ref) - np.min(ref))
    ref_rms = float(np.sqrt(np.sum(w * ref**2)))
    rmse = float(np.sqrt(np.sum(w * diff**2)))
    covariance = float(np.sum(w * (ref - ref_mean) * (est - est_mean)))
    std_product = float(np.sqrt(np.sum(w * (ref - ref_mean)**2)
                                * np.sum(w * (est - est_mean)**2)))
    result.update({
        "bias": float(np.sum(w * diff)),
        "mae": float(np.sum(w * np.abs(diff))),
        "rmse": rmse,
        "nrmse_reference_range": rmse / ref_range if ref_range > 0 else np.nan,
        "nrmse_reference_rms": rmse / ref_rms if ref_rms > 0 else np.nan,
        "pearson_r": float(np.clip(covariance / std_product, -1, 1))
        if std_product > 0 else np.nan,
        "reference_mean": ref_mean,
        "estimate_mean": est_mean,
        "reference_range": ref_range,
        "reference_rms": ref_rms,
    })
    return result


def section_flow(
    velocity_m_s: np.ndarray,
    area_m2: np.ndarray | float,
    normal: np.ndarray = (1.0, 0.0, 0.0),
    valid: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Integrate a single section; positive means along the supplied normal.

    Forward and reverse magnitudes partition spatial flux at this instant, so
    both can be positive concurrently. Missing samples contribute no fabricated
    value: returned flow is a partial-area integral with explicit coverage.
    """
    u = _vectors(velocity_m_s)
    area = _weights(area_m2, u.shape[:-1])
    requested = _mask(valid, u.shape[:-1]) & (area > 0)
    keep = requested & np.all(np.isfinite(u), axis=-1)
    requested_area = float(np.sum(area[requested]))
    sampled_area = float(np.sum(area[keep]))
    flux = (u[keep] @ _normal(normal)) * area[keep]
    return {
        "net_m3_s": float(np.sum(flux)) if sampled_area else np.nan,
        "forward_m3_s": float(np.sum(np.maximum(flux, 0))) if sampled_area else np.nan,
        "reverse_m3_s": float(np.sum(np.maximum(-flux, 0))) if sampled_area else np.nan,
        "sampled_area_m2": sampled_area,
        "requested_area_m2": requested_area,
        "sampled_area_fraction": sampled_area / requested_area if requested_area else np.nan,
        "n_valid": int(np.count_nonzero(keep)),
    }


def midpoint_cycle_integrals(flow_m3_s: np.ndarray, period_s: float) -> dict[str, float]:
    """Integrate one finite, uniformly spaced full cycle of bin-midpoint values.

    Forward/reverse here partition *net section flow in time*, unlike spatial
    forward/reverse flux returned by section_flow. To integrate simultaneous
    forward/reverse transport, call this on each corresponding flux waveform.
    """
    flow = np.asarray(flow_m3_s, dtype=float)
    if flow.ndim != 1 or flow.size == 0 or not np.all(np.isfinite(flow)):
        raise ValueError("A complete finite one-dimensional cycle is required.")
    if not np.isfinite(period_s) or period_s <= 0:
        raise ValueError("Cycle period must be finite and positive.")
    dt = float(period_s) / flow.size
    return {
        "net_m3": float(np.sum(flow) * dt),
        "forward_m3": float(np.sum(np.maximum(flow, 0)) * dt),
        "reverse_m3": float(np.sum(np.maximum(-flow, 0)) * dt),
        "midpoint_bin_duration_s": dt,
    }


def region_kinetic_energy(
    velocity_m_s: np.ndarray,
    volume_m3: np.ndarray | float,
    valid: np.ndarray | None = None,
    density_kg_m3: float | None = None,
) -> dict[str, float]:
    """Resolved kinetic energy, never turbulence; missing support is excluded.

    The property-free integral is KE/rho in m^5/s^2. Only an explicitly supplied
    positive density produces energy in joules.
    """
    u = _vectors(velocity_m_s)
    volume = _weights(volume_m3, u.shape[:-1])
    requested = _mask(valid, u.shape[:-1]) & (volume > 0)
    keep = requested & np.all(np.isfinite(u), axis=-1)
    requested_volume = float(np.sum(volume[requested]))
    sampled_volume = float(np.sum(volume[keep]))
    energy_per_density = (float(0.5 * np.sum(np.sum(u[keep]**2, axis=-1) * volume[keep]))
                          if sampled_volume else np.nan)
    result = {
        "energy_per_density_m5_s2": energy_per_density,
        "sampled_volume_m3": sampled_volume,
        "requested_volume_m3": requested_volume,
        "sampled_volume_fraction": sampled_volume / requested_volume if requested_volume else np.nan,
    }
    if density_kg_m3 is not None:
        if not np.isfinite(density_kg_m3) or density_kg_m3 <= 0:
            raise ValueError("Density must be finite and positive.")
        result["energy_j"] = energy_per_density * density_kg_m3
    return result


def velocity_gradient(
    velocity_zyx3: np.ndarray,
    spacing_xyz_m: np.ndarray,
    valid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Central-difference gradient du_i/dx_j on a Cartesian (z,y,x,3) grid.

    A result requires all velocity components at the center and its six axial
    neighbors to be valid. No one-sided, wall-imposed, or zero-filled gradients
    are invented. Output shape is (z,y,x,3,3), axes (component, coordinate xyz).
    The common stencil mask is returned separately; unsupported values are NaN.
    """
    u = _vectors(velocity_zyx3)
    if u.ndim != 4:
        raise ValueError("Velocity grid shape must be (nz,ny,nx,3).")
    spacing = np.asarray(spacing_xyz_m, dtype=float)
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError("Spacing must contain positive finite dx,dy,dz in metres.")
    finite = _mask(valid, u.shape[:-1]) & np.all(np.isfinite(u), axis=-1)
    stencil = np.zeros(u.shape[:-1], dtype=bool)
    gradient = np.full(u.shape + (3,), np.nan, dtype=float)
    if min(u.shape[:3]) < 3:
        return gradient, stencil
    center = (slice(1, -1),) * 3
    stencil[center] = (finite[center] & finite[:-2, 1:-1, 1:-1]
                       & finite[2:, 1:-1, 1:-1] & finite[1:-1, :-2, 1:-1]
                       & finite[1:-1, 2:, 1:-1] & finite[1:-1, 1:-1, :-2]
                       & finite[1:-1, 1:-1, 2:])
    for coordinate, array_axis in enumerate((2, 1, 0)):
        before, after = list(center), list(center)
        before[array_axis], after[array_axis] = slice(None, -2), slice(2, None)
        gradient[center + (slice(None), coordinate)] = (
            u[tuple(after)] - u[tuple(before)]) / (2 * spacing[coordinate])
    gradient[~stencil] = np.nan
    return gradient, stencil


def differential_metrics(
    gradient: np.ndarray,
    velocity: np.ndarray | None = None,
    viscosity_pa_s: float | None = None,
) -> dict[str, np.ndarray]:
    """Curl, divergence, strain, helicity and Newtonian dissipation density.

    Inputs use SI and gradient[...,i,j]=du_i/dx_j. Dissipation/mu is 2*S:S
    (s^-2); with explicit dynamic viscosity, dissipation is W/m^3. This is the
    resolved Newtonian incompressible-fluid expression, not turbulent loss.
    """
    grad = np.asarray(gradient, dtype=float)
    if grad.ndim < 2 or grad.shape[-2:] != (3, 3):
        raise ValueError("Gradient must end in (3,3).")
    strain = 0.5 * (grad + np.swapaxes(grad, -1, -2))
    vorticity = np.stack((grad[..., 2, 1] - grad[..., 1, 2],
                          grad[..., 0, 2] - grad[..., 2, 0],
                          grad[..., 1, 0] - grad[..., 0, 1]), axis=-1)
    enstrophy_twice = np.sum(vorticity**2, axis=-1)
    dissipation_per_mu = 2 * np.sum(strain**2, axis=(-2, -1))
    result = {
        "vorticity_s_inv": vorticity,
        "vorticity_magnitude_s_inv": np.sqrt(enstrophy_twice),
        "divergence_s_inv": np.trace(grad, axis1=-2, axis2=-1),
        "strain_rate_s_inv": strain,
        "dissipation_per_viscosity_s_inv2": dissipation_per_mu,
    }
    if velocity is not None:
        u = _vectors(velocity)
        if u.shape != grad.shape[:-1]:
            raise ValueError("Velocity and gradient grid shapes must match.")
        helicity = np.sum(u * vorticity, axis=-1)
        denominator = np.linalg.norm(u, axis=-1) * np.sqrt(enstrophy_twice)
        normalized = np.full(denominator.shape, np.nan)
        np.divide(helicity, denominator, out=normalized, where=denominator > 0)
        result["helicity_m_s_inv2"] = helicity
        result["relative_helicity"] = np.clip(normalized, -1, 1)
    if viscosity_pa_s is not None:
        if not np.isfinite(viscosity_pa_s) or viscosity_pa_s <= 0:
            raise ValueError("Dynamic viscosity must be finite and positive.")
        result["dissipation_w_m3"] = viscosity_pa_s * dissipation_per_mu
    return result


def jet_metrics(
    velocity_m_s: np.ndarray,
    positions_m: np.ndarray,
    area_m2: np.ndarray | float,
    normal: np.ndarray = (1.0, 0.0, 0.0),
    valid: np.ndarray | None = None,
    minimum_forward_speed_m_s: float = 0.0,
) -> dict[str, float | np.ndarray]:
    """Forward-flux-weighted section center, RMS radial width and direction.

    Without an explicit input selection this summarizes all forward flow, not
    a threshold-segmented jet. ``rms_radius_m`` is neither diameter nor FWHM.
    Supply positions on one plane. An optional absolute normal-speed threshold
    must be defined identically for reference and estimate; it is never fitted
    separately to each field. The default is the complete forward-flow region.
    """
    u, positions = _vectors(velocity_m_s), _vectors(positions_m)
    if u.shape != positions.shape:
        raise ValueError("Positions and velocity shapes must match.")
    area = _weights(area_m2, u.shape[:-1])
    n = _normal(normal)
    if not np.isfinite(minimum_forward_speed_m_s) or minimum_forward_speed_m_s < 0:
        raise ValueError("Forward-speed threshold must be finite and nonnegative.")
    keep = (_mask(valid, u.shape[:-1]) & (area > 0)
            & np.all(np.isfinite(u), axis=-1) & np.all(np.isfinite(positions), axis=-1))
    selected_u, selected_p = u[keep], positions[keep]
    normal_speed = selected_u @ n
    weights = np.where(normal_speed > minimum_forward_speed_m_s,
                       normal_speed, 0) * area[keep]
    flux = float(np.sum(weights)) if np.any(keep) else np.nan
    result = {"forward_flux_m3_s": flux,
              "centroid_m": np.full(3, np.nan), "rms_radius_m": np.nan,
              "direction_unit": np.full(3, np.nan), "angle_to_normal_deg": np.nan}
    if not np.isfinite(flux) or flux == 0:
        return result
    weights /= flux
    centroid = np.sum(selected_p * weights[:, None], axis=0)
    offset = selected_p - centroid
    perpendicular = offset - (offset @ n)[:, None] * n
    mean_u = np.sum(selected_u * weights[:, None], axis=0)
    direction = mean_u / np.linalg.norm(mean_u)
    result.update({
        "centroid_m": centroid,
        "rms_radius_m": float(np.sqrt(np.sum(weights * np.sum(perpendicular**2, axis=-1)))),
        "direction_unit": direction,
        "angle_to_normal_deg": float(np.degrees(np.arccos(np.clip(direction @ n, -1, 1)))),
    })
    return result


def wall_cycle_metrics(
    shear_vectors: np.ndarray,
    period_s: float,
    *,
    complete_cycle: bool,
    valid: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """TAWSS and OSI for uniformly sampled complete midpoint shear-vector cycles.

    Axis zero is time. All bins must be finite/valid at a wall point; no missing
    bins are dropped. TAWSS has the supplied shear units (Pa if physical WSS).
    OSI is undefined where integrated shear magnitude is zero. A synthetic
    closure, if present, must be disclosed by the caller, not inferred here.
    """
    shear = _vectors(shear_vectors)
    if shear.ndim < 2 or shear.shape[0] == 0:
        raise ValueError("Shear must have a nonempty leading time axis.")
    if not complete_cycle:
        raise ValueError("TAWSS/OSI require an explicitly complete cycle.")
    if not np.isfinite(period_s) or period_s <= 0:
        raise ValueError("Cycle period must be finite and positive.")
    finite = np.all(np.isfinite(shear), axis=-1) & _mask(valid, shear.shape[:-1])
    complete = np.all(finite, axis=0)
    tawss = np.mean(np.linalg.norm(shear, axis=-1), axis=0)
    mean_vector = np.mean(shear, axis=0)
    denominator = tawss
    ratio = np.full(denominator.shape, np.nan)
    np.divide(np.linalg.norm(mean_vector, axis=-1), denominator,
              out=ratio, where=complete & (denominator > 0))
    osi = np.clip(0.5 * (1 - ratio), 0, 0.5)
    return {"tawss": np.where(complete, tawss, np.nan),
            "osi": osi, "mean_shear_vector": np.where(complete[..., None], mean_vector, np.nan),
            "complete_cycle_valid": complete}
