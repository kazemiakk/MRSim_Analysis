"""Discrete, fixed-domain resolved-energy decompositions; never a TKE estimator."""
from __future__ import annotations

import numpy as np


def cycle_energy_decomposition(velocity, volume_m3, valid=None, *, time_weights=None,
                               density_kg_m3=None):
    """Split resolved bin-averaged KE into mean-vector and coherent variation.

    Input axes are (time, ..., component=3). A fixed spatial domain is required:
    missing vectors anywhere on it fail closed instead of changing denominators
    across bins. Unselected data can be NaN. Time weights define a discrete
    average; they do not create additional acquired samples. Cardiac variation
    includes coherent pulsatility and measurement error, not isolated turbulence.
    """
    u = np.asarray(velocity, dtype=float)
    if u.ndim < 3 or u.shape[-1] != 3 or u.shape[0] < 1:
        raise ValueError('Velocity must have shape (time, spatial..., 3)')
    shape = u.shape[1:-1]
    volume = np.broadcast_to(np.asarray(volume_m3, dtype=float), shape)
    if not np.isfinite(volume).all() or np.any(volume < 0):
        raise ValueError('Volume weights must be finite and nonnegative')
    mask = np.ones(shape, dtype=bool) if valid is None else np.broadcast_to(np.asarray(valid, bool), shape)
    keep = mask & (volume > 0)
    if not keep.any():
        raise ValueError('Fixed spatial domain is empty')
    selected = u[:, keep, :]
    if not np.isfinite(selected).all():
        raise ValueError('Every selected vector must be finite in every bin')
    weights = np.ones(len(u)) if time_weights is None else np.asarray(time_weights, dtype=float)
    if weights.shape != (len(u),) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError('Time weights must be positive finite and match bins')
    weights = weights / weights.sum()
    mean = np.einsum('t,tnc->nc', weights, selected)
    variance = np.einsum('t,tnc->nc', weights, (selected - mean)**2)
    v = volume[keep]
    per_bin = .5*np.einsum('tnc,n,tnc->t', selected, v, selected)
    total = float(weights @ per_bin)
    steady = float(.5*np.einsum('nc,n,nc->', mean, v, mean))
    coherent = float(.5*np.sum(variance*v[:, None]))
    mean_grid = np.full((*shape, 3), np.nan)
    var_grid = np.full((*shape, 3), np.nan)
    mean_grid[keep] = mean
    var_grid[keep] = variance
    result = dict(
        definition='Discrete resolved KE = KE of mean vector + coherent bin variation; NOT TKE',
        bins=len(u), voxels=int(keep.sum()), sampled_volume_m3=float(v.sum()),
        normalized_time_weights=weights,
        per_bin_ke_per_density_m5_s2=per_bin,
        mean_resolved_ke_per_density_m5_s2=total,
        mean_vector_ke_per_density_m5_s2=steady,
        coherent_variation_ke_per_density_m5_s2=coherent,
        coherent_fraction=coherent/total if total > 0 else np.nan,
        closure_per_density_m5_s2=total-steady-coherent,
        relative_closure=abs(total-steady-coherent)/total if total > 0 else 0.,
        mean_velocity_m_s=mean_grid,
        coherent_component_variance_m2_s2=var_grid,
    )
    if density_kg_m3 is not None:
        if not np.isfinite(density_kg_m3) or density_kg_m3 <= 0:
            raise ValueError('Density must be positive finite')
        result.update(mean_resolved_ke_j=total*density_kg_m3,
                      mean_vector_ke_j=steady*density_kg_m3,
                      coherent_variation_ke_j=coherent*density_kg_m3)
    return result
