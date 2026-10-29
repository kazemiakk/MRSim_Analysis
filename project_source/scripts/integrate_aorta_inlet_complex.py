"""E-036: separate, signal-gated complex-image inlet quadrature derivative."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import tomllib
import uuid

import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from scripts.aorta_hemodynamic_reference import AortaGrid, sequence_timing
from scripts.build_aorta_inlet_picker import sha

ENCODINGS = ('REF', 'X', 'Y', 'Z')


def complex_velocity_samples(images, axes, points, venc, threshold=.05):
    """Interpolate complex signals first; never interpolate zero-filled velocity.

    Returns wrapped velocity, validity and relative signal (bin, point, encoding).
    Invalid velocities are NaN, including phase cancellation or absent signal.
    """
    if set(images) != set(ENCODINGS) or not np.isfinite(venc) or venc <= 0:
        raise ValueError('Require REF/X/Y/Z and positive finite VENC')
    if not 0 < threshold < 1:
        raise ValueError('Require a relative signal threshold between zero and one')
    shape = images['REF'].shape
    if len(shape) != 4 or shape[1:] != tuple(map(len, axes)):
        raise ValueError('Expected bin/z/y/x images matching the grid axes')
    signals = []; ratios = []
    for encoding in ENCODINGS:
        data = np.asarray(images[encoding])
        if data.shape != shape or not np.isfinite(data).all():
            raise ValueError('Complex images must have matching shapes and finite values')
        sampled = np.array([RegularGridInterpolator(axes, image, bounds_error=True)(points) for image in data])
        maximum = np.abs(data).max(axis=(1, 2, 3))[:, None]
        ratios.append(np.divide(abs(sampled), maximum, out=np.zeros(sampled.shape), where=maximum > 0))
        signals.append(sampled)
    ratios = np.stack(ratios, axis=-1)
    valid = np.all(np.isfinite(ratios) & (ratios >= threshold), axis=-1)
    velocity = np.stack([np.angle(signal * signals[0].conj()) * (venc / np.pi)
                         for signal in signals[1:]], axis=-1)
    velocity[~valid] = np.nan
    return velocity, valid, ratios


def integrate_flow(velocity, area_mm2, normal):
    """m/s times mm2 is mL/s; no area-ratio extrapolation is applied."""
    velocity = np.asarray(velocity); area = np.asarray(area_mm2); normal = np.asarray(normal)
    if velocity.shape[-2:] != (len(area), 3) or normal.shape != (3,):
        raise ValueError('Velocity/area/normal shape mismatch')
    if not np.isfinite(velocity).all() or not np.isfinite(area).all() or np.any(area <= 0):
        raise ValueError('Cannot integrate missing velocity or invalid areas')
    if not np.isfinite(normal).all() or not np.isclose(np.linalg.norm(normal), 1, atol=1e-6, rtol=0):
        raise ValueError('Require a unit inlet normal')
    return np.einsum('...nc,c,n->...', velocity, normal, area)


def synthetic_controls():
    axes = (np.arange(2, dtype=float),) * 3
    amplitude = np.zeros((2, 2, 2)); amplitude[:, :, 1] = [[1, 1.2], [.8, 1.1]]
    ref = (amplitude * np.exp(1j * np.arange(8).reshape(2, 2, 2) * .09))[None]
    target = np.array([.12, -.035, .08]); venc = 2.5
    images = {'REF': ref, **{e: ref * np.exp(1j * np.pi * v / venc) for e, v in zip(ENCODINGS[1:], target)}}
    points = np.array([[.25, .25, .25], [.75, .5, .5], [.2, .75, .75]])
    velocity, valid, _ = complex_velocity_samples(images, axes, points, venc)
    error = float(np.max(abs(velocity - target)))
    if not valid.all() or error >= 1e-6:
        raise AssertionError('Partial-volume constant-velocity control failed')
    no_signal = {e: np.zeros_like(ref) for e in ENCODINGS}
    missing, valid_missing, _ = complex_velocity_samples(no_signal, axes, points, venc)
    if valid_missing.any() or not np.isnan(missing).all():
        raise AssertionError('Missing signal must remain unavailable')
    cancel = {e: np.ones_like(ref) for e in ENCODINGS}; cancel['X'][:, :, :, 1] = -1
    missing, valid_cancel, _ = complex_velocity_samples(cancel, axes, [[.5, .5, .5]], venc)
    if valid_cancel.any() or not np.isnan(missing).all():
        raise AssertionError('Encoded cancellation was not rejected')
    wrapped = {e: np.ones_like(ref) for e in ENCODINGS}
    wrapped['X'][:, :, :, 0] = np.exp(1j * (np.pi - .01))
    wrapped['X'][:, :, :, 1] = np.exp(-1j * (np.pi - .01))
    result, valid_wrapped, _ = complex_velocity_samples(wrapped, axes, [[.5, .5, .5]], venc)
    if not valid_wrapped.all() or not np.isclose(abs(result[0, 0, 0]), venc, atol=1e-12):
        raise AssertionError('Circular interpolation control failed')
    flux = integrate_flow(np.array([[[.2, 0, 0], [.2, 0, 0]]]), [3, 7], [1, 0, 0])
    if not np.allclose(flux, [2], rtol=0, atol=1e-12):
        raise AssertionError('Flux units control failed')
    return dict(status='PASS_SYNTHETIC_CONTROLS',partial_volume_max_velocity_error_m_s=error,
                missing_signal_unavailable=True,encoded_cancellation_unavailable=True,
                circular_phase_correct=True,flux_units_correct=True)


def metrics(cfd, koma, rr):
    difference = koma - cfd
    return dict(rmse_ml_s=float(np.sqrt(np.mean(difference ** 2))),
        range_normalized_rmse=float(np.sqrt(np.mean(difference ** 2)) / np.ptp(cfd)),
        correlation=float(np.corrcoef(cfd, koma)[0, 1]),
        maximum_absolute_bin_error_ml_s=float(np.max(abs(difference))),
        nominal_bin_cycle_volume_CFD_ml=float(cfd.sum() * rr / len(cfd)),
        nominal_bin_cycle_volume_Koma_ml=float(koma.sum() * rr / len(koma)),
        nominal_bin_cycle_volume_ratio=float(koma.sum() / cfd.sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    controls = synthetic_controls()  # Must precede real-data flow evaluation.
    root = Path(__file__).resolve().parents[1]; base = root / 'outputs/aorta_inlet_replenishment'
    rawdir = base / 'e034_100k_mri_v2'; recovery = base / 'e034_100k_reconstruction_v1'
    protocol = base / 'E036_COMPLEX_INTEGRATION_PROTOCOL.md'
    sim_path = recovery / 'simulation_report.toml'; sim = tomllib.loads(sim_path.read_text())
    zpath = root / 'outputs/aorta_inlet_flow/e031_v1/inlet_numerical_evidence.npz'
    if sha(zpath) != '11a76b06940052d0b13361560b8ec85581c9c0a701eaaf5e895afdea701ca5c0':
        raise ValueError('Inlet quadrature/source evidence changed')
    recon = recovery / 'pc_reconstruction.h5'
    if sha(recon) != sim['reconstruction_sha256']: raise ValueError('Preserved reconstruction changed')
    with h5py.File(recon, 'r') as f:
        venc = float(f.attrs['Venc_m_per_s']); rr = float(f.attrs['RR_s'])
    images = {}; paths = [protocol, sim_path, zpath, recon]
    for encoding in ENCODINGS:
        path = rawdir / (encoding.lower() + '_result.jld2')
        if sha(path) != sim['raw_result_sha256'][encoding]: raise ValueError('Raw encoding identity changed')
        with h5py.File(path, 'r') as f:
            a = f['image'][:]
            if f['encoding'][()].decode() != encoding: raise ValueError('Wrong raw encoding label')
        images[encoding] = a['re'] + 1j * a['im']; paths.append(path)
    z = np.load(zpath, allow_pickle=False); grid = AortaGrid()
    query = ((z['triangle_centroids_source_mm'] / 1000 - grid.center)[:, grid.order])[:, ::-1]
    seqdir = root / 'outputs/aorta_dp2_mri_pipeline/koma/multibin_8x4mm/sequences'
    weights = None; timing = None
    for encoding in ENCODINGS:
        path = seqdir / (encoding.lower() + '.seq'); paths.append(path)
        if sha(path) != sim['sequence_sha256'][encoding]: raise ValueError('Sequence identity changed')
        current_timing, current_weights, _ = sequence_timing(path, rr)
        if not np.allclose(current_weights, z['frame_weights'], rtol=0, atol=2e-12):
            raise ValueError('Actual ADC timing differs from the bound reference')
        if encoding == 'REF': weights = current_weights; timing = current_timing
    old_evidence = base / 'e034_100k_audit_v1/paired_partial_flow_evidence.npz'; paths.append(old_evidence)
    old_manifest = json.loads((old_evidence.parent / 'MANIFEST.json').read_text())
    if sha(old_evidence) != old_manifest[old_evidence.name]['sha256']: raise ValueError('Old support evidence changed')
    old_common = np.load(old_evidence, allow_pickle=False)['common']
    staging = args.output.with_name(args.output.name + '.building-' + uuid.uuid4().hex[:8]); staging.mkdir(parents=True)
    registration = dict(experiment='E-036',created_utc=datetime.now(timezone.utc).isoformat(),
        method='complex interpolation, then phase difference; fixed point-signal support',threshold=.05,
        synthetic_controls=controls,code_sha256=sha(__file__),python=platform.python_version(),numpy=np.__version__,
        inputs={str(p):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in paths},
        CFD_flow_not_yet_evaluated=True,no_new_acquisition=True)
    (staging / 'registration.json').write_text(json.dumps(registration,indent=2)+'\n',encoding='utf-8')
    velocity, valid, ratios = complex_velocity_samples(images, grid.axes()[::-1], query, venc)
    fixed = valid.all(axis=0)
    if not fixed.any(): raise ValueError('No common signal-supported area; flow is unavailable')
    area = z['triangle_area_mm2']; normal = z['normal_source'][list(grid.order)]
    scale = .72 / rr
    # CFD enters only after signal support and the estimator have been fixed.
    native_vn = np.einsum('ncf,c->nf', z['face_velocities_source_m_s'], z['normal_source'])
    triangle_vn = native_vn[z['inferred_triangles']].mean(axis=1)
    native_full = area @ triangle_vn; native_paired = area[fixed] @ triangle_vn[fixed]
    cfd = (weights @ native_paired) * scale
    koma = integrate_flow(velocity[:, fixed], area[fixed], normal)
    old_joint = fixed & old_common
    cfd_old = (weights @ (area[old_joint] @ triangle_vn[old_joint])) * scale
    koma_old = integrate_flow(velocity[:, old_joint], area[old_joint], normal)
    centers = np.mod(np.angle(weights @ np.exp(2j * np.pi * np.arange(72) / 72)), 2 * np.pi) * .72 / (2 * np.pi)
    report = dict(experiment='E-036',status='COMPLETE_EXPLORATORY_NEAR_FULL_COMPLEX_INTEGRATION',
        synthetic_controls=controls,threshold=.05,coverage_fraction=float(area[fixed].sum()/area.sum()),
        paired_area_mm2=float(area[fixed].sum()),inferred_face_area_mm2=float(area.sum()),
        excluded_area_mm2=float(area[~fixed].sum()),full_face_supported=bool(fixed.all()),
        retained_triangles=int(fixed.sum()),total_triangles=len(area),physical_RR_s=rr,
        known_source_time_velocity_scale=scale,original_partial_support_retained=bool(np.all(fixed[old_common])),
        original_partial_area_fraction=float(area[old_joint].sum()/area.sum()),
        physical_units_metrics=metrics(cfd,koma,rr),original_partial_area_complex_metrics=metrics(cfd_old,koma_old,rr),
        CFD_physical_ml_s=cfd.tolist(),Koma_physical_ml_s=koma.tolist(),
        CFD_native_time_ml_s=(cfd/scale).tolist(),Koma_native_time_ml_s=(koma/scale).tolist(),
        native_source_phase_centers_s=centers.tolist(),native_phase_bin_order_one_based=(np.argsort(centers)+1).tolist(),
        CFD_full_native_waveform_ml_s=native_full.tolist(),CFD_paired_native_waveform_ml_s=native_paired.tolist(),
        CFD_excluded_native_transport_fraction=(1-native_paired.sum()/native_full.sum()),
        same_area_metrics_are_internal_consistency=True,registration_sha256=sha(staging/'registration.json'),
        limitations=['Post hoc estimator test, not independent velocity or boundary validation',
            'Inferred convex-hull cap and approximate point-cloud motion',
            'Eight acquired bins only; no recovered 72-frame MRI waveform',
            'Known source-phase/time-unit conversion is not a fitted alignment or gain',
            'Signal support is not proof of phase accuracy; no full-face claim while any area is excluded',
            'No new MRI acquisition and no1M run authorized'])
    (staging/'complex_inlet_report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    np.savez_compressed(staging/'complex_inlet_evidence.npz',velocity_mri_m_s=velocity,valid=valid,
        relative_complex_signal=ratios,fixed=fixed,area_mm2=area,CFD_physical_ml_s=cfd,Koma_physical_ml_s=koma,
        CFD_old_support_ml_s=cfd_old,Koma_old_support_ml_s=koma_old,frame_weights=weights,
        native_phase_centers_s=centers,normal_mri=normal,query_mri_zyx_m=query)
    with (staging/'inlet_flow_bins.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream);writer.writerow(['MRI_bin','nominal_MRI_time_s','native_CFD_phase_s',
            'CFD_physical_ml_s','Koma_physical_ml_s','CFD_native_time_ml_s','Koma_native_time_ml_s'])
        for b in range(8):writer.writerow([b+1,timing['bin_timing'][b]['nominal_midpoint_s'],centers[b],cfd[b],koma[b],cfd[b]/scale,koma[b]/scale])
    manifest={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in staging.iterdir()}
    (staging/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    staging.rename(args.output)
    print(json.dumps({key:report[key] for key in ('status','coverage_fraction','excluded_area_mm2',
        'full_face_supported','physical_units_metrics','original_partial_area_complex_metrics',
        'CFD_physical_ml_s','Koma_physical_ml_s','native_source_phase_centers_s')},indent=2))


if __name__ == '__main__': main()
