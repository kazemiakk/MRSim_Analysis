"""E-045: fixed-estimator inlet comparison for the native-zero 16-phase pilot."""
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

from scripts.aorta_hemodynamic_reference import AortaGrid
from scripts.build_aorta_inlet_picker import sha
from scripts.integrate_aorta_inlet_complex import (
    ENCODINGS, complex_velocity_samples, integrate_flow, metrics, synthetic_controls,
)


def validate_contract(sim, timing, images, weights):
    if sim['status'] != 'COMPLETE_E045_NATIVE_ZERO_16BIN_PILOT':
        raise ValueError('Completed native-zero acquisition required')
    if sim['particle_count'] != 100000 or sim['source_start_frame'] != 0 or sim['cardiac_bins'] != 16:
        raise ValueError('Authorized frame0/100k/16-bin pilot only')
    if timing['status'] != 'PASS_NATIVE_ZERO_TIMING' or timing['native_source_start_frame'] != 0:
        raise ValueError('Native-zero timing gate required')
    if weights.shape != (16, 72) or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError('Expected nonnegative 16x72 actual ADC weights')
    if not np.allclose(weights.sum(axis=1), 1, rtol=0, atol=1e-12):
        raise ValueError('Temporal weights must conserve a constant signal')
    if set(images) != set(ENCODINGS) or any(a.shape != (16, 32, 32, 72) for a in images.values()):
        raise ValueError('Expected four native-zero 16-bin images on the preserved grid')
    if not all(np.isfinite(a).all() for a in images.values()):
        raise ValueError('Nonfinite complex image')


def export_figure(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42})
    t = np.asarray(report['native_source_phase_centers_s'])
    cfd = np.asarray(report['CFD_native_time_ml_s'])
    koma = np.asarray(report['Koma_native_time_ml_s'])
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.4), sharex=True,
                            gridspec_kw={'height_ratios': [3, 1]})
    ax = axes[0]
    ax.plot(np.arange(72)*.01, report['CFD_full_native_waveform_ml_s'],
            color='#235f82', lw=1.7, label='Native CFD')
    ax.plot(t, cfd, 'o', color='#235f82', mfc='white', ms=6, label='CFD ADC average')
    ax.plot(t, koma, 's--', color='#c66d1b', ms=5, lw=1.2, label='Koma (16 phases)')
    ax.set_ylabel('Inward flow (mL/s)')
    ax.set_title('Aortic inlet flow | native CFD time zero', loc='left', pad=15)
    ax.legend(frameon=False, ncol=3, loc='upper right', fontsize=10)
    ax.set_ylim(bottom=min(0, float(koma.min())-1))
    ax.grid(axis='y', color='#e2e5e9', lw=.7)
    m = report['physical_units_metrics']
    ax.text(.985, .81, f"RMSE {m['rmse_ml_s']:.2f} mL/s  |  r = {m['correlation']:.4f}",
            ha='right', transform=ax.transAxes, color='#424b59', fontsize=10)
    axes[1].axhline(0, color='#747b84', lw=.8)
    axes[1].plot(t, koma-cfd, 's-', color='#c66d1b', ms=4, lw=1)
    axes[1].set_ylabel('Koma - CFD\n(mL/s)')
    axes[1].set_xlabel('Native CFD elapsed time (s)')
    axes[1].grid(axis='y', color='#e2e5e9', lw=.7)
    axes[1].set_xlim(0, .72)
    axes[1].set_xticks([0, .1, .2, .3, .4, .5, .6, .72])
    note = (f"100k pilot; fixed 5% signal support: {report['coverage_fraction']:.3%} of inferred face. "
            "Approximate point-cloud motion.\n"
            f"First MRI window center: {t[0]*1000:.2f} ms; no measured MRI point at zero. "
            "No fitted shift/gain; lines connect acquired values.")
    fig.text(.10, .025, note, fontsize=9, color='#424b59', va='bottom')
    fig.subplots_adjust(left=.10, right=.95, bottom=.16, top=.94, hspace=.12)
    for suffix in ('png', 'pdf'):
        fig.savefig(output/f'native_zero_inlet_waveform.{suffix}', dpi=600)
    plt.close(fig)


def analyze(mri, timing_path, output):
    if output.exists():
        raise FileExistsError(output)
    controls = synthetic_controls()
    root = Path(__file__).resolve().parents[1]
    base = root/'outputs/aorta_inlet_replenishment'
    protocol = base/'E045_NATIVE_ZERO_PROTOCOL.md'
    sim_path = mri/'simulation_report.toml'
    sim = tomllib.loads(sim_path.read_text(encoding='utf-8'))
    if sha(root/'koma_aorta/native_zero_pilot.jl') != sim['script_sha256']:
        raise ValueError('Acquisition script identity changed')
    phantom_report = tomllib.loads((mri/'phantom_report.toml').read_text())
    if sha(mri/'inlet_pilot.phantom') != phantom_report['phantom_sha256']:
        raise ValueError('Phantom identity changed')
    if sha(mri/'reset_preflight.toml') != sim['reset_preflight_sha256']:
        raise ValueError('Reset preflight identity changed')
    tracks = base/'e045_native0_100k_tracking_v1'
    if sha(tracks/'tracks.h5') != phantom_report['tracks_sha256']:
        raise ValueError('Tracking-to-phantom chain changed')
    track_registration = json.loads((tracks/'registration.json').read_text())
    if sha(protocol) != track_registration['protocol_sha256'] or track_registration['start_frame'] != 0:
        raise ValueError('Native-zero tracking protocol changed')
    timing = json.loads(timing_path.read_text(encoding='utf-8'))
    weights_path = timing_path.with_suffix('.npz')
    weights = np.load(weights_path, allow_pickle=False)['native_frame_weights']
    zpath = root/'outputs/aorta_inlet_flow/e031_v1/inlet_numerical_evidence.npz'
    if sha(zpath) != '11a76b06940052d0b13361560b8ec85581c9c0a701eaaf5e895afdea701ca5c0':
        raise ValueError('Original inferred inlet evidence changed')
    recon = mri/'pc_reconstruction.h5'
    if sha(recon) != sim['reconstruction_sha256']:
        raise ValueError('Reconstruction identity changed')
    with h5py.File(recon, 'r') as f:
        venc = float(f.attrs['Venc_m_per_s']); rr = float(f.attrs['RR_s'])
    if rr != timing['realized_Float32_motion_period_s']:
        raise ValueError('Motion period differs from audited timing')
    images = {}; paths = [protocol, sim_path, timing_path, weights_path, zpath, recon,
                          mri/'phantom_report.toml', mri/'reset_preflight.toml',
                          mri/'reference_viability.toml', mri/'inlet_pilot.phantom',
                          root/'koma_aorta/native_zero_pilot.jl', tracks/'tracks.h5',
                          tracks/'tracking_report.json', tracks/'registration.json',
                          root/'koma_aorta/Project.toml', root/'koma_aorta/Manifest.toml']
    for encoding in ENCODINGS:
        path = mri/(encoding.lower()+'_result.jld2')
        if sha(path) != sim['raw_result_sha256'][encoding]:
            raise ValueError('Raw encoding identity changed')
        with h5py.File(path, 'r') as f:
            a = f['image'][:]
            if f['encoding'][()].decode() != encoding:
                raise ValueError('Wrong encoding label')
        images[encoding] = a['re']+1j*a['im']; paths.append(path)
        sequence = Path(sim['sequence_directory'])/(encoding.lower()+'.seq')
        if sha(sequence) != sim['sequence_sha256'][encoding] or sha(sequence) != timing['sequence_sha256'][encoding]:
            raise ValueError('Sequence/timing identity changed')
        paths.append(sequence)
    validate_contract(sim, timing, images, weights)
    for name, status in [('reset_preflight.toml', 'PASS_RESET_PREFLIGHT'),
                         ('reference_viability.toml', 'PASS_REFERENCE_VIABILITY')]:
        if tomllib.loads((mri/name).read_text())['status'] != status:
            raise ValueError('Failed acquisition gate')
    z = np.load(zpath, allow_pickle=False); grid = AortaGrid()
    query = ((z['triangle_centroids_source_mm']/1000-grid.center)[:, grid.order])[:, ::-1]
    staging = output.with_name(output.name+'.building-'+uuid.uuid4().hex[:8])
    staging.mkdir(parents=True)
    registration = dict(experiment='E-045', created_utc=datetime.now(timezone.utc).isoformat(),
        python=platform.python_version(), numpy=np.__version__, h5py=h5py.__version__,
        code_sha256=sha(__file__), estimator_code_sha256=sha(root/'scripts/integrate_aorta_inlet_complex.py'),
        mri_directory=str(mri.resolve()), timing_path=str(timing_path.resolve()),
        inputs={str(p.resolve()): dict(bytes=p.stat().st_size, sha256=sha(p)) for p in paths},
        threshold=.05, synthetic_controls=controls, CFD_flow_not_yet_evaluated=True,
        estimator='E036 unchanged complex interpolation before phase; all-bin/all-encoding support',
        no_fitted_shift_or_gain=True, production_rerun_authorized=False)
    (staging/'registration.json').write_text(json.dumps(registration, indent=2)+'\n', encoding='utf-8')
    velocity, valid, ratios = complex_velocity_samples(images, grid.axes()[::-1], query, venc, threshold=.05)
    fixed = valid.all(axis=0)
    if not fixed.any():
        raise ValueError('No fixed signal-supported inlet area')
    area = z['triangle_area_mm2']; normal = z['normal_source'][list(grid.order)]
    # Evaluate CFD only after unchanged signal support has been frozen.
    native_vn = np.einsum('ncf,c->nf', z['face_velocities_source_m_s'], z['normal_source'])
    triangle_vn = native_vn[z['inferred_triangles']].mean(axis=1)
    native_full = area@triangle_vn; native_paired = area[fixed]@triangle_vn[fixed]
    scale = .72/rr
    cfd = (weights@native_paired)*scale
    koma = integrate_flow(velocity[:, fixed], area[fixed], normal)
    centers = np.array([r['source_center_s'] for r in timing['encodings']['REF']['source_bins']])
    errors = koma-cfd
    relative = np.divide(errors, cfd, out=np.full_like(errors, np.nan), where=abs(cfd)>1e-12)
    peak = int(np.argmax(cfd)); native_peak = int(np.argmax(native_full))
    report = dict(experiment='E-045', status='COMPLETE_NATIVE_ZERO_16BIN_INTERNAL_COMPARISON',
        particle_count=100000, source_start_frame=0, cardiac_bins=16,
        coverage_fraction=float(area[fixed].sum()/area.sum()), paired_area_mm2=float(area[fixed].sum()),
        inferred_face_area_mm2=float(area.sum()), excluded_area_mm2=float(area[~fixed].sum()),
        full_face_supported=bool(fixed.all()), retained_triangles=int(fixed.sum()), total_triangles=len(area),
        threshold=.05, physical_RR_s=rr, known_source_time_velocity_scale=scale,
        physical_units_metrics=metrics(cfd, koma, rr),
        CFD_physical_ml_s=cfd.tolist(), Koma_physical_ml_s=koma.tolist(),
        CFD_native_time_ml_s=(cfd/scale).tolist(), Koma_native_time_ml_s=(koma/scale).tolist(),
        native_source_phase_centers_s=centers.tolist(),
        CFD_full_native_waveform_ml_s=native_full.tolist(), CFD_paired_native_waveform_ml_s=native_paired.tolist(),
        signed_bin_error_ml_s=errors.tolist(), relative_bin_error=[None if not np.isfinite(x) else float(x) for x in relative],
        matched_CFD_peak_bin_one_based=peak+1, matched_CFD_peak_ml_s=float(cfd[peak]),
        Koma_at_matched_peak_ml_s=float(koma[peak]), peak_bin_error_fraction=float(relative[peak]),
        native_instantaneous_CFD_peak_ml_s=float(native_full[native_peak]), native_CFD_peak_time_s=native_peak*.01,
        first_MRI_window_center_s=float(centers[0]), MRI_measurement_at_exact_time_zero=False,
        registration_sha256=sha(staging/'registration.json'), synthetic_controls=controls,
        evidence_class='Internal consistency of approximate same-source CFD-to-Koma pipeline',
        limitations=['Inferred cap connectivity and physical boundary normal unverified',
            '16 acquired phases, not recovered 72-frame MRI; no measured instantaneous t=0 point',
            'Origin, bin count and idle timing changed together; not isolated temporal convergence',
            'Approximate point-cloud motion, 2ms RK4/identity interpolation and wrapped phase',
            'Controlled repeated trajectory cycle with all-spin cycle-end magnetization reset; not validated native multicycle CFD',
            'Signal support does not certify phase accuracy or absence of aliasing',
            'No fitted shift/gain, no independent physical validation, no new1M authorization'])
    (staging/'native_zero_report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    np.savez_compressed(staging/'native_zero_evidence.npz', velocity_mri_m_s=velocity, valid=valid,
        relative_complex_signal=ratios, fixed=fixed, area_mm2=area, normal_mri=normal,
        query_mri_zyx_m=query, frame_weights=weights, native_phase_centers_s=centers,
        CFD_physical_ml_s=cfd, Koma_physical_ml_s=koma)
    with (staging/'inlet_flow_bins.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['MRI_bin', 'native_CFD_phase_s', 'CFD_ml_s', 'Koma_ml_s', 'signed_error_ml_s', 'relative_error'])
        for b in range(16):
            writer.writerow([b+1, centers[b], cfd[b]/scale, koma[b]/scale, errors[b]/scale, report['relative_bin_error'][b]])
    export_figure(report, staging)
    manifest = dict(report_sha256=sha(staging/'native_zero_report.json'), plotter_sha256=sha(__file__),
        files={p.name: dict(bytes=p.stat().st_size, sha256=sha(p)) for p in staging.iterdir()})
    (staging/'MANIFEST.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    staging.rename(output)
    print(json.dumps({k: report[k] for k in ['status', 'coverage_fraction', 'physical_units_metrics',
        'peak_bin_error_fraction', 'first_MRI_window_center_s']}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mri', type=Path, required=True)
    p.add_argument('--timing', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); analyze(a.mri, a.timing, a.output)
