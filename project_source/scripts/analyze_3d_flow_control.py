"""E051 full-volume known-flow gates; REF-conditional phase is not magnitude validation."""
import argparse
import csv
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import tomllib
import uuid

import h5py
import numpy as np

from scripts.prepare_3d_flow_control import ROOT, BASE, ENC, SHAPE, SPACING, dump
from scripts.analyze_phantom_precision import sort_profiles, fft_image
from scripts.analyze_remapped_confirmation import velocities, fixed_mask
from scripts.build_aorta_inlet_picker import sha

MODELS = ('static_landmarks', 'plug_positive', 'plug_negative')


def predefined_masks(model, indices):
    geometry = np.zeros(SHAPE[1:], bool)
    geometry[tuple(indices.T)] = True
    core = geometry.copy()
    if model != 'static_landmarks':
        core[:] = False; core[13:17, 22:28, :] = True
    return geometry, core


def known_velocity_metrics(images, expected, geometry, core):
    mask = fixed_mask(images, .05)
    fields = velocities(images, 5.)
    keep = mask&core
    if keep.any():
        delta = fields[:, keep].astype(float)-expected
        metric = dict(vector_rmse_m_s=float(np.sqrt(np.mean(np.sum(delta**2, axis=-1)))),
            maximum_component_error_m_s=float(abs(delta).max()),
            mean_velocity_m_s=fields[:, keep].mean(axis=(0, 1), dtype=float).tolist(),
            first_last_cardiac_bin_axial_delta_m_s=float(abs(fields[0, keep, 0].mean(dtype=float)-fields[-1, keep, 0].mean(dtype=float))))
    else:
        metric = None
    bins = []
    for b in range(15):
        valid = core.copy()
        for image in images.values():
            mag = abs(image[b]); valid &= (mag >= .05*mag.max())&(mag > 0)
        bins.append(dict(bin=b+1, core_coverage_fraction=float(valid.sum()/core.sum()),
            retained_core_voxels=int(valid.sum()),
            mean_velocity_m_s=None if not valid.any() else fields[b, valid].mean(axis=0, dtype=float).tolist()))
    return dict(fixed_core_coverage_fraction=float(keep.sum()/core.sum()),
        fixed_geometry_coverage_fraction=float((mask&geometry).sum()/geometry.sum()),
        fixed_retained_core_voxels=int(keep.sum()), predefined_core_voxels=int(core.sum()),
        metrics=metric, bins=bins), fields, mask


def conditional_error(ref, observed, rf_times, delta0, delta1, velocity):
    rows = []; total_delta = total_reference = 0.
    profiles = len(rf_times)
    a = ref.reshape(profiles, 400); b = observed.reshape(profiles, 400)
    first = delta1@velocity; zero = delta0@velocity
    for quartile, ids in enumerate(np.array_split(np.arange(profiles), 4), 1):
        phase = -2*np.pi*(first[None]+rf_times[ids, None]*zero[None])
        predicted = a[ids]*np.exp(1j*phase)
        delta = b[ids]-predicted
        numerator = float(np.vdot(delta.ravel(), delta.ravel()).real)
        denominator = float(np.vdot(a[ids].ravel(), a[ids].ravel()).real)
        if denominator == 0:
            raise ValueError('No REF signal in acquisition quartile')
        rows.append(dict(quartile=quartile, relative_complex_L2=float(np.sqrt(numerator/denominator))))
        total_delta += numerator; total_reference += denominator
    return dict(relative_complex_L2=float(np.sqrt(total_delta/total_reference)), acquisition_quartiles=rows)


def static_localization(image, indices):
    rows = []
    for b in range(15):
        for number, point in enumerate(indices, 1):
            low = np.maximum(point-3, 0); high = np.minimum(point+4, SHAPE[1:])
            slices = tuple(slice(a, z) for a, z in zip(low, high))
            observed = np.array(np.unravel_index(abs(image[b][slices]).argmax(), tuple(high-low)))+low
            rows.append(dict(bin=b+1, landmark=number, expected_index=point.tolist(), observed_index=observed.tolist(),
                             exact_match=bool(np.array_equal(observed, point))))
    return rows


def figures(folder, summaries, views, positions):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
                         'axes.spines.top':False, 'axes.spines.right':False})
    names = ('Static landmarks', '+1.50 m/s plug flow', '−0.75 m/s plug flow')
    fig = plt.figure(figsize=(14, 4.5))
    for i, (name, title) in enumerate(zip(MODELS, names), 1):
        ax = fig.add_subplot(1, 3, i, projection='3d')
        p = positions[name]
        stride = 1 if name == 'static_landmarks' else 16
        p = p[::stride]
        # These are prescribed sample coordinates; color is reconstructed
        # axial velocity sampled from cardiac bin1 at those grid centers.
        c = views[name]['sample_velocity'][::stride]
        ax.scatter(p[:, 0]*1000, p[:, 1]*1000, p[:, 2]*1000, c='#b4b8bd', s=5, alpha=.3)
        ax.scatter(p[:, 0]*1000, p[:, 1]*1000, p[:, 2]*1000, c=c, cmap='coolwarm', vmin=-1.6, vmax=1.6,
                   s=42 if name == 'static_landmarks' else 3, alpha=.9)
        ax.set(title=title, xlabel='MRI x (mm)', ylabel='MRI y (mm)', zlabel='MRI z (mm)')
        if name != 'static_landmarks':
            ax.set_box_aspect((4, 1, 1))
        ax.view_init(elev=20, azim=-62)
    fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(-1.6, 1.6), cmap='coolwarm'), ax=fig.axes,
                 fraction=.02, pad=.06, label='Reconstructed axial velocity (m/s)')
    fig.suptitle('3D known-flow controls | bin 1 velocity at prescribed samples; display aspect adjusted', y=.99)
    fig.subplots_adjust(left=.02, right=.87, bottom=.12, top=.85, wspace=.2)
    fig.savefig(folder/'known_flow_3d.png', dpi=300); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    ref = views['static_landmarks']['reference_magnitude']
    for ax, (axis, title, labels, extent) in zip(axes, [
        (0, 'XY magnitude projection', ('MRI x (mm)', 'MRI y (mm)'), [-250.625, 249.375, -40.625, 39.375]),
        (1, 'XZ magnitude projection', ('MRI x (mm)', 'MRI z (mm)'), [-250.625, 249.375, -41.25, 38.75]),
        (2, 'YZ magnitude projection', ('MRI y (mm)', 'MRI z (mm)'), [-40.625, 39.375, -41.25, 38.75])]):
        ax.imshow(ref.max(axis=axis), origin='lower', extent=extent, cmap='gray', aspect='equal', interpolation='nearest')
        p = positions['static_landmarks']*1000
        x, y = ((0, 1) if axis == 0 else ((0, 2) if axis == 1 else (1, 2)))
        ax.scatter(p[:, x], p[:, y], s=35, facecolors='none', edgecolors='#db9a32', linewidths=1)
        ax.set(title=title, xlabel=labels[0], ylabel=labels[1])
    fig.suptitle('3D static localization | circles mark the five prescribed coordinates')
    fig.tight_layout(); fig.savefig(folder/'static_localization.png', dpi=300); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    y = (np.arange(64)-32)*1.25
    for ax, name, title in zip(axes, MODELS[1:], names[1:]):
        profile = views[name]['phase_profile']
        expected = np.full(64, np.nan); expected[21:29] = summaries[name]['velocity_mri_m_s'][0]
        ax.plot(y, expected, 'k--', lw=1.5, label='Known')
        ax.plot(y, profile.mean(axis=0), 'o-', ms=3, color='#235f82', label='Koma mean')
        ax.fill_between(y, profile.min(axis=0), profile.max(axis=0), color='#235f82', alpha=.18, label='15-bin range')
        ax.axhline(0, color='.7', lw=.8); ax.set(title=title, xlabel='MRI y (mm)', xlim=(-25, 5), ylim=(-1.1, 1.8))
        ax.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('Axial velocity (m/s)'); axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle('Matched transverse profiles | MRI x = 0 mm, z = −5 mm; fixed 5% support')
    fig.tight_layout(); fig.savefig(folder/'known_velocity_profiles.png', dpi=300); plt.close(fig)


def analyze(inputs, acquisition, output):
    if output.exists():
        raise FileExistsError(output)
    contract = json.loads((inputs/'contract.json').read_text())
    runtime = tomllib.loads((acquisition/'control_report.toml').read_text())
    if runtime['status'] != 'COMPLETE_E051_3D_ACQUISITIONS' or sha(inputs/'controls.h5') != runtime['input_sha256']:
        raise ValueError('Acquisition/input identity mismatch')
    for path, expected in [(inputs/'contract.json', runtime['contract_sha256']),
                           (BASE/'PROTOCOL.md', runtime['protocol_sha256']),
                           (ROOT/'koma_migration_e046/known_flow_3d_e051.jl', runtime['script_sha256'])]:
        if sha(path) != expected:
            raise ValueError('Registered source changed')
    stage = output.with_name(output.name+'.building-'+uuid.uuid4().hex[:8]); stage.mkdir(parents=True)
    sources = [Path(__file__).resolve(), inputs/'contract.json', inputs/'controls.h5', acquisition/'control_report.toml',
        BASE/'PROTOCOL.md', ROOT/'scripts/prepare_3d_flow_control.py', ROOT/'scripts/analyze_phantom_precision.py',
        ROOT/'scripts/analyze_remapped_confirmation.py', ROOT/'scripts/build_aorta_inlet_picker.py',
        ROOT/'outputs/phantom_precision_e050/analysis_v1/report.json',
        ROOT/'outputs/phantom_precision_e050/analysis_v1/validation.json']
    sources += [acquisition/(name+'_'+e.lower()+'.h5') for name in MODELS for e in ENC]
    dump(stage/'registration.json', dict(status='REGISTERED_BEFORE_3D_METRICS', experiment='E-051',
        created_utc=datetime.now(timezone.utc).isoformat(), inputs={str(p):sha(p) for p in sources}))
    summaries = {}; gates = []; views = {}; positions = {}; fft_audits = []
    with h5py.File(inputs/'controls.h5') as source:
        labels = source['labels'][:]; rf_times = source['rf_start_s'][:]
        for model in MODELS:
            print('ANALYZE_E051', model, flush=True)
            images = {}; raw_ref = None; signal_checks = []
            indices = source[model+'/indices'][:]; p = source[model+'/positions'][:]
            velocity = source[model+'/velocity'][:]
            for e in ENC:
                path = acquisition/(model+'_'+e.lower()+'.h5')
                if sha(path) != runtime['result_sha256'][model+'_'+e.lower()]:
                    raise ValueError('Acquired signal hash differs')
                with h5py.File(path) as f:
                    raw = f['raw_signal_f64'][:].ravel(); image = f['image'][:]
                if raw.size != 12288000 or raw.dtype != np.complex128 or image.shape != SHAPE or image.dtype != np.complex64:
                    raise ValueError('Invalid 3D result schema')
                if not np.isfinite(raw).all() or not np.isfinite(image).all():
                    raise ValueError('Nonfinite acquisition')
                k = sort_profiles(raw.astype(np.complex64), labels, SHAPE)
                independent = fft_image(k)
                relative = float(abs(image-independent).max()/max(float(abs(image).max()), 1e-30))
                fft_audits.append(dict(model=model, encoding=e, independent_FFT_relative_max=relative))
                gates.append(dict(name=model+' '+e+' independent FFT', passed=relative <= 3e-6, observed=relative, limit=3e-6))
                if e == 'REF':
                    raw_ref = raw
                else:
                    check = conditional_error(raw_ref, raw, rf_times, source[e+'_delta_m0'][:], source[e+'_delta_m1'][:], velocity)
                    signal_checks.append(dict(encoding=e, **check))
                    gates.append(dict(name=model+' '+e+' REF-conditional known-phase signal', passed=check['relative_complex_L2'] <= .003,
                                      observed=check['relative_complex_L2'], limit=.003))
                images[e] = image
                del k, independent, raw
                gc.collect()
            geom, core = predefined_masks(model, indices)
            result, fields, mask = known_velocity_metrics(images, velocity, geom, core)
            metric = result['metrics']
            limit = .002 if model == 'static_landmarks' else .005
            gates.append(dict(name=model+' vector RMSE', passed=metric is not None and metric['vector_rmse_m_s'] <= limit,
                              observed=None if metric is None else metric['vector_rmse_m_s'], limit=limit))
            maxlimit = .005 if model == 'static_landmarks' else .010
            gates.append(dict(name=model+' maximum component error', passed=metric is not None and metric['maximum_component_error_m_s'] <= maxlimit,
                              observed=None if metric is None else metric['maximum_component_error_m_s'], limit=maxlimit))
            if model != 'static_landmarks':
                gates.append(dict(name=model+' fixed predefined core coverage', passed=result['fixed_core_coverage_fraction'] >= .95,
                                  observed=result['fixed_core_coverage_fraction'], minimum=.95))
                gates.append(dict(name=model+' first-last cardiac-bin delta', passed=metric is not None and metric['first_last_cardiac_bin_axial_delta_m_s'] <= .002,
                                  observed=None if metric is None else metric['first_last_cardiac_bin_axial_delta_m_s'], limit=.002))
            locations = static_localization(images['REF'], indices) if model == 'static_landmarks' else []
            if locations:
                gates.append(dict(name='static exact 3D localization all75checks', passed=all(r['exact_match'] for r in locations),
                                  observed=sum(r['exact_match'] for r in locations), required=len(locations)))
            summaries[model] = dict(velocity_mri_m_s=velocity.tolist(), spins=len(p), **result,
                                    conditional_signal_checks=signal_checks, localization=locations)
            profile = fields[:, 14, :, 200, 0].copy()
            profile[:, ~mask[14, :, 200]] = np.nan
            sample_velocity = fields[0][tuple(indices.T)][:, 0].copy()
            sample_velocity[~mask[tuple(indices.T)]] = np.nan
            views[model] = dict(reference_magnitude=abs(images['REF'][0]), phase_profile=profile,
                sample_velocity=sample_velocity)
            positions[model] = p
            del images, fields, raw_ref
            gc.collect()
    figures(stage, summaries, views, positions)
    report = dict(experiment='E-051', status='PASS_3D_KNOWN_FLOW_IMAGE_GATES' if all(g['passed'] for g in gates) else 'FAIL_3D_KNOWN_FLOW_IMAGE_GATES',
        created_utc=datetime.now(timezone.utc).isoformat(), models=summaries, gates=gates, independent_FFT_audits=fft_audits,
        prediction='Independent Pulseq gradient-phase law conditional on measured Koma REF; NOT independent magnitude/Bloch validation',
        limitations=['Analytic numerical verification only; not independent physical validation.',
            'Deterministic lattice axial flow in periodic FOV, not random particle-density convergence.',
            'No nonzero through-slab flow, acceleration, shear, inlet replenishment, wall, TKE or CFD-accuracy certification.',
            'No fitting, threshold relaxation, phase filling or automatic300k/1M promotion.',
            'First/last cardiac bins are not early/late absolute acquisition times; raw acquisition quartiles and separate late-clock prefix address clock sensitivity.'])
    dump(stage/'report.json', report)
    rows = []
    for name, data in summaries.items():
        for row in data['bins']:
            values = row['mean_velocity_m_s']
            rows.append(dict(model=name, bin=row['bin'], core_coverage_fraction=row['core_coverage_fraction'],
                vx_m_s=None if values is None else values[0], vy_m_s=None if values is None else values[1], vz_m_s=None if values is None else values[2]))
    with (stage/'known_velocity_bins.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    dump(stage/'MANIFEST.json', dict(outputs={p.name:sha(p) for p in stage.iterdir() if p.is_file()}))
    stage.rename(output)
    print(json.dumps(dict(status=report['status'], models={k:{q:v[q] for q in ('fixed_core_coverage_fraction', 'metrics')} for k, v in summaries.items()},
                         failed_gates=[g for g in gates if not g['passed']])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--acquisition', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); analyze(args.inputs, args.acquisition, args.output)
