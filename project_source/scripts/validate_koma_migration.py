"""Independent E046 source/FFT/mask/phase/flux checks, without analyzer helpers."""
import argparse
import itertools
import json
from pathlib import Path
import tomllib

import h5py
import numpy as np

from scripts.build_aorta_inlet_picker import sha

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'outputs/upstream_migration_e046'


def interpolate_manual(a, axes, points):
    indices = [np.searchsorted(ax, points[:, i], side='right')-1 for i, ax in enumerate(axes)]
    indices = [np.clip(k, 0, len(ax)-2) for k, ax in zip(indices, axes)]
    fractions = [(points[:, i]-ax[k])/(ax[k+1]-ax[k]) for i, (ax, k) in enumerate(zip(axes, indices))]
    if any((p < 0).any() or (p > 1).any() for p in fractions):
        raise ValueError('Query outside image grid')
    result = np.zeros((a.shape[0], len(points)), complex)
    for offsets in itertools.product([0, 1], repeat=3):
        weight = np.prod([f if o else 1-f for f, o in zip(fractions, offsets)], axis=0)
        result += a[:, indices[0]+offsets[0], indices[1]+offsets[1], indices[2]+offsets[2]]*weight
    return result


def validate(directories, output):
    if output.exists():
        raise FileExistsError(output)
    checks = 0
    def check(value, message):
        nonlocal checks
        if not value:
            raise AssertionError(message)
        checks += 1
    legacy = json.loads((ROOT/'outputs/aorta_inlet_replenishment/e046_legacy_snapshot_v1/registration.json').read_text())
    for name, identity in legacy['inputs'].items():
        p = ROOT/name
        check(p.stat().st_size == identity['bytes'] and sha(p) == identity['sha256'], f'Legacy changed: {name}')
    reports = {}
    for directory in directories:
        report = json.loads((directory/'report.json').read_text())
        registration = json.loads((directory/'registration.json').read_text())
        manifest = json.loads((directory/'MANIFEST.json').read_text())
        check(report['status'] == 'COMPLETE_CONTROLLED_MIGRATION_SENSITIVITY', 'Incomplete analysis')
        for name, ident in registration['inputs'].items():
            p = ROOT/name
            check(p.stat().st_size == ident['bytes'] and sha(p) == ident['sha256'], f'Analysis source changed: {name}')
        for name, ident in manifest.items():
            p = directory/name
            check(p.stat().st_size == ident['bytes'] and sha(p) == ident['sha256'], 'Analysis export changed')
        case = report['case']; arms = ['legacy', 'updated_nomap', 'updated_map']
        inputs = BASE/('aorta_inputs_v1' if case == 'aorta' else 'phantom_inputs_v3')
        mapdir = BASE/f'{case}_map_v1'
        mr = json.loads((mapdir/'map_report.json').read_text())
        with h5py.File(mapdir/'cycle_map.h5') as f:
            mapping = f['cycle_map'][:]; distance = f['distance_m'][:]
        with h5py.File(inputs/'case_inputs.h5') as f:
            basepoints = np.column_stack([f[c][:20000] for c in ['x', 'y', 'z']])
            initial = basepoints+np.column_stack([f[c][0, :20000] for c in ['dx', 'dy', 'dz']])
            final = basepoints+np.column_stack([f[c][-1, :20000] for c in ['dx', 'dy', 'dz']])
        check(mapping.shape == (20001,) and mapping[-1] == 20001 and mapping[:20000].min() >= 1 and mapping[:20000].max() <= 20000, 'Map indices/sentinel')
        independent_distance = np.sqrt(np.sum((initial.astype(float)-final[mapping[:20000]-1].astype(float))**2, axis=-1))
        check(np.allclose(independent_distance, distance, rtol=1e-6, atol=2e-8), 'Map distance direction')
        for i in np.linspace(0, 19999, 20).astype(int):
            nearest = np.sqrt(np.min(np.sum((final.astype(float)-initial[i].astype(float))**2, axis=-1)))
            check(abs(nearest-independent_distance[i]) < 1e-7, 'Independent sampled nearest-neighbor minimum')
        counts = np.bincount(mapping[:20000]-1, minlength=20000)
        check(counts.max() == mr['maximum_source_multiplicity'] and np.mean(counts > 0) == mr['unique_source_fraction'], 'Map multiplicity')
        images = {}; mags = {}; masks = {}; velocities = {}; point_signals = {}; point_masks = {}
        flow = report['flow']
        if case == 'aorta':
            z = np.load(ROOT/'outputs/aorta_inlet_flow/e031_v1/inlet_numerical_evidence.npz')
            center = np.array([.038389891386032104, .015447176992893219, -.023723918944597244])
            points = ((z['triangle_centroids_source_mm']/1000-center)[:, [2, 0, 1]])[:, ::-1]
            axes = tuple((np.arange(n)-n/2)*f/n for n, f in [(32, .128), (32, .128), (72, .28)])
        venc = 2.5 if case == 'aorta' else 5.
        for arm in arms:
            run = (f'{case}_updated_nomap_v2' if arm == 'updated_nomap' else f'{case}_{arm}_v1')
            folder = BASE/'phantom_inputs_v3' if case == 'phantom' and arm == 'legacy' else BASE/run
            arrays = {}
            for encoding in ['ref', 'x', 'y', 'z']:
                suffix = '_legacy_result.h5' if case == 'phantom' and arm == 'legacy' else '_result.h5'
                with h5py.File(folder/(encoding+suffix)) as f:
                    k = f['rawsort'][:]; a = f['image'][:]
                reconstructed = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(k, axes=(1, 2, 3)), axes=(1, 2, 3)), axes=(1, 2, 3))
                check(abs(a-reconstructed).max() <= 3e-6*abs(a).max(), 'FFT identity')
                arrays[encoding] = a
            valid = np.ones(arrays['ref'].shape, bool)
            for a in arrays.values():
                power = (a.real*a.real+a.imag*a.imag)
                # Amplitude gate independently expressed as squared power.
                valid &= (power >= .0025*power.max(axis=(1, 2, 3), keepdims=True)) & (power > 0)
            masks[arm] = valid.min(axis=0)
            mags[arm] = abs(arrays['ref'])
            rv = []
            for e in ['x', 'y', 'z']:
                re = arrays[e].real*arrays['ref'].real+arrays[e].imag*arrays['ref'].imag
                im = arrays[e].imag*arrays['ref'].real-arrays[e].real*arrays['ref'].imag
                rv.append(np.arctan2(im, re)*venc/np.pi)
            velocities[arm] = np.stack(rv, axis=-1)
            if case == 'aorta':
                samples = {e: interpolate_manual(a, axes, points) for e, a in arrays.items()}
                validp = np.ones(samples['ref'].shape, bool)
                for e, sample in samples.items():
                    validp &= abs(sample) >= .05*abs(arrays[e]).max(axis=(1, 2, 3))[:, None]
                point_signals[arm] = samples; point_masks[arm] = validp.all(axis=0)
            del arrays
        union = np.any(list(masks.values()), axis=0); joint = np.all(list(masks.values()), axis=0)
        with np.load(directory/'support_evidence.npz') as n:
            check(np.array_equal(joint, n['joint']) and np.array_equal(union, n['union']), 'Joint/union masks')
            for arm in arms:
                check(np.array_equal(masks[arm], n[arm+'_mask']), 'Per-arm mask')
        check(int(union.sum()) == report['union_voxels'] and int(joint.sum()) == report['joint_voxels'], 'Support totals')
        for pair, d in report['differences'].items():
            a, b = pair.split('_to_')
            if not joint.any():
                check(d['magnitude_relative_rmse'] is None, 'Empty support must remain null')
                continue
            av, bv = mags[a][:, joint].astype(float), mags[b][:, joint].astype(float)
            rms = np.sqrt(np.sum((bv-av)**2)/np.sum(av**2))
            check(np.isclose(rms, d['magnitude_relative_rmse'], rtol=2e-7), 'Magnitude relative RMSE')
            # Principal angle difference via complex unit vectors, not modulo expression.
            angle = (velocities[b][:, joint].astype(float)-velocities[a][:, joint].astype(float))*np.pi/venc
            difference = np.arctan2(np.sin(angle), np.cos(angle))*venc/np.pi
            check(np.isclose(np.sqrt(np.mean(difference**2)), d['velocity_wrapped_component_rmse_m_s'], rtol=2e-5, atol=2e-8), 'Circular velocity RMSE')
        if case == 'aorta':
            jointp = np.all(list(point_masks.values()), axis=0); area = z['triangle_area_mm2']
            check(np.isclose(area[jointp].sum()/area.sum(), flow['joint_coverage_fraction']), 'Aortic paired coverage')
            for arm in arms:
                samples = point_signals[arm]
                v = np.stack([np.angle(samples[e]/samples['ref'])*venc/np.pi for e in ['x', 'y', 'z']], axis=-1)
                normal = z['normal_source'][[2, 0, 1]]
                q = np.sum(np.sum(v[:, jointp]*normal, axis=-1)*area[jointp], axis=-1)
                check(np.allclose(q, flow['Koma_ml_s'][arm], rtol=1e-6, atol=1e-5), 'Independent manual complex integration')
        else:
            with h5py.File(ROOT/'outputs/dp150_hemodynamic_comparison/e025_v1/results_v2/matched_reference.h5') as f:
                area = f['plane_area_fraction_2'][:, :, 70]*3.125
            chosen = joint[:, :, 70] & (area > 0)
            check(np.isclose(area[chosen].sum()/area.sum(), flow['joint_coverage_fraction']), 'Phantom section coverage')
            for arm in arms:
                q = np.sum(velocities[arm][:, :, :, 70, 0][:, chosen]*area[chosen], axis=-1)
                check(np.allclose(q, flow['Koma_ml_s'][arm], rtol=1e-6, atol=1e-5), 'Independent section flux')
        for arm in arms:
            if flow['Koma_ml_s'][arm] is None:
                continue
            ref, est = np.array(flow['CFD_ml_s']), np.array(flow['Koma_ml_s'][arm])
            check(np.isclose(np.linalg.norm(est-ref)/np.sqrt(len(ref)), flow['metrics'][arm]['rmse_ml_s']), 'Flow RMSE')
            check(np.isclose(est.sum()/ref.sum(), flow['metrics'][arm]['nominal_bin_cycle_volume_ratio']), 'Cycle transport ratio')
        reports[case] = dict(report_sha256=sha(directory/'report.json'), joint_support_voxels=int(joint.sum()))
    result = dict(status='PASS_INDEPENDENT_E046_CHECKS', checks=checks, legacy_inputs_unchanged=len(legacy['inputs']),
                  reports=reports, validator_sha256=sha(__file__), PNG_visual_review='Separate manual view_image check required', PDF_rendering='Not performed')
    output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis', type=Path, action='append', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); validate(a.analysis, a.output)
