"""E050: source-bound old/new 100k precision comparison; no accuracy promotion.

Run only after all four Float64 acquisitions finish. The saved E048 CFD
comparators are kept distinct from each other and from reconstructed phase.
"""
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

from scripts.analyze_remapped_confirmation import fixed_mask, velocities, weighted_metrics, flow_metrics
from scripts.build_aorta_inlet_picker import sha
from scripts.compare_dp150_hemodynamics import REGIONS, PLANES

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'outputs/phantom_precision_e050'
OLD = ROOT/'outputs/remapped_confirmation_e047/phantom_mapped_v1'
INPUTS = OLD.parent/'phantom_inputs_v1'
REF = ROOT/'outputs/phantom_diagnosis_e048/reference_v1'
LOCAL = ROOT/'outputs/phantom_diagnosis_e048/local_v1'
ENC = ('REF', 'X', 'Y', 'Z')
ARMS = ('saved_f32', 'candidate_f64')
THRESHOLDS = (.05, .10, .15, .20)


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def toml(path):
    return tomllib.loads(Path(path).read_text(encoding='utf-8'))


def fft_image(k):
    axes = (1, 2, 3)
    values = np.asarray(k, dtype=np.complex128)
    return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(values, axes=axes), axes=axes), axes=axes)


def sort_profiles(signal, labels, shape):
    """Independent Python sort, including completeness/uniqueness checks."""
    labels = np.asarray(labels)
    if len(shape) != 4 or labels.ndim != 2 or labels.shape[1] != 3:
        raise ValueError('Expected bin/slice/phase/readout shape and LIN/PAR/PHS labels')
    nb, nz, ny, nx = shape
    if len(labels) != nb*nz*ny or np.asarray(signal).size != nx*len(labels):
        raise ValueError('Incomplete profile/sample count')
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError('Integer labels required')
    y, z, b = labels.T
    if not (np.all((0 <= y)&(y < ny)) and np.all((0 <= z)&(z < nz)) and np.all((0 <= b)&(b < nb))):
        raise ValueError('Out-of-range labels')
    if len(np.unique(np.ravel_multi_index((b, z, y), (nb, nz, ny)))) != len(labels):
        raise ValueError('Duplicate labels')
    result = np.empty(shape, dtype=np.asarray(signal).dtype)
    result[b, z, y] = np.asarray(signal).reshape(len(labels), nx)
    return result


def coverage(fraction, mask, eligible=None):
    eligible = fraction > 0 if eligible is None else np.asarray(eligible)&(fraction > 0)
    denominator = float(np.sum(fraction[eligible], dtype=float))
    return None if denominator == 0 else float(np.sum(fraction[mask&eligible], dtype=float)/denominator)


def region_rows(reference, fields, masks, fraction, source_z, threshold):
    rows = []
    pair = np.logical_and.reduce(list(masks.values()))
    for name, (low, high) in REGIONS.items():
        eligible = (fraction > 0)&((source_z >= low)&(source_z < high))[None, None, :]
        for arm, field in fields.items():
            for policy, mask in (('own', masks[arm]), ('paired', pair)):
                keep = mask&eligible
                metric = weighted_metrics(reference, field, fraction, keep)
                rows.append(dict(threshold=threshold, region=name, arm=arm, support=policy,
                    coverage_fraction=coverage(fraction, keep, eligible), metrics=metric))
    return rows


def partial_flow(reference, estimate, area, mask, period):
    keep = np.asarray(mask)&(area > 0)
    retained = float(area[keep].sum())
    full = float(area.sum())
    if retained == 0:
        return dict(coverage_fraction=0.0 if full else None, retained_area_mm2=0.0,
                    geometric_area_mm2=full, CFD_ml_s=None, Koma_ml_s=None, metrics=None)
    a = reference[:, keep]@area[keep]
    b = estimate[:, keep]@area[keep]
    return dict(coverage_fraction=retained/full, retained_area_mm2=retained,
                geometric_area_mm2=full, CFD_ml_s=a.tolist(), Koma_ml_s=b.tolist(),
                metrics=flow_metrics(a, b, period))


def bind_reference(folder, name, expected):
    path = folder/name
    manifest = json.loads((folder/'MANIFEST.json').read_text())['outputs']
    if manifest[name] != expected or sha(path) != expected:
        raise ValueError('Frozen E048 reference changed: '+str(path))
    return path


def load_arm(folder, report, candidate=False):
    images, audits = {}, []
    with h5py.File(INPUTS/'legacy_timing.h5') as timing:
        for e in ENC:
            path = folder/(e.lower()+'_result.h5')
            if sha(path) != report['result_sha256'][e]:
                raise ValueError('Result changed: '+str(path))
            with h5py.File(path) as f:
                k, image = f['rawsort'][:], f['image'][:]
                if k.shape != (15, 32, 64, 400) or image.shape != k.shape:
                    raise ValueError('Unexpected result shape')
                if k.dtype != np.complex64 or image.dtype != np.complex64 or not np.isfinite(k).all() or not np.isfinite(image).all():
                    raise ValueError('Invalid primary reconstruction')
                independent = fft_image(k)
                error = float(abs(image-independent).max()/max(float(abs(image).max()), 1e-30))
                if error > 3e-6:
                    raise ValueError('Saved reconstruction differs from independent FFT')
                audit = dict(encoding=e, independent_fft_relative_max=error)
                if candidate:
                    raw = f['raw_signal_f64'][:]
                    if raw.dtype != np.complex128 or not np.isfinite(raw).all():
                        raise ValueError('Finite ComplexF64 raw acquisition required')
                    k64 = sort_profiles(raw, timing[e]['labels'][:], k.shape)
                    if not np.array_equal(k, k64.astype(np.complex64)):
                        raise ValueError('Raw Float64 signal does not reproduce saved sort')
                    independent64 = fft_image(k64)
                    audit['raw_signal_sort_exact_after_f32_cast'] = True
                    audit['f32_vs_f64_reconstruction_relative_max'] = float(abs(image-independent64).max()/max(float(abs(independent64).max()), 1e-30))
                    if audit['f32_vs_f64_reconstruction_relative_max'] > 3e-6:
                        raise ValueError('Unexpected reconstruction precision effect')
                    del raw, k64, independent64
                images[e] = image
                audits.append(audit)
                del k, independent
            gc.collect()
    return images, audits


def temporal_kspace_difference(old, new):
    """Old raw time series was not saved; compare its exactly labeled profiles."""
    rows = []
    with h5py.File(INPUTS/'legacy_timing.h5') as t:
        for e in ENC:
            labels = t[e]['labels'][:]
            times = t[e]['adc_first_s'][:]
            with h5py.File(old/(e.lower()+'_result.h5')) as a, h5py.File(new/(e.lower()+'_result.h5')) as b:
                ka, kb = a['rawsort'][:], b['rawsort'][:]
            for quartile, ids in enumerate(np.array_split(np.argsort(times), 4), 1):
                y, z, phase = labels[ids].T
                va, vb = ka[phase, z, y].astype(np.complex128), kb[phase, z, y].astype(np.complex128)
                denominator = float(np.linalg.norm(va))
                rows.append(dict(encoding=e, acquisition_quartile=quartile,
                    first_ADC_time_s=float(times[ids].min()), last_ADC_time_s=float(times[ids].max()),
                    relative_complex_L2=None if denominator == 0 else float(np.linalg.norm(vb-va)/denominator),
                    saved_f32_signal_L2=denominator, profiles=len(ids)))
    return rows


def analyze(candidate, output):
    if output.exists():
        raise FileExistsError(output)
    reports = {ARMS[0]: toml(OLD/'control_report.toml'), ARMS[1]: toml(candidate/'control_report.toml')}
    if reports[ARMS[1]]['status'] != 'COMPLETE_E050_F64_100K_MAPPED_ACQUISITIONS':
        raise ValueError('Float64 acquisition incomplete')
    if reports[ARMS[0]]['status'] != 'COMPLETE_E047_100K_REMAPPED':
        raise ValueError('Saved Float32 arm incomplete')
    registration = toml(candidate/'registration.toml')
    info = toml(INPUTS/'case_info.toml')
    preflight = toml(BASE/'preflight_v2/report.toml')
    for key in ('input_sha256', 'map_sha256', 'manifest_sha256', 'helper_sha256'):
        if reports[ARMS[0]][key] != reports[ARMS[1]][key]:
            raise ValueError('Precision comparison source differs: '+key)
    for key in ('script_sha256', 'protocol_sha256', 'input_sha256', 'map_sha256', 'timing_sha256', 'manifest_sha256', 'motion_timing_contract'):
        if registration[key] != reports[ARMS[1]][key] or registration[key] != preflight[key]:
            raise ValueError('Registration/preflight identity mismatch: '+key)
    for path, digest in [(ROOT/'koma_migration_e046/precision_confirmation_e050.jl', registration['script_sha256']),
                         (BASE/'PROTOCOL.md', registration['protocol_sha256']),
                         (INPUTS/'case_inputs.h5', registration['input_sha256']),
                         (INPUTS/'legacy_timing.h5', registration['timing_sha256'])]:
        if sha(path) != digest:
            raise ValueError('Registered source changed: '+str(path))
    reference_path = bind_reference(REF, 'reference_fields.h5', 'f4c4ec3153d30a2a9dfd360d8ff1c21811e14e6f942697cdd990b669fec75694')
    local_path = bind_reference(LOCAL, 'local_fields.h5', '6c57a8708ea4a5aff09f61db232b0ba1528ae4d000bf60d9ca00524516a27885')
    paths = [Path(__file__).resolve(), BASE/'PROTOCOL.md', candidate/'registration.toml', candidate/'control_report.toml',
             OLD/'control_report.toml', INPUTS/'case_info.toml', INPUTS/'legacy_timing.h5', reference_path, local_path,
             ROOT/'scripts/analyze_remapped_confirmation.py', ROOT/'scripts/compare_dp150_hemodynamics.py',
             ROOT/'scripts/build_aorta_inlet_picker.py']
    paths += [folder/(e.lower()+'_result.h5') for folder in (OLD, candidate) for e in ENC]
    stage = output.with_name(output.name+'.building-'+uuid.uuid4().hex[:8])
    stage.mkdir(parents=True)
    dump(stage/'registration.json', dict(experiment='E-050', status='REGISTERED_BEFORE_METRICS',
        created_utc=datetime.now(timezone.utc).isoformat(), inputs={str(p):sha(p) for p in paths}))
    images, fields, audits = {}, {}, {}
    for arm, folder in zip(ARMS, (OLD, candidate)):
        print('E050_ANALYZE', arm, flush=True)
        images[arm], audits[arm] = load_arm(folder, reports[arm], arm == ARMS[1])
        fields[arm] = velocities(images[arm], info['Venc_m_s'])
    with h5py.File(reference_path) as f:
        reference, fraction = f['fft_adc'][:], f['fraction_fft'][:]
    with h5py.File(local_path) as f:
        source_z = f['source_z_fft_m'][:]
    rows, sections, bins = [], [], []
    masks_by_threshold = {}
    for threshold in THRESHOLDS:
        masks = {a:fixed_mask(images[a], threshold) for a in ARMS}
        masks_by_threshold[threshold] = masks
        rows += region_rows(reference, fields, masks, fraction, source_z, threshold)
        for arm in ARMS:
            magnitude_pass = np.ones(reference.shape[:-1], bool)
            for image in images[arm].values():
                mag = abs(image)
                magnitude_pass &= (mag >= threshold*mag.max(axis=(1, 2, 3), keepdims=True))&(mag > 0)
            bins += [dict(arm=arm, threshold=threshold, bin=b+1,
                all_encoding_coverage_fraction=coverage(fraction, magnitude_pass[b])) for b in range(15)]
    del images
    # Load one zero-thickness section at a time; never sum a volume slab as flow.
    with h5py.File(local_path) as f:
        for name, target in PLANES.items():
            index = int(np.argmin(abs(source_z-target)))
            area = f['plane_fraction'][:, :, index]*3.125
            plane_reference = f['plane_reference'][:, :, :, index, 0]
            for threshold, masks in masks_by_threshold.items():
                pair = masks[ARMS[0]][:, :, index]&masks[ARMS[1]][:, :, index]
                for arm in ARMS:
                    for policy, keep in (('own', masks[arm][:, :, index]), ('paired', pair)):
                        sections.append(dict(section=name, index=index, source_z_m=float(source_z[index]),
                            threshold=threshold, arm=arm, support=policy,
                            **partial_flow(plane_reference, fields[arm][:, :, :, index, 0], area, keep, info['period_s'])))
    # All choices below use identical support for both arms AND every comparator.
    sensitivity = []
    with h5py.File(reference_path) as f:
        common = (fraction > 0)&(f['fraction_half'][:] > 0)
        paired = masks_by_threshold[.05][ARMS[0]]&masks_by_threshold[.05][ARMS[1]]&common
        for choice in ('half_mid', 'fft_mid', 'fft_adc'):
            ref_choice = f[choice][:]
            chosen = {a:paired for a in ARMS}
            values = region_rows(ref_choice, fields, chosen, fraction, source_z, .05)
            sensitivity += [dict(comparator=choice, **r) for r in values if r['support'] == 'paired']
    primary = [r for r in rows if r['threshold'] == .05 and r['region'] in ('whole', 'throat')]
    report = dict(experiment='E-050', status='COMPLETE_PRECISION_COMPARISON_NOT_ACCURACY_PROMOTION',
        created_utc=datetime.now(timezone.utc).isoformat(), physical_particles=100000,
        evidence_class='Same-source internal consistency and numerical precision sensitivity',
        primary_reference='E048 centered-FFT physical grid, saved actual-ADC averaged CFD',
        motion_timing_contract=registration['motion_timing_contract'], primary=primary, regions=rows,
        sections=sections, bin_coverage=bins, comparator_sensitivities=sensitivity,
        reconstruction_audits=audits, acquisition_quartile_differences=temporal_kspace_difference(OLD, candidate),
        limitations=['Not independent physical validation; not a whole-lumen accuracy certificate.',
            'Float64 promotes motion interpolation, resets and clock arithmetic together; not phase accumulation alone.',
            'Literal endpoint promotion differs from the exported Float32-subtracted comparator period by59.6ns/cycle.',
            'Comparison choices share fixed support and weights; their effects are not additive causal fractions.',
            'CFD mean velocity and encoded complex-image phase are different observables under acceleration/mixing.',
            '10ms CFD, synthetic final80ms closure, scanner-derived geometry-adapted VENC-adjusted separate encodings.',
            'No fitting, smoothing, imputation, mask relaxation, new1M, TKE, wall or pressure validation.'])
    dump(stage/'report.json', report)
    flat = []
    for r in rows:
        metric = r['metrics'] or {}
        flat.append({k:r[k] for k in ('threshold', 'region', 'arm', 'support', 'coverage_fraction')}|
            {k:metric.get(k) for k in ('voxels', 'vector_rmse_m_s', 'speed_rmse_m_s', 'mean_speed_bias_m_s')})
    with (stage/'regions.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
    flow_rows = []
    for r in sections:
        for b in range(15):
            flow_rows.append({k:r[k] for k in ('threshold', 'section', 'source_z_m', 'arm', 'support', 'coverage_fraction')}|
                dict(bin=b+1, CFD_ml_s=None if r['CFD_ml_s'] is None else r['CFD_ml_s'][b],
                     Koma_ml_s=None if r['Koma_ml_s'] is None else r['Koma_ml_s'][b]))
    with (stage/'sections.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flow_rows[0])); writer.writeheader(); writer.writerows(flow_rows)
    with h5py.File(stage/'paired_fields.h5', 'w') as f:
        for arm in ARMS:
            f.create_dataset(arm, data=fields[arm], compression='gzip', compression_opts=1)
            f.create_dataset(arm+'_fixed5_mask', data=masks_by_threshold[.05][arm])
        f.create_dataset('paired_fixed5_mask', data=masks_by_threshold[.05][ARMS[0]]&masks_by_threshold[.05][ARMS[1]])
        f.attrs['reference_file'] = str(reference_path)
        f.attrs['reference_sha256'] = sha(reference_path)
    dump(stage/'MANIFEST.json', dict(outputs={p.name:sha(p) for p in stage.iterdir() if p.is_file()}))
    stage.rename(output)
    print(json.dumps(dict(status=report['status'], primary=primary)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    analyze(args.candidate, args.output)
