"""Independent all-bin E-038 arithmetic and plan-coverage verification."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import binary_erosion, generate_binary_structure

from scripts.complete_advanced_comparisons import load_inputs
from scripts.extend_hemodynamic_comparisons import ROOT, sha


def independent_fields(u, spacing):
    g = np.stack([np.stack(np.gradient(u[..., c], *spacing[::-1]), axis=-1)[..., ::-1] for c in range(3)], axis=-2)
    w = np.stack([g[..., 2, 1]-g[..., 1, 2], g[..., 0, 2]-g[..., 2, 0], g[..., 1, 0]-g[..., 0, 1]], axis=-1)
    strain = (g+np.swapaxes(g, -1, -2))/2
    s2 = np.sum(strain*strain, axis=(-2, -1)); omega = np.linalg.norm(w, axis=-1)
    helicity = np.sum(u*w, axis=-1); den = np.linalg.norm(u, axis=-1)*omega
    relative = np.full(den.shape, np.nan)
    np.divide(helicity, den, out=relative, where=den > 0)
    return dict(vorticity_magnitude_s_inv=omega, helicity_m_s_inv2=helicity,
        relative_helicity=np.clip(relative, -1, 1), strain_frobenius_s_inv=np.sqrt(s2),
        divergence_s_inv=np.trace(g, axis1=-2, axis2=-1),
        dissipation_per_viscosity_s_inv2=2*s2, enstrophy_s_inv2=.5*omega*omega)


def validate(run, figures):
    r = json.loads((run/'report.json').read_text()); checks = []
    def check(name, ok): checks.append(dict(name=name, passed=bool(ok)))
    def close(name, a, b, rtol=4e-10, atol=1e-12):
        check(name, np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True))
    manifest = json.loads((run/'MANIFEST.json').read_text())
    for name, row in manifest.items():
        check('artifact:'+name, sha(run/name) == row['sha256'] and (run/name).stat().st_size == row['bytes'])
    for row in r['inputs']:
        check('source_hash:'+row['path'], sha(ROOT/row['path']) == row['sha256'])
        check('matched_source_scope:'+row['path'], 'ansys/' not in row['path'].lower() and 'additional_native' not in row['path'].lower())
    inventory = r['plan_completion']
    check('supported_families_16_in_each_case', sum(x['status'] == 'COMPLETED_SUPPORTED_SCOPE' for x in inventory) == 48)
    check('conditional_families_not_called_complete', sum(x['status'] == 'UNAVAILABLE_NOT_COMPUTED' for x in inventory) == 12)
    _, cases, dp, ao, ext, refs, fraction = load_inputs()
    with h5py.File(run/'advanced_fields.h5') as f:
        check('fields_not_TKE', not f.attrs['is_tke'])
        for case in cases:
            name = case['name']; result = r['cases'][name]['gradients']
            mask = binary_erosion(case['gradient_mask'], structure=generate_binary_structure(3, 1))
            ids = np.flatnonzero(mask); w = case['volume'][mask]
            check(name+':fixed_stencil', np.array_equal(f[name+'/flat_indices'][:], ids))
            close(name+':fixed_volume', f[name+'/volume_weights_m3'][:], w)
            pools = {k: [[], []] for k in ['vorticity_magnitude_s_inv', 'helicity_m_s_inv2',
                'strain_frobenius_s_inv', 'divergence_s_inv', 'dissipation_per_viscosity_s_inv2']}
            for b in range(len(case['reference'])):
                a = independent_fields(case['reference'][b], case['spacing'])
                c = independent_fields(case['estimate'][b], case['spacing'])
                for metric in a:
                    av, cv = a[metric][mask], c[metric][mask]
                    for method, expected in [('CFD', av), ('Koma', cv)]:
                        saved = f[name+'/'+method+'/'+metric][b]
                        # Only the stored field is f32. Reports below must match
                        # f64 recomputation without loosening their tolerance.
                        close(f'{name}:field:{b}:{method}:{metric}', saved, expected, rtol=2e-6, atol=2e-7)
                    keep = np.isfinite(av) & np.isfinite(cv)
                    rr = next(x for x in result['per_bin_errors'] if x['bin'] == b+1 and x['metric'] == metric)
                    if keep.any():
                        rmse = np.sqrt(np.average((cv[keep]-av[keep])**2, weights=w[keep]))
                        close(f'{name}:f64_RMSE:{b}:{metric}', rr['rmse'], rmse)
                        close(f'{name}:paired_weight:{b}:{metric}', rr['weight_sum'], w[keep].sum())
                    else:
                        check(f'{name}:undefined_not_zero:{b}:{metric}', rr['rmse'] is None)
                    if metric in pools:
                        pools[metric][0].append(av); pools[metric][1].append(cv)
                for region, roi in case['regions'].items():
                    m = mask & roi
                    if not m.any(): continue
                    for method, values in [('CFD', a), ('Koma', c)]:
                        row = next(x for x in result['region_integrals'] if x['bin'] == b+1 and x['region'] == region and x['method'] == method)
                        v = case['volume'][m]
                        close(f'{name}:loss:{b}:{region}:{method}', row['dissipation_per_viscosity_m3_s2'], np.sum(values['dissipation_per_viscosity_s_inv2'][m]*v))
                        close(f'{name}:helicity:{b}:{region}:{method}', row['signed_helicity_m4_s2'], np.sum(values['helicity_m_s_inv2'][m]*v))
                        close(f'{name}:enstrophy:{b}:{region}:{method}', row['enstrophy_m3_s2'], np.sum(values['enstrophy_s_inv2'][m]*v))
            for metric, (av, cv) in pools.items():
                av, cv = np.concatenate(av), np.concatenate(cv)
                weights = np.tile(w, len(case['reference']))
                close(name+':pooled_weighted:'+metric, result['weighted_pooled'][metric]['rmse'], np.sqrt(np.average((cv-av)**2, weights=weights)))
                close(name+':pooled_unweighted:'+metric, result['unweighted_pooled'][metric]['rmse'], np.sqrt(np.mean((cv-av)**2)))
                if name == 'aorta_1m':
                    old = ao['derivatives']['primary']['pooled'][metric]['rmse']
                elif name == 'aorta_100k_corrected':
                    old = ext['aorta_pilot_derivatives']['primary']['pooled'][metric]['rmse']
                else:
                    oldkey = 'dissipation_w_m3' if metric == 'dissipation_per_viscosity_s_inv2' else metric
                    old = dp['derivative_error_summary']['interior'][oldkey]['rmse']
                    if metric == 'dissipation_per_viscosity_s_inv2': old /= .0035
                actual = result['weighted_pooled' if name == 'phantom_1m' else 'unweighted_pooled'][metric]['rmse']
                close(name+':old_metric_reproduction:'+metric, actual, old, rtol=2e-6)
            for row in result['cycle_integrals']:
                series = [x['dissipation_per_viscosity_m3_s2'] for x in result['region_integrals'] if x['region'] == row['region'] and x['method'] == row['method']]
                close(name+':time_integral:'+row['region']+row['method'], row['nominal_bin_loss_integral_per_viscosity_m3_s'], np.sum(series)*case['period']/len(series))
                if case['viscosity'] is None:
                    check(name+':no_assumed_viscosity:'+row['region']+row['method'], 'nominal_bin_resolved_loss_j' not in row)
    pilot = cases[-1]
    for row in r['corrected_pilot_reference_sensitivity']:
        reference = refs[row['reference']]; b = row['bin']-1
        m = pilot['gradient_mask'] & np.isfinite(reference).all(axis=(0, 4)); w = pilot['volume'][m]
        diff = pilot['estimate'][b][m]-reference[b][m]
        close('pilot_reference:'+row['reference']+str(b), row['vector_rmse_m_s'], np.sqrt(np.average(np.sum(diff*diff, axis=-1), weights=w)))
    erosion = binary_erosion(pilot['gradient_mask'], structure=np.ones((3, 3, 3)))
    stencil = binary_erosion(erosion, structure=generate_binary_structure(3, 1))
    check('pilot_erosion_stencil', r['corrected_pilot_eroded_derivatives']['stencil_voxels_per_bin'] == stencil.sum())
    erosion_pools = {}
    for b in range(8):
        a = independent_fields(pilot['reference'][b], pilot['spacing'])
        c = independent_fields(pilot['estimate'][b], pilot['spacing'])
        for metric in r['corrected_pilot_eroded_derivatives']['pooled']:
            erosion_pools.setdefault(metric, []).append((c[metric][stencil]-a[metric][stencil])**2)
    for metric, squared in erosion_pools.items():
        if stencil.any():
            close('pilot_erosion_RMSE:'+metric, np.sqrt(np.mean(np.concatenate(squared))), r['corrected_pilot_eroded_derivatives']['pooled'][metric]['rmse'])
        else:
            check('pilot_erosion_unavailable_not_zero:'+metric,
                  r['corrected_pilot_eroded_derivatives']['pooled'][metric]['rmse'] is None)
    source_sections = {
        'phantom_1m': [dict(**x, method={'cfd_voxel_common': 'CFD', 'koma_hybrid_common': 'Koma'}[x['dataset']])
            for x in dp['section_values'] if x['dataset'] in ['cfd_voxel_common', 'koma_hybrid_common']],
        'aorta_1m': ao['section_series'],
        'aorta_100k_corrected': ext['aorta_pilot_sections']['series']}
    periods = {c['name']: c['period'] for c in cases}
    for row in r['flow_comparison']:
        values = sorted([x for x in source_sections[row['case']] if x['section'] == row['section'] and x['method'] == row['method']], key=lambda x: x['bin'])
        q = np.array([x['net_m3_s'] for x in values]); dt = periods[row['case']]/len(q)
        prefix = 'flow:'+row['case']+row['section']+row['method']
        close(prefix+':net', row['net_nominal_cycle_ml'], q.sum()*dt*1e6)
        close(prefix+':simultaneous_forward', row['simultaneous_forward_volume_ml'], sum(x['forward_m3_s'] for x in values)*dt*1e6)
        close(prefix+':simultaneous_reverse', row['simultaneous_reverse_volume_ml'], sum(x['reverse_m3_s'] for x in values)*dt*1e6)
        close(prefix+':net_reverse', row['net_reverse_time_partition_ml'], -np.minimum(q, 0).sum()*dt*1e6)
        close(prefix+':peak', row['absolute_peak_ml_s'], abs(q).max()*1e6)
        if 'paired_error_ml_s' in row:
            source = [x for x in source_sections[row['case']] if x['section'] == row['section'] and x['method'] == 'CFD']
            source = sorted(source, key=lambda x: x['bin'])
            cq = np.array([x['net_m3_s'] for x in source])
            close(prefix+':paired_RMSE', row['paired_error_ml_s']['rmse'], np.sqrt(np.mean((q-cq)**2))*1e6)
    source_cores = {'phantom_1m': dp['jet_values'], 'aorta_1m': ao['flow_core_series'],
                    'aorta_100k_corrected': ext['aorta_pilot_sections']['flow_cores']}
    for row in r['flow_core_comparison']:
        originals = [x for x in source_cores[row['case']] if all(x[k] == row[k] for k in ['section', 'bin', 'threshold_fraction']) and x.get('direction_sign', 1) == row['direction_sign']]
        pair = {x.get('method', {'cfd': 'CFD', 'koma': 'Koma'}.get(x.get('dataset'))): x for x in originals}
        prefix = 'core:'+str((row['case'], row['section'], row['bin'], row['threshold_fraction'], row['direction_sign']))
        vectors = []
        for method in ['CFD', 'Koma']:
            x = pair[method]
            center = x.get('centroid_m', [x.get('centroid_X_m'), x.get('centroid_Y_m'), x.get('centroid_Z_m')])
            direction = x.get('direction_unit', [x.get('direction_X'), x.get('direction_Y'), x.get('direction_Z')])
            vectors.append((np.array(center, float), np.array(direction, float)))
        valid = bool(np.isfinite(np.array(vectors)).all())
        check(prefix+':availability', row['paired_core_available'] == valid)
        if valid:
            close(prefix+':centroid', row['centroid_error_mm'], np.sqrt(np.sum((vectors[1][0]-vectors[0][0])**2))*1000)
            close(prefix+':direction', row['direction_error_deg'], np.arccos(np.clip(np.sum(vectors[0][1]*vectors[1][1]), -1, 1))*180/np.pi)
        else:
            check(prefix+':missing_not_zero', row['centroid_error_mm'] is None and row['direction_error_deg'] is None)
    fm = json.loads((figures/'MANIFEST.json').read_text())
    check('six_PNGs_six_PDFs', sum(n.endswith('.png') for n in fm) == 6 and sum(n.endswith('.pdf') for n in fm) == 6)
    for name, row in fm.items():
        check('presentation_hash:'+name, sha(figures/name) == row['sha256'])
        if name.endswith('.pdf'): check('pdf_signature:'+name, (figures/name).read_bytes().startswith(b'%PDF-'))
    return dict(status='PASS' if all(x['passed'] for x in checks) else 'FAIL',
        passed=sum(x['passed'] for x in checks), total=len(checks), checks=checks,
        evidence_class='Independent arithmetic implementation on same inputs; not independent physical validation',
        visual_review='Recorded separately after viewing PNGs; PDFs checked by signature/hash, not rendered')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--run', type=Path, required=True)
    p.add_argument('--figures', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    result = validate(a.run, a.figures)
    with a.output.open('x', encoding='utf-8') as f: json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['checks']}, indent=2))
    if result['status'] != 'PASS': raise SystemExit(1)
