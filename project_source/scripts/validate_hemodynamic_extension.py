"""Independent arithmetic checks of E-037 saved fields and immutable inputs."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import binary_erosion, generate_binary_structure

from scripts.extend_hemodynamic_comparisons import AO, DP, DP_H5, PILOT, ROOT, sha


def validate(run):
    report = json.loads((run/'report.json').read_text())
    checks = []

    def check(name, ok):
        checks.append(dict(name=name, passed=bool(ok)))

    def close(name, a, b, **kw):
        check(name, np.allclose(a, b, rtol=kw.get('rtol', 2e-10), atol=kw.get('atol', 1e-14), equal_nan=True))

    manifest = json.loads((run/'MANIFEST.json').read_text())
    for name, row in manifest.items():
        check('artifact:'+name, sha(run/name) == row['sha256'] and (run/name).stat().st_size == row['bytes'])
    for row in report['inputs']:
        path = ROOT/row['path']
        check('input:'+row['path'], path.stat().st_size == row['bytes'] and sha(path) == row['sha256'])
    audit = json.loads((run/'turbulence_source_audit.json').read_text())
    check('73native_frames', len(audit['dp150']['frames']) == 73)
    check('no_qualified_TKE', not audit['dp150']['tke_comparison_available'] and not audit['aorta']['tke_comparison_available'])
    check('no_turbulence_candidate_fields', audit['dp150']['candidate_fields'] == [])
    for row in audit['dp150']['frames']:
        path = ROOT/row['path']
        check('native_hash:'+path.name, sha(path) == row['sha256'])
    with h5py.File(DP/'matched_reference.h5') as f:
        ref_dp = f['cfd_voxel_midpoint'][:].astype(float)
        dp_vol = f['volume_fraction_2'][:].astype(float)*(.00125*.00125*.0025)
        dp_common = f['common_mask'][:]; dp_direct = f['direct_common_mask'][:]
    with h5py.File(DP_H5) as f:
        dp_est = np.moveaxis(f['velocity_corrected_m_per_s'][:], 0, -1).astype(float)
        dp_raw = np.moveaxis(f['velocity_raw_m_per_s'][:], 0, -1).astype(float)
    with h5py.File(AO/'matched_reference.h5') as f:
        ref_ao = f['adc_time_average'][:].astype(float); legacy = f['Koma'][:]
        fraction = f['fraction'][:]; plane_fraction = f['plane_fraction'][:]
        oldmask = f['masks/core15'][:]
        section_rois = {k: f['sections/'+k][:] for k in f['sections']}
    with h5py.File(PILOT/'pc_reconstruction.h5') as f:
        pilot = np.moveaxis(f['velocity'][:], 0, -1).astype(float)
        magnitude = f['reference_magnitude'][:]; stored = f['mask'][:].astype(bool)
    shape = ref_ao.shape[1:-1]; spacing = np.array([.28/72, .004, .004])
    volume = fraction.astype(float)*np.prod(spacing)
    pilot_masks = {f'core{round(th*100):02d}': np.all(stored & (magnitude >= th*magnitude.max(axis=(1, 2, 3), keepdims=True)), axis=0) & (fraction > 0)
                   for th in [.05, .1, .15, .2]}
    pilot_masks['legacy_pilot_intersection15'] = pilot_masks['core15'] & oldmask
    pilot_masks['eroded15'] = binary_erosion(pilot_masks['core15'], structure=np.ones((3, 3, 3)))
    cases = {
        'dp150_1m_hybrid': (ref_dp, dp_est, dp_vol, {'common': dp_common}),
        'dp150_1m_raw_direct': (ref_dp, dp_raw, dp_vol, {'direct_partial': dp_direct}),
        'aorta_corrected_100k_pilot': (ref_ao, pilot, volume, pilot_masks),
        'aorta_legacy_1m': (ref_ao, legacy, volume, {'legacy_core15': oldmask, 'legacy_pilot_intersection15': pilot_masks['legacy_pilot_intersection15']}),
    }
    with h5py.File(run/'energy_fields.h5') as fields:
        check('energy_fields_not_TKE', not fields.attrs['is_tke'])
        for case, (ref, est, vol, masks) in cases.items():
            for support, mask in masks.items():
                pair = report['cases'][case]['supports'][support]
                if not mask.any():
                    check(case+support+'empty', pair['status'] == 'UNAVAILABLE_EMPTY_SUPPORT'); continue
                for method, values in [('CFD', ref), ('Koma', est)]:
                    u = values[:, mask]; w = vol[mask]
                    mean = u.mean(axis=0); var = np.mean((u-mean)**2, axis=0)
                    per_bin = .5*np.sum(np.sum(u*u, axis=-1)*w, axis=1)
                    steady = .5*np.sum(np.sum(mean*mean, axis=-1)*w)
                    pulsatile = .5*np.sum(np.sum(var, axis=-1)*w)
                    r = pair[method]; prefix = case+'/'+support+'/'+method
                    close(prefix+'per_bin', per_bin, r['per_bin_ke_per_density_m5_s2'])
                    close(prefix+'total', per_bin.mean(), r['mean_resolved_ke_per_density_m5_s2'])
                    close(prefix+'steady', steady, r['mean_vector_ke_per_density_m5_s2'])
                    close(prefix+'coherent', pulsatile, r['coherent_variation_ke_per_density_m5_s2'])
                    close(prefix+'identity', per_bin.mean(), steady+pulsatile)
                    close(prefix+'mean_field', fields[prefix+'/mean_velocity_m_s'][:][mask], mean)
                    close(prefix+'variance_field', fields[prefix+'/coherent_component_variance_m2_s2'][:][mask], var)
                    check(prefix+'missing_outside_domain', np.isnan(fields[prefix+'/mean_velocity_m_s'][:][~mask]).all())
                    if case.startswith('dp150'):
                        close(prefix+'density', r['mean_resolved_ke_j'], per_bin.mean()*1060)
                    else:
                        check(prefix+'no_assumed_density', 'mean_resolved_ke_j' not in r)
                error = est[:, mask]-ref[:, mask]
                rmse = np.sqrt(np.sum(np.sum(error*error, axis=-1)*vol[mask])/(len(ref)*vol[mask].sum()))
                close(case+'/'+support+'/vector_RMSE', rmse, pair['velocity_error']['vector_rmse_m_s'])
    native = json.loads((DP/'report.json').read_text())['native_values']
    fine = np.array([x['quadrature_unaveraged_ke_common_j'] for x in native])
    coarse = np.array(report['cases']['dp150_1m_hybrid']['supports']['common']['CFD']['per_bin_ke_per_density_m5_s2'])*1060
    close('spatial_difference', fine-coarse, report['dp150_spatial_dispersion']['per_bin_j'])
    check('spatial_difference_not_TKE', report['dp150_spatial_dispersion']['is_tke'] is False)
    axes = [(np.arange(n)-n//2)*d for n, d in zip(shape[::-1], spacing)]
    for row in report['aorta_pilot_sections']['series']:
        i = int(np.argmin(abs(axes[0]-row['x_m'])))
        mask = section_rois[row['section']] & pilot_masks['core15'][:, :, i] & (plane_fraction[:, :, i] > 0)
        area = plane_fraction[:, :, i].astype(float)*spacing[1]*spacing[2]
        u = (ref_ao if row['method'] == 'CFD' else pilot)[row['bin']-1, :, :, i, 0]
        flux = u[mask]*area[mask]
        close('section_flux:'+str(row['bin'])+row['section']+row['method'], flux.sum()*1e6, row['net_ml_s'], rtol=3e-7)
        close('section_forward:'+str(row['bin'])+row['section']+row['method'], np.maximum(flux, 0).sum(), row['forward_m3_s'], rtol=3e-7)
        close('section_reverse:'+str(row['bin'])+row['section']+row['method'], np.maximum(-flux, 0).sum(), row['reverse_m3_s'], rtol=3e-7)
    # Independent full-grid np.gradient implementation; only common six-neighbor
    # interior stencils are compared. No package differential routine is called.
    mask = binary_erosion(pilot_masks['core15'], structure=generate_binary_structure(3, 1))
    metric_fields = []
    for u in [ref_ao, pilot]:
        arrays = {k: [] for k in ['vorticity_magnitude_s_inv', 'helicity_m_s_inv2', 'divergence_s_inv', 'strain_frobenius_s_inv', 'dissipation_per_viscosity_s_inv2']}
        for b in range(8):
            grad = np.stack([np.stack(np.gradient(u[b, ..., c], *spacing[::-1]), axis=-1)[..., ::-1] for c in range(3)], axis=-2)
            curl = np.stack([grad[..., 2, 1]-grad[..., 1, 2], grad[..., 0, 2]-grad[..., 2, 0], grad[..., 1, 0]-grad[..., 0, 1]], axis=-1)
            strain = (grad+np.swapaxes(grad, -1, -2))/2
            strain2 = np.sum(strain*strain, axis=(-2, -1))
            values = dict(vorticity_magnitude_s_inv=np.linalg.norm(curl, axis=-1),
                helicity_m_s_inv2=np.sum(u[b]*curl, axis=-1), divergence_s_inv=np.trace(grad, axis1=-2, axis2=-1),
                strain_frobenius_s_inv=np.sqrt(strain2), dissipation_per_viscosity_s_inv2=2*strain2)
            for name, field in values.items():
                arrays[name].append(field[mask])
        metric_fields.append({k: np.concatenate(v) for k, v in arrays.items()})
    for name in metric_fields[0]:
        rmse = np.sqrt(np.mean((metric_fields[1][name]-metric_fields[0][name])**2))
        close('pilot_gradient_RMSE:'+name, rmse, report['aorta_pilot_derivatives']['primary']['pooled'][name]['rmse'])
    return dict(status='PASS' if all(c['passed'] for c in checks) else 'FAIL',
                checks=checks, passed=sum(c['passed'] for c in checks), total=len(checks),
                independent_physical_validation=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = validate(args.run)
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'checks'}, indent=2))
    if result['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
