"""Independent arithmetic/provenance checks for E048 saved-data diagnostics."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import tomllib

import h5py
import numpy as np
from PIL import Image

from scripts.build_aorta_inlet_picker import sha
from scripts.diagnose_phantom_signal import BASE, E047, ROOT, PROD, ENC
from scripts.compare_dp150_hemodynamics import REGIONS
from scripts.diagnose_phantom_encoding_kernel import phase_kernel


def validate(output):
    if output.exists():
        raise FileExistsError(output)
    checks = []
    hash_cache = {}

    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append(name)

    def close(name, actual, expected, atol=2e-7):
        check(name, np.allclose(actual, expected, atol=atol, rtol=2e-6))

    def digest(path):
        path = Path(path).resolve()
        if path not in hash_cache:
            hash_cache[path] = sha(path)
        return hash_cache[path]

    def identities(entries, parent, label):
        for name, value in entries.items():
            path = Path(name)
            if not path.is_absolute():
                path = parent / path
            expected = value['sha256'] if isinstance(value, dict) else value
            check(f'{label}:{name}', digest(path) == expected)

    def load(name):
        return json.loads((BASE / name / 'report.json').read_text())

    for folder in ('basic_v1', 'reference_v1', 'local_v1', 'kernel_v2', 'figures_v1'):
        path = BASE / folder
        manifest = json.loads((path / 'MANIFEST.json').read_text())
        identities(manifest.get('inputs', {}), ROOT, folder + ':input')
        identities(manifest['outputs'], path, folder + ':output')
        if (path / 'registration.json').exists():
            registration = json.loads((path / 'registration.json').read_text())
            identities(registration.get('inputs', registration.get('input_sha256', {})), ROOT, folder + ':registration')
    # E047's final delivery also hashes mutable governance; do not claim those
    # intentionally updated documents still equal their historical snapshots.
    identities(json.loads((E047 / 'source_registration.json').read_text())['inputs'], ROOT, 'E047-frozen-source')
    old = ROOT / 'outputs/aorta_inlet_replenishment/e046_legacy_snapshot_v1/registration.json'
    legacy = json.loads(old.read_text())['inputs']
    check('39 legacy identities declared', len(legacy) == 39)
    identities(legacy, ROOT, 'preserved-legacy')
    print('E048 provenance and preserved source hashes checked', flush=True)
    frozen = BASE / 'kernel_v1/diagnose_phantom_encoding_kernel_frozen_v1.py'
    check('rejected kernel v1 original code retained', digest(frozen) == '70dcb0ce193cd555db80fe9b89aa69c19c46e6491870f25e0b08836c6244a603')
    basic, audit, local, kernel = [load(p) for p in ('basic_v1', 'reference_v1', 'local_v1', 'kernel_v2')]
    with h5py.File(BASE / 'basic_v1/diagnostic_fields.h5') as f:
        old_fraction = f['volume_fraction'][:]
        mapped_mask = f['map100k_mask'][:].astype(bool)
        million_mask = f['legacy1m_mask'][:].astype(bool)
    folders = dict(map20k=ROOT/'outputs/upstream_migration_e046/phantom_updated_map_v1',
                   legacy100k=E047/'phantom_inputs_v1', map100k=E047/'phantom_mapped_v1', legacy1m=PROD)
    masks = {}
    for arm, folder in folders.items():
        all_bins = np.ones((15, *old_fraction.shape), bool)
        fixed = np.ones(old_fraction.shape, bool)
        for enc in ENC:
            suffix = '_result.jld2' if arm == 'legacy1m' else '_legacy_result.h5' if arm == 'legacy100k' else '_result.h5'
            with h5py.File(folder / (enc.lower() + suffix)) as f:
                a = f['image'][:]
            if a.dtype.names:
                a = a['re'] + 1j * a['im']
            mag = np.abs(a)
            valid = (mag >= .05 * np.max(mag, axis=(1, 2, 3), keepdims=True)) & (mag > 0)
            enc_fixed = np.all(valid, axis=0)
            close(f'{arm}:{enc}:fixed signal coverage', np.sum(old_fraction[enc_fixed], dtype=float)/old_fraction.sum(),
                  basic['cases'][arm]['fixed_encoding_volume_fraction'][enc])
            all_bins &= valid
            fixed &= enc_fixed
            del a, mag, valid
        masks[arm] = fixed
        close(f'{arm}:fixed all coverage', np.sum(old_fraction[fixed], dtype=float)/old_fraction.sum(),
              basic['cases'][arm]['fixed_all_bin_all_encoding_volume_fraction'])
        for b in range(15):
            close(f'{arm}:bin{b+1}:coverage', np.sum(old_fraction[all_bins[b]], dtype=float)/old_fraction.sum(),
                  basic['cases'][arm]['all_encoding_per_bin'][b]['all_encoding_volume_fraction'])
        close(f'{arm}:failure buckets sum to one', sum(r['volume_fraction'] for r in basic['cases'][arm]['failure_patterns']), 1)
    check('mapped mask independently reproduced', np.array_equal(masks['map100k'], mapped_mask))
    check('saved million mask independently reproduced', np.array_equal(masks['legacy1m'], million_mask))
    print('E048 raw-image signal support reproduced', flush=True)
    del all_bins

    with h5py.File(E047/'phantom_analysis_v1/paired_fields.h5') as f:
        fields = {a: f[a][:] for a in ('legacy100k', 'map100k')}
    with h5py.File(BASE/'local_v1/local_fields.h5') as f:
        fields['legacy1m'] = f['legacy1m_velocity'][:]
        z = f['source_z_fft_m'][:]
        plane = f['plane_reference'][:]
        plane_fraction = f['plane_fraction'][:]
        shared = f['shared_mask'][:].astype(bool)
        fft_counts = f['FFT_bin_center_counts'][:]
    with h5py.File(BASE/'reference_v1/reference_fields.h5') as f:
        fraction = f['fraction_fft'][:]
        common = f['common_mask'][:].astype(bool)
        common_fraction = f['common_fraction'][:]
        references = {k: f[k][:] for k in ('half_mid','half_adc','fft_mid','fft_adc','fft_adc_f32','half_q3_mid','half_3sample')}
        close('ADC weights unity', np.sum(f['adc_weights'][:], axis=1), np.ones(15), atol=1e-12)
    check('three-arm support exactly reproduced', np.array_equal(shared, masks['legacy100k'] & masks['map100k'] & masks['legacy1m'] & (fraction>0)))

    def independent_rmse(ref, estimate, weights, mask):
        good = mask & (weights > 0) & np.isfinite(ref).all(axis=(0,4)) & np.isfinite(estimate).all(axis=(0,4))
        error = np.linalg.norm(estimate[:,good].astype(float)-ref[:,good].astype(float), axis=-1)**2
        return float(np.sqrt(np.mean(np.average(error, weights=weights[good].astype(float), axis=1))))

    # Old-grid boundaries deliberately remain fixed in the coordinate sensitivity.
    old_z = z + .000625
    for row in audit['results']:
        lo, hi = REGIONS[row['region']]
        m = common & ((old_z >= lo) & (old_z < hi))[None,None,:]
        close(f"reference:{row['reference']}:{row['arm']}:{row['region']}",
              independent_rmse(references[row['reference']], fields[row['arm']], common_fraction, m), row['metrics']['vector_rmse_m_s'])
    for row in local['regions']:
        lo, hi = REGIONS[row['region']]
        region = (fraction>0) & ((z>=lo)&(z<hi))[None,None,:]
        for policy, mask in [('own', masks[row['arm']] & region), ('shared', shared & region)]:
            close(f"local:{row['arm']}:{row['region']}:{policy}:coverage", fraction[mask].sum(dtype=float)/fraction[region].sum(dtype=float), row[policy+'_coverage_fraction'])
            close(f"local:{row['arm']}:{row['region']}:{policy}:RMSE", independent_rmse(references['fft_adc'], fields[row['arm']], fraction, mask), row[policy+'_metrics']['vector_rmse_m_s'])
    info = tomllib.loads((E047/'phantom_inputs_v1/case_info.toml').read_text())
    for row in local['sections']:
        i = row['index']
        area = plane_fraction[:,:,i].astype(float)*3.125
        m = (area>0) & (masks[row['arm']][:,:,i] if row['support']=='own' else shared[:,:,i])
        check(f"{row['section']}:{row['arm']}:{row['support']}:nonempty", area[m].sum()>0)
        a = np.sum(plane[:,:,:,i,0][:,m].astype(float)*area[m], axis=1)
        b = np.sum(fields[row['arm']][:,:,:,i,0][:,m].astype(float)*area[m], axis=1)
        prefix = f"section:{row['section']}:{row['arm']}:{row['support']}"
        close(prefix+':area', area[m].sum(), row['area_mm2'])
        close(prefix+':CFD', a, row['CFD_ml_s'], atol=3e-5)
        close(prefix+':Koma', b, row['Koma_ml_s'], atol=3e-5)
        close(prefix+':RMSE', np.sqrt(np.mean((b-a)**2)), row['metrics']['rmse_ml_s'], atol=3e-5)
        close(prefix+':cycle', b.mean()/a.mean()-1, row['metrics']['cycle_volume_error_fraction'])

    # A different algorithm (histogramdd, not ravel/bincount) checks occupancy.
    n = np.array(fraction.shape)
    d = np.array([.0025,.00125,.00125])
    edges = [(np.arange(k+1)-k//2-.5)*s for k,s in zip(n,d)]
    with h5py.File(E047/'phantom_inputs_v1/case_inputs.h5') as f:
        base = np.column_stack([f[c][:100000] for c in ('x','y','z')])
        for b in range(15):
            i = 48+96*b
            points = base+np.column_stack([f[c][i,:100000] for c in ('dx','dy','dz')])
            histogram = np.histogramdd(points[:,::-1], bins=edges)[0]
            check(f'FFT occupancy bin{b+1}', np.array_equal(histogram,fft_counts[b]))
    print('E048 regional errors, true-plane flows and occupancy reproduced', flush=True)

    a,b,knots,_ = phase_kernel(Path(info['sequence_directory']))
    tau = knots-a['echo']
    g = b['gradient'](knots)-a['gradient'](knots)
    # Analytic polynomial antiderivatives, independent of Gauss quadrature.
    moments = []
    for order in range(3):
        slope = np.diff(g)/np.diff(tau)
        intercept = g[:-1]-slope*tau[:-1]
        moments.append(float(np.sum(slope*np.diff(tau**(order+2))/(order+2)+intercept*np.diff(tau**(order+1))/(order+1))))
    for i, key in enumerate(('gradient_M0_cycles_m','gradient_M1_s_m','gradient_M2_s2_m')):
        close('analytic '+key, moments[i], kernel[key], atol=1e-8 if i==0 else 1e-11)
    close('analytic acceleration offset', moments[2]/(2*moments[1]), kernel['equivalent_acceleration_time_offset_s'], atol=1e-11)
    with h5py.File(BASE/'kernel_v2/kernel_fields.h5') as f:
        proxy = f['gradient_velocity_proxy'][:]
        instant = f['instantaneous_centered_velocity'][:]
        clean_counts = f['clean_counts'][:]
    for row in kernel['regional']:
        lo,hi = REGIONS[row['region']]
        m = mapped_mask[None] & (fraction>0)[None] & ((z>=lo)&(z<hi))[None,None,None,:] & np.isfinite(proxy) & np.isfinite(instant)
        w = np.broadcast_to(fraction, m.shape)[m].astype(float)
        target = fields['map100k'][...,0][m].astype(float)
        check('kernel sample count '+row['region'], int(m.sum()) == row['space_time_samples'])
        for key, p in [('Koma_vs_CFD_axial_rmse_m_s',references['fft_adc'][...,0]),
                       ('Koma_vs_instant_particles_axial_rmse_m_s',instant),
                       ('Koma_vs_gradient_proxy_axial_rmse_m_s',proxy)]:
            close('kernel '+row['region']+key, np.sqrt(np.average((target-p[m])**2,weights=w)), row[key])
    # Sampled numerical verification of the trajectory integration using 30,001
    # uniform points and trapezoids rather than exact split Gauss integration.
    trajectory_controls = []
    with h5py.File(E047/'phantom_inputs_v1/case_inputs.h5') as f:
        for bin_index, region in [(3,'throat'),(3,'upstream'),(9,'throat')]:
            lo,hi = REGIONS[region]
            centers = base+np.column_stack([f[c][48+96*bin_index,:100000] for c in ('dx','dy','dz')])
            hist_ids = np.column_stack([np.searchsorted(e, centers[:,::-1][:,j],side='right')-1 for j,e in enumerate(edges)])
            valid = np.all((hist_ids>=0)&(hist_ids<n),axis=1)
            u = hist_ids[valid]
            flat_ids = np.ravel_multi_index(u.T, fraction.shape)
            candidate = np.argwhere((clean_counts[bin_index]>0)&(clean_counts[bin_index]<=3)&mapped_mask&(z[None,None,:]>=lo)&(z[None,None,:]<hi))
            control = None
            for cell in candidate:
                voxel = np.ravel_multi_index(cell,fraction.shape)
                spins = np.flatnonzero(valid)[flat_ids==voxel]
                center = (bin_index+.5)/15*info['period_s']
                times = np.linspace(center+knots[0]-a['echo'], center, 30001)
                step = info['period_s']/1440
                left,right = int(np.floor(times[0]/step)), int(np.ceil(times[-1]/step))
                reset = f['reset'][left:48+96*bin_index+2,:100000][:,spins]
                if reset.any() or len(spins)!=clean_counts[(bin_index,*cell)]:
                    continue
                gradient = b['gradient'](times-center+a['echo'])-a['gradient'](times-center+a['echo'])
                values = []
                for spin in spins:
                    track = base[spin,0]+f['dx'][left:right+1,spin].astype(float)
                    positions = np.interp(times,np.arange(left,right+1)*step,track)
                    integrand = gradient*positions
                    values.append(-10*np.sum(np.diff(times)*(integrand[:-1]+integrand[1:])/2))
                stored = float(proxy[(bin_index,*cell)])
                actual = float(np.mean(values))
                close(f'numerical trajectory control:{bin_index}:{region}', actual,stored,atol=2e-4)
                control = dict(bin=bin_index+1,region=region,voxel=cell.tolist(),spins=spins.tolist(),
                               trapezoid_m_s=actual,saved_Gauss_m_s=stored,absolute_delta_m_s=abs(actual-stored))
                break
            check(f'trajectory control found:{region}:{bin_index}',control is not None)
            trajectory_controls.append(control)
    for png in sorted((BASE/'figures_v1').glob('*.png')):
        with Image.open(png) as im:
            im.load()
            check(png.name+':decoded dimensions', im.width>=5000 and im.height>=2000)
        with png.with_suffix('.pdf').open('rb') as stream:
            check(png.stem+':PDF signature', stream.read(5)==b'%PDF-')
    result = dict(experiment='E-048',status='PASS_INDEPENDENT_ARITHMETIC_AND_PROVENANCE',
                  created_utc=datetime.now(timezone.utc).isoformat(),checks=len(checks),check_names=checks,
                  distinct_files_hashed=len(hash_cache),legacy_identities_unchanged=len(legacy),
                  trajectory_controls=trajectory_controls,
                  PNG_review='All five final PNGs separately opened and visually inspected in the agent session; labels, denominators and caveats readable.',
                  PDF_review='Signature and hash checked; PDF rendering not inspected.',
                  limitation='Independent calculations on the same sources; not independent physical validation or a complete full-Bloch response model.',
                  source_sha256=digest(Path(__file__)))
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','checks','distinct_files_hashed','legacy_identities_unchanged','trajectory_controls')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    validate(p.parse_args().output)
