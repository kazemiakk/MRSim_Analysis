"""E-038: complete the original matched-data comparison, never another archive."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile

import h5py
import numpy as np
from scipy.ndimage import binary_erosion

from mrsimtracks.hemodynamics import differential_metrics, velocity_gradient, weighted_error_metrics
from scripts.compare_dp150_hemodynamics import (REGIONS, combine_error_summaries,
    matched_smooth, paired_velocity_metrics, safe_json)
from scripts.compare_aorta_hemodynamics import derivative_set, dump, table
from scripts.extend_hemodynamic_comparisons import (AO, DP, DP_H5, PILOT, PINNED,
    ROOT, bind, sha)

EXT = ROOT/'outputs/hemodynamic_extension/e037_v2'
PROTOCOL = ROOT/'outputs/hemodynamic_extension/E038_PROTOCOL.md'
FIELD_NAMES = ('vorticity_magnitude_s_inv', 'helicity_m_s_inv2', 'relative_helicity',
               'strain_frobenius_s_inv', 'divergence_s_inv',
               'dissipation_per_viscosity_s_inv2', 'enstrophy_s_inv2')


def scalar_fields(u, spacing, mask):
    """Same central-stencil operators as the original plan; no wall imputation."""
    gradient, valid = velocity_gradient(u, spacing, mask)
    fields = differential_metrics(gradient, u)
    fields['strain_frobenius_s_inv'] = np.sqrt(np.sum(fields['strain_rate_s_inv']**2, axis=(-2, -1)))
    fields['enstrophy_s_inv2'] = .5*fields['vorticity_magnitude_s_inv']**2
    return {name: fields[name] for name in FIELD_NAMES}, valid


def scalar_region_integrals(fields, volume, valid):
    """Resolved interior integrals; dissipation/mu has units m3/s2."""
    v = np.asarray(volume, float)[valid]
    h = fields['helicity_m_s_inv2'][valid]
    return dict(sampled_volume_m3=float(v.sum()),
        signed_helicity_m4_s2=float(np.sum(h*v)),
        absolute_helicity_m4_s2=float(np.sum(abs(h)*v)),
        enstrophy_m3_s2=float(np.sum(fields['enstrophy_s_inv2'][valid]*v)),
        dissipation_per_viscosity_m3_s2=float(np.sum(fields['dissipation_per_viscosity_s_inv2'][valid]*v)))


def compare_gradients(group, case):
    ref, est, volume = case['reference'], case['estimate'], case['volume']
    compact = None; datasets = {}; weights = None
    errors = []; integrals = []; pooled = {name: [] for name in FIELD_NAMES}
    pooled_unweighted = {name: [] for name in FIELD_NAMES}
    for b in range(len(ref)):
        a, va = scalar_fields(ref[b], case['spacing'], case['gradient_mask'])
        c, vc = scalar_fields(est[b], case['spacing'], case['gradient_mask'])
        valid = va & vc
        if compact is None:
            compact = np.flatnonzero(valid); weights = volume.ravel()[compact]
            group.create_dataset('flat_indices', data=compact)
            group.create_dataset('volume_weights_m3', data=weights)
            group.attrs.update(shape_spr=case['shape'], spacing_xyz_m=case['spacing'],
                period_s=case['period'], coordinate_convention=case['coordinate_convention'],
                storage_precision='float32 scalar fields; float64 accumulation', is_tke=False)
            for method in ('CFD', 'Koma'):
                for name in FIELD_NAMES:
                    datasets[method, name] = group.create_dataset(method+'/'+name,
                        shape=(len(ref), len(compact)), dtype='f4', compression='gzip')
        elif not np.array_equal(compact, np.flatnonzero(valid)):
            raise ValueError('Derivative domain changes between bins')
        for name in FIELD_NAMES:
            av, cv = a[name].ravel()[compact], c[name].ravel()[compact]
            datasets['CFD', name][b] = av; datasets['Koma', name][b] = cv
            weighted = weighted_error_metrics(av, cv, weights)
            unweighted = weighted_error_metrics(av, cv)
            errors.append(dict(bin=b+1, metric=name, **weighted))
            pooled[name].append(weighted); pooled_unweighted[name].append(unweighted)
        for region, selection in case['regions'].items():
            keep = valid & selection
            if not keep.any():
                continue
            for method, values in [('CFD', a), ('Koma', c)]:
                row = dict(bin=b+1, region=region, method=method,
                           **scalar_region_integrals(values, volume, keep))
                if case['viscosity'] is not None:
                    row['resolved_dissipation_w'] = row['dissipation_per_viscosity_m3_s2']*case['viscosity']
                integrals.append(row)
        print(case['name'], 'scalar bin', b+1, '/', len(ref), flush=True)
    summary = dict(stencil_voxels=len(compact), stencil_volume_m3=float(weights.sum()),
        weighted_pooled={k: combine_error_summaries(v) for k, v in pooled.items()},
        unweighted_pooled={k: combine_error_summaries(v) for k, v in pooled_unweighted.items()},
        per_bin_errors=errors, region_integrals=integrals, cycle_integrals=[])
    for region in case['regions']:
        for method in ('CFD', 'Koma'):
            rows = [r for r in integrals if r['region'] == region and r['method'] == method]
            if len(rows) != len(ref):
                continue
            loss = np.array([r['dissipation_per_viscosity_m3_s2'] for r in rows])
            row = dict(region=region, method=method,
                mean_resolved_dissipation_per_viscosity_m3_s2=float(loss.mean()),
                nominal_bin_loss_integral_per_viscosity_m3_s=float(loss.mean()*case['period']),
                sampled_volume_m3=rows[0]['sampled_volume_m3'],
                interpretation='Interior resolved molecular loss, not total turbulent loss')
            if case['viscosity'] is not None:
                row.update(mean_resolved_dissipation_w=float(loss.mean()*case['viscosity']),
                           nominal_bin_resolved_loss_j=float(loss.mean()*case['period']*case['viscosity']))
            summary['cycle_integrals'].append(row)
    return summary


def pair_cores(rows):
    """Pair only identical section/bin/threshold/direction; missing core stays missing."""
    groups = {}
    for row in rows:
        key = (row['bin'], row['section'], row['threshold_fraction'], row.get('direction_sign', 1))
        method = row.get('method', {'cfd': 'CFD', 'koma': 'Koma'}.get(row.get('dataset')))
        if method not in ('CFD', 'Koma') or method in groups.setdefault(key, {}):
            raise ValueError('Duplicate or unknown flow-core method')
        groups[key][method] = row
    result = []
    for (b, section, threshold, sign), pair in groups.items():
        if set(pair) != {'CFD', 'Koma'}:
            raise ValueError('Unpaired flow core')
        a, c = pair['CFD'], pair['Koma']
        def vector(row, prefix, fields):
            if prefix in row:
                return np.asarray(row[prefix], float)
            return np.array([row.get(k) for k in fields], float)
        ac = vector(a, 'centroid_m', ['centroid_X_m', 'centroid_Y_m', 'centroid_Z_m'])
        cc = vector(c, 'centroid_m', ['centroid_X_m', 'centroid_Y_m', 'centroid_Z_m'])
        ad = vector(a, 'direction_unit', ['direction_X', 'direction_Y', 'direction_Z'])
        cd = vector(c, 'direction_unit', ['direction_X', 'direction_Y', 'direction_Z'])
        finite = bool(np.isfinite([ac, cc, ad, cd]).all())
        result.append(dict(bin=b, section=section, threshold_fraction=threshold, direction_sign=sign,
            paired_core_available=finite,
            centroid_error_mm=float(np.linalg.norm(cc-ac)*1e3) if finite else np.nan,
            direction_error_deg=float(np.degrees(np.arccos(np.clip(ad@cd, -1, 1)))) if finite else np.nan,
            CFD_radius_mm=float(a['rms_radius_m']*1000) if a['rms_radius_m'] is not None else np.nan,
            Koma_radius_mm=float(c['rms_radius_m']*1000) if c['rms_radius_m'] is not None else np.nan))
    return result


def flow_summaries(rows, period):
    result = []
    sections = sorted({r['section'] for r in rows})
    for section in sections:
        waves = {}
        for method in ('CFD', 'Koma'):
            selected = sorted([r for r in rows if r['section'] == section and r['method'] == method], key=lambda r: r['bin'])
            if not selected or [r['bin'] for r in selected] != list(range(1, len(selected)+1)):
                raise ValueError('Incomplete or duplicate section bins')
            q = np.array([r['net_m3_s'] for r in selected]); dt = period/len(q)
            waves[method] = q
            result.append(dict(section=section, method=method, bins=len(q),
                net_nominal_cycle_ml=float(q.sum()*dt*1e6),
                net_forward_time_partition_ml=float(np.maximum(q, 0).sum()*dt*1e6),
                net_reverse_time_partition_ml=float(np.maximum(-q, 0).sum()*dt*1e6),
                simultaneous_forward_volume_ml=float(sum(r['forward_m3_s'] for r in selected)*dt*1e6),
                simultaneous_reverse_volume_ml=float(sum(r['reverse_m3_s'] for r in selected)*dt*1e6),
                absolute_peak_ml_s=float(np.max(abs(q))*1e6), absolute_peak_bin=int(np.argmax(abs(q))+1),
                sampled_area_mm2=float(selected[0]['sampled_area_m2']*1e6)))
        if len(waves['CFD']) != len(waves['Koma']):
            raise ValueError('Different section time grids')
        result[-1]['paired_error_ml_s'] = weighted_error_metrics(waves['CFD']*1e6, waves['Koma']*1e6)
    return result


def load_inputs():
    inputs = [bind(p, expected) for p, expected in PINNED.items()]
    manifest_path = EXT/'MANIFEST.json'
    inputs.append(bind(manifest_path, '463ce755d3cd2e383d4e00461dbcea41f05af4f8b41bd513534aef3c3df7e50d'))
    manifest = json.loads(manifest_path.read_text())
    inputs.append(bind(EXT/'report.json', manifest['report.json']['sha256']))
    dp, ao, ext = [json.loads(p.read_text()) for p in [DP/'report.json', AO/'report.json', EXT/'report.json']]
    with h5py.File(DP/'matched_reference.h5') as f:
        ref = f['cfd_voxel_midpoint'][:].astype(float); fraction = f['volume_fraction_2'][:].astype(float)
        common = f['common_mask'][:]; z = f['source_z_m'][:]
        times_dp = f['time_s'][:]
    with h5py.File(DP_H5) as f:
        estimate = np.moveaxis(f['velocity_corrected_m_per_s'][:], 0, -1).astype(float)
    regions = {name: common & ((z >= lo) & (z < hi))[None, None, :] for name, (lo, hi) in REGIONS.items()}
    cases = [dict(name='phantom_1m', reference=ref, estimate=estimate,
        volume=fraction*np.prod([.00125, .00125, .0025]), gradient_mask=common & (fraction >= .999999),
        spacing=np.array([.00125, .00125, .0025]), shape=common.shape, regions=regions,
        viscosity=.0035, period=dp['timing']['effective_motion_period_s'], times=times_dp,
        coordinate_convention='MRI R/P/S half-voxel centers; source Z/X/Y transform unchanged')]
    with h5py.File(AO/'matched_reference.h5') as f:
        ref = f['adc_time_average'][:].astype(float); legacy = f['Koma'][:]
        fraction = f['fraction'][:].astype(float); oldmask = f['masks/core15'][:]
        references = {k: f[k][:].astype(float) for k in ['adc_time_average', 'nominal_midpoint', 'box3', 'quadrature3', 'legacy_half_voxel']}
    with h5py.File(PILOT/'pc_reconstruction.h5') as f:
        pilot = np.moveaxis(f['velocity'][:], 0, -1).astype(float)
    # Mask is regenerated from the unchanged, preregistered E-037 definition.
    with h5py.File(EXT/'energy_fields.h5') as f:
        inputs.append(bind(EXT/'energy_fields.h5', manifest['energy_fields.h5']['sha256']))
        pilotmask = f['aorta_corrected_100k_pilot/primary_mask'][:]
    spacing = np.array([.28/72, .004, .004]); x = (np.arange(72)-36)*spacing[0]
    for name, estimate, mask in [('aorta_1m', legacy, oldmask), ('aorta_100k_corrected', pilot, pilotmask)]:
        regions = dict(whole=mask, readout_below_minus45=mask & (x < -.045)[None, None, :],
            readout_middle=mask & ((x >= -.045) & (x <= .045))[None, None, :],
            readout_above45=mask & (x > .045)[None, None, :])
        cases.append(dict(name=name, reference=ref, estimate=estimate, volume=fraction*np.prod(spacing),
            gradient_mask=mask, spacing=spacing, shape=mask.shape, regions=regions, viscosity=None,
            period=ao['period_s'], times=np.array(ao['time_s']), coordinate_convention='Centered inverse-DFT samples; source phase18, actual ADC average'))
    for path in [Path(__file__), PROTOCOL, ROOT/'mrsimtracks/hemodynamics.py', ROOT/'scripts/compare_dp150_hemodynamics.py',
                 ROOT/'scripts/compare_aorta_hemodynamics.py']:
        inputs.append(bind(path))
    return inputs, cases, dp, ao, ext, references, fraction


def comparison_inventory(dp, ao, ext, report):
    supported = [
        ('Source identity, timing and coordinate alignment', 'inputs; E-025/E-027/E-037 registrations'),
        ('Component/vector/speed errors and coverage', 'E-025 aggregate; E-027 primary_pooled; E-037 cases'),
        ('Regional mean/p99/max speed', 'E-025 regional_values; E-027 regional_series; E-037 cases.regions'),
        ('Section flow waveforms and nominal-bin volumes', 'flow_comparison.csv; source section_values/section_series'),
        ('Simultaneous forward/reverse transport', 'flow_comparison.csv'),
        ('Velocity profiles', 'E-025 profiles.csv; E-027 profile_series.csv; E-037 aorta_pilot_profiles.csv'),
        ('Flow-core centroid/width/direction and thresholds', 'flow_core_comparison.csv; original flow-core rows'),
        ('Resolved KE waveforms and regional energy', 'E-025/E-027 regional series; E-037 energy_summary.csv'),
        ('Mean-vector/coherent energy decomposition (NOT TKE)', 'E-037 energy_summary.csv; energy_fields.h5'),
        ('Vorticity', 'scalar_comparisons.csv; advanced_fields.h5'),
        ('Signed helicity and relative helicity', 'scalar_comparisons.csv; advanced_fields.h5'),
        ('Strain rate and divergence', 'scalar_comparisons.csv; advanced_fields.h5'),
        ('Resolved molecular dissipation and interior loss integrals', 'region_integrals.csv; cycle_loss.csv'),
        ('Temporal-reference and spatial-quadrature sensitivity', 'E-025 sensitivity; E-027 sensitivity_series; corrected_pilot_reference_sensitivity.json'),
        ('Mask, erosion and identical-smoothing sensitivity', 'E-025 derivative_error_summary; E-027 derivatives; E-037 supports; corrected_pilot_eroded_derivatives.json'),
        ('All-bin numeric fields and representative scientific figures', 'advanced_fields.h5; presentation derivative')]
    rows = []
    for metric, evidence in supported:
        for case in report['cases']:
            rows.append(dict(metric=metric, case=case, status='COMPLETED_SUPPORTED_SCOPE', evidence=evidence))
    for metric, reason in [
        ('True TKE', 'Matched CFD has no qualified fluctuation statistics; no qualified MRI turbulence estimator'),
        ('WSS / TAWSS / OSI', 'A defensible wall-gradient estimator/resolution is absent; aortic native wall connectivity is absent'),
        ('Pressure / pressure drop', 'No qualified operating-condition-consistent pressure reference/estimator'),
        ('Independent physical validation', 'No independently measured matched velocity/flow reference')]:
        for case in report['cases']:
            rows.append(dict(metric=metric, case=case, status='UNAVAILABLE_NOT_COMPUTED', evidence=reason))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', type=Path, required=True)
    target = p.parse_args().output.resolve()
    if target.exists():
        raise FileExistsError(target)
    inputs, cases, dp, ao, ext, references, fraction = load_inputs()
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=target.name+'.partial-', dir=target.parent))
    dump(stage/'registration.json', dict(experiment='E-038', created_utc=datetime.now(timezone.utc).isoformat(),
        inputs=inputs, source_scope='Only original matched CFD/Koma artifacts; separate archive excluded',
        new_acquisition=False, numpy=np.__version__))
    report = dict(experiment='E-038', status='SUPPORTED_PLAN_COMPLETED_CONDITIONAL_ENDPOINTS_UNAVAILABLE',
        evidence_class='Internal consistency; not independent physical validation', cases={},
        source_restriction='Other phantom CFD archive excluded by user', inputs=inputs)
    scalar_rows = []; integral_rows = []; loss_rows = []; flow_rows = []; core_rows = []
    with h5py.File(stage/'advanced_fields.h5', 'x') as f:
        f.attrs.update(experiment='E-038', is_tke=False, evidence_class=report['evidence_class'])
        for case in cases:
            result = compare_gradients(f.create_group(case['name']), case)
            report['cases'][case['name']] = dict(shape=case['shape'], period_s=case['period'],
                times_s=case['times'], viscosity_pa_s=case['viscosity'], gradients=result)
            scalar_rows += [dict(case=case['name'], **r) for r in result['per_bin_errors']]
            integral_rows += [dict(case=case['name'], **r) for r in result['region_integrals']]
            loss_rows += [dict(case=case['name'], **r) for r in result['cycle_integrals']]
    # Complete the corrected pilot's original five-reference sensitivity family.
    pilot = cases[-1]; mask = pilot['gradient_mask']; volume = pilot['volume']
    sensitivities = []
    for name, ref in references.items():
        keep = mask & np.isfinite(ref).all(axis=(0, 4))
        for b in range(8):
            sensitivities.append(dict(reference=name, bin=b+1, voxels=int(keep.sum()),
                paired_volume_m3=float(volume[keep].sum()),
                **paired_velocity_metrics(ref[b], pilot['estimate'][b], volume, keep)))
    dump(stage/'corrected_pilot_reference_sensitivity.json', sensitivities)
    eroded = binary_erosion(mask, structure=np.ones((3, 3, 3)))
    erosion = derivative_set(pilot['reference'], pilot['estimate'], eroded, pilot['spacing'])
    dump(stage/'corrected_pilot_eroded_derivatives.json', erosion)
    report['corrected_pilot_reference_sensitivity'] = sensitivities
    report['corrected_pilot_eroded_derivatives'] = erosion
    report['preserved_derivative_sensitivities'] = dict(phantom=dp['derivative_error_summary'],
        aorta_1m=ao['derivatives'], aorta_100k_corrected=ext['aorta_pilot_derivatives'])
    dp_sections = [dict(**r, method={'cfd_voxel_common': 'CFD', 'koma_hybrid_common': 'Koma'}[r['dataset']])
        for r in dp['section_values'] if r['dataset'] in ['cfd_voxel_common', 'koma_hybrid_common']]
    for case, sections, cores in [(cases[0], dp_sections, dp['jet_values']),
            (cases[1], ao['section_series'], ao['flow_core_series']),
            (cases[2], ext['aorta_pilot_sections']['series'], ext['aorta_pilot_sections']['flow_cores'])]:
        flow_rows += [dict(case=case['name'], **r) for r in flow_summaries(sections, case['period'])]
        core_rows += [dict(case=case['name'], **r) for r in pair_cores(cores)]
    table(stage/'scalar_comparisons.csv', scalar_rows); table(stage/'region_integrals.csv', integral_rows)
    table(stage/'cycle_loss.csv', loss_rows); table(stage/'flow_comparison.csv', flow_rows)
    table(stage/'flow_core_comparison.csv', core_rows)
    inventory = comparison_inventory(dp, ao, ext, report)
    table(stage/'plan_completion.csv', inventory)
    report['plan_completion'] = inventory; report['flow_comparison'] = flow_rows
    report['flow_core_comparison'] = core_rows
    dump(stage/'report.json', report)
    lines = ['# E-038: matched CFD–Koma advanced comparisons', '',
        'Other phantom archive excluded. No new simulation. Internal consistency only.', '',
        '| Analysis family | Phantom 1M | Aorta 1M | Aorta corrected 100k |',
        '|---|---|---|---|']
    for metric in dict.fromkeys(r['metric'] for r in inventory):
        rows = [r for r in inventory if r['metric'] == metric]
        lines.append('| '+metric+' | '+' | '.join('Complete within stated support' if r['status'].startswith('COMPLETED') else '**Unavailable, not computed**' for r in rows)+' |')
    lines += ['', '## Meaning of completion', '',
        'The complete supported plan includes original E-025/E-027 results, E-037 energy/pilot results and this new missing-work closure. See plan_completion.csv for the exact evidence mapping. Conditional metrics are not counted as completed. The two aortic acquisitions are distinct case versions, not a controlled spin-count experiment.', '',
        'The new per-bin scalar comparisons use volume weighting; unweighted summaries reproduce prior aortic reports. Scalar maps contain only central interior stencils, not wall shear. Aortic dissipation/loss is per viscosity; no material property was inherited. The resolved loss integral excludes walls/missing stencils and is not total turbulent energy loss.', '',
        'All bins are stored in advanced_fields.h5. Representative static maps do not imply an all-bin display. Undefined relative helicity and absent threshold-defined flow cores remain missing, not zero. Nominal-bin time integration is a discrete summary, not an exact continuous-cycle integral.', '',
        'Aortic anonymous sections are partial computational sections. E-036 remains the separate near-full inferred-inlet comparison. Native full-face CFD and eight MRI ADC averages are different temporal observations; no additional MRI phases or CFD-fitting shifts were invented.', '',
        '## Remaining conditional endpoints', '',
        '- True TKE: no qualified turbulent fluctuation statistics/forward model in the matched package. Cardiac-bin variation and unresolved spatial mean-field energy are not TKE.',
        '- WSS/TAWSS/OSI: no qualified wall estimator/resolution; aortic native wall/cap connectivity is absent.',
        '- Pressure/drop: no operating-condition-consistent matched pressure method/reference.',
        '- Independent physical validation: no independent matched velocity/flow payload.', '',
        'These gaps cannot be removed by another arithmetic pass over the saved mean-velocity images. Their prerequisites are the original plan conditions, not a request to study another archive.']
    (stage/'READOUT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    dump(stage/'MANIFEST.json', {p.name: dict(bytes=p.stat().st_size, sha256=sha(p)) for p in stage.iterdir() if p.is_file()})
    if target.exists():
        raise FileExistsError(target)
    stage.rename(target)
    print(json.dumps(safe_json(dict(output=str(target), scalar_error_rows=len(scalar_rows),
        integral_rows=len(integral_rows), loss_rows=len(loss_rows), flow_rows=len(flow_rows),
        core_pairs=len(core_rows), supported_families=16, conditional_families=4)), indent=2))


if __name__ == '__main__':
    main()
