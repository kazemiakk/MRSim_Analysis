"""E-037 saved-data energy analysis and corrected aortic pilot comparisons."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import tomllib

import h5py
import numpy as np
from scipy.ndimage import binary_erosion

from mrsimtracks.hemodynamics import (jet_metrics, midpoint_cycle_integrals,
    region_kinetic_energy, section_flow, weighted_error_metrics)
from mrsimtracks.resolved_energy import cycle_energy_decomposition
from scripts.aorta_hemodynamic_reference import AortaGrid, sequence_timing
from scripts.audit_aorta_replenishment_mri import reconstruction_checks
from scripts.compare_aorta_hemodynamics import (derivative_set, dump, masks_for,
    section_components, table as write_csv)
from scripts.compare_dp150_hemodynamics import (REGIONS, finite_summary,
    paired_velocity_metrics, safe_json)

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'outputs/hemodynamic_extension'
DP = ROOT/'outputs/dp150_hemodynamic_comparison/e025_v1/results_v2'
AO = ROOT/'outputs/aorta_hemodynamic_comparison/e027_v1/results_v2'
PILOT = ROOT/'outputs/aorta_inlet_replenishment/e034_100k_reconstruction_v1'
MULTI = ROOT/'outputs/aorta_dp2_mri_pipeline/koma/multibin_8x4mm'
DP_H5 = ROOT/'outputs/dp150_mri_pipeline/koma/simulation_motion_1m_scanner_matched_venc5_seam_safe_v1/final_image_analysis/support_constrained_phase/dp150_seam_safe_support_phase.h5'
PINNED = {
    DP/'matched_reference.h5': 'aee4a2856fd8beea53f28be550a4c8d7f281e0557270473040307c7c309123fc',
    DP/'report.json': '908035a463428f44876ba1b91a467b798cdf29cffb5343558a6149b53a7bbe1f',
    AO/'matched_reference.h5': 'ade84dcad431246d86d13f429824168700dc2c1e306c5e6fa83126bdd0208afe',
    AO/'report.json': 'a54cf57e0e1fa27cf6e0238df338bcb16c6292095ad4225540ec441bbd1c9660',
    DP_H5: 'ed8f4157e14bc31e0e76200ca297fdec070578468d3292313254057d625d1adc',
    PILOT/'pc_reconstruction.h5': '3c8b09a7cc1a3228f49e8f2c7f60b653b222d87b3fcb3ccbc58f64adcf69ee11',
}


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def bind(path, expected=None):
    path = Path(path)
    actual = sha(path)
    if expected is not None and actual != expected:
        raise ValueError('Input identity mismatch: '+str(path))
    return dict(path=os.path.relpath(path, ROOT).replace('\\', '/'),
                bytes=path.stat().st_size, sha256=actual)


def require_same_reference(grid, phantom, period, weights, saved_weights):
    """Fail closed on the contract needed to reuse the E-027 CFD reference."""
    tests = {
        'center': np.array_equal(phantom['fixed_source_center_m'], grid.center),
        'axis_order': np.array_equal(np.array(phantom['axis_order_one_based'])-1, grid.order),
        'period': phantom['realized_rr_s'] == period,
        'source_cycle': phantom['source_cycle_s'] == .72,
        'adc_weights': weights.shape == saved_weights.shape and np.allclose(weights, saved_weights, rtol=0, atol=1e-12),
    }
    if not all(tests.values()):
        raise ValueError('Reference-reuse contract failed: '+str(tests))
    return {k: bool(v) for k, v in tests.items()}


def audit_turbulence_sources():
    native = ROOT.parent/'external_inputs/dp150_fluent'
    paths = sorted(native.glob('FFF-18-*.dat.h5'))
    if len(paths) != 73:
        raise ValueError('Expected 73 original dp150 frames')
    rows = []
    candidates = set()
    pattern = re.compile(r'SV_(?:K(?:_.*)?|OMEGA|EPSILON|D|UU|VV|WW|UV|UW|VW|.*TKE.*|.*SGS.*|.*REYN.*)$')
    for path in paths:
        datasets = []
        with h5py.File(path) as f:
            f.visititems(lambda n, x: datasets.append(n) if isinstance(x, h5py.Dataset) else None)
            fields = sorted({n.split('/')[-2] for n in datasets if '/SV_' in n})
            found = [x for x in fields if pattern.fullmatch(x)]
            candidates.update(found)
            mu_t = f['results/1/phase-1/cells/SV_MU_T/1'][:]
            rows.append(dict(**bind(path), field_names=fields,
                turbulence_candidate_fields=found,
                eddy_viscosity_min_pa_s=float(mu_t.min()),
                eddy_viscosity_max_pa_s=float(mu_t.max())))
    aorta = ROOT.parent/'external_inputs/aorta_pointcloud/processed/dp2'
    metadata = json.loads((aorta/'metadata.json').read_text(encoding='utf-8'))
    return dict(dp150=dict(frames=rows, candidate_fields=sorted(candidates),
            schema_scope='All HDF5 dataset names in all73 source frames; candidate variable names require semantic qualification',
            model='Unresolved conflicting saved model flags; eddy viscosity alone is not TKE',
            qualified_solver_k_available=False,
            tke_comparison_available=False),
        aorta=dict(files=sorted(p.name for p in aorta.iterdir() if p.is_file()),
            metadata=metadata, velocity_shape=list(np.load(aorta/'velocities.npy', mmap_mode='r', allow_pickle=False).shape),
            turbulence_fields_in_verified_four_file_contract=False,
            pickle_loaded=False, tke_comparison_available=False))


def energy_case(out_h5, name, reference, estimate, volume, masks, *, density=None,
                regions=None):
    result = dict(supports={}, regions=[])
    for support, mask in masks.items():
        if not mask.any():
            result['supports'][support] = dict(status='UNAVAILABLE_EMPTY_SUPPORT')
            continue
        pair = {}
        for method, u in [('CFD', reference), ('Koma', estimate)]:
            dec = cycle_energy_decomposition(u, volume, mask, density_kg_m3=density)
            if dec['relative_closure'] > 1e-10:
                raise ValueError('Energy decomposition did not close')
            for field in ['mean_velocity_m_s', 'coherent_component_variance_m2_s2']:
                out_h5.create_dataset(f'{name}/{support}/{method}/{field}', data=dec.pop(field), compression='gzip')
            pair[method] = dec
        w = np.broadcast_to(volume, reference.shape[:-1])
        m = np.broadcast_to(mask, reference.shape[:-1])
        pair['velocity_error'] = paired_velocity_metrics(reference, estimate, w, m)
        for quantity in ['mean_resolved_ke', 'mean_vector_ke', 'coherent_variation_ke']:
            key = quantity+'_per_density_m5_s2'
            pair[quantity+'_ratio_Koma_over_CFD'] = pair['Koma'][key]/pair['CFD'][key] if pair['CFD'][key] > 0 else None
        result['supports'][support] = pair
    for region, region_mask in (regions or {}).items():
        if not region_mask.any():
            continue
        for b in range(len(reference)):
            for method, u in [('CFD', reference), ('Koma', estimate)]:
                result['regions'].append(dict(region=region, bin=b+1, method=method,
                    **finite_summary(np.linalg.norm(u[b], axis=-1), volume, region_mask),
                    **region_kinetic_energy(u[b], volume, valid=region_mask, density_kg_m3=density)))
    return result


def pilot_sections(reference, estimate, fraction, common, grid, period):
    rows, summaries, profiles, cores = [], [], [], []
    for section in section_components(fraction, grid):
        i = section['readout_index']; roi = section['roi']
        valid = roi & common[:, :, i] & (fraction[:, :, i] > 0)
        area = fraction[:, :, i]*grid.spacing[1]*grid.spacing[2]
        waves = {}
        for method, u in [('CFD', reference), ('Koma', estimate)]:
            waves[method] = []
            for b in range(len(u)):
                flow = section_flow(u[b, :, :, i], area, valid=valid)
                rows.append(dict(section=section['id'], bin=b+1, method=method,
                    x_m=section['x_m'], net_ml_s=1e6*flow['net_m3_s'], **flow))
                waves[method].append(flow['net_m3_s'])
                for axis in (0, 1):
                    for k in range(valid.shape[axis]):
                        sl = [slice(None), slice(None)]; sl[axis] = k; sl = tuple(sl)
                        keep = valid[sl]; w = area[sl][keep]
                        value = float(np.average(u[b, :, :, i, 0][sl][keep], weights=w)) if w.sum() else np.nan
                        profiles.append(dict(section=section['id'], bin=b+1, method=method,
                            axis='slice_Z' if axis == 0 else 'phase_Y',
                            coordinate_mm=float(grid.axes()[2 if axis == 0 else 1][k]*1000),
                            normal_velocity_m_s=value))
        cfd, koma = np.array(waves['CFD']), np.array(waves['Koma'])
        row = dict(section=section['id'], x_m=section['x_m'], paired_pixels=int(valid.sum()),
                   numerical_section_area_mm2=float(area[roi].sum()*1e6),
                   paired_area_mm2=float(area[valid].sum()*1e6))
        if valid.any():
            row.update(flow_error_ml_s=weighted_error_metrics(cfd*1e6, koma*1e6),
                CFD_cycle=midpoint_cycle_integrals(cfd, period),
                Koma_cycle=midpoint_cycle_integrals(koma, period))
            row['net_cycle_volume_ratio'] = koma.sum()/cfd.sum() if abs(cfd.sum()) > 1e-12 else None
        else:
            row['status'] = 'UNAVAILABLE_EMPTY_SUPPORT'
        summaries.append(row)
        for b in range(len(reference)):
            p99 = finite_summary(abs(reference[b, :, :, i, 0]), area, valid)['p99']
            if not np.isfinite(p99):
                continue
            for factor in (.25, .5, .75):
                for sign in (-1, 1):
                    for method, u in [('CFD', reference), ('Koma', estimate)]:
                        core = jet_metrics(u[b, :, :, i], grid.points()[:, :, i], area,
                            normal=(sign, 0, 0), valid=valid, minimum_forward_speed_m_s=factor*p99)
                        centroid = core.pop('centroid_m')
                        direction = core.pop('direction_unit')
                        cores.append(dict(section=section['id'], bin=b+1, method=method,
                            threshold_fraction=factor, threshold_m_s=factor*p99, direction_sign=sign,
                            centroid_X_m=centroid[0], centroid_Y_m=centroid[1], centroid_Z_m=centroid[2],
                            direction_X=direction[0], direction_Y=direction[1], direction_Z=direction[2], **core))
    return dict(series=rows, summary=summaries, profiles=profiles, flow_cores=cores,
                definition='Separate +MRI-X partial numerical-support sections; not the confirmed physical inlet')


def write_readout(out, report):
    lines = ['# E-037: resolved energy and corrected aortic pilot hemodynamics', '',
        'Internal consistency only. True TKE, wall shear and pressure comparisons remain unsupported.', '',
        '| Case / support | Vector RMSE (m/s) | Resolved KE Koma/CFD | Coherent variation KE Koma/CFD |',
        '|---|---:|---:|---:|']
    for name, case in report['cases'].items():
        for support, pair in case['supports'].items():
            if 'CFD' not in pair:
                continue
            lines.append(f"| {name} / {support} | {pair['velocity_error']['vector_rmse_m_s']:.6f} | {pair['mean_resolved_ke_ratio_Koma_over_CFD']:.4f} | {pair['coherent_variation_ke_ratio_Koma_over_CFD']:.4f} |")
    lines += ['', 'Coherent variation measures changes between cardiac bins, including errors; it is NOT TKE.',
        'dp150 uses verified density1060kg/m3. Aortic energy remains per density; no material property was borrowed.',
        'Different supports/cases are not controlled spin-count or replenishment comparisons.', '',
        '## What is newly completed', '',
        '- Fixed-domain cycle-mean/coherent resolved-energy decomposition for dp150 and both aortic reconstructions.',
        '- dp150 same-quadrature spatial unresolved mean-field energy, explicitly not turbulent energy.',
        '- Corrected100k aortic regional speed/KE, partial-section waveforms, forward/reverse flux, nominal-bin volumes, profiles, directional flow cores and interior gradient metrics.',
        '- Aortic5/10/15/20% mask, fixed legacy/pilot intersection and identical4mm-smoothing sensitivities.',
        '- All73 original Fluent frame schemas checked for turbulence quantities; restored aortic package inventoried.', '',
        '## Why true TKE is not reported', '',
        'TKE requires velocity fluctuations relative to an appropriate ensemble/phase mean, or a qualified MRI intravoxel-variance measurement. A single cardiac cycle of mean vectors does not isolate these fluctuations. Eddy viscosity is not turbulent kinetic energy. Current trajectory generation has no imported Reynolds-stress/SGS variance or specified stochastic fluctuation process.', '',
        'The existence of complex REF/X/Y/Z images alone does not qualify magnitude attenuation as turbulent variance. Spatial shear, motion, partial volume, reset/magnetization history and imaging-gradient moments must be separated with controls. No negative apparent variance was clipped into a TKE map and no missing TKE was replaced by zero.', '',
        '## Inputs needed to unlock the gated endpoints', '',
        '1. TKE: recover native solver model/settings plus k/Reynolds stresses/SGS fields or adequately resolved repeated-cycle velocities; export units and geometry linkage. For MRI TKE, agree a fluctuation-aware forward model and calibrate the combined encoding/readout response before a small pilot.',
        '2. WSS/TAWSS/OSI: native wall/cap geometry, valid normals and wall distances, sufficiently resolved wall gradients and a tested wall estimator. Aortic point-cloud numerical support does not supply these.',
        '3. Pressure/drop: native pressure with an operating-condition-consistent CFD comparator or an explicitly qualified velocity-to-pressure method. dp150 source pressure is not automatically valid after retiming/synthetic closure.',
        '4. Aortic absolute energy/loss: verify density and viscosity from source solver settings; no blood-property default is assumed.', '',
        '## Interpretation and preserved evidence', '',
        'Aortic pilot reuses the E-027 ADC-average comparator only after exact source/grid/sequence/timing checks. That comparator is a kinematic average, not the MRI complex forward response. Legacy1M and corrected100k have multiple differences and are not a controlled density experiment. Anonymous section flow is not the confirmed inlet: the separate E-036 near-full inferred-inlet result remains unchanged.', '',
        'The dp150 spatial-dispersion difference uses identical quadrature/support and15 midpoint samples; it contains mean-field spatial gradients, not a CFD turbulence closure. Synthetic-tail bins14/15 remain disclosed. No new acquisition, resampling to fictitious bins, fitted shift/gain or source overwrite occurred.', '',
        'Primary-method reference: [McGrath etal., ISMRM2023: encoding/readout interplay in turbulence imaging](https://doi.org/10.58530/2023/3646). Their turbulence simulations explicitly add stochastic fluctuations; that is not implemented in the current deterministic saved trajectories.', '',
        'Numeric results: report.json, energy_summary.csv, regional_series.csv, aorta_pilot_sections.csv, aorta_pilot_profiles.csv, aorta_pilot_flow_cores.csv, energy_fields.h5 and turbulence_source_audit.json.']
    (out/'READOUT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    target = p.parse_args().output.resolve()
    if target.exists():
        raise FileExistsError('Refusing to overwrite '+str(target))
    inputs = [bind(path, expected) for path, expected in PINNED.items()]
    dp_report = json.loads((DP/'report.json').read_text())
    ao_report = json.loads((AO/'report.json').read_text())
    for path in [Path(__file__), ROOT/'mrsimtracks/resolved_energy.py', ROOT/'mrsimtracks/hemodynamics.py',
                 ROOT/'scripts/compare_aorta_hemodynamics.py', ROOT/'scripts/compare_dp150_hemodynamics.py',
                 ROOT/'scripts/aorta_hemodynamic_reference.py', ROOT/'scripts/audit_aorta_replenishment_mri.py', BASE/'E037_PROTOCOL.md']:
        inputs.append(bind(path))
    # Recheck the original aortic source identities, rather than relying on its directory name.
    for row in ao_report['inputs']:
        if '/external_inputs/' in row['path'].replace('\\', '/'):
            inputs.append(bind(ROOT/row['path'], row['sha256']))
    sim = tomllib.loads((PILOT/'simulation_report.toml').read_text())
    acq = Path(sim['acquisition_directory'])
    phantom_path = acq/'phantom_report.toml'
    inputs += [bind(PILOT/'simulation_report.toml'), bind(phantom_path, sim['phantom_report_sha256']),
               bind(acq/'inlet_pilot.phantom', sim['phantom_sha256'])]
    phantom = tomllib.loads(phantom_path.read_text())
    registration_path = ROOT/'outputs/aorta_inlet_replenishment/e033_100k_v1/registration.json'
    tracking_registration = json.loads(registration_path.read_text()); inputs.append(bind(registration_path))
    if tracking_registration['start_frame'] != 18 or tracking_registration['recorded_cycles'] != 1:
        raise ValueError('Unexpected source phase/cycle contract')
    inputs.append(bind(ROOT/'outputs/aorta_inlet_replenishment/e033_100k_v1/tracks.h5', phantom['tracks_sha256']))
    grid = AortaGrid(); period = ao_report['period_s']
    for encoding in ('REF', 'X', 'Y', 'Z'):
        seq = MULTI/f'sequences/{encoding.lower()}.seq'
        inputs.append(bind(seq, sim['sequence_sha256'][encoding]))
        inputs.append(bind(acq/(encoding.lower()+'_result.jld2'), sim['raw_result_sha256'][encoding]))
    _, weights, _ = sequence_timing(MULTI/'sequences/ref.seq', period)
    with h5py.File(AO/'matched_reference.h5') as f:
        reuse = require_same_reference(grid, phantom, period, weights, f['frame_weights'][:8])
        ref_ao = f['adc_time_average'][:].astype(float); legacy = f['Koma'][:]
        fraction_ao = f['fraction'][:]; plane_fraction = f['plane_fraction'][:]
        oldmask = f['masks/core15'][:]
    with h5py.File(PILOT/'pc_reconstruction.h5') as f:
        mag = f['reference_magnitude'][:]; stored = f['mask'][:]
        pilot = np.moveaxis(f['velocity'][:], 0, -1).astype(float)
        checks = reconstruction_checks(mag, pilot, stored, f['speed'][:], float(f.attrs['Venc_m_per_s']))
        checks['fov'] = bool(np.allclose(f.attrs['FOV_m'], grid.fov, rtol=1e-6, atol=0))
        checks['period'] = bool(np.isclose(float(f.attrs['RR_s']), period, rtol=0, atol=1e-7))
    if not all(checks.values()):
        raise ValueError('Pilot reconstruction checks failed: '+str(checks))
    masks = masks_for(mag, stored.astype(bool), fraction_ao)
    masks['legacy_pilot_intersection15'] = masks['core15'] & oldmask
    masks['eroded15'] = binary_erosion(masks['core15'], structure=np.ones((3, 3, 3)))
    if not np.isfinite(ref_ao[:, masks['core05']]).all():
        raise ValueError('Reference missing on pilot support')
    target.parent.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix=target.name+'.partial-', dir=target.parent))
    dump(out/'registration.json', dict(experiment='E-037', created_utc=datetime.now(timezone.utc).isoformat(),
        inputs=inputs, reference_reuse=reuse, reconstruction_checks=checks, post_hoc=True,
        runtime=dict(python=sys.version, numpy=np.__version__, h5py=h5py.__version__),
        accumulation='float64 including velocity-error intermediates and SI volume weights'))
    print('Inputs, sequence/timing reuse and reconstruction checks passed; auditing native turbulence fields', flush=True)
    audit = audit_turbulence_sources(); dump(out/'turbulence_source_audit.json', audit)
    report = dict(experiment='E-037', evidence_class='Post hoc internal consistency; not independent physical validation',
        status='SUPPORTED_CALCULATIONS_COMPLETE_CONDITIONAL_ENDPOINTS_UNAVAILABLE', cases={},
        aorta_reference_reuse=reuse, pilot_reconstruction_checks=checks,
        unavailable={'TKE': 'No qualified turbulence fields/ensemble or MRI fluctuation model; not zero',
            'WSS_TAWSS_OSI': 'Wall geometry, resolution and estimator not qualified',
            'pressure': 'No operating-condition-consistent matched pressure comparator',
            'aortic_absolute_energy_and_loss': 'Density/viscosity unverified'})
    with h5py.File(out/'energy_fields.h5', 'x') as fields:
        fields.attrs.update(experiment='E-037', evidence_class=report['evidence_class'], is_tke=False)
        with h5py.File(DP/'matched_reference.h5') as f:
            ref_dp = f['cfd_voxel_midpoint'][:].astype(float); common = f['common_mask'][:]
            direct = f['direct_common_mask'][:]
            volume_dp = f['volume_fraction_2'][:].astype(float)*(.00125*.00125*.0025)
            z = f['source_z_m'][:]
        with h5py.File(DP_H5) as f:
            est_dp = np.moveaxis(f['velocity_corrected_m_per_s'][:], 0, -1).astype(float)
            raw_dp = np.moveaxis(f['velocity_raw_m_per_s'][:], 0, -1).astype(float)
        regions = {name: common & (z[None, None, :] >= lo) & (z[None, None, :] <= hi) for name, (lo, hi) in REGIONS.items()}
        report['cases']['dp150_1m_hybrid'] = energy_case(fields, 'dp150_1m_hybrid', ref_dp, est_dp, volume_dp,
            {'common': common}, density=1060., regions=regions)
        report['cases']['dp150_1m_raw_direct'] = energy_case(fields, 'dp150_1m_raw_direct', ref_dp, raw_dp, volume_dp,
            {'direct_partial': direct}, density=1060.)
        mean_ke = np.array(report['cases']['dp150_1m_hybrid']['supports']['common']['CFD']['per_bin_ke_per_density_m5_s2'])*1060.
        fine_ke = np.array([r['quadrature_unaveraged_ke_common_j'] for r in dp_report['native_values']])
        dispersion = fine_ke-mean_ke
        if np.min(dispersion) < -1e-8*max(float(fine_ke.max()), 1e-30):
            raise ValueError('Inconsistent spatial energy moments')
        report['dp150_spatial_dispersion'] = dict(is_tke=False,
            definition='Same-quadrature unaveraged minus voxel-mean converted-field KE on identical common support',
            per_bin_j=dispersion, mean_j=float(dispersion.mean()),
            mean_unaveraged_j=float(fine_ke.mean()), mean_voxel_resolved_j=float(mean_ke.mean()),
            fraction_of_unaveraged=float(dispersion.mean()/fine_ke.mean()),
            includes_synthetic_tail_bins=[14, 15])
        del ref_dp, est_dp, raw_dp
        volume_ao = fraction_ao*np.prod(grid.spacing)
        x = grid.points()[..., 0]; primary = masks['core15']
        regions_ao = dict(whole=primary, readout_below_minus45=primary & (x < -.045),
            readout_middle=primary & (x >= -.045) & (x <= .045), readout_above45=primary & (x > .045))
        report['cases']['aorta_corrected_100k_pilot'] = energy_case(fields, 'aorta_corrected_100k_pilot',
            ref_ao, pilot, volume_ao, masks, regions=regions_ao)
        report['cases']['aorta_legacy_1m'] = energy_case(fields, 'aorta_legacy_1m', ref_ao, legacy, volume_ao,
            {'legacy_core15': oldmask, 'legacy_pilot_intersection15': masks['legacy_pilot_intersection15']})
        for case, mask in [('aorta_corrected_100k_pilot', primary), ('aorta_legacy_1m', oldmask)]:
            fields.create_dataset(case+'/primary_mask', data=mask, compression='gzip')
        fields.create_dataset('aorta_corrected_100k_pilot/velocity_m_s', data=pilot, compression='gzip')
        fields.create_dataset('aorta_corrected_100k_pilot/reference_m_s', data=ref_ao, compression='gzip')
        fields.create_dataset('aorta_corrected_100k_pilot/fraction', data=fraction_ao, compression='gzip')
        fields.attrs['aorta_grid'] = json.dumps(asdict(grid))
    print('Energy decomposition complete; calculating corrected aortic section and gradient endpoints', flush=True)
    report['aorta_pilot_sections'] = pilot_sections(ref_ao, pilot, plane_fraction, primary, grid, period)
    report['aorta_pilot_derivatives'] = dict(primary=derivative_set(ref_ao, pilot, primary, grid.spacing),
        matched_4mm_smoothing=derivative_set(ref_ao, pilot, primary, grid.spacing, sigma=.004))
    report['coverage'] = dict(pilot_core15_voxels=int(primary.sum()), legacy_core15_voxels=int(oldmask.sum()),
        same_support_voxels=int(masks['legacy_pilot_intersection15'].sum()),
        area_volume_semantics='Approximate nearest-point support, not certified physical lumen')
    report['inputs'] = inputs
    dump(out/'report.json', report)
    energy_rows = []
    regional_rows = []
    for case, result in report['cases'].items():
        regional_rows += [dict(case=case, **r) for r in result['regions']]
        for support, pair in result['supports'].items():
            for method in ['CFD', 'Koma']:
                if method in pair:
                    energy_rows.append(dict(case=case, support=support, method=method,
                        **{k: v for k, v in pair[method].items() if not isinstance(v, (np.ndarray, list))}))
    write_csv(out/'energy_summary.csv', safe_json(energy_rows))
    write_csv(out/'regional_series.csv', safe_json(regional_rows))
    for name, key in [('sections', 'series'), ('profiles', 'profiles'), ('flow_cores', 'flow_cores')]:
        write_csv(out/f'aorta_pilot_{name}.csv', safe_json(report['aorta_pilot_sections'][key]))
    write_readout(out, report)
    dump(out/'MANIFEST.json', {p.name: dict(bytes=p.stat().st_size, sha256=sha(p)) for p in out.iterdir() if p.is_file()})
    if target.exists():
        raise FileExistsError(target)
    out.rename(target)
    print(json.dumps(safe_json(dict(output=str(target), coverage=report['coverage'],
        pilot_primary=report['cases']['aorta_corrected_100k_pilot']['supports']['core15']['velocity_error'],
        spatial_dispersion=report['dp150_spatial_dispersion'])), indent=2), flush=True)


if __name__ == '__main__':
    main()
