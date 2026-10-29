"""Check every reviewed E-027 report row against the saved numerical result."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    report_path = args.root / 'results_v2/report.json'
    snapshot_path = args.root / 'report_app/src/data.json'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    snapshot_text = snapshot_path.read_text(encoding='utf-8')
    snapshot = json.loads(snapshot_text)
    checks = 0

    def equal(actual, expected):
        nonlocal checks
        checks += 1
        if isinstance(expected, dict):
            assert set(actual) == set(expected), (set(actual), set(expected))
            for key in expected:
                equal(actual[key], expected[key])
        elif isinstance(expected, list):
            assert len(actual) == len(expected), (len(actual), len(expected))
            for a, e in zip(actual, expected):
                equal(a, e)
        elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
            assert actual is not None and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-14), (actual, expected)
        else:
            assert actual == expected, (actual, expected)

    queries = snapshot['queries']
    rows = lambda key: queries[key]['rows']
    digest = hashlib.sha256(report_path.read_bytes()).hexdigest()
    equal(snapshot['id'], 'report:fb92123c-4dba-45f3-99d2-b0be42d4922a')
    equal(snapshot['buildStatus'], 'complete')
    equal(snapshot['report']['sourceReportSha256'], digest)
    equal(snapshot['report']['coverage'], report['coverage'])
    equal(snapshot['report']['vectorRmse'], report['primary_pooled']['vector_rmse_m_s'])
    equal(snapshot['report']['vectorNrmsePct'], report['primary_pooled']['vector_nrmse_reference_rms'] * 100)
    for key, source in [('flow', 'section_series'), ('profiles', 'profile_series')]:
        equal(rows(key), report[source])
    for row in rows('flow'):
        for named, si in [('net_ml_s', 'net_m3_s'), ('forward_ml_s', 'forward_m3_s'), ('reverse_ml_s', 'reverse_m3_s'), ('sampled_area_mm2', 'sampled_area_m2')]:
            equal(row[named], row[si] * 1e6)
    equal(len(rows('sections')), len(report['section_summary']))
    for row, source in zip(rows('sections'), report['section_summary']):
        expected = dict(section=source['section'], x_mm=source['x_m'] * 1000,
            paired_pixels=source['paired_pixels'], flow_rmse_ml_s=source['flow_error']['rmse'] * 1e6,
            correlation=source['flow_error']['pearson_r'], CFD_cycle_ml=source['CFD_cycle']['net_m3'] * 1e6,
            Koma_cycle_ml=source['Koma_cycle']['net_m3'] * 1e6,
            cycle_error_pct=(source['net_cycle_volume_ratio'] - 1) * 100,
            peak_error_pct=(source['peak_absolute_flow_ratio'] - 1) * 100)
        equal(row, expected)
    equal(rows('regions'), [dict(x, energy_scaled=x['energy_per_density_m5_s2'] * 1e6) for x in report['regional_series']])
    whole = [x for x in report['regional_series'] if x['region'] == 'whole']
    equal(snapshot['report']['energyRatio'],
        sum(x['energy_per_density_m5_s2'] for x in whole if x['method'] == 'Koma') /
        sum(x['energy_per_density_m5_s2'] for x in whole if x['method'] == 'CFD'))
    expected_velocity = []
    for component in ['X', 'Y', 'Z', 'speed']:
        source = report['primary_pooled'][component]
        expected_velocity.append(dict(quantity=component, bias=source['bias'], mae=source['mae'],
            rmse=source['rmse'], nrmse_pct=source['nrmse_reference_rms'] * 100))
    expected_velocity.append(dict(quantity='vector', bias=None, mae=None,
        rmse=report['primary_pooled']['vector_rmse_m_s'],
        nrmse_pct=report['primary_pooled']['vector_nrmse_reference_rms'] * 100))
    equal(rows('velocity'), expected_velocity)
    metrics = dict(vorticity_magnitude_s_inv='Vorticity magnitude (1/s)',
        helicity_m_s_inv2='Helicity density (m/s²)', divergence_s_inv='Divergence (1/s)',
        strain_frobenius_s_inv='Strain norm (1/s)', dissipation_per_viscosity_s_inv2='Dissipation / viscosity (1/s²)')
    modes = dict(primary='Common interior', eroded='Extra-eroded interior', matched_4mm_smoothing='Matched 4 mm smoothing')
    derivative_rows, pooled = [], []
    for mode, source in report['derivatives'].items():
        for metric, values in source['pooled'].items():
            pooled.append(dict(mode=modes[mode], metric=metrics[metric], rmse=values['rmse'],
                nrmse_pct=values['nrmse_reference_rms'] * 100, stencil_voxels=source['stencil_voxels_per_bin']))
        for values in source['per_bin']:
            derivative_rows.append(dict(bin=values['bin'], mode=modes[mode], metric=metrics[values['metric']],
                rmse=values['rmse'], nrmse_pct=values['nrmse_reference_rms'] * 100))
    equal(rows('derivatives'), derivative_rows)
    equal(rows('derivative_summary'), pooled)
    labels = dict(adc_time_average='Actual ADC times', nominal_midpoint='Nominal midpoint',
        box3='Nominal 3-time box', quadrature3='3³ spatial samples', legacy_half_voxel='Legacy half-voxel grid')
    equal(rows('sensitivities'), [dict(bin=x['bin'], reference=labels[x['reference']],
        rmse=x['vector_rmse_m_s'], nrmse_pct=x['vector_nrmse_reference_rms'] * 100) for x in report['sensitivity_series']])
    equal(rows('masks'), [dict(mask=x['mask'], voxels=x['voxels'], rmse=x['pooled']['vector_rmse_m_s'],
        nrmse_pct=x['pooled']['vector_nrmse_reference_rms'] * 100) for x in report['mask_sensitivity']])
    equal(rows('cores'), [dict(section=x['section'], bin=x['bin'], threshold=str(x['threshold_fraction']),
        direction=str(x['direction_sign']), method=x['method'],
        radius_mm=None if x['rms_radius_m'] is None else x['rms_radius_m'] * 1000,
        angle_deg=x['angle_to_normal_deg'], flux_ml_s=x['forward_flux_m3_s'] * 1e6,
        centroid_Y_mm=None if x['centroid_m'][1] is None else x['centroid_m'][1] * 1000,
        centroid_Z_mm=None if x['centroid_m'][2] is None else x['centroid_m'][2] * 1000) for x in report['flow_core_series']])
    equal(rows('timing'), [dict(encoding=encoding, bin=x['bin'], profiles=x['profiles'],
        adc_phase_circular_mean_ms=x['phase_circular_mean_s'] * 1000,
        nominal_center_ms=x['nominal_midpoint_s'] * 1000,
        offset_ms=(x['phase_circular_mean_s'] - x['nominal_midpoint_s']) * 1000,
        seam_crossing_profiles=x['seam_crossing_profiles'])
        for encoding, source in report['timing_audit'].items() for x in source['bin_timing']])
    equal(snapshot['report']['timingOffsetMs'], rows('timing')[0]['offset_ms'])
    equal(rows('gates'), [dict(parameter=k, reason=v) for k, v in report['gated'].items()])
    equal(len(queries), 12)
    equal(bool(re.search(r'[A-Za-z]:[\\/]|(?:Bearer\s|api[_-]?key\s*[:=])', snapshot_text)), False)
    for query in queries.values():
        equal(query['source']['files'], [snapshot['report']['sourceReport']])
        equal(digest in query['source']['evidenceFlow'][0]['detail'], True)
    output = dict(status='PASS', checks=checks, failed_checks=0,
        created_utc=datetime.now(timezone.utc).isoformat(),
        report_sha256=digest, snapshot_sha256=hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
        query_rows={k: len(v['rows']) for k, v in queries.items()},
        scope='Every reviewed row, units, null retention, headline metrics, provenance and completion state; not physical validation')
    destination = args.root / 'validation/report_snapshot_validation.json'
    with destination.open('x', encoding='utf-8') as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
    print(json.dumps(output))


if __name__ == '__main__':
    main()
