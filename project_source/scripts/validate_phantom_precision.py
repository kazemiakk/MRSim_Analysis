"""Independent E050 primary-metric, provenance and coverage checks."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from scripts.analyze_phantom_precision import ARMS, BASE, REF, ROOT
from scripts.build_aorta_inlet_picker import sha


def validate(folder):
    target = folder/'validation.json'
    if target.exists():
        raise FileExistsError(target)
    checks = []
    def require(value, name):
        if not value:
            raise AssertionError(name)
        checks.append(name)
    manifest = json.loads((folder/'MANIFEST.json').read_text())['outputs']
    for name, digest in manifest.items():
        require(sha(folder/name) == digest, 'output hash '+name)
    registration = json.loads((folder/'registration.json').read_text())['inputs']
    for name, digest in registration.items():
        require(sha(Path(name)) == digest, 'input hash '+name)
    report = json.loads((folder/'report.json').read_text())
    require(report['status'] == 'COMPLETE_PRECISION_COMPARISON_NOT_ACCURACY_PROMOTION', 'bounded evidence class')
    with h5py.File(REF/'reference_fields.h5') as f:
        ref, fraction = f['fft_adc'][:], f['fraction_fft'][:]
    with h5py.File(folder/'paired_fields.h5') as f:
        fields = {a:f[a][:] for a in ARMS}
        masks = {a:f[a+'_fixed5_mask'][:].astype(bool) for a in ARMS}
        pair = f['paired_fixed5_mask'][:].astype(bool)
    require(np.array_equal(pair, masks[ARMS[0]]&masks[ARMS[1]]), 'paired mask intersection')
    for row in report['primary']:
        region = np.ones(fraction.shape, bool)
        if row['region'] == 'throat':
            from scripts.analyze_phantom_precision import LOCAL
            with h5py.File(LOCAL/'local_fields.h5') as f:
                z = f['source_z_fft_m'][:]
            region &= ((z >= -.01)&(z < .01))[None, None, :]
        mask = masks[row['arm']] if row['support'] == 'own' else pair
        eligible = region&(fraction > 0)
        keep = mask&eligible
        weight = fraction[keep].astype(float)
        coverage = weight.sum()/fraction[eligible].sum()
        require(np.isclose(coverage, row['coverage_fraction'], atol=1e-10, rtol=1e-7), 'primary coverage '+str(row))
        if not keep.any():
            require(row['metrics'] is None, 'empty metric unavailable')
            continue
        a, b = ref[:, keep].astype(float), fields[row['arm']][:, keep].astype(float)
        require(np.isfinite(a).all() and np.isfinite(b).all(), 'finite paired field')
        independent = np.sqrt(np.sum((b-a)**2*weight[None, :, None])/(len(a)*weight.sum()))
        require(np.isclose(independent, row['metrics']['vector_rmse_m_s'], atol=1e-10, rtol=1e-7), 'independent vector RMSE '+row['arm']+row['region']+row['support'])
    for threshold in (.05, .10, .15, .20):
        for region in ('whole', 'upstream', 'throat', 'near_downstream', 'mid_downstream', 'far_downstream'):
            rows = [r for r in report['regions'] if r['threshold'] == threshold and r['region'] == region]
            require(len(rows) == 4, 'complete regional coverage/error rows')
            shared = [r for r in rows if r['support'] == 'paired']
            require(shared[0]['coverage_fraction'] == shared[1]['coverage_fraction'], 'same paired coverage')
    for row in report['sections']:
        require(row['coverage_fraction'] is None or 0 <= row['coverage_fraction'] <= 1+1e-12, 'section coverage range')
        if row['retained_area_mm2'] == 0:
            require(row['metrics'] is None and row['CFD_ml_s'] is None and row['Koma_ml_s'] is None, 'empty section is not zero flow')
        else:
            a, b = np.array(row['CFD_ml_s']), np.array(row['Koma_ml_s'])
            require(np.isclose(np.sqrt(np.mean((b-a)**2)), row['metrics']['rmse_ml_s'], atol=1e-10), 'independent partial flow RMSE')
    value = dict(status='PASS_INDEPENDENT_E050_CHECKS_NOT_PHYSICAL_VALIDATION', checks=len(checks),
                 report_sha256=sha(folder/'report.json'), check_names=checks)
    target.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:value[k] for k in ('status', 'checks', 'report_sha256')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    validate(parser.parse_args().analysis)
