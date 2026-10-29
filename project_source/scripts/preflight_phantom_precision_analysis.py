"""Reproduce E048's saved100k centered-grid baseline before interpreting E050."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from scripts.analyze_phantom_precision import (OLD, REF, LOCAL, INPUTS, ROOT, ENC, toml,
    load_arm, fixed_mask, velocities, region_rows, dump, bind_reference)
from scripts.build_aorta_inlet_picker import sha


def preflight(output):
    if output.exists():
        raise FileExistsError(output)
    source = bind_reference(REF, 'reference_fields.h5', 'f4c4ec3153d30a2a9dfd360d8ff1c21811e14e6f942697cdd990b669fec75694')
    local = bind_reference(LOCAL, 'local_fields.h5', '6c57a8708ea4a5aff09f61db232b0ba1528ae4d000bf60d9ca00524516a27885')
    expected = json.loads((LOCAL/'report.json').read_text())
    images, audits = load_arm(OLD, toml(OLD/'control_report.toml'))
    field = velocities(images, toml(INPUTS/'case_info.toml')['Venc_m_s'])
    mask = fixed_mask(images, .05)
    with h5py.File(source) as f:
        reference, fraction = f['fft_adc'][:], f['fraction_fft'][:]
    with h5py.File(local) as f:
        z = f['source_z_fft_m'][:]
    rows = region_rows(reference, {'saved_f32':field}, {'saved_f32':mask}, fraction, z, .05)
    checks = 0
    for row in rows:
        if row['support'] != 'own':
            continue
        match = next(r for r in expected['regions'] if r['region'] == row['region'] and r['arm'] == 'map100k')
        if not np.isclose(row['coverage_fraction'], match['own_coverage_fraction'], rtol=1e-6, atol=1e-9):
            raise AssertionError('Saved coverage reproduction '+row['region'])
        if not np.isclose(row['metrics']['vector_rmse_m_s'], match['own_metrics']['vector_rmse_m_s'], rtol=1e-6, atol=1e-9):
            raise AssertionError('Saved regional velocity reproduction '+row['region'])
        checks += 2
    primary = next(r for r in rows if r['region'] == 'whole' and r['support'] == 'own')
    keep = mask&(fraction > 0)
    weight = fraction[keep].astype(float)
    direct = np.sqrt(np.sum((field[:, keep].astype(float)-reference[:, keep].astype(float))**2*weight[None, :, None])/(15*weight.sum()))
    if not np.isclose(direct, primary['metrics']['vector_rmse_m_s'], atol=1e-12):
        raise AssertionError('Independent baseline RMSE')
    paths = [Path(__file__).resolve(), ROOT/'scripts/analyze_phantom_precision.py', source, local, LOCAL/'report.json', OLD/'control_report.toml']
    paths += [OLD/(e.lower()+'_result.h5') for e in ENC]
    report = dict(status='PASS_SAVED_BASELINE_REPRODUCTION_NOT_NEW_RESULT', checks=checks+1,
        primary=primary, regions=rows, reconstruction_audits=audits,
        sources={str(p):sha(p) for p in paths}, limitation='Reproduces previous same-source results; no new candidate or accuracy claim.')
    output.parent.mkdir(parents=True, exist_ok=True)
    dump(output, report)
    print(json.dumps(dict(status=report['status'], checks=report['checks'], primary=primary)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    preflight(parser.parse_args().output)
