"""Build a bounded, source-bound E-041 report snapshot; never alter E-040."""
from pathlib import Path
import csv
import hashlib
import json
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'outputs/hemodynamic_extension/e040_concise_png_v3'
TARGET = ROOT / 'outputs/hemodynamic_extension/e041_review_report_v1'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(refresh=False):
    if TARGET.exists() and not refresh:
        raise FileExistsError(TARGET)
    manifest = json.loads((SOURCE / 'MANIFEST.json').read_text())
    for name, meta in manifest.items():
        if digest(SOURCE / name) != meta['sha256']:
            raise ValueError('Changed E-040 source: ' + name)
    def read(name):
        with (SOURCE / name).open(encoding='utf-8', newline='') as stream:
            result = list(csv.DictReader(stream))
        for row in result:
            for k, value in row.items():
                if value == '': row[k] = None
                elif value in ('True', 'False'): row[k] = value == 'True'
                else:
                    try: row[k] = float(value)
                    except (ValueError, TypeError): pass
        return result
    regional = read('regional_speeds.csv')
    queries = {}
    for key, filename in [('regional','regional_speeds.csv'),('longitudinal','longitudinal_profiles.csv'),('choices','analysis_choices.csv'),('derivatives','derivative_sensitivity.csv'),('cores','core_errors.csv'),('core_width','core_width.csv')]:
        queries[key] = dict(rows=read(filename), source=dict(label='Original matched CFD–Koma comparison: ' + filename,
            files=[dict(path='e040_concise_png_v3/' + filename, sha256=manifest[filename]['sha256'])],
            evidenceFlow=[dict(title='Saved-data comparison', detail='Original matched phantom 1M, aortic baseline 1M and corrected aortic 100k; E-040 source-bound tables, unchanged. Review date 2026-10-22; simulation date is not inferred.')],
            caveats=['Internal consistency, not independent validation.','Aortic masks and regions describe numerical support, not certified physical lumen or anatomical branches.','Different acquisition versions and supports are not a controlled particle-count experiment.']))
    summary = []
    for case in ['phantom_1m','aorta_1m','aorta_100k_corrected']:
        b = 4 if case == 'phantom_1m' else 8
        pair = {r['method']: r for r in regional if r['case'] == case and r['region'] == 'whole' and r['bin'] == b}
        summary.append(dict(case=case,bin=b,mean_cfd=pair['CFD']['mean'],mean_koma=pair['Koma']['mean'],
            mean_bias_pct=100*(pair['Koma']['mean']/pair['CFD']['mean']-1),p99_cfd=pair['CFD']['p99'],p99_koma=pair['Koma']['p99'],
            p99_bias_pct=100*(pair['Koma']['p99']/pair['CFD']['p99']-1)))
    queries['peak_summary'] = dict(rows=summary, source=dict(label='Matched whole-region anchor-bin speed comparison',
        files=[dict(path='e040_concise_png_v3/regional_speeds.csv',sha256=manifest['regional_speeds.csv']['sha256'])],
        metricDefinitions=[dict(label='Relative bias',definition='100 × (Koma / CFD − 1), using matched whole-region weighted mean or weighted p99 in the same anchor bin.',componentIds=['peak-summary','aortic-finding'])],
        evidenceFlow=[dict(title='Select paired anchor bins',detail='Phantom upstream CFD peak-flow bin4; aortic anonymous X+000_C1 peak-flow bin8. No fitted alignment. Means use numerical voxel-volume weights; p99 uses the inverse weighted empirical CDF.')]))
    queries['figures'] = dict(rows=[dict(file=n,sha256=m['sha256'],bytes=m['bytes']) for n,m in manifest.items() if n.endswith('.png')],
        source=dict(label='Unchanged E-040 PNG figures',files=[dict(path='e040_concise_png_v3/MANIFEST.json',sha256=digest(SOURCE/'MANIFEST.json'))],
            caveats=['Native PNGs retained without cropping or replotting.','Transverse phantom profiles are central lines; aortic profiles are partial-section strip means.','Longitudinal profiles are readout-axis aggregates, not aortic anatomical centerlines.']))
    from scripts.complete_advanced_comparisons import load_inputs
    from scripts.extend_hemodynamic_comparisons import DP
    import h5py
    import numpy as np
    _, cases, *_ = load_inputs()
    case = cases[0]; b = 3; i = 116
    roi = case['regions']['whole'][:,:,i]
    speed = np.linalg.norm(case['estimate'][b,:,:,i],axis=-1)
    z,y = np.where(roi); order=np.argsort(speed[z,y])[-2:]
    with h5py.File(DP/'matched_reference.h5') as f: direct=f['direct_common_mask'][:]
    outliers=[dict(bin=4,slice_index=i,z_index=int(z[n]),y_index=int(y[n]),
        koma_speed=float(speed[z[n],y[n]]),cfd_speed=float(np.linalg.norm(case['reference'][b,z[n],y[n],i])),
        koma_x=float(case['estimate'][b,z[n],y[n],i,0]),direct_supported=bool(direct[z[n],y[n],i])) for n in order[::-1]]
    queries['outliers'] = dict(rows=outliers,source=dict(label='Phantom local velocity spot check',
        files=[dict(path='E-025 matched_reference.h5: direct_common_mask'),dict(path='E-022 support-constrained phase HDF5: velocity_corrected_m_per_s')],
        evidenceFlow=[dict(title='Inspect the local p99 spike',detail='Bin4, readout index116 (MRI X=-104.375 mm), whole common support. Two largest Koma speed voxels selected; Euclidean component norm, compared to the same CFD voxels. Direct-support status is inherited from the fixed all-bin mask, not a guarantee of local signal accuracy.')],
        caveats=['Selected outliers, not a prevalence estimate.','Phase wrapping/weak-signal explanations remain untested hypotheses.']))
    snapshot = dict(surface='report', title='Mean speeds agree more closely than local peaks and gradients',
        generatedAt=datetime.now(timezone.utc).isoformat(),status='reviewed',buildStatus='creating',filters=[],
        report=dict(asOf='2026-10-22'),queries=queries)
    TARGET.mkdir(parents=True,exist_ok=refresh)
    (TARGET/'reviewed_snapshot.json').write_text(json.dumps(snapshot,ensure_ascii=False,indent=2),encoding='utf-8')
    app_data=TARGET/'report_app/src/data.json'
    if refresh and app_data.exists():
        existing=json.loads(app_data.read_text(encoding='utf-8'))
        snapshot['id']=existing['id']
        if 'theme' in existing: snapshot['theme']=existing['theme']
        app_data.write_text(json.dumps(snapshot,ensure_ascii=False,indent=2),encoding='utf-8')
    (TARGET/'source_manifest.json').write_text(json.dumps(dict(experiment='E-041',source_manifest_sha256=digest(SOURCE/'MANIFEST.json'),
        source_png_count=24,source_files=manifest,source_scope='Original matched sources only; no other phantom archive; no new simulation'),indent=2),encoding='utf-8')
    print(TARGET/'reviewed_snapshot.json')


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--refresh-snapshot',action='store_true')
    main(parser.parse_args().refresh_snapshot)
