"""E047 regional companion and compact source-backed handoff; no input mutation."""
import csv
import json
from pathlib import Path
import tomllib
import h5py
import numpy as np
from scripts.analyze_remapped_confirmation import weighted_metrics
from scripts.build_aorta_inlet_picker import sha

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/remapped_confirmation_e047'


def summarize():
    directory=BASE/'regional_v1'
    if directory.exists():raise FileExistsError(directory)
    directory.mkdir();allrows=[];summary={};sources=[]
    for case in ('aorta','phantom'):
        analysis=BASE/f'{case}_analysis_v1';report=json.loads((analysis/'report.json').read_text())
        masks=np.load(analysis/'masks.npz',allow_pickle=False)
        with h5py.File(analysis/'paired_fields.h5') as f:
            ref=f['reference'][:];fraction=f['numerical_volume_fraction'][:]
            fields={arm:f[arm][:] for arm in ('legacy100k','map20k','map100k')}
        if case=='phantom':
            from scripts.compare_dp150_hemodynamics import REGIONS
            source=ROOT/'outputs/dp150_hemodynamic_comparison/e025_v1/results_v2/matched_reference.h5'
            with h5py.File(source) as f:x=f['source_z_m'][:]
            regions=REGIONS;definition='Original CFD source z intervals in metres, identical to E025'
        else:
            from scripts.aorta_hemodynamic_reference import AortaGrid
            x=AortaGrid().axes()[0]
            regions=dict(whole=(-np.inf,np.inf),readout_below_minus45=(-np.inf,-.045),
                         readout_middle=(-.045,.045),readout_above45=(.045,np.inf))
            definition='MRI readout intervals, not anatomical branch identities'
        region_result=[]
        for region,(low,high) in regions.items():
            eligible=np.broadcast_to(((x>=low)&(x<high))[None,None,:],fraction.shape)&(fraction>0)
            denominator=float(fraction[eligible].sum())
            for arm,b in fields.items():
                retained=eligible&masks[arm]
                metrics=weighted_metrics(ref,b,fraction,retained)
                # Check the region result with a separate direct weighted vector calculation.
                if retained.any():
                    squares=np.sum((ref[:,retained].astype(float)-b[:,retained].astype(float))**2,axis=-1)
                    expected=np.sqrt(np.sum(squares*fraction[retained])/len(ref)/fraction[retained].sum())
                    if not np.isclose(expected,metrics['vector_rmse_m_s'],rtol=1e-6):raise AssertionError('Regional arithmetic')
                row=dict(case=case,region=region,arm=arm,voxels=int(retained.sum()),
                         own_coverage_fraction=None if denominator==0 else float(fraction[retained].sum()/denominator),
                         vector_rmse_m_s=None if metrics is None else metrics['vector_rmse_m_s'],
                         speed_rmse_m_s=None if metrics is None else metrics['speed_rmse_m_s'])
                allrows.append(row);region_result.append(row)
        summary[case]=dict(region_definition=definition,regions=region_result,
                           primary=report['thresholds'][0],mapping=report['mapping'])
        sources.extend([analysis/'report.json',analysis/'paired_fields.h5',analysis/'masks.npz'])
    with (directory/'regional_errors_coverage.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(allrows[0]));writer.writeheader();writer.writerows(allrows)
    for case in summary:
        report=json.loads((BASE/f'{case}_analysis_v1/report.json').read_text())
        rows=[]
        for threshold in report['thresholds']:
            for arm in ('legacy100k','map20k','map100k'):
                f=threshold['flow']['own'][arm];v=threshold['volume']['own'][arm]
                rows.append(dict(threshold=threshold['signal_threshold'],arm=arm,
                    section_coverage_fraction=f['coverage_fraction'],volume_coverage_fraction=v['coverage_fraction'],
                    flow_rmse_ml_s=None if f['metrics'] is None else f['metrics']['rmse_ml_s'],
                    vector_rmse_m_s=None if v['metrics'] is None else v['metrics']['vector_rmse_m_s']))
        with (directory/f'{case}_sensitivity.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    output=dict(experiment='E-047',status='COMPLETE_REGIONAL_COMPANION',cases=summary,
                input_sha256={str(p):sha(p) for p in sources},script_sha256=sha(__file__),
                limitations='Each region reports its own retained support. Errors across different coverage are not a matched-area improvement claim.')
    (directory/'regional_report.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=output['status'],regional_rows=len(allrows))))


if __name__=='__main__':summarize()
