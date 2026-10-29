"""Software QA only: reproduce existing 100k numbers; not a 300k result."""
import ast
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import tomllib
import h5py
import numpy as np
from analyze_300k import scaleup_decision
from scripts.analyze_phantom_precision import load_arm,region_rows
from scripts.analyze_remapped_confirmation import fixed_mask,velocities

repo=Path('D:/MRSim/MRSIM/MRSimTracks')
run=Path('D:/MRSim/outputs/phantom_density_e055')
target=run/'analysis_software_precheck.json'
if target.exists():raise FileExistsError(target)
checks=0
for source in Path(__file__).parent.glob('*.py'):
    ast.parse(source.read_text(encoding='utf-8'));checks+=1
for cov,local,flows,expected in [
    ([.2,.5],[.3,.2],[dict(no_worse_RMSE=True)],True),
    ([.2,.5],[.3,.4],[dict(no_worse_RMSE=True)],False),
    ([.2,.2],[.3,.2],[dict(no_worse_RMSE=True)],False),
    ([.2,.5],[.3,.2],[dict(no_worse_RMSE=False)],False),
    ([.2,.5],[None,None],[dict(no_worse_RMSE=True)],False),
    ([None,None],[.3,.2],[dict(no_worse_RMSE=True)],False),
]:
    decision=scaleup_decision(cov,local,flows)
    assert decision['single_comparison_numeric_gates_pass']==expected
    assert decision['new_1m_started'] is False and decision['status'].startswith('HOLD_1M_')
    checks+=2
base=repo/'outputs/phantom_precision_e050'
folder=base/'mapped_f64_v1'
report=tomllib.loads((folder/'control_report.toml').read_text())
images,audits=load_arm(folder,report,candidate=True)
field=velocities(images,5.0)
with h5py.File(repo/'outputs/phantom_diagnosis_e048/reference_v1/reference_fields.h5') as f:
    reference=f['fft_adc'][:];fraction=f['fraction_fft'][:]
with h5py.File(repo/'outputs/phantom_diagnosis_e048/local_v1/local_fields.h5') as f:source_z=f['source_z_fft_m'][:]
previous=json.loads((base/'analysis_v1/report.json').read_text())
rows=[]
for threshold in [.05,.10,.15,.20]:
    mask=fixed_mask(images,threshold)
    rows+=region_rows(reference,{'candidate_f64':field},{'candidate_f64':mask},fraction,source_z,threshold)
for row in rows:
    if row['support']!='own':continue
    original=next(r for r in previous['regions'] if all(r[k]==row[k] for k in ['region','arm','support','threshold']))
    np.testing.assert_allclose(row['coverage_fraction'],original['coverage_fraction'],atol=1e-12,rtol=0);checks+=1
    if row['metrics'] is None:assert original['metrics'] is None;checks+=1;continue
    for key in ['vector_rmse_m_s','mean_speed_bias_m_s','speed_rmse_m_s']:
        np.testing.assert_allclose(row['metrics'][key],original['metrics'][key],atol=1e-12,rtol=0);checks+=1
receipt=dict(status='PASS_SOFTWARE_BASELINE_REPRODUCTION_NOT_300K_RESULT',checks=checks,
    utc=datetime.now(timezone.utc).isoformat(),audits=audits,
    baseline_report_sha256=hashlib.sha256((base/'analysis_v1/report.json').read_bytes()).hexdigest(),
    one_million_started=False)
target.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt),flush=True)
