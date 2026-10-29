"""Fail-closed independent repeated-RF predictor gate before full E056 scans."""
from pathlib import Path
from datetime import datetime,timezone
import json
import h5py
import numpy as np
from analyze_physics import load_group,predict,complex_array,sha,write_json

run=Path('D:/MRSim/outputs/acquisition_physics_e056')
target=run/'prefix_v1/independent_gate.json'
if target.exists():raise FileExistsError(target)
with h5py.File(run/'inputs_v1/controls.h5') as f:
    positions=f['positions'][:];rho=f['rho'][:];u=f['u_parabolic_osc'][:]
    t1=float(f['T1'][0]);t2=float(f['T2'][0]);times=f['path_time_s'][:]
    factor=f['path_basis_s'][:];period=float(f['motion_period_s'][()])
rows=[]
with h5py.File(run/'sequences_v1/waveforms.h5') as f:
    for encoding in ['REF','X']:
        metadata=load_group(f['baseline'][encoding],80,16,32)
        metadata['gamma']=float(f['gamma_Hz_T'][()])
        for name in ['labels','compensation','profile_rf_index0','adc_first_s','dwell_s']:
            metadata[name]=metadata[name][:15]
        last=int(metadata['profile_rf_index0'][-1])+1
        for name in ['rf_start_s','rf_scale','rf_phase_rad']:metadata[name]=metadata[name][:last]
        raw,_=predict(metadata,positions,rho,u,t1,t2,times,factor,period,(80,16,32,15),require_complete=False)
        for step in ['production','fine']:
            path=run/'prefix_v1'/('parabolic_osc_baseline_'+encoding.lower()+'_'+step+'.h5')
            with h5py.File(path) as result:actual=complex_array(result['raw_signal_f64']).ravel()
            relative=float(np.linalg.norm(actual-raw)/np.linalg.norm(raw))
            rows.append(dict(encoding=encoding,RF_step=step,relative_complex_L2=relative,passed=relative<=.01))
            print(encoding,step,relative,flush=True)
required=[r for r in rows if r['RF_step']=='fine']
receipt=dict(experiment='E-056',utc=datetime.now(timezone.utc).isoformat(),
    status='PASS_INDEPENDENT_REPEATED_RF_PREFIX' if all(r['passed'] for r in required) else 'FAIL_INDEPENDENT_REPEATED_RF_PREFIX',
    rows=rows,new_1m=False,prefix_report_sha256=sha(run/'prefix_v1/report.toml'))
write_json(target,receipt)
if receipt['status']=='PASS_INDEPENDENT_REPEATED_RF_PREFIX':
    (run/'prefix_v1/independent_gate.toml').write_text('status = "PASS_INDEPENDENT_REPEATED_RF_PREFIX"\n',encoding='utf-8')
print(json.dumps(receipt),flush=True)
if not all(r['passed'] for r in required):raise SystemExit('Independent fine-RF prefix gate failed; do not run full acquisition')
