"""Independent finite-RF, volume-weighted complex-signal checks for E-054."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tomllib

import h5py
import numpy as np
from scripts.analyze_phantom_encoding import independent_bloch
from scripts.diagnose_phantom_signal import read_complex
from scripts.prepare_phantom_encoding import shot_contract, gradient


def sha(path):
    with open(path,'rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def main(inputs,cpu,output):
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    registration=json.loads((inputs/'registration.json').read_text())
    runtime=tomllib.loads((cpu/'report.toml').read_text())
    assert runtime['status']=='COMPLETE_CPU_MULTI_SPIN_SIGNALS' and runtime['cpu_only']
    assert runtime['input_sha256']==registration['input_sha256']==sha(inputs/'controls.h5')
    assert runtime['signals_sha256']==sha(cpu/'signals.h5')
    for p,digest in registration['source_sha256'].items():assert sha(Path(p))==digest,p
    models=registration['models'];names=[m['name'] for m in models]
    signal={};independent={};wave_checks=[]
    with h5py.File(inputs/'controls.h5') as f,h5py.File(cpu/'signals.h5') as raw:
        time=f['time_s'][:];adc=f['adc_time_s'][:]
        positions=np.concatenate([np.stack([f[name+'/'+c][:].T for c in 'xyz'],axis=-1) for name in names])
        t1=np.concatenate([f[name+'/T1'][:] for name in names]);t2=np.concatenate([f[name+'/T2'][:] for name in names])
        weights=np.stack([f[name+'/weights'][:] for name in names])
        for encoding in ['REF','X','Y','Z']:
            wave=raw[encoding+'_waveform']
            g=np.column_stack([wave[c+'_T_m'][:] for c in ['gx','gy','gz']])
            b1=read_complex(wave['b1_T'])
            parsed=gradient(shot_contract(Path(registration['sequence_directory'])/(encoding.lower()+'.seq')),time)
            nonzero=abs(g)>1e-6;gamma=float(np.median(parsed[nonzero]/g[nonzero]))
            delta=float(np.max(abs(parsed/gamma-g)))
            assert 42e6<gamma<43e6 and delta<=1e-9
            assert np.max(abs(wave['rf_frequency_Hz'][:]))<1e-9 and np.max(abs(wave['rf_phase_rad'][:]))<1e-9
            pred=independent_bloch(time,positions,b1,g,adc,t1,t2,gamma)
            pred*=read_complex(wave['adc_phase_compensation'])[None,:]
            independent[encoding]=np.sum(pred.reshape(len(names),32,len(adc))*weights[:,:,None],axis=1)
            for mode in ['fine','production','late_production']:
                signal[encoding,mode]=np.stack([read_complex(raw[name+'_'+mode+'/'+encoding]) for name in names])
            wave_checks.append(dict(encoding=encoding,gamma_Hz_T=gamma,gradient_max_delta_T_m=delta))
            print('E054 independent finite-RF prediction '+encoding,flush=True)
    center=len(adc)//2
    vel=lambda values: np.stack([np.angle(values[e]*values['REF'].conj())*5/np.pi for e in ['X','Y','Z']],axis=-1)
    expected=vel(independent)
    velocities={mode:vel({e:signal[e,mode] for e in ['REF','X','Y','Z']}) for mode in ['fine','production','late_production']}
    rows=[];gates=[]
    for i,model in enumerate(models):
        for mode in ['fine','production','late_production']:
            errors=[float(np.linalg.norm(signal[e,mode][i]-independent[e][i])/np.linalg.norm(independent[e][i])) for e in ['REF','X','Y','Z']]
            minimum=min(float(abs(signal[e,mode][i,center])/np.max(abs(signal[e,mode][i]))) for e in ['REF','X','Y','Z'])
            discrepancy=float(np.max(abs(velocities[mode][i,center]-expected[i,center])))
            late_delta=float(np.max(abs(velocities['late_production'][i,center]-velocities['production'][i,center])))
            row=dict(model=model['name'],mode=mode,max_relative_signal_L2=max(errors),central_minimum_relative_signal=minimum,
                     max_encoded_component_error_m_s=discrepancy,late_early_component_change_m_s=late_delta,
                     echo_mean_m_s=model['instantaneous_echo_mean_m_s'],expected_encoded_m_s=expected[i,center].tolist(),
                     Koma_encoded_m_s=velocities[mode][i,center].tolist(),
                     echo_mean_vector_difference_m_s=float(np.linalg.norm(velocities[mode][i,center]-model['instantaneous_echo_mean_m_s'])))
            passed=minimum>.05 and discrepancy<=(.002 if mode=='fine' else .005)
            if mode=='fine':passed=passed and max(errors)<=.003
            if mode=='late_production':passed=passed and late_delta<=.002
            rows.append(row);gates.append(dict(model=model['name'],mode=mode,passed=bool(passed)))
    report=dict(experiment='E-054',created_utc=datetime.now(timezone.utc).isoformat(),
                status='PASS_SMALL_ACCELERATION_SHEAR_CONTROLS' if all(g['passed'] for g in gates) else 'FAIL_SMALL_CONTROL_GATE',
                rows=rows,gates=gates,waveform_checks=wave_checks,
                maximum_fine_signal_relative_L2=max(r['max_relative_signal_L2'] for r in rows if r['mode']=='fine'),
                maximum_fine_phase_component_error_m_s=max(r['max_encoded_component_error_m_s'] for r in rows if r['mode']=='fine'),
                maximum_production_phase_component_error_m_s=max(r['max_encoded_component_error_m_s'] for r in rows if r['mode']=='production'),
                maximum_late_early_component_change_m_s=max(r['late_early_component_change_m_s'] for r in rows),
                limitations=['One equilibrium RF shot, not repeated-RF magnitude or full 3D image formation.',
                             'Deterministic prescribed volume-preserving shear, not realistic inlet/wall/CFD tracking validation.',
                             'Independent Bloch algorithm shares exported RF waveforms as input; gradients are separately parsed.',
                             'No 300k or 1M acquisition was performed by this control.'])
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    with (output/'comparison.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    text='# E-054 acceleration/shear multi-spin controls\n\nStatus: **'+report['status']+'**.\n\n'
    text+='| Model | Fine error (m/s) | Production error (m/s) | Late/early change (m/s) |\n|---|---:|---:|---:|\n'
    for name in names:
        fine=next(r for r in rows if r['model']==name and r['mode']=='fine');prod=next(r for r in rows if r['model']==name and r['mode']=='production')
        text+=f"| {name} | {fine['max_encoded_component_error_m_s']:.8g} | {prod['max_encoded_component_error_m_s']:.8g} | {prod['late_early_component_change_m_s']:.8g} |\n"
    text+='\n## Limits\n\n'+'\n'.join('- '+s for s in report['limitations'])+'\n'
    (output/'README.md').write_text(text,encoding='utf-8')
    sources=[inputs/'registration.json',inputs/'controls.h5',cpu/'report.toml',cpu/'signals.h5',Path(__file__)]
    manifest=dict(inputs={str(p):sha(p) for p in sources},outputs={p.name:sha(p) for p in output.iterdir() if p.is_file()})
    (output/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k.startswith('maximum_') or k=='status'}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--cpu',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.inputs,a.cpu,a.output)
