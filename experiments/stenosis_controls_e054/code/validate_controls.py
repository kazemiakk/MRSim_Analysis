"""Reproduce E054 saved-signal metrics and actual path/weight contracts."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import tomllib
import h5py
import numpy as np


def digest(path):
    with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def complex_value(dataset):
    a=dataset[:]
    if np.iscomplexobj(a):return a
    return a['r']+1j*a['i']


def main(run):
    report_path=run/'analysis_v1/report.json'
    report=json.loads(report_path.read_text());registration=json.loads((run/'inputs_v1/registration.json').read_text())
    runtime=tomllib.loads((run/'cpu_v1/report.toml').read_text());checks=0
    assert digest(run/'cpu_v1/signals.h5')==runtime['signals_sha256'];checks+=1
    assert digest(run/'inputs_v1/controls.h5')==runtime['input_sha256']==registration['input_sha256'];checks+=1
    with h5py.File(run/'inputs_v1/controls.h5') as inputs,h5py.File(run/'cpu_v1/signals.h5') as signals:
        times=inputs['time_s'][:];adc=inputs['adc_time_s'][:];center=len(adc)//2;tau=times-adc[center]
        for model in registration['models']:
            name=model['name'];positions=np.stack([inputs[name+'/'+c][:].T for c in 'xyz'],axis=-1)
            weights=inputs[name+'/weights'][:]
            np.testing.assert_allclose(weights.sum(),1,atol=1e-12,rtol=0);checks+=1
            np.testing.assert_allclose(np.ptp(positions[:,:,1:],axis=1),0,atol=1e-15,rtol=0);checks+=1
            initial=np.stack([np.interp(adc[center],times,p[:,0]) for p in positions])
            # Quadratic paths are represented by a fine piecewise-linear grid;
            # solve x0 from the exact registered expression, not an echo interp.
            v=model['velocity_base_m_s']+model['shear_s_1']*positions[:,0,1]
            x0=positions[:,0,0]-v*tau[0]-.5*model['acceleration_m_s2']*tau[0]**2
            np.testing.assert_allclose(positions[:,:,0],x0[:,None]+v[:,None]*tau+.5*model['acceleration_m_s2']*tau**2,atol=1e-14,rtol=0);checks+=1
            values={}
            for mode in ['fine','production','late_production']:
                s={e:complex_value(signals[name+'_'+mode+'/'+e]) for e in ['REF','X','Y','Z']}
                assert all(x.shape==(400,) and np.isfinite(x).all() for x in s.values());checks+=4
                velocity=np.array([np.angle(s[e][center]*s['REF'][center].conjugate())*5/np.pi for e in ['X','Y','Z']])
                row=next(r for r in report['rows'] if r['model']==name and r['mode']==mode)
                np.testing.assert_allclose(velocity,row['Koma_encoded_m_s'],atol=1e-12,rtol=0);checks+=1
                error=float(np.max(abs(velocity-np.array(row['expected_encoded_m_s']))))
                np.testing.assert_allclose(error,row['max_encoded_component_error_m_s'],atol=1e-12,rtol=0);checks+=1
                assert error<=(.002 if mode=='fine' else .005);checks+=1
                support=min(abs(x[center])/abs(x).max() for x in s.values())
                np.testing.assert_allclose(support,row['central_minimum_relative_signal'],atol=1e-12,rtol=0);assert support>.05;checks+=2
                values[mode]=velocity
            assert np.max(abs(values['late_production']-values['production']))<=.002;checks+=1
    assert len(report['gates'])==24 and all(g['passed'] for g in report['gates']);checks+=1
    assert report['status']=='PASS_SMALL_ACCELERATION_SHEAR_CONTROLS';checks+=1
    receipt=dict(status='PASS_INDEPENDENT_SAVED_SIGNAL_AND_PATH_CHECKS',experiment='E-054',checks=checks,
                 utc=datetime.now(timezone.utc).isoformat(),report_sha256=digest(report_path),
                 independent_finite_RF_reference_recomputed_by='analysis_v1; separate validator reproduces stored signal metrics and prescribed geometry',
                 new_300k_started=False,new_1m_started=False)
    target=run/'validation_v1.json'
    if target.exists():raise FileExistsError(target)
    target.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);main(p.parse_args().run)
