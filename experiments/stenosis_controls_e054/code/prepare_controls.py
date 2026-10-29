"""Freeze E-054 prescribed multi-spin inputs without touching the source tree."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tomllib

import h5py
import numpy as np
from scripts.prepare_phantom_encoding import shot_contract


def sha(path):
    return hashlib.file_digest(open(path, 'rb'), 'sha256').hexdigest()


def main(repo, code, output):
    if not sys.dont_write_bytecode:
        raise RuntimeError('PYTHONDONTWRITEBYTECODE=1 required')
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    info = tomllib.loads((repo/'outputs/remapped_confirmation_e047/phantom_inputs_v1/case_info.toml').read_text())
    sequence_dir = Path(info['sequence_directory'])
    shots = {e: shot_contract(sequence_dir/(e.lower()+'.seq')) for e in ['REF','X','Y','Z']}
    adc = shots['REF']['adc_times']
    for shot in shots.values():
        np.testing.assert_allclose(shot['adc_times'], adc, atol=1e-9, rtol=0)
    time = np.linspace(0, shots['REF']['duration'], int(np.ceil(shots['REF']['duration']/0.25e-6))+1)
    tau = time - adc[len(adc)//2]
    xyz = np.array(np.meshgrid((np.arange(4)+0.5-2)*0.00125/4,
                              (np.arange(4)+0.5-2)*0.00125/4,
                              (np.arange(2)+0.5-1)*0.0025/2, indexing='ij')).reshape(3,-1).T
    models = [('static',0.,0.,0.),('positive',1.5,0.,0.),('negative',-.75,0.,0.),
              ('accel_positive',1.5,200.,0.),('accel_negative',1.5,-200.,0.),
              ('shear_positive',1.5,0.,1000.),('shear_negative',1.5,0.,-1000.),
              ('accel_shear',1.5,200.,1000.)]
    cases = []
    with h5py.File(output/'controls.h5.partial','w') as f:
        f['time_s'] = time
        f['adc_time_s'] = adc
        f['late_clock_s'] = 2047*info['period_s']
        for name,v,a,shear in models:
            group=f.create_group(name)
            velocity = v+shear*xyz[:,1]
            position=np.broadcast_to(xyz[:,None,:], (len(xyz),len(time),3)).copy()
            position[:,:,0] += velocity[:,None]*tau + 0.5*a*tau**2
            weights = np.full(len(xyz),1/len(xyz))
            assert abs(weights.sum()-1)<1e-12
            for c,axis in enumerate('xyz'):
                group.create_dataset(axis,data=position[:,:,c].T,compression='gzip')
            group['weights']=weights
            group['T1']=np.full(len(xyz),1.3)
            group['T2']=np.full(len(xyz),.1)
            cases.append(dict(name=name,spins=len(xyz),velocity_base_m_s=v,acceleration_m_s2=a,
                              shear_s_1=shear,instantaneous_echo_mean_m_s=[float(np.sum(weights*velocity)),0.,0.],
                              weight_sum=float(weights.sum()),transverse_motion_max_m=0.))
    (output/'controls.h5.partial').rename(output/'controls.h5')
    sources = [code/'PROTOCOL.md',code/'prepare_controls.py',code/'run_controls.jl',code/'analyze_controls.py',
               repo/'koma_migration_e046/Manifest.toml', repo/'scripts/analyze_phantom_encoding.py',
               repo/'scripts/prepare_phantom_encoding.py']
    sources += [sequence_dir/(e.lower()+'.seq') for e in ['REF','X','Y','Z']]
    registration = dict(experiment='E-054',decision='D-057',status='REGISTERED_BEFORE_SIGNAL',
                        created_utc=datetime.now(timezone.utc).isoformat(),sequence_directory=str(sequence_dir),
                        source_repo=str(repo),models=cases,physical_spins_per_case=32,
                        input_sha256=sha(output/'controls.h5'),source_sha256={str(p):sha(p) for p in sources},
                        precision='f64',late_clock_s=2047*info['period_s'],new_300k=False,new_1m=False,
                        pending_canonical_governance_reason='E053 every-file snapshot remains frozen before source retirement')
    (output/'registration.json').write_text(json.dumps(registration,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:registration[k] for k in ['experiment','status','physical_spins_per_case','late_clock_s']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--code',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.repo,a.code,a.output)
