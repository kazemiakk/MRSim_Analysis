"""Early full-history signal checks on completed static encodings; no promotion."""
from pathlib import Path
from time import perf_counter
import h5py
import numpy as np
from analyze_physics import load_group,predict,complex_array

root=Path('D:/MRSim/outputs/acquisition_physics_e056')
with h5py.File(root/'inputs_v1/controls.h5') as f:
    positions=f['positions'][:];rho=f['rho'][:];u=f['u_static'][:]
    t1=float(f['T1'][0]);t2=float(f['T2'][0]);times=f['path_time_s'][:]
    factor=f['path_basis_s'][:];period=float(f['motion_period_s'][()])
with h5py.File(root/'reference_contract_v2/waveforms.h5') as f:
    for enc in ['REF','X']:
        start=perf_counter();print('START_EARLY_STATIC_NUMERICAL',enc,flush=True)
        m=load_group(f['baseline'][enc],80,16,32);m['gamma']=float(f['gamma_Hz_T'][()])
        raw,image=predict(m,positions,rho,u,t1,t2,times,factor,period,(80,16,32,15))
        with h5py.File(root/'acquisition_v1'/f'static_baseline_{enc.lower()}.h5') as result:
            actual=complex_array(result['raw_signal_f64']).ravel();actual_image=complex_array(result['image'])
        r=float(np.linalg.norm(actual-raw)/np.linalg.norm(raw))
        i=float(np.linalg.norm(actual_image-image)/np.linalg.norm(image))
        print(dict(encoding=enc,raw_relative_L2=r,image_relative_L2=i,seconds=perf_counter()-start,
                   passed=r<=.01 and i<=.01,scope='Early completed static encodings only'),flush=True)
        assert r<=.01 and i<=.01,'Full-history static numerical gate failed'
