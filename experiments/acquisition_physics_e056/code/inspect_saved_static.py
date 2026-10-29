"""Read-only early zero-flow sanity check; full independent gates still apply."""
from pathlib import Path
import h5py
import numpy as np
from analyze_physics import complex_array

root=Path('D:/MRSim/outputs/acquisition_physics_e056/acquisition_v1')
x=(np.arange(80)-40)*.00125;y=(np.arange(16)-8)*.00125;z=(np.arange(32)-16)*.0025
zz,yy,xx=np.meshgrid(z,y,x,indexing='ij')
core=(yy**2+zz**2<=(.004579049785927147-.00125)**2)&(abs(xx)<=.0025)
for arm in ['baseline','bw700_te274','bw700_te210']:
    paths=[root/f'static_{arm}_{e}.h5' for e in ['ref','x','y','z']]
    if not all(p.exists() for p in paths):continue
    images=[]
    for p in paths:
        with h5py.File(p) as f:images.append(complex_array(f['image']))
    a=np.stack(images);mag=abs(a)
    support=np.all(mag>=.05*mag.max(axis=(2,3,4),keepdims=True),axis=(0,1))
    v=np.angle(a[1:]*a[0].conj())*5/np.pi
    mask=core&support
    print(dict(arm=arm,predefined_core_voxels=int(core.sum()),retained_core_voxels=int(mask.sum()),
               static_vector_RMSE_m_s=float(np.sqrt(np.mean(np.sum(v[:,:,mask]**2,axis=0)))),
               limitation='Early zero-flow sanity only; not independent complex-signal clearance'),flush=True)
