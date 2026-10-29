"""Numerical checker QA on old baseline only, not a new MRI result."""
from pathlib import Path
from datetime import datetime,timezone
import json
import h5py
import numpy as np

run=Path('D:/MRSim/outputs/phantom_density_e055')
target=run/'phase_checker_precheck.json'
if target.exists():raise FileExistsError(target)
folder=Path('D:/MRSim/MRSIM/MRSimTracks/outputs/phantom_precision_e050/mapped_f64_v1')
with h5py.File(folder/'ref_result.h5') as f:ref=f['image'][:]
rows=[]
for e in ['x','y','z']:
    with h5py.File(folder/(e+'_result.h5')) as f:im=f['image'][:]
    saved=np.angle(im*ref.conj())*(5.0/np.pi)
    cross=im.astype(np.complex128)*ref.conj()
    independent=np.arctan2(cross.imag,cross.real)*(5.0/np.pi)
    error=float(abs((independent-saved+5.0)%10.0-5.0).max())
    assert error<=3e-6
    rows.append(dict(encoding=e,maximum_wrapped_error_m_s=error,voxels_compared=saved.size))
receipt=dict(status='PASS_BASELINE_INDEPENDENT_PHASE_FORMULA',
    utc=datetime.now(timezone.utc).isoformat(),rows=rows,new_300k_accuracy_result=False)
target.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt),flush=True)
