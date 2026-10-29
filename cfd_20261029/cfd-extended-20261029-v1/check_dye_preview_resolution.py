"""Compare early agglomerated dye fields against preserved native-cell reference."""
from pathlib import Path
import csv,sys,numpy as np
sys.path.insert(0,r'D:\FluentAgent\automation\20261029-extended-hemodynamics-v1')
from extended_common import *
out=PROJECT/'results/20261029-virtual-angiography-connected-250um-v1'
reference=PROJECT/'results/20261029-virtual-angiography-right-left-v3'
mapping=np.load(PREVIOUS/'comparisons/fluid-ROI-mapping.npz');v=mapping['volume_m3'];mask=mapping['ellipsoid_mask']
rows=[]
for us in [10000,75000,105000,135000,185000]:
    name='C-t%07dus.npz'%us
    with np.load(out/'fields'/name) as approx,np.load(reference/'fields'/name) as native:
        for side in ['right','left']:
            a=native[side].astype(float);b=approx[side].astype(float)
            error=float(np.sqrt(np.dot(v,(a-b)**2)/np.dot(v,a*a)))
            av=float(np.dot(v[mask],a[mask])/v[mask].sum());bv=float(np.dot(v[mask],b[mask])/v[mask].sum())
            rows.append({'injection':side,'solver_time_s':us/1e6,'elapsed_after_injection_s':us/1e6-.005,
                'volume_weighted_relative_concentration_L2_difference':error,'native_ROI_mean_C':av,
                'preview_ROI_mean_C':bv,'ROI_mean_absolute_C_difference':abs(bv-av),
                'meaning':'Limited early-time spatial-resolution sensitivity; preview NOT transport-converged'})
with (out/'early-native-resolution-sensitivity.csv').open('x',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(rows[-2:],flush=True)
