"""Independent constant-field and saved-tracer checks for the corrected scheme."""
from pathlib import Path
import sys,h5py,numpy as np
sys.path.insert(0,r'D:\FluentAgent\automation\20261029-extended-hemodynamics-v1')
from extended_common import *
from dye_bolus import Transport

out=PROJECT/'results/20261029-virtual-angiography-right-left-v2'
cfg=load(RUN/'patient_specific/input-configuration.json');steps=load(RUN/'patient_specific/progress.json')['accepted_steps']
mapping=np.load(PREVIOUS/'comparisons/fluid-ROI-mapping.npz');mass=mapping['volume_m3']*1060
with h5py.File(cfg['source_case'],'r') as h:
    c0=h['meshes/1/faces/c0/1'][:].astype(int)-1;c1=h['meshes/1/faces/c1/1'][:].astype(int)-1
ss=[min(steps,key=lambda s:abs(s['time_s']-t)) for t in [.060,.065]];ff=[]
for s in ss:
    with h5py.File(s['output_stem']+'.dat.h5','r') as h:ff.append(h['results/1/phase-1/faces/SV_FLUX/1'][:])
flux=.5*(ff[0]+ff[1]);ni=len(c1)
incoming=np.bincount(c0[ni:],weights=np.maximum(-flux[ni:],0),minlength=len(mass))
op=Transport(mass,c0,c1,flux,.005,True)
new,audit=op.advance(np.ones(len(mass)),incoming)
error=float(np.max(abs(new-1)))
assert error<1e-8
save(out/'constant-concentration-preservation-check.json',{'status':'passed_scoped_test',
    'flow_interval_s':[.060,.065],'max_absolute_constant_field_error':error,'checks':audit,
    'original_conservative_test_error':load(PROJECT/'results/20261029-virtual-angiography-right-left-v1/constant-preservation-diagnostic.json')['constant_C1_test_max_absolute_error'],
    'physical_flow_modified':False,'scheme':'Advective-form concentration with explicitly recorded native continuity defect',
    'scope':'One representative problematic flux interval; not mesh/time-step convergence or physical contrast validation'})
print('CONSTANT_PRESERVATION_CHECK',error,flush=True)
