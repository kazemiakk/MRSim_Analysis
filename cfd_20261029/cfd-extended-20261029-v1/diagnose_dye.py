"""Check local carrier continuity at the first rejected dye interval."""
import sys
from pathlib import Path
import h5py,numpy as np
sys.path.insert(0,r'D:\FluentAgent\automation\20261029-extended-hemodynamics-v1')
from dye_bolus import Transport
from extended_common import *

out=PROJECT/'results/20261029-virtual-angiography-right-left-v1'
cfg=load(RUN/'patient_specific/input-configuration.json');steps=load(RUN/'patient_specific/progress.json')['accepted_steps']
mapping=np.load(PREVIOUS/'comparisons/fluid-ROI-mapping.npz');mass=mapping['volume_m3']*1060
with h5py.File(cfg['source_case'],'r') as h:
    c0=h['meshes/1/faces/c0/1'][:].astype(int)-1;c1=h['meshes/1/faces/c1/1'][:].astype(int)-1
    z=h['meshes/1/faces/zoneTopology'];zones={n:(int(lo)-1,int(hi)) for n,lo,hi in zip(z['name'][0].decode().split(';'),z['minId'][:],z['maxId'][:])}
ss=[min(steps,key=lambda s:abs(s['time_s']-t)) for t in [.060,.065]];ff=[]
for s in ss:
    with h5py.File(s['output_stem']+'.dat.h5','r') as h:ff.append(h['results/1/phase-1/faces/SV_FLUX/1'][:])
flux=.5*(ff[0]+ff[1]);ni=len(c1);div=np.bincount(c0,weights=flux,minlength=len(mass))-np.bincount(c1,weights=flux[:ni],minlength=len(mass))
op=Transport(mass,c0,c1,flux,.005)
incoming=np.bincount(c0[ni:],weights=np.maximum(-flux[ni:],0),minlength=len(mass))
constant,check=op.advance(np.ones(len(mass)),incoming)
native=np.load(out/'fields/C-t0060000us.npz')['right'];a,b=zones['inlet_b1'];inc=np.bincount(c0[a:b],weights=-flux[a:b],minlength=len(mass))
candidate,audit=op.advance(native,inc);j=int(np.argmax(candidate));owned=np.flatnonzero(c0==j);other=np.flatnonzero(c1==j)
grid=pv.read(PROJECT/'visualizations/20261002-measured-ica-velocity-context-v6/original-fluid-tetrahedra-mm.vtu')
tet=grid.cells.reshape(-1,5)[:,1:]
result={'interval_s':[.060,.065],'constant_C1_test_max_absolute_error':float(np.max(abs(constant-1))),
        'constant_C1_test_max':float(constant.max()),'right_pulse_test_max':float(candidate[j]),'cell_index0':j,
        'cell_centroid_mm':grid.points[tet[j]].mean(0).tolist(),'cell_native_mass_flux_divergence_kg_s':float(div[j]),
        'cell_native_relative_carrier_mass_defect_over_5ms':float(.005*div[j]/mass[j]),
        'global_net_carrier_mass_flow_kg_s':float(flux[ni:].sum()),'linear_solve':audit,
        'interpretation':'A spatially uniform passive concentration should remain uniform. The native conservative scalar operator fails that test because saved carrier flux is not cellwise divergence-free. No original flow or boundary condition modified.'}
save(out/'constant-preservation-diagnostic.json',result)
print(result,flush=True)
