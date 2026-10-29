"""Benchmark flow-graph ordered preconditioning; same scalar equation and tolerances."""
import sys,time
from pathlib import Path
import numpy as np,h5py
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spilu,LinearOperator,gmres
sys.path.insert(0,r'D:\FluentAgent\automation\20261029-extended-hemodynamics-v1')
from extended_common import *
from dye_bolus import Transport

out=PROJECT/'results/20261029-virtual-angiography-right-left-v2'
cfg=load(RUN/'patient_specific/input-configuration.json');steps=load(RUN/'patient_specific/progress.json')['accepted_steps'];ff=[]
with h5py.File(cfg['source_case'],'r') as h:
    c0=h['meshes/1/faces/c0/1'][:].astype(int)-1;c1=h['meshes/1/faces/c1/1'][:].astype(int)-1
    z=h['meshes/1/faces/zoneTopology'];zones={n:(int(lo)-1,int(hi)) for n,lo,hi in zip(z['name'][0].decode().split(';'),z['minId'][:],z['maxId'][:])}
for s in steps[:2]:
    with h5py.File(s['output_stem']+'.dat.h5','r') as h:ff.append(h['results/1/phase-1/faces/SV_FLUX/1'][:])
flux=.5*(ff[0]+ff[1]);mass=1060*np.load(PREVIOUS/'comparisons/fluid-ROI-mapping.npz')['volume_m3']
op=Transport(mass,c0,c1,flux,.005,True);n=len(mass);ni=len(c1);positive=flux[:ni]>=0
donor=np.where(positive,c0[:ni],c1);receiver=np.where(positive,c1,c0[:ni]);keep=abs(flux[:ni])>0
start=time.monotonic();graph=csr_matrix((np.ones(keep.sum(),np.int8),(donor[keep],receiver[keep])),shape=(n,n))
nc,labels=connected_components(graph,directed=True,connection='strong');perm=np.argsort(labels,kind='stable');sizes=np.bincount(labels)
ilu=spilu(op.a[perm,:][:,perm].tocsc(),drop_tol=1e-4,fill_factor=1.5,permc_spec='NATURAL',diag_pivot_thresh=0)
def apply(v):
    y=np.empty_like(v);y[perm]=ilu.solve(v[perm]);return y
preconditioner=LinearOperator((n,n),matvec=apply);setup=time.monotonic()-start
a,b=zones['inlet_b1'];incoming=np.bincount(c0[a:b],weights=-flux[a:b],minlength=n)
rhs=incoming/op.diag;counter=[0]
def counter_step(_):counter[0]+=1
start=time.monotonic();answer,info=gmres(op.a,rhs,M=preconditioner,rtol=1e-10,atol=1e-13,restart=40,maxiter=10,callback=counter_step,callback_type='pr_norm')
elapsed=time.monotonic()-start;residual=float(np.linalg.norm(op.a@answer-rhs)/np.linalg.norm(rhs))
reference=np.load(out/'fields/C-t0010000us.npz')['right'];difference=float(np.max(abs(answer-reference)))
assert info==0 and residual<1e-8 and difference<1e-6
result={'strong_components':nc,'largest_component_cells':int(sizes.max()),'singleton_components':int((sizes==1).sum()),
        'ordered_setup_s':setup,'solve_s':elapsed,'gmres_iterations':counter[0],'relative_residual':residual,
        'max_difference_from_existing_float32_same_equation_solution':difference,'equation_or_tolerances_changed':False}
save(out/'flow-ordered-solver-benchmark.json',result);print(result,flush=True)
