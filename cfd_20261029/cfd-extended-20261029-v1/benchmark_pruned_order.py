"""Weighted flow ordering changes ONLY the preconditioner, never scalar fluxes."""
import sys,time
import h5py,numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spilu,LinearOperator,gmres
sys.path.insert(0,r'D:\FluentAgent\automation\20261029-extended-hemodynamics-v1')
from extended_common import *
from dye_bolus import Transport
out=PROJECT/'results/20261029-virtual-angiography-right-left-v2'
cfg=load(RUN/'patient_specific/input-configuration.json');steps=load(RUN/'patient_specific/progress.json')['accepted_steps']
with h5py.File(cfg['source_case'],'r') as h:
    c0=h['meshes/1/faces/c0/1'][:].astype(int)-1;c1=h['meshes/1/faces/c1/1'][:].astype(int)-1
    z=h['meshes/1/faces/zoneTopology'];zones={n:(int(lo)-1,int(hi)) for n,lo,hi in zip(z['name'][0].decode().split(';'),z['minId'][:],z['maxId'][:])}
ff=[]
for t in [.095,.105]:
    step=min(steps,key=lambda s:abs(s['time_s']-t))
    with h5py.File(step['output_stem']+'.dat.h5','r') as h:ff.append(h['results/1/phase-1/faces/SV_FLUX/1'][:])
flux=.5*(ff[0]+ff[1]);mass=1060*np.load(PREVIOUS/'comparisons/fluid-ROI-mapping.npz')['volume_m3']
op=Transport(mass,c0,c1,flux,.005,True);n=len(mass);ni=len(c1);positive=flux[:ni]>=0
donor=np.where(positive,c0[:ni],c1);receiver=np.where(positive,c1,c0[:ni]);f=abs(flux[:ni]);outflux=np.bincount(donor,weights=f,minlength=n)
start=time.monotonic();keep=f>.15*outflux[donor]
graph=csr_matrix((np.ones(keep.sum(),np.int8),(donor[keep],receiver[keep])),shape=(n,n))
nc,labels=connected_components(graph,directed=True,connection='strong');perm=np.argsort(labels,kind='stable');sizes=np.bincount(labels)
ilu=spilu(op.a[perm,:][:,perm].tocsc(),drop_tol=1e-4,fill_factor=1.5,permc_spec='NATURAL',diag_pivot_thresh=0)
def apply(v):
    y=np.empty_like(v);y[perm]=ilu.solve(v[perm]);return y
preconditioner=LinearOperator((n,n),matvec=apply);setup=time.monotonic()-start
a,b=zones['inlet_b1'];incoming=np.bincount(c0[a:b],weights=-flux[a:b],minlength=n)
c=np.load(out/'fields/C-t0095000us.npz')['right'].astype(float);count=[0];residuals=[]
def called(_):count[0]+=1
start=time.monotonic()
for j in range(2):
    rhs=(mass/.005*c+incoming)/op.diag
    c,info=gmres(op.a,rhs,x0=c,M=preconditioner,rtol=1e-10,atol=1e-13,restart=40,maxiter=10,callback=called,callback_type='pr_norm')
    residuals.append(float(np.linalg.norm(op.a@c-rhs)/np.linalg.norm(rhs)));assert info==0 and residuals[-1]<1e-8
elapsed=time.monotonic()-start
reference=np.load(out/'fields/C-t0105000us.npz')['right'];difference=float(np.max(abs(c-reference)));assert difference<1e-6
result={'interval_s':[.095,.105],'preconditioner_only_graph_relative_edge_threshold':.15,'strong_components':nc,
        'largest_component_cells':int(sizes.max()),'ordered_setup_s':setup,'two_solve_s':elapsed,'gmres_iterations':count[0],
        'relative_residuals':residuals,'max_difference_from_saved_same_equation_solution':difference,
        'actual_scalar_fluxes_removed_or_changed':False,'equation_or_tolerances_changed':False}
save(out/'weighted-flow-order-benchmark.json',result);print(result,flush=True)
