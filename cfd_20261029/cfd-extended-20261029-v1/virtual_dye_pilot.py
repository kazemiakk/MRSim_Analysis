"""Bounded conservative pure-advection diagnostic on original finite-volume cells."""
import argparse,time
from pathlib import Path
import h5py,numpy as np
from scipy.sparse import coo_matrix,diags
from scipy.sparse.linalg import bicgstab,LinearOperator
from extended_common import *


def backward_step(c,mass,c0,c1,flux,dt):
    ni=len(c1);positive=flux[:ni]>=0
    donor=np.where(positive,c0[:ni],c1);receiver=np.where(positive,c1,c0[:ni]);f=np.abs(flux[:ni])
    out=np.bincount(donor,weights=f,minlength=len(c))
    out+=np.bincount(c0[ni:],weights=np.maximum(flux[ni:],0),minlength=len(c))
    diag=mass/dt+out
    a=coo_matrix((-f,(receiver,donor)),shape=(len(c),len(c))).tocsr()+diags(diag)
    # Row equilibration removes enormous mesh-volume scaling without altering equations.
    scaled=diags(1/diag)@a
    new,info=bicgstab(scaled,(mass/dt*c)/diag,x0=c,rtol=1e-8,atol=1e-12,maxiter=600)
    if info!=0:raise RuntimeError('Scalar linear solve did not converge: '+str(info))
    loss=dt*float(np.dot(np.maximum(flux[ni:],0),new[c0[ni:]]))
    residual=float(abs(np.dot(mass,new)-np.dot(mass,c)+loss))
    return new,loss,residual


def main(out,case,wall_limit):
    out.mkdir(exist_ok=False);clock=time.monotonic()
    # Two well-mixed cells, closed end: exactly conservative known backward-Euler result.
    q,_,err=backward_step(np.array([1.,0.]),np.ones(2),np.array([0]),np.array([1]),np.array([1.]),.1)
    assert np.allclose(q,[1/1.1,.1/1.1]) and err<1e-12
    cfg=load(RUN/case/'input-configuration.json');steps=load(RUN/case/'progress.json')['accepted_steps']
    with h5py.File(cfg['source_case'],'r') as h:
        c0=h['meshes/1/faces/c0/1'][:].astype(int)-1;c1=h['meshes/1/faces/c1/1'][:].astype(int)-1
    mapping=np.load(PREVIOUS/'comparisons/fluid-ROI-mapping.npz');vol=mapping['volume_m3'];roi=mapping['ellipsoid_mask'];mass=vol*cfg['density_kg_m3']
    c=roi.astype(float);initial=float(np.dot(mass,c));loss_total=0.;t=START;history=[];status='bounded_partial_pilot';error=None
    try:
        for left,right in zip(steps[:-1],steps[1:]):
            if time.monotonic()-clock>wall_limit:break
            t1=left['time_s'];t2=right['time_s']
            with h5py.File(left['output_stem']+'.dat.h5','r') as h:fa=h['results/1/phase-1/faces/SV_FLUX/1'][:]
            with h5py.File(right['output_stem']+'.dat.h5','r') as h:fb=h['results/1/phase-1/faces/SV_FLUX/1'][:]
            count=int(np.ceil((t2-t1)/.001-1e-9));dt=(t2-t1)/count
            for j in range(count):
                if time.monotonic()-clock>wall_limit:break
                alpha=(j+.5)/count;flux=(1-alpha)*fa+alpha*fb
                c,lost,res=backward_step(c,mass,c0,c1,flux,dt);loss_total+=lost;t=t1+(j+1)*dt
                minimum=float(c.min());maximum=float(c.max());balance=abs(float(np.dot(mass,c))+loss_total-initial)/initial
                history.append({'time_s':float(t),'dt_s':float(dt),'ROI_tracer_remaining_fraction':float(np.dot(mass[roi],c[roi])/initial),
                    'whole_domain_tracer_remaining_fraction':float(np.dot(mass,c)/initial),
                    'domain_exited_fraction':loss_total/initial,'mass_balance_relative_error':balance,'C_min':minimum,'C_max':maximum})
                if minimum< -1e-6 or maximum>1.001:
                    status='concentration_bounds_failed';raise RuntimeError('Native nonzero local divergence or transport error violates concentration bounds; no clipping applied.')
                if balance>1e-5:
                    status='mass_balance_failed';raise RuntimeError('Scalar mass accounting did not pass.')
            print('DYE_PILOT',case,round(t,3),round(time.monotonic()-clock,1),flush=True)
            if time.monotonic()-clock>wall_limit:break
        if abs(t-PERIOD)<1e-10:status='one_first_cycle_transport_pilot_complete'
    except Exception as e:error=str(e)
    if history:csvsave(out/'virtual-dye-pilot-history.csv',history)
    np.savez_compressed(out/'last-transport-state.npz',concentration=c,time_s=t,fluent_cell_index0=np.arange(len(c)))
    save(out/'pilot-result.json',{'status':status,'case':case,'last_time_s':float(t),'wall_seconds':time.monotonic()-clock,'error':error,
        'requested_wall_limit_s':wall_limit,'method':'Conservative backward-Euler first-order upwind advection of a passive numerical label; actual native internal mass fluxes, zero incoming label, outlet advection. No diffusion coefficient invented, no momentum feedback.',
        'initial_condition':'C=1 only in the previously approved ellipsoidal fluid ROI, C=0 elsewhere.',
        'new_Fluent_sessions':0,'source_flow_modified':False,'periodicity_validated':False,
        'not_completed':'Ten-cycle washout and transport mesh/timestep/numerical-diffusion validation. Pilot failures are not repaired by clipping concentration or changing native fluxes.'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--case',choices=KINDS,default='patient_specific');p.add_argument('--wall-limit',type=float,default=120);a=p.parse_args()
    main(a.out/('virtual-dye-pilot-v2-'+a.case),a.case,a.wall_limit)
