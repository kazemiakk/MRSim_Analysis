"""Independent right/left ICA virtual boluses on immutable native CFD fluxes."""
from pathlib import Path
import argparse,csv,json,os,sys,time,hashlib,shutil
import h5py,numpy as np
from scipy.sparse import coo_matrix,diags,csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spilu,LinearOperator,bicgstab,gmres
sys.path.insert(0,r'D:\FluentAgent\automation\20261029-extended-hemodynamics-v1')
from extended_common import RUN,PREVIOUS,PROJECT,load,save,sha


class Transport:
    def __init__(self,mass,c0,c1,flux,dt,continuity_correction=False,flow_ordered=False):
        start=time.monotonic();self.mass=mass;self.c0=c0;self.c1=c1;self.flux=flux;self.dt=dt
        ni=len(c1);n=len(mass);positive=flux[:ni]>=0
        donor=np.where(positive,c0[:ni],c1);receiver=np.where(positive,c1,c0[:ni]);f=np.abs(flux[:ni])
        out=np.bincount(donor,weights=f,minlength=n)+np.bincount(c0[ni:],weights=np.maximum(flux[ni:],0),minlength=n)
        self.divergence=np.bincount(c0,weights=flux,minlength=n)-np.bincount(c1,weights=flux[:ni],minlength=n)
        self.continuity_correction=continuity_correction
        self.diag=mass/dt+out-(self.divergence if continuity_correction else 0)
        self.a=coo_matrix((-f/self.diag[receiver],(receiver,donor)),shape=(n,n)).tocsr()+diags(np.ones(n))
        self.flow_ordered=flow_ordered
        if flow_ordered:
            keep=f>0
            graph=csr_matrix((np.ones(keep.sum(),np.int8),(donor[keep],receiver[keep])),shape=(n,n))
            _,labels=connected_components(graph,directed=True,connection='strong')
            perm=np.argsort(labels,kind='stable')
            ilu=spilu(self.a[perm,:][:,perm].tocsc(),drop_tol=1e-4,fill_factor=1.5,permc_spec='NATURAL',diag_pivot_thresh=0)
            def apply(v):
                y=np.empty_like(v);y[perm]=ilu.solve(v[perm]);return y
            self.preconditioner=LinearOperator((n,n),matvec=apply)
        else:
            ilu=spilu(self.a.tocsc(),drop_tol=.001,fill_factor=3,permc_spec='COLAMD',diag_pivot_thresh=0)
            self.preconditioner=LinearOperator((n,n),matvec=ilu.solve)
        self.setup_seconds=time.monotonic()-start

    def advance(self,c,incoming):
        ni=len(self.c1);start=time.monotonic()
        rhs=(self.mass/self.dt*c+incoming)/self.diag
        count=[0]
        def called(_):count[0]+=1
        if self.flow_ordered:
            new,info=gmres(self.a,rhs,x0=c,M=self.preconditioner,rtol=1e-10,atol=1e-16,
                           restart=40,maxiter=10,callback=called,callback_type='pr_norm')
            first_info='not_used_flow_ordered'
        else:
            new,info=bicgstab(self.a,rhs,x0=c,M=self.preconditioner,rtol=1e-10,atol=1e-16,maxiter=200,callback=called)
            first_info=int(info)
        if info and not self.flow_ordered:
            new,info=gmres(self.a,rhs,x0=new,M=self.preconditioner,rtol=1e-10,atol=1e-16,
                           restart=40,maxiter=10,callback=called,callback_type='pr_norm')
        if info:raise RuntimeError('Unconverged scalar linear solve after bounded GMRES fallback: '+str(info))
        residual=np.linalg.norm(self.a@new-rhs)/max(np.linalg.norm(rhs),1e-30)
        if residual>1e-8:
            new,info=gmres(self.a,rhs,x0=new,M=self.preconditioner,rtol=1e-11,atol=0.0,
                           restart=40,maxiter=10,callback=called,callback_type='pr_norm')
            residual=np.linalg.norm(self.a@new-rhs)/max(np.linalg.norm(rhs),1e-30)
        if residual>1e-8:raise RuntimeError('Independent scalar linear residual failed: '+str(residual))
        loss=self.dt*float(np.dot(np.maximum(self.flux[ni:],0),new[self.c0[ni:]]))
        added=self.dt*float(incoming.sum())
        balance=float(np.dot(self.mass,new-c)+loss-added)
        continuity_defect=self.dt*float(np.dot(self.divergence,new)) if self.continuity_correction else 0.
        return new,{'linear_relative_residual':float(residual),'solver_iterations':count[0],'initial_bicgstab_info':first_info,
                    'injected_kg_label':added,'exited_kg_label':loss,'step_balance_kg_label':balance,
                    'inherited_continuity_defect_kg_label':continuity_defect,
                    'corrected_accounting_residual_kg_label':balance-continuity_defect,
                    'solve_seconds':time.monotonic()-start}


def unit_checks():
    op=Transport(np.ones(2),np.array([0,1]),np.array([1]),np.array([1.,1.]),.1)
    q,r=op.advance(np.array([1.,0.]),np.zeros(2))
    assert np.allclose(q,[1/1.1,.1/1.1**2]) and abs(r['step_balance_kg_label'])<1e-12
    op=Transport(np.ones(1),np.array([0,0]),np.array([],dtype=int),np.array([-1.,1.]),.1)
    q,r=op.advance(np.zeros(1),np.ones(1))
    assert np.allclose(q,[.1/1.1]) and abs(r['step_balance_kg_label'])<1e-12
    op=Transport(np.ones(1),np.array([0,0]),np.array([],dtype=int),np.array([-2.,1.]),.1,True)
    q,r=op.advance(np.ones(1),np.array([2.]))
    assert np.allclose(q,[1.]) and abs(r['corrected_accounting_residual_kg_label'])<1e-12
    return {'two_cell_advection_with_exit':'passed','one_cell_inlet_and_exit':'passed',
            'uniform_concentration_with_known_carrier_continuity_defect':'passed'}


def main(out,wall_limit,dtmax,pilot_only,continuity_correction=False,flow_ordered=False,resume=None,agglomerate_mm=0.,case='patient_specific'):
    started=time.monotonic();out.mkdir(exist_ok=False);(out/'fields').mkdir()
    save(out/'unit-checks.json',unit_checks())
    cfg=load(RUN/case/'input-configuration.json');steps=load(RUN/case/'progress.json')['accepted_steps']
    mapping=np.load(PREVIOUS/'comparisons/fluid-ROI-mapping.npz');vol=mapping['volume_m3'];roi=mapping['ellipsoid_mask'];mass=1060*vol
    with h5py.File(cfg['source_case'],'r') as h:
        mesh=h['meshes/1'];c0=mesh['faces/c0/1'][:].astype(int)-1;c1=mesh['faces/c1/1'][:].astype(int)-1
        z=mesh['faces/zoneTopology'];zones={n:(int(lo)-1,int(hi)) for n,lo,hi in zip(z['name'][0].decode().split(';'),z['minId'][:],z['maxId'][:])}
    ni=len(c1);assert len(mass)==1169992
    native_ni=ni;native_internal_keep=None;native_to_scalar=None;roi_volume=vol*roi
    if agglomerate_mm:
        from scipy.sparse.csgraph import connected_components
        import pyvista as pv
        grid=pv.read(PROJECT/'visualizations/20261002-measured-ica-velocity-context-v6/original-fluid-tetrahedra-mm.vtu')
        center=grid.points[grid.cells.reshape(-1,5)[:,1:]].mean(1)
        xyz=np.floor((center-center.min(0))/agglomerate_mm).astype(np.int64);dims=xyz.max(0)+1
        bins=xyz[:,0]+dims[0]*(xyz[:,1]+dims[1]*xyz[:,2])
        same=bins[c0[:ni]]==bins[c1]
        graph=csr_matrix((np.ones(same.sum(),np.int8),(c0[:ni][same],c1[same])),shape=(len(mass),len(mass)))
        count,labels=connected_components(graph,directed=False)
        # Only face-connected native cells inside the same centroid voxel can merge.
        minimum=np.full(count,np.iinfo(np.int64).max,np.int64);maximum=np.full(count,-1,np.int64)
        np.minimum.at(minimum,labels,bins);np.maximum.at(maximum,labels,bins);assert np.array_equal(minimum,maximum)
        coarse0=labels[c0[:ni]];coarse1=labels[c1];native_internal_keep=coarse0!=coarse1
        c0=np.r_[coarse0[native_internal_keep],labels[c0[ni:]]];c1=coarse1[native_internal_keep];ni=len(c1)
        zones={n:(a-native_ni+ni,b-native_ni+ni) for n,(a,b) in zones.items() if a>=native_ni}
        oldvolume=float(vol.sum());roi_volume=np.bincount(labels,weights=vol*roi)
        vol=np.bincount(labels,weights=vol);mass=1060*vol;assert abs(vol.sum()/oldvolume-1)<1e-12
        native_to_scalar=labels
        np.savez_compressed(out/'connected-scalar-agglomeration.npz',native_cell_to_scalar_cell=labels,scalar_volume_m3=vol,scalar_ROI_volume_m3=roi_volume)
        save(out/'agglomeration-checks.json',{'original_fluid_cells':len(labels),'scalar_cells':count,'centroid_bin_mm':agglomerate_mm,
            'total_fluid_volume_preserved':True,'all_groups_within_one_bin':True,
            'merge_edges':'Only original shared fluid-cell faces; separate wall-separated passages cannot merge',
            'geometry_or_original_flow_mesh_changed':False,'new_scalar_geometry':'Union of original tetrahedra, not a voxel replacement boundary',
            'directed_fluxes':'All native crossing-face fluxes retained; opposing flows are NOT canceled before upwinding'})
        print('CONNECTED_SCALAR_CELLS',count,'from',len(labels),flush=True)
    source_files=[cfg['source_case']]+[s['output_stem']+'.dat.h5' for s in steps]
    source_hashes=[{'file':p,'sha256':sha(p)} for p in source_files]
    save(out/'source-manifest.json',source_hashes)
    start_t=float(steps[0]['time_s']);end_t=float(steps[-1]['time_s']);pulse_end=start_t+.100
    protocol={'case':case,'injections':['Right ICA: inlet_b1','Left ICA: inlet_b2'],
              'initial_condition':'C=0 everywhere','inlet_condition':'C=1 at selected inlet for first 100ms; C=0 subsequently and at all other incoming boundaries',
              'concentration':'Dimensionless passive numerical label, NOT clinical iodine concentration',
              'pulse_start_solver_time_s':start_t,'pulse_end_solver_time_s':pulse_end,
              'maximum_scalar_step_s':dtmax,'native_flow':'Frozen midpoint-average face flux during each original 5/10ms interval',
              'scheme':'Backward Euler, first-order spatial upwind; ILU-preconditioned BiCGSTAB with bounded GMRES fallback',
              'carrier_continuity_correction':continuity_correction,
              'linear_solver':'Flow-graph SCC-ordered ILU and GMRES' if flow_ordered else 'ILU BiCGSTAB / bounded GMRES fallback',
              'native_flow_cells':1169992,'scalar_transport_cells':len(mass),'connected_scalar_agglomeration_mm':agglomerate_mm,
              'scalar_equation':'m dC/dt + sum(face_flux*C_upwind) - C*sum(face_flux)=0' if continuity_correction else 'm dC/dt + sum(face_flux*C_upwind)=0',
              'mass_accounting':'Raw label mass defect and inherited C*div(F) are both reported; corrected ledger tolerance 1e-5, raw cumulative defect must remain below 1e-3 of injected label',
              'diffusion':'No physical diffusion specified; numerical diffusion remains unvalidated',
              'source_flow_unchanged':True,'new_Fluent_sessions':0,'periodicity_validated':False,
              'molecular_or_injection_momentum_effects':False,'wall_limit_seconds':wall_limit,
              'timing':'Starts from first saved 5ms CFD state; no invented pre-start velocity field'}
    save(out/'protocol.json',protocol)
    concentrations=np.zeros((2,len(mass)));injected=np.zeros(2);exited=np.zeros(2);inherited_defect=np.zeros(2)
    history=[];frames=[];t=start_t;status='bounded_partial';error=None
    def flux(step):
        with h5py.File(step['output_stem']+'.dat.h5','r') as h:f=h['results/1/phase-1/faces/SV_FLUX/1'][:]
        return f if native_internal_keep is None else np.r_[f[:native_ni][native_internal_keep],f[native_ni:]]
    def record_frame():
        name='fields/C-t%07dus.npz'%round(t*1e6)
        native=concentrations if native_to_scalar is None else concentrations[:,native_to_scalar]
        np.savez_compressed(out/name,right=native[0].astype('float32'),left=native[1].astype('float32'),time_s=t)
        frames.append({'time_s':t,'file':name})
    right_test_passed=False
    if resume is not None:
        previous=load(resume/'result.json');prior_protocol=load(resume/'protocol.json')
        assert previous['status']=='bounded_partial' or (previous['status']=='stopped_with_error' and previous['error'].startswith('TimeoutError:'))
        assert previous['original_sources_unchanged'] and prior_protocol['carrier_continuity_correction']==continuity_correction
        assert prior_protocol['maximum_scalar_step_s']==dtmax
        assert prior_protocol.get('connected_scalar_agglomeration_mm',0)==agglomerate_mm
        frames=load(resume/'frames.json');t=frames[-1]['time_s']
        with np.load(resume/frames[-1]['file']) as p:concentrations=np.stack([p['right'],p['left']]).astype(float)
        if native_to_scalar is not None:
            concentrations=np.array([np.bincount(native_to_scalar,weights=v)/np.bincount(native_to_scalar) for v in concentrations])
        for f in frames:shutil.copy2(resume/f['file'],out/f['file'])
        history=list(csv.DictReader((resume/'transport-history.csv').open()))
        unfinished=[r for r in history if float(r['solver_time_s'])>t+1e-10]
        history=[r for r in history if float(r['solver_time_s'])<=t+1e-10]
        for i,side in enumerate(['right','left']):
            row=[r for r in history if r['injection']==side][-1]
            injected[i]=float(row['injected_label_volume_ml'])*1060/1e6
            exited[i]=float(row['exited_label_volume_ml'])*1060/1e6
            inherited_defect[i]=float(row['cumulative_inherited_continuity_defect_kg_label'])
        shutil.copy2(resume/'right-inlet-initial-test.json',out/'right-inlet-initial-test.json')
        right_test_passed=True
        save(out/'resume-source.json',{'source':str(resume),'starting_solver_time_s':t,
            'initial_label_fields':'Copied saved cell concentrations; no second injection or initialization',
            'prior_result':previous,'previous_pulse_start_s':start_t,
            'uncheckpointed_partial_interval_rows_recomputed':len(unfinished)})
    else:record_frame()
    run_steps=[s for s in steps if s['time_s']>=t-1e-10]
    assert abs(run_steps[0]['time_s']-t)<1e-10
    fa=flux(run_steps[0])
    try:
        for interval,(old,new) in enumerate(zip(run_steps[:-1],run_steps[1:])):
            if time.monotonic()-started>=wall_limit-30:break
            fb=flux(new);fmid=.5*(fa+fb);lo=float(old['time_s']);hi=float(new['time_s'])
            assert abs(t-lo)<1e-10
            count=int(np.ceil((hi-lo)/dtmax-1e-9));dt=(hi-lo)/count
            op=Transport(mass,c0,c1,fmid,dt,continuity_correction,flow_ordered)
            if interval==0 and resume is None:
                all_incoming=np.bincount(c0[ni:],weights=np.maximum(-fmid[ni:],0),minlength=len(mass))
                if continuity_correction:
                    uniform,uniform_audit=op.advance(np.ones(len(mass)),all_incoming)
                    uniform_error=float(np.max(abs(uniform-1)));assert uniform_error<1e-8
                    save(out/'actual-scalar-grid-constant-check.json',{'maximum_constant_concentration_error':uniform_error,
                        'scalar_cells':len(mass),'passed':True,'audit':uniform_audit})
            boundary_sources=[]
            for inlet in ['inlet_b1','inlet_b2']:
                a,b=zones[inlet];assert a>=ni and np.max(fmid[a:b])<0
                boundary_sources.append(np.bincount(c0[a:b],weights=-fmid[a:b],minlength=len(mass)))
            for sub in range(count):
                if time.monotonic()-started>=wall_limit-20:raise TimeoutError('Transport wall-time bound reached; partial states saved')
                left_t=lo+sub*dt;right_t=lo+(sub+1)*dt
                # Exact rectangular-pulse overlap, including an interval crossing the pulse end.
                fraction=max(0.,min(right_t,pulse_end)-max(left_t,start_t))/dt
                for side in range(2):
                    incoming=boundary_sources[side]*fraction
                    candidate,audit=op.advance(concentrations[side],incoming)
                    minimum=float(candidate.min());maximum=float(candidate.max())
                    if not np.isfinite(candidate).all() or minimum< -1e-6 or maximum>1.001:
                        raise RuntimeError('Scalar bounds failed: '+str((minimum,maximum))+'; no clipping permitted')
                    total_added=injected[side]+audit['injected_kg_label'];total_exited=exited[side]+audit['exited_kg_label']
                    signed_balance=float(np.dot(mass,candidate))+total_exited-total_added
                    balance=abs(signed_balance)/max(total_added,1e-30)
                    corrected_defect=inherited_defect[side]+audit['inherited_continuity_defect_kg_label']
                    accounting_error=abs(signed_balance-corrected_defect)/max(total_added,1e-30)
                    if accounting_error>1e-5:raise RuntimeError('Cumulative scalar accounting failed: '+str(accounting_error))
                    if balance>(1e-3 if continuity_correction else 1e-5):raise RuntimeError('Raw tracer mass defect exceeded declared limit: '+str(balance))
                    if side==0 and not right_test_passed:
                        right_test_passed=True
                        save(out/'right-inlet-initial-test.json',{'status':'passed_initial_step_only','time_s':right_t,
                             'concentration_min':minimum,'concentration_max':maximum,'relative_mass_balance_error':balance,**audit})
                        print('RIGHT_INITIAL_TEST_PASSED',flush=True)
                    concentrations[side]=candidate;injected[side]=total_added;exited[side]=total_exited;inherited_defect[side]=corrected_defect
                    history.append({'injection':'right' if side==0 else 'left','solver_time_s':right_t,'time_since_injection_s':right_t-start_t,
                        'dt_s':dt,'C_min':minimum,'C_max':maximum,'injected_label_volume_ml':total_added/1060*1e6,
                        'exited_label_volume_ml':total_exited/1060*1e6,'whole_domain_label_volume_ml':float(np.dot(vol,candidate))*1e6,
                        'ellipsoidal_ROI_label_volume_ml':float(np.dot(roi_volume,candidate))*1e6,
                        'ellipsoidal_ROI_mean_C':float(np.dot(roi_volume,candidate)/roi_volume.sum()),
                        'relative_mass_balance_error':balance,'corrected_ledger_relative_error':accounting_error,
                        'cumulative_inherited_continuity_defect_kg_label':corrected_defect,'factor_setup_seconds':op.setup_seconds,**audit})
                t=right_t
            record_frame();fa=fb
            update={'status':'running','last_time_s':t,'accepted_native_intervals':len(frames)-1,'elapsed_seconds':time.monotonic()-started}
            with (out/'progress.json').open('w') as h:json.dump(update,h,indent=2)
            print('DYE_BOLUS t=%.3f wall=%.1fs setup=%.2fs maxC=%.6f'%(t,time.monotonic()-started,op.setup_seconds,concentrations.max()),flush=True)
            if pilot_only:status='initial_interval_pilot_complete';break
        if abs(t-end_t)<1e-10:status='available_first_cycle_transport_complete'
    except Exception as e:error=type(e).__name__+': '+str(e);status='stopped_with_error'
    if history:
        with (out/'transport-history.csv').open('x',newline='') as h:
            w=csv.DictWriter(h,fieldnames=list(history[0]));w.writeheader();w.writerows(history)
    save(out/'frames.json',frames)
    unchanged=all(sha(r['file'])==r['sha256'] for r in source_hashes)
    save(out/'result.json',{'status':status,'last_time_s':t,'duration_since_injection_s':t-start_t,
        'wall_seconds':time.monotonic()-started,'error':error,'original_sources_unchanged':unchanged,
        'saved_frames':len(frames),'injected_label_volumes_ml':(injected/1060*1e6).tolist(),
        'source_periodicity_validated':False,'transport_convergence_validated':False,
        'incomplete':'No clinical injection calibration, physical contrast effects, full washout or mesh/time-step convergence established'})
    print('DYE_BOLUS_FINISHED '+str(out)+' '+status+' '+str(error),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--wall-limit',type=float,default=1500)
    p.add_argument('--dt-max',type=float,default=.0025);p.add_argument('--pilot-only',action='store_true')
    p.add_argument('--continuity-correction',action='store_true');p.add_argument('--flow-ordered',action='store_true')
    p.add_argument('--resume',type=Path);p.add_argument('--agglomerate-mm',type=float,default=0.)
    p.add_argument('--case',default='patient_specific');a=p.parse_args()
    main(a.out,a.wall_limit,a.dt_max,a.pilot_only,a.continuity_correction,a.flow_ordered,a.resume,a.agglomerate_mm,a.case)
