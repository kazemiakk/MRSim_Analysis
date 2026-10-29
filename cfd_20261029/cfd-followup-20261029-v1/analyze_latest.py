"""Native-cell comparisons and saved-track exploratory transport; all outputs new."""
import argparse, importlib.util, sys, time
from pathlib import Path
import h5py
import numpy as np
import pyvista as pv
from common_followup import *


def wall_module():
    p = PROJECT/'automation/20261023-systole-first-wss-v1/wss_compare.py'
    spec = importlib.util.spec_from_file_location('existing_wall_reader', p)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def fluid_comparison(a,b,pa,pb,volume):
    mse = float(np.average(np.sum((a-b)**2,axis=1),weights=volume))
    ref = float(np.average(np.sum(a*a,axis=1),weights=volume))
    dp = pb-pa
    return {'velocity_mse_m2_s2':mse, 'ps_velocity_energy_m2_s2':ref,
            'vector_velocity_rmse_m_s':mse**.5,
            'normalized_velocity_rmse_percent':100*(mse/ref)**.5 if ref>0 else None,
            'speed_rmse_m_s':float(np.average((np.linalg.norm(a,axis=1)-np.linalg.norm(b,axis=1))**2,weights=volume))**.5,
            'mean_pressure_difference_pa':float(np.average(dp,weights=volume)),
            'pressure_mse_pa2':float(np.average(dp*dp,weights=volume)),
            'pressure_rmse_pa':float(np.average(dp*dp,weights=volume))**.5}


def geometry(output, cfg):
    grid = pv.read(PROJECT/'visualizations/20261002-measured-ica-velocity-context-v6/original-fluid-tetrahedra-mm.vtu')
    assert np.array_equal(grid['FluentCellIndex0'],np.arange(1169992))
    tet = grid.cells.reshape(-1,5)[:,1:]
    xyz = grid.points/1000
    p = xyz[tet]
    volume = np.abs(np.einsum('ij,ij->i',p[:,1]-p[:,0],np.cross(p[:,2]-p[:,0],p[:,3]-p[:,0])))/6
    centers = p.mean(1)
    inside = in_ellipsoid(centers)
    assert inside.any() and (~inside).any() and np.all(volume>0)
    assert abs(volume.sum()/2.16468663609535e-6-1)<1e-10
    for kind in KINDS:
        with h5py.File(cfg[kind]['source_case'],'r') as f:
            native = next(iter(f['meshes/1/nodes/coords'].values()))[:]
            assert np.allclose(native, xyz,rtol=0,atol=1e-15)
    roi = grid.extract_cells(inside); roi.save(output/'ellipsoidal-sac-fluid-ROI.vtu')
    np.savez_compressed(output/'fluid-ROI-mapping.npz', fluent_cell_index0=np.arange(len(volume)),
                        ellipsoid_mask=inside, volume_m3=volume, center_mm=CENTER_MM, radii_mm=RADII_MM)
    # Explicit original wall and volume-ROI boundary, not a modified CFD geometry.
    folders = selected(); plot = pv.Plotter(off_screen=True,window_size=(1400,900));plot.set_background('white')
    for zone in ['wall','aneurysm_sac']:
        s = pv.read(folders['patient_specific']/(zone+'.vtu'));s.points*=1000
        plot.add_mesh(s,color='#94a3b8',opacity=.15)
    plot.add_mesh(roi.extract_surface(algorithm='dataset_surface'),color='#d97706',opacity=.7)
    plot.add_mesh(pv.ParametricEllipsoid(*RADII_MM).translate(CENTER_MM),color='#2563eb',style='wireframe',opacity=.15)
    plot.add_text('Approved ellipsoidal fluid ROI (orange)',font_size=16,color='#172554')
    plot.add_text('Exploratory selection; not an anatomical neck segmentation',position='lower_left',font_size=11,color='#172554')
    plot.camera_position=[CENTER_MM+np.array([12,-65,20]),CENTER_MM,[0,0,1]];plot.enable_parallel_projection();plot.camera.parallel_scale=25
    plot.screenshot(output/'ellipsoidal-sac-ROI-overlay.png');plot.close()
    return volume, {'sac_ellipsoid':inside,'surrounding_fluid':~inside,'whole_fluid':np.ones(len(volume),bool)}


def native_fields(step):
    with h5py.File(step['output_stem']+'.dat.h5','r') as f:
        cells = f['results/1/phase-1/cells']
        v=np.column_stack([cells[n+'/1'][:].reshape(-1) for n in ['SV_U','SV_V','SV_W']])
        p=cells['SV_P/1'][:].reshape(-1)
    assert v.shape==(1169992,3) and p.shape==(1169992,) and np.isfinite(v).all() and np.isfinite(p).all()
    return v,p


def analyze_fields(output):
    output.mkdir(exist_ok=False)
    cfg={k:load(RUN/k/'input-configuration.json') for k in KINDS}
    steps={k:load(RUN/k/'progress.json')['accepted_steps'] for k in KINDS}
    times=np.array([s['time_s'] for s in steps['patient_specific']]); times[-1]=PERIOD
    assert len(times)==95 and all(np.allclose([s['time_s'] for s in steps[k]],times,rtol=0,atol=1e-10) for k in KINDS)
    volume,masks=geometry(output,cfg)
    wall=wall_module(); xyz,zones=wall.geometry(cfg['patient_specific']['source_case'])
    qtime=np.linspace(0,PERIOD,4000,endpoint=False);flow={};wr=[];wm=[]
    for k in KINDS:
        flow[k]={i:np.interp(qtime,cfg[k]['time_s'],cfg[k]['inlets'][i]['flow_m3_s'])*1e6 for i in ['inlet_b1','inlet_b2']}
        flow[k]['combined']=flow[k]['inlet_b1']+flow[k]['inlet_b2']
    for inlet in ['inlet_b1','inlet_b2','combined']:
        a=flow['patient_specific'][inlet]
        for j,t in enumerate(qtime): wr.append({'time_s':float(t),'inlet':inlet,**{k:float(flow[k][inlet][j]) for k in KINDS}})
        for k in KINDS[1:]:
            b=flow[k][inlet];rmse=float(np.sqrt(np.mean((b-a)**2)))
            wm.append({'candidate':k,'inlet':inlet,'rmse_ml_s':rmse,'nrmse_percent':100*rmse/np.ptp(a),
                       'ps_range_ml_s':float(np.ptp(a)),'ps_mean_ml_s':float(a.mean()),'candidate_mean_ml_s':float(b.mean()),'samples':len(a)})
    csvsave(output/'latest-waveforms-mL-s.csv',wr);csvsave(output/'latest-waveform-NRMSE.csv',wm)
    peak_times={k:float(qtime[np.argmax(flow[k]['combined'])]) for k in KINDS}
    peak_idx={k:int(np.argmin(abs(times-t))) for k,t in peak_times.items()}
    fluid_rows=[]; wall_rows=[]; wallcmp=[]; native_checks=[]
    wall_integral={k:{z:[] for z in zones} for k in KINDS}
    provenance=[]
    for j,t in enumerate(times):
        state={}
        for k in KINDS:
            step=steps[k][j]; v,p=native_fields(step);tau=wall.shear_vectors(step['output_stem']+'.dat.h5',zones)
            state[k]=(v,p,tau)
            path=Path(step['output_stem']+'.dat.h5')
            provenance.append({'case':k,'time_s':float(t),'file':str(path),'bytes':path.stat().st_size,'sha256':sha(path)})
            for z,g in zones.items():
                m=np.linalg.norm(tau[z],axis=1);ar=g['area'];native=step['comparison_wall_reports'][z]
                error=max(abs(np.average(m,weights=ar)/native['area_mean_wss_pa']-1),abs(m.max()/native['max_wss_pa']-1))
                assert error<1e-5
                native_checks.append(error);wall_integral[k][z].append(tau[z])
                wall_rows.append({'case':k,'time_s':float(t),'region':z,'area_mean_wss_pa':float(np.average(m,weights=ar)),
                                  'max_wss_pa':float(m.max()),'p99_wss_pa':wall.weighted_quantile(m,ar,.99)})
        a,pa,ta=state['patient_specific']
        for k in KINDS[1:]:
            b,pb,tb=state[k]
            for region,mask in masks.items():
                fluid_rows.append({'candidate':k,'time_s':float(t),'region':region,**fluid_comparison(a[mask],b[mask],pa[mask],pb[mask],volume[mask])})
            for z,g in zones.items():
                ma=np.linalg.norm(ta[z],axis=1);mb=np.linalg.norm(tb[z],axis=1);ar=g['area']
                am=float(np.average(ma,weights=ar));bm=float(np.average(mb,weights=ar));da=ma-am;db=mb-bm
                denom=float(np.sqrt(np.average(da*da,weights=ar)*np.average(db*db,weights=ar)))
                wallcmp.append({'candidate':k,'time_s':float(t),'region':z,'ps_mean_wss_pa':am,'candidate_mean_wss_pa':bm,
                    'mean_wss_signed_difference_pa':bm-am,'mean_wss_absolute_difference_percent':100*abs(bm-am)/am if am>.05 else None,
                    'ps_max_wss_pa':float(ma.max()),'candidate_max_wss_pa':float(mb.max()),
                    'area_weighted_wss_rmse_pa':float(np.average((mb-ma)**2,weights=ar))**.5,
                    'spatial_wss_correlation':float(np.average(da*db,weights=ar))/denom if denom>0 else None})
        if j%15==0: print('FIELDS',j,95,flush=True)
    csvsave(output/'regional-velocity-pressure-all-times.csv',fluid_rows)
    csvsave(output/'sac-and-vessel-WSS-all-times.csv',wall_rows)
    csvsave(output/'regional-WSS-comparisons-all-times.csv',wallcmp)
    save(output/'native-data-manifest.json',provenance)
    cycle=[];peak=[];cache={}
    for k in KINDS:
        for z,g in zones.items():
            tau=np.asarray(wall_integral[k][z]);mag=np.linalg.norm(tau,axis=2)
            tawss=periodic_average(mag,times);vec=periodic_average(tau,times)
            osi=.5*(1-np.linalg.norm(vec,axis=1)/np.maximum(tawss,1e-30))
            assert np.all((osi>=-1e-12)&(osi<=.5+1e-12))
            row={'case':k,'wall_region':z,'area_mm2':float(g['area'].sum()*1e6),
                 'area_mean_TAWSS_pa':float(np.average(tawss,weights=g['area'])),
                 'max_local_TAWSS_pa':float(tawss.max()),'max_instantaneous_WSS_pa':float(mag.max()),
                 'area_mean_OSI':float(np.average(osi,weights=g['area'])),
                 'max_OSI':float(osi.max()),'integration':'5-868 ms samples plus assumed 5-ms closure; first cycle'}
            cycle.append(row);cache[k+'_'+z+'_tawss']=tawss;cache[k+'_'+z+'_osi']=osi
    csvsave(output/'wall-cycle-TAWSS-OSI-summary.csv',cycle)
    np.savez_compressed(output/'wall-cycle-fields.npz',**cache)
    for k in KINDS[1:]:
        for region in masks:
            rows=[r for r in fluid_rows if r['candidate']==k and r['region']==region]
            mse=periodic_average(np.array([r['velocity_mse_m2_s2'] for r in rows]),times)
            ref=periodic_average(np.array([r['ps_velocity_energy_m2_s2'] for r in rows]),times)
            dp=periodic_average(np.array([r['mean_pressure_difference_pa'] for r in rows]),times)
            pm=periodic_average(np.array([r['pressure_mse_pa2'] for r in rows]),times)
            peak.append({'candidate':k,'region':region,'window':'cycle_assumed_5ms_closure','time_s':None,
                         'vector_velocity_rmse_m_s':float(np.sqrt(mse)), 'normalized_velocity_rmse_percent':float(100*np.sqrt(mse/ref)),
                         'mean_pressure_difference_pa':float(dp),'pressure_rmse_pa':float(np.sqrt(pm))})
            for clock,index in [('PS_inlet_peak',peak_idx['patient_specific']),('candidate_inlet_peak',peak_idx[k])]:
                r=rows[index]
                peak.append({'candidate':k,'region':region,'window':clock,'time_s':r['time_s'],
                             **{n:r[n] for n in ['vector_velocity_rmse_m_s','normalized_velocity_rmse_percent','mean_pressure_difference_pa','pressure_rmse_pa']}})
    csvsave(output/'peak-and-cycle-regional-comparisons.csv',peak)
    # WSS comparisons at the same reference clock, not independently phase-aligned fields.
    csvsave(output/'reference-peak-WSS-comparisons.csv',[r for r in wallcmp if abs(r['time_s']-times[peak_idx['patient_specific']])<1e-10])
    save(output/'analysis-summary.json',{'status':'completed_exploratory','source_run':str(RUN),
        'native_states_per_case':95,'periodicity_validated':False,'new_Fluent_cycles':0,
        'ROI_approval':'User approved existing ellipsoid for exploratory fluid measurements on 2026-10-29.',
        'ROI_center_mm':CENTER_MM.tolist(),'ROI_radii_mm':RADII_MM.tolist(),
        'fluid_regions':{n:{'cells':int(m.sum()),'volume_mm3':float(volume[m].sum()*1e9)} for n,m in masks.items()},
        'wall_regions':'Existing native aneurysm_sac and wall zones; no new wall segmentation.',
        'region_caveat':'Fluid ellipsoid and native sac wall patch are related selections, not an identical neck-bounded anatomical region. Surrounding fluid is the entire complementary fluid domain; untagged numerical extensions cannot be excluded separately.',
        'peak_target_s':peak_times,'peak_saved_s':{k:float(times[v]) for k,v in peak_idx.items()},
        'native_WSS_max_relative_error':max(native_checks),
        'pressure':'Gauge pressure, same B4 reference; not systemic blood pressure.',
        'cycle_caveat':'Period-average metrics assume a linear 5-ms closure. CFD cycle periodicity and mesh/time independence are unvalidated.'})


def residence(output):
    output.mkdir(exist_ok=False); summaries=[];all_survival=[];curves=[];cohorts=[]
    for kind,folder in selected().items():
        meta=load(folder/'tracking-audit.json'); n=meta['released']
        birth=np.full(n,np.nan);labels=np.zeros(n,int);first=np.full(n,np.nan);exit1=np.full(n,np.nan)
        last=np.full(n,np.nan);cumulative=np.zeros(n);entries=np.zeros(n,int);slow1=np.zeros(n);slow5=np.zeros(n)
        intervals={};prev=None
        with h5py.File(folder/'tracks.h5','r') as h:
            for j,key in enumerate(h['frames']):
                f=h['frames/'+key];t=float(f.attrs['time_s']);ids=f['id'][:].astype(int);p=f['position_m'][:];speed=f['speed_m_s'][:]
                birth[ids]=f['birth_time_s'][:];labels[ids]=f['inlet_id'][:];last[ids]=t
                assert np.all(~in_ellipsoid(p[np.abs(birth[ids]-t)<1e-10])), 'Inlet births unexpectedly inside sac ROI'
                if prev is not None:
                    oldt,oldid,oldp,oldspeed=prev
                    common,ia,ib=np.intersect1d(oldid,ids,return_indices=True)
                    lo,hi,ok=interval_inside(oldp[ia],p[ib]);ci=common[ok];start=oldt+(t-oldt)*lo[ok];end=oldt+(t-oldt)*hi[ok]
                    dt=end-start;new=~np.isfinite(first[ci]); first[ci[new]]=start[new]
                    isentry=(lo[ok]>1e-10)|(~in_ellipsoid(oldp[ia[ok]]))
                    # First detected residence counts even if starting exactly on the selection boundary.
                    entries[ci]+=isentry.astype(int)
                    entries[ci[new & (entries[ci]==0)]]=1
                    cumulative[ci]+=dt
                    avspeed=(oldspeed[ia[ok]]+speed[ib[ok]])/2
                    slow1[ci]+=dt*(avspeed<.01);slow5[ci]+=dt*(avspeed<.05)
                    isexit=(hi[ok]<1-1e-10)&(~np.isfinite(exit1[ci]));exit1[ci[isexit]]=end[isexit]
                    for ident,a,b in zip(ci,start,end):intervals.setdefault(int(ident),[]).append((float(a),float(b)))
                prev=(t,ids,p,speed)
                if j%250==0:print('RESIDENCE',kind,j,flush=True)
        assert np.isfinite(birth).all() and set(labels)=={1,2}
        rc=release_cycle(birth); event=np.isfinite(exit1);entered=np.isfinite(first)
        assert np.all(~event|entered) and np.all(exit1[event]>=first[event]-1e-10)
        rows=[]
        for i in range(n):
            has=bool(entered[i]);observed=bool(event[i]);stop=exit1[i] if observed else last[i]
            rows.append({'case':kind,'particle_id':i,'inlet_id':int(labels[i]),'release_cycle':int(rc[i]),'birth_time_s':float(birth[i]),
                'entered_ROI':has,'first_entry_s':float(first[i]) if has else None,'first_exit_s':float(exit1[i]) if observed else None,
                'first_exit_observed':observed,'first_pass_duration_or_censor_s':float(stop-first[i]) if has else None,
                'last_observed_s':float(last[i]),'cumulative_observed_ROI_time_s':float(cumulative[i]),
                'observed_entry_episodes':int(entries[i]),'observed_reentry':bool(entries[i]>=2),
                'ROI_time_below_0p01_m_s':float(slow1[i]),'ROI_time_below_0p05_m_s':float(slow5[i])})
        csvsave(output/(kind+'-particle-residence.csv'),rows)
        for cycle in range(1,11):
            mask=rc==cycle
            cohorts.append({'case':kind,'release_cycle':cycle,'released':int(mask.sum()),'entered_ROI':int((mask&entered).sum()),
                            'first_exits_observed':int((mask&event).sum()),'entered_but_first_exit_censored':int((mask&entered&~event).sum())})
        exterior_exits={e['id'] for e in meta['exit_events']}
        for scope,mask in [('all_entrants',entered),('first_replay_cycle_entrants',entered&(first<=START+PERIOD+1e-10))]:
            ix=np.flatnonzero(mask);duration=np.where(event[ix],exit1[ix],last[ix])-first[ix]
            sr=km(duration,event[ix]);all_survival.extend([{'case':kind,'cohort':scope,**r} for r in sr])
            clearance=lambda level:next((r['age_s'] for r in sr if r['survival']<=level),None)
            summaries.append({'case':kind,'cohort':scope,'injected_tracers':n,'ROI_entrants':int(len(ix)),
                'first_exits_observed':int(event[ix].sum()),'right_censored':int((~event[ix]).sum()),
                't50_first_exit_s':clearance(.5),'t90_first_exit_s':clearance(.1),
                'mean_observed_cumulative_ROI_time_s':float(cumulative[ix].mean()) if len(ix) else None,
                'reentry_tracer_fraction':float(np.mean(entries[ix]>=2)) if len(ix) else None,
                'low_speed_ROI_time_fraction_0p01_m_s':float(slow1[ix].sum()/cumulative[ix].sum()) if cumulative[ix].sum()>0 else None,
                'low_speed_ROI_time_fraction_0p05_m_s':float(slow5[ix].sum()/cumulative[ix].sum()) if cumulative[ix].sum()>0 else None,
                'interpretation':'Unweighted visual tracers in approved ellipsoid; first-cycle replay; individualized tracking unconverged.'})
            if scope=='first_replay_cycle_entrants':
                for age in np.linspace(0,8*PERIOD,201):
                    inside_count=0;unknown=0
                    for i in ix:
                        when=first[i]+age
                        if when>last[i]+1e-10:
                            # Exterior-domain exits are known terminal; final survivors are censored.
                            if i not in exterior_exits:unknown+=1
                            continue
                        inside_count+=any(a-1e-10<=when<=b+1e-10 for a,b in intervals.get(int(i),[]))
                    curves.append({'case':kind,'age_since_first_sac_entry_s':float(age),'cohort':'first_replay_cycle_entrants',
                        'cohort_size':len(ix),'inside':inside_count,'unknown':unknown,
                        'occupancy_lower_fraction':inside_count/len(ix) if len(ix) else None,
                        'occupancy_upper_fraction':(inside_count+unknown)/len(ix) if len(ix) else None})
        save(output/(kind+'-residence-metadata.json'),{'source_tracks':str(folder/'tracks.h5'),'sha256':sha(folder/'tracks.h5'),
             'geometry':'approved existing ellipsoid; pointwise selection, not a neck surface',
             'trajectory_interpolation':'Analytic ellipsoid crossings of straight segments between saved positions. Curved paths or repeated crossings within a saved interval can be missed.',
             'right_censoring':'Unknown first exit censored at last observed track time; never assigned an artificial ten-cycle residence.',
             'weighting':'Equal tracer weights, not mass/flux weighted; do not interpret as fraction of blood volume.',
             'clearance':'Descriptive Kaplan-Meier first-exit survival since first ROI entry; reentries reported separately.',
             'low_speed_thresholds_m_s':[.01,.05],'low_speed_caveat':'Operational sensitivity measures, not physiological definitions of stagnation.',
             'recirculation_caveat':'Reentry indicates return across the ROI boundary, not proof of a closed vortex.',
             'periodicity_validated':False,'tracking_refinement_passed':kind!='individualized'})
    csvsave(output/'particle-release-cohorts.csv',cohorts);csvsave(output/'residence-summary.csv',summaries)
    csvsave(output/'first-exit-retention-curves.csv',all_survival);csvsave(output/'first-cycle-entry-occupancy-curves.csv',curves)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--stage',choices=['fields','residence','all'],default='all');args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    if not (args.out/'self-tests.json').exists():save(args.out/'self-tests.json',self_tests())
    if args.stage in ['all','fields']:analyze_fields(args.out/'comparisons')
    if args.stage in ['all','residence']:residence(args.out/'residence')


if __name__=='__main__':main()
