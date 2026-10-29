"""E047: errors WITH retained coverage for frozen100k remapped confirmations.

No fitting, phase imputation, source modification or changed support criterion.
Saved references and masks make the numerical calculations independently auditable.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import tomllib
import uuid

import h5py
import numpy as np
from scipy.spatial import cKDTree

from scripts.build_aorta_inlet_picker import sha
from scripts.aorta_hemodynamic_reference import AortaGrid, build_lookup, temporal_node_values
from scripts.integrate_aorta_inlet_complex import complex_velocity_samples

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'outputs/remapped_confirmation_e047'
OLD = ROOT/'outputs/upstream_migration_e046'
ENC = ('REF','X','Y','Z')
ARMS = ('legacy100k','map20k','map100k')
LABEL = dict(legacy100k='Legacy100k',map20k='Map20k',map100k='Map100k')
COLOR = dict(legacy100k='#626973',map20k='#235f82',map100k='#bc721d')


def toml(path):
    return tomllib.loads(Path(path).read_text(encoding='utf-8'))


def dump(path, value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def fixed_mask(images, threshold):
    if set(images)!=set(ENC) or len({a.shape for a in images.values()})!=1:
        raise ValueError('Four matched encodings required')
    valid = np.ones(images['REF'].shape[1:],bool)
    for image in images.values():
        magnitude = abs(image)
        valid &= ((magnitude >= threshold*magnitude.max(axis=(1,2,3),keepdims=True)) & (magnitude>0)).all(axis=0)
    return valid


def velocities(images, venc):
    return np.stack([np.angle(images[e]*images['REF'].conj())*(venc/np.pi) for e in ENC[1:]],axis=-1)


def weighted_metrics(reference, estimate, fraction, mask):
    good = mask & (fraction>0) & np.isfinite(reference).all(axis=(0,4)) & np.isfinite(estimate).all(axis=(0,4))
    if not good.any():
        return None
    a=reference[:,good].astype(float); b=estimate[:,good].astype(float); w=fraction[good].astype(float)
    total=w.sum()*len(a); delta=b-a
    rms=lambda v:float(np.sqrt(np.sum(v*w[None,:])/total))
    vector=rms(np.sum(delta**2,axis=-1)); ref_rms=rms(np.sum(a**2,axis=-1))
    speed_a=np.linalg.norm(a,axis=-1);speed_b=np.linalg.norm(b,axis=-1)
    return dict(voxels=int(good.sum()),vector_rmse_m_s=vector,
                vector_nrmse_reference_rms=None if ref_rms==0 else vector/ref_rms,
                reference_vector_rms_m_s=ref_rms,speed_rmse_m_s=rms((speed_b-speed_a)**2),
                mean_speed_bias_m_s=float(np.sum((speed_b-speed_a)*w[None,:])/total),
                component_rmse_m_s=[rms(delta[...,c]**2) for c in range(3)])


def flow_metrics(a,b,period):
    a=np.asarray(a,float);b=np.asarray(b,float)
    if len(a)==0 or not np.isfinite(a).all() or not np.isfinite(b).all():
        return None
    cycle_a=a.sum()*period/len(a);cycle_b=b.sum()*period/len(b)
    return dict(rmse_ml_s=float(np.sqrt(np.mean((b-a)**2))),
                correlation=None if a.std()==0 or b.std()==0 else float(np.corrcoef(a,b)[0,1]),
                cycle_CFD_ml=float(cycle_a),cycle_Koma_ml=float(cycle_b),
                cycle_volume_error_fraction=None if cycle_a==0 else float(cycle_b/cycle_a-1),
                maximum_absolute_bin_error_ml_s=float(np.max(abs(b-a))),
                range_nrmse=None if np.ptp(a)==0 else float(np.sqrt(np.mean((b-a)**2))/np.ptp(a)))


def load_images(case,arm,inputs,paths):
    folder=inputs if arm=='legacy100k' else (OLD/f'{case}_updated_map_v1' if arm=='map20k' else BASE/f'{case}_mapped_v1')
    info=toml(inputs/'case_info.toml')
    report=None if arm=='legacy100k' else toml(folder/'control_report.toml')
    if report:
        if report['case']!=case or report['arm']!='updated_map' or report['physical_particles']!=(20000 if arm=='map20k' else 100000):
            raise ValueError('Control identity mismatch')
        if arm=='map100k' and (report['status']!='COMPLETE_E047_100K_REMAPPED' or report['input_sha256']!=sha(inputs/'case_inputs.h5')):
            raise ValueError('Incomplete/unmatched100k control')
        paths.append(folder/'control_report.toml')
    images={}; fft={}
    for e in ENC:
        p=folder/(e.lower()+('_legacy_result.h5' if arm=='legacy100k' else '_result.h5'))
        if report and sha(p)!=report['result_sha256'][e]:
            raise ValueError('Result identity changed')
        with h5py.File(p) as f:
            k=f['rawsort'][:]; a=f['image'][:]
            if not np.isfinite(k).all() or not np.isfinite(a).all():
                raise ValueError('Nonfinite saved data')
            transformed=np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(k,axes=(1,2,3)),axes=(1,2,3)),axes=(1,2,3))
            fft[e]=float(abs(a-transformed).max()/max(float(abs(a).max()),1e-30))
            if fft[e]>3e-6:
                raise ValueError('Independent FFT mismatch')
            if arm=='legacy100k':
                original=Path(info['source_output'])/(e.lower()+'_result.jld2')
                stored=f.attrs['source_result_sha256'];stored=stored.decode() if isinstance(stored,bytes) else stored
                if sha(original)!=info['legacy_result_sha256'][e] or stored!=info['legacy_result_sha256'][e]:
                    raise ValueError('Legacy result provenance mismatch')
                paths.append(original)
            images[e]=a
        paths.append(p)
    return images,fft


def reference(case, info, stage, paths):
    if case=='phantom':
        source=ROOT/'outputs/dp150_hemodynamic_comparison/e025_v1/results_v2/matched_reference.h5'
        manifest=source.parent/'MANIFEST.json'
        identities={row['file']:row['sha256'] for row in json.loads(manifest.read_text())}
        if sha(source)!=identities[source.name]:
            raise ValueError('Phantom reference changed')
        paths.extend([source,manifest])
        with h5py.File(source) as f:
            if float(f.attrs['effective_period_s'])!=info['period_s']:
                raise ValueError('Reference period changed')
            ref=f['cfd_voxel_midpoint'][:]; alternate=f['cfd_voxel_three_sample_mean'][:]
            fraction=f['volume_fraction_2'][:]; area=f['plane_area_fraction_2'][:,:,70]*3.125
            times=f['time_s'][:]; readout=f['source_z_m'][:]
        return ref,alternate,fraction,dict(area_mm2=area,time_s=times,readout_m=readout,
                 reference='CFD midpoint; three-sample timing sensitivity (not ADC-exact)',section_index=70,
                 spacing_m=[.00125,.00125,.0025])
    # New native-zero16-bin reference, not the old eight-bin acquisition average.
    from scripts.compare_aorta_hemodynamics import EXPECTED,DATA
    for name in ('point_coordinates.npy','velocities.npy'):
        if sha(DATA/name)!=EXPECTED[name]:
            raise ValueError('Original aortic source changed')
        paths.append(DATA/name)
    timing=ROOT/'outputs/aorta_inlet_replenishment/e045_native0_timing_v1.json'
    weight=timing.with_suffix('.npz');paths.extend([timing,weight])
    weights=np.load(weight,allow_pickle=False)['native_frame_weights']
    if weights.shape!=(16,72) or not np.allclose(weights.sum(axis=1),1,atol=1e-12):
        raise ValueError('Native-zero temporal weights invalid')
    grid=AortaGrid();coords=np.load(DATA/'point_coordinates.npy',mmap_mode='r',allow_pickle=False)
    values=np.load(DATA/'velocities.npy',mmap_mode='r',allow_pickle=False)
    lookup=build_lookup(cKDTree(coords),(grid),(2,2,2))
    ref=lookup.evaluate(temporal_node_values(values,weights,order=grid.order,scale=.72/info['period_s']))
    fraction=lookup.fraction
    times=np.array([b['source_center_s'] for b in json.loads(timing.read_text())['encodings']['REF']['source_bins']])
    with h5py.File(stage/'aorta_native0_reference.h5','w') as f:
        f['cfd_actual_adc_average']=ref;f['numerical_fraction']=fraction;f['time_s']=times;f['weights']=weights
        f.attrs['evidence_class']='Same-source, approximate point-cloud numerical support; NOT native lumen'
    return ref,None,fraction,dict(time_s=times,readout_m=grid.axes()[0],spacing_m=grid.spacing.tolist(),
                   reference='CFD actual ADC-time average, native frame0,2x2x2 numerical quadrature')


def flow_aorta(images, info, threshold, paths):
    inlet=ROOT/'outputs/aorta_inlet_flow/e031_v1/inlet_numerical_evidence.npz'
    if sha(inlet)!='11a76b06940052d0b13361560b8ec85581c9c0a701eaaf5e895afdea701ca5c0':
        raise ValueError('Inferred inlet changed')
    paths.append(inlet);z=np.load(inlet,allow_pickle=False);grid=AortaGrid()
    points=((z['triangle_centroids_source_mm']/1000-grid.center)[:,grid.order])[:,::-1]
    area=z['triangle_area_mm2'];normal=z['normal_source'][list(grid.order)]
    vn=np.einsum('ncf,c->nf',z['face_velocities_source_m_s'],z['normal_source'])
    native=vn[z['inferred_triangles']].mean(axis=1)
    weights=np.load(ROOT/'outputs/aorta_inlet_replenishment/e045_native0_timing_v1.npz',allow_pickle=False)['native_frame_weights']
    sample={};masks={}
    for arm in ARMS:
        v,ok,_=complex_velocity_samples(images[arm],grid.axes()[::-1],points,info['Venc_m_s'],threshold)
        sample[arm]=v;masks[arm]=ok.all(axis=0)
    joint=np.logical_and.reduce(list(masks.values()))
    def evaluate(mask):
        if not mask.any():
            return None,{a:None for a in ARMS}
        cfd=weights@(area[mask]@native[mask])*(.72/info['period_s'])
        q={a:np.einsum('bnc,c,n->b',sample[a][:,mask],normal,area[mask]) for a in ARMS}
        return cfd,q
    return area,masks,joint,evaluate,dict(method='Complex interpolation BEFORE phase on inferred origin face; actual ADC CFD',
                full_native_CFD_ml_s=(area@native).tolist(),native_time_s=(np.arange(72)*.01).tolist())


def flow_phantom(fields,masks,ref,alternate,metadata):
    area=metadata['area_mm2'];own={a:masks[a][:,:,70]&(area>0) for a in ARMS}
    joint=np.logical_and.reduce(list(own.values()))
    def evaluate(mask):
        if not mask.any():
            return None,{a:None for a in ARMS}
        cfd=ref[:,:,:,70,0][:,mask]@area[mask]
        return cfd,{a:fields[a][:,:,:,70,0][:,mask]@area[mask] for a in ARMS}
    return area,own,joint,evaluate,dict(method='Original upstream section70; midpoint CFD, not newly measured inlet',
             alternate_CFD_full_ml_s=(alternate[:,:,:,70,0][:,area>0]@area[area>0]).tolist())


def plots(case,result,images,fields,reference_field,masks,fraction,metadata,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):
        fig.savefig(out/f'{case}_{name}.png',dpi=400)
        fig.savefig(out/f'{case}_{name}.pdf')
        plt.close(fig)
    primary=result['thresholds'][0];flow=primary['flow'];t=np.asarray(result['time_s'])
    fig,axs=plt.subplots(2,2,figsize=(12,8))
    for arm in ('legacy100k','map100k'):
        q=flow['pair100k_Koma_ml_s'][arm]
        if q is not None:
            axs[0,0].plot(t,q,'o--',label=LABEL[arm],color=COLOR[arm],ms=3)
            axs[0,1].plot(t,np.asarray(q)-flow['pair100k_CFD_ml_s'],'o-',color=COLOR[arm],ms=3)
    if flow['pair100k_CFD_ml_s'] is not None:
        axs[0,0].plot(t,flow['pair100k_CFD_ml_s'],'k-',label='CFD',lw=1.5)
    axs[0,0].set(title=f"100k paired flow | shared area {flow['pair100k_coverage_fraction']:.1%}",ylabel='Flow (mL/s)',xlabel='Cycle time (s)')
    axs[0,0].legend(frameon=False,fontsize=8,ncol=2)
    axs[0,1].axhline(0,color='.5',lw=.7)
    axs[0,1].set(title='Flow residual | same100k paired area',ylabel='Koma − CFD (mL/s)',xlabel='Cycle time (s)')
    for arm in ARMS:
        coverage=[row['volume']['own'][arm]['coverage_fraction']*100 for row in result['thresholds']]
        err=[row['volume']['own'][arm]['metrics'] for row in result['thresholds']]
        axs[1,0].plot([5,10,15,20],coverage,'o-',color=COLOR[arm],label=LABEL[arm])
        axs[1,1].plot([5,10,15,20],[np.nan if v is None else v['vector_rmse_m_s'] for v in err],'o-',color=COLOR[arm],label=LABEL[arm])
    axs[1,0].set(title='Own numerical-volume support',xlabel='Signal threshold (%)',ylabel='Retained volume (%)',ylim=(0,105))
    axs[1,0].legend(frameon=False,fontsize=8)
    axs[1,1].set(title='Own-support velocity error | coverage varies',xlabel='Signal threshold (%)',ylabel='Vector RMSE (m/s)')
    for ax in axs.flat:ax.grid(axis='y',alpha=.2)
    fig.suptitle(f'{case.capitalize()} | 100k remapped confirmation',x=.08,ha='left')
    fig.text(.08,.015,'Same-source internal consistency. Coverage-dependent errors are not whole-lumen accuracy or independent validation.',fontsize=8,color='.35')
    fig.tight_layout(rect=(0,.045,1,.95));save(fig,'confirmation')
    # One physical plane, fixed speed limits, no projections or missing-phase filling.
    joint=masks['legacy100k']&masks['map100k']&(fraction>0)
    cfd_speed=np.linalg.norm(reference_field,axis=-1)
    score=np.nansum(np.where(joint,cfd_speed,0)*fraction,axis=(1,2,3))
    peak=int(np.argmax(score));low=int(np.argmin(score))
    plane=reference_field.shape[1]//2
    # Geometry-only slice selection: maximum supported native quadrature per plane.
    plane=int(np.argmax(fraction.sum(axis=(1,2))))
    vmax=max(float(np.nanmax(cfd_speed[b][joint])) for b in (peak,low)) if joint.any() else 1
    for a in ('legacy100k','map100k'):
        if joint.any():vmax=max(vmax,max(float(np.max(np.linalg.norm(fields[a][b][joint],axis=-1))) for b in (peak,low)))
    fig,axs=plt.subplots(2,3,figsize=(13,5.8),squeeze=False)
    for row,b in enumerate((peak,low)):
        data=[np.where(joint[plane],cfd_speed[b,plane],np.nan)]+[
              np.where(joint[plane],np.linalg.norm(fields[a][b,plane],axis=-1),np.nan) for a in ('legacy100k','map100k')]
        for col,(title,values) in enumerate(zip(['CFD','Legacy100k','Map100k'],data)):
            im=axs[row,col].imshow(values,origin='lower',cmap='viridis',vmin=0,vmax=vmax,aspect='auto',interpolation='nearest')
            axs[row,col].set(title=f'{title} | bin{b+1}',xlabel='Readout index',ylabel='Phase index')
    fig.colorbar(im,ax=axs.ravel().tolist(),label='Speed (m/s)',fraction=.02,pad=.02)
    fig.suptitle(f'{case.capitalize()} | true MRI slice {plane}, CFD-selected peak/low;100k paired support',x=.06,ha='left')
    fig.subplots_adjust(left=.06,right=.87,bottom=.12,top=.86,hspace=.5,wspace=.35)
    save(fig,'velocity_slices')


def analyze(case, output):
    if output.exists():raise FileExistsError(output)
    inputs=BASE/f'{case}_inputs_v1';info=toml(inputs/'case_info.toml')
    if info['physical_particles']!=100000 or sha(inputs/'case_inputs.h5')!=info['neutral_input_sha256']:
        raise ValueError('Frozen100k input required')
    frozen=json.loads((BASE/'source_registration.json').read_text())
    for path,identity in frozen['inputs'].items():
        if sha(ROOT/path)!=identity['sha256']:raise ValueError('Frozen source changed: '+path)
    stage=output.with_name(output.name+'.building-'+uuid.uuid4().hex[:8]);stage.mkdir(parents=True)
    paths=[BASE/'PROTOCOL.md',BASE/'source_registration.json',inputs/'case_info.toml',inputs/'case_inputs.h5',
           inputs/'legacy_timing.h5',Path(__file__).resolve(),BASE/f'{case}_map_v1/map_report.json',BASE/f'{case}_map_v1/cycle_map.h5']
    images={};fft={};fields={}
    for arm in ARMS:
        images[arm],fft[arm]=load_images(case,arm,inputs,paths)
        fields[arm]=velocities(images[arm],info['Venc_m_s'])
    ref,alternate,fraction,meta=reference(case,info,stage,paths)
    if any(field.shape!=ref.shape for field in fields.values()):raise ValueError('Reference/grid mismatch')
    result=dict(experiment='E-047',case=case,status='COMPLETE_100K_SAME_SOURCE_CONFIRMATION',
        created_utc=datetime.now(timezone.utc).isoformat(),evidence_class='Internal consistency and descriptive density/remapping sensitivity',
        physical_particles=dict(legacy100k=100000,map20k=20000,map100k=100000),FFT_max_relative_errors=fft,
        time_s=meta['time_s'].tolist(),reference=meta['reference'],thresholds=[],
        numerical_support_volume_ml=float(fraction.sum()*np.prod(meta['spacing_m'])*1e6),
        limitations=['Map plus endpoint-reset-policy change, not version-only effect.',
                     'NN ensemble remapping is many-to-one, not exact conservation.',
                     'Support-dependent comparisons are not independent physical validation.',
                     'No1M run authorized; pressure/WSS/TKE remain unavailable.'])
    if case=='aorta':result['limitations'].append('Aortic support/face is inferred, not native boundary-resolved geometry.')
    if case=='phantom':result['limitations'].append('Scanner-derived, geometry-adapted,VENC-adjusted; CFD timing is midpoint/three-sample, synthetic tail preserved.')
    save_masks={}
    for threshold in (.05,.10,.15,.20):
        masks={a:fixed_mask(images[a],threshold) for a in ARMS}
        pair=masks['legacy100k']&masks['map100k']&(fraction>0)
        joint=np.logical_and.reduce(list(masks.values()))&(fraction>0)
        own={a:dict(voxels=int((m&(fraction>0)).sum()),
                   coverage_fraction=float(fraction[m].sum()/fraction.sum()),
                   metrics=weighted_metrics(ref,fields[a],fraction,m)) for a,m in masks.items()}
        volume=dict(own=own,pair100k_coverage_fraction=float(fraction[pair].sum()/fraction.sum()),
                    pair100k_metrics={a:weighted_metrics(ref,fields[a],fraction,pair) for a in ('legacy100k','map100k')},
                    three_arm_coverage_fraction=float(fraction[joint].sum()/fraction.sum()),
                    three_arm_metrics={a:weighted_metrics(ref,fields[a],fraction,joint) for a in ARMS})
        flow_parts=flow_aorta(images,info,threshold,paths) if case=='aorta' else flow_phantom(fields,masks,ref,alternate,meta)
        area,fmasks,fjoint,evaluate,extra=flow_parts
        cfd,q=evaluate(fjoint)
        fpair=fmasks['legacy100k']&fmasks['map100k'];pcfd,pq=evaluate(fpair)
        flow=dict(whole_section_area_mm2=float(area.sum()),three_arm_area_mm2=float(area[fjoint].sum()),
                  three_arm_coverage_fraction=float(area[fjoint].sum()/area.sum()),
                  three_arm_CFD_ml_s=None if cfd is None else cfd.tolist(),
                  three_arm_Koma_ml_s={a:None if q[a] is None else q[a].tolist() for a in ARMS},
                  three_arm_metrics={a:None if cfd is None else flow_metrics(cfd,q[a],info['period_s']) for a in ARMS},
                  pair100k_coverage_fraction=float(area[fpair].sum()/area.sum()),
                  pair100k_CFD_ml_s=None if pcfd is None else pcfd.tolist(),
                  pair100k_Koma_ml_s={a:None if pq[a] is None else pq[a].tolist() for a in ('legacy100k','map100k')},
                  pair100k_metrics={a:None if pcfd is None else flow_metrics(pcfd,pq[a],info['period_s']) for a in ('legacy100k','map100k')},
                  own={},**extra)
        for arm,mask in fmasks.items():
            own_cfd,own_q=evaluate(mask)
            flow['own'][arm]=dict(coverage_fraction=float(area[mask].sum()/area.sum()),area_mm2=float(area[mask].sum()),
                CFD_ml_s=None if own_cfd is None else own_cfd.tolist(),Koma_ml_s=None if own_q[arm] is None else own_q[arm].tolist(),
                metrics=None if own_cfd is None else flow_metrics(own_cfd,own_q[arm],info['period_s']))
        result['thresholds'].append(dict(signal_threshold=threshold,volume=volume,flow=flow))
        if threshold==.05:
            for a in ARMS:save_masks[a]=masks[a]
            save_masks['pair100k']=pair;save_masks['three_arm']=joint
            primary_masks=masks
    np.savez_compressed(stage/'masks.npz',**save_masks)
    with h5py.File(stage/'paired_fields.h5','w') as f:
        f['reference']=ref;f['numerical_volume_fraction']=fraction;f['time_s']=meta['time_s']
        for arm in ARMS:f[arm]=fields[arm]
        f.attrs['array_axes']='bin,slice,phase,readout,component';f.attrs['velocity_units']='m/s'
    mapreport=json.loads((BASE/f'{case}_map_v1/map_report.json').read_text())
    oldmap=json.loads((OLD/f'{case}_map_v1/map_report.json').read_text())
    result['mapping']={n:dict(map_distance_m=r['map_distance_m'],unique_source_fraction=r['unique_source_fraction'],
                         maximum_source_multiplicity=r['maximum_source_multiplicity']) for n,r in [('20k',oldmap),('100k',mapreport)]}
    result['mapping']['median_distance_to_min_voxel_ratio_100k']=mapreport['map_distance_m']['median']/min(meta['spacing_m'])
    dump(stage/'report.json',result)
    plots(case,result,images,fields,ref,primary_masks,fraction,meta,stage)
    registration=dict(experiment='E-047',case=case,input_identities={str(p):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in dict.fromkeys(paths)},
                      outputs={p.name:sha(p) for p in stage.iterdir() if p.is_file()})
    dump(stage/'MANIFEST.json',registration);stage.rename(output)
    print(json.dumps(dict(case=case,status=result['status'],primary_flow=result['thresholds'][0]['flow']['own']['map100k'],
                         primary_velocity=result['thresholds'][0]['volume']['own']['map100k'])))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',choices=['aorta','phantom'],required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();analyze(a.case,a.output)
