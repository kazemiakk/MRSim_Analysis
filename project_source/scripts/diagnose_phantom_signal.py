"""E048 read-only source diagnosis: signal-gate decomposition and spin occupancy."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import tomllib
import uuid

import h5py
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.stats import spearmanr
from scripts.build_aorta_inlet_picker import sha
from scripts.analyze_remapped_confirmation import velocities, weighted_metrics

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/phantom_diagnosis_e048'
E047=ROOT/'outputs/remapped_confirmation_e047'
PROD=ROOT/'outputs/dp150_mri_pipeline/koma/simulation_motion_1m_scanner_matched_venc5_seam_safe_v1'
REFERENCE=ROOT/'outputs/dp150_hemodynamic_comparison/e025_v1/results_v2/matched_reference.h5'
ENC=('REF','X','Y','Z')


def dump(path,data):
    Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def read_complex(dataset):
    value=dataset[:]
    if value.dtype.fields is not None:
        if set(value.dtype.names)!= {'re','im'}:raise ValueError('Unknown complex representation')
        value=value['re']+1j*value['im']
    if not np.iscomplexobj(value) or not np.isfinite(value).all():raise ValueError('Finite complex data required')
    return value


def read_text(dataset):
    value=dataset[()]
    return value.decode() if isinstance(value,bytes) else str(value)


def signal_decomposition(images,fraction,area,threshold=.05):
    masks={}; minimum={}; binpass={}
    for e,image in images.items():
        magnitude=abs(image); maximum=magnitude.max(axis=(1,2,3),keepdims=True)
        ratio=magnitude/maximum
        binpass[e]=(ratio>=threshold)&(magnitude>0)
        masks[e]=binpass[e].all(axis=0);minimum[e]=ratio.min(axis=0)
    valid=fraction>0;total=float(fraction.sum());shared=np.logical_and.reduce(list(masks.values()))
    w=lambda m:float(np.sum(fraction[m],dtype=float)/total)
    patterns=np.zeros(fraction.shape,np.uint8)
    for i,e in enumerate(ENC):patterns|=((~masks[e]).astype(np.uint8)<<i)
    buckets=[]
    for value in range(16):
        m=valid&(patterns==value)
        buckets.append(dict(failed_encodings=[e for i,e in enumerate(ENC) if value&(1<<i)],
                            voxels=int(m.sum()),volume_fraction=w(m)))
    perbin=[]
    for b in range(len(images['REF'])):
        all_enc=np.logical_and.reduce([binpass[e][b] for e in ENC])
        perbin.append(dict(bin=b+1,all_encoding_volume_fraction=w(all_enc),
                          **{e:w(binpass[e][b]) for e in ENC}))
    own=shared&(area>0)[:,:,None] if False else shared[:,:,70]&(area>0)
    result=dict(fixed_all_bin_all_encoding_volume_fraction=w(shared),
                fixed_upstream_area_fraction=float(area[own].sum()/area.sum()),
                fixed_encoding_volume_fraction={e:w(masks[e]) for e in ENC},
                REF_failure_volume_fraction=w(~masks['REF']),
                REF_pass_but_encoding_failure_fraction=w(masks['REF']&~shared),
                all_encoding_per_bin=perbin,failure_patterns=buckets,
                failed_bin_histogram=[])
    pass_all=np.logical_and.reduce(list(binpass.values()))
    failed=(~pass_all).sum(axis=0)
    for n in range(len(images['REF'])+1):result['failed_bin_histogram'].append(dict(failed_bins=n,volume_fraction=w(failed==n)))
    return result,shared,minimum,pass_all


def voxel_indices(points,shape,fov,offset=.5):
    """Nearest physical center. offset=.5 is old reference,0 is centered FFT."""
    size=np.asarray(shape[::-1]);spacing=np.asarray(fov)/size
    index=np.floor(points/spacing+size/2-offset+.5).astype(np.int64)
    good=np.all((index>=0)&(index<size),axis=1)
    flat=np.ravel_multi_index(index[good][:,::-1].T,shape)
    return flat,good


def occupancy(inputfile,shape,fov,period,offset=.5):
    n=100000;size=int(np.prod(shape));counts=[];means=[];dephasing=[];excluded=[]
    with h5py.File(inputfile) as f:
        base=np.column_stack([f[c][:n] for c in ('x','y','z')]);steps=f['dx'].shape[0]-1
        for b in range(15):
            phase=(b+.5)/15;where=phase*steps;left=int(np.floor(where));alpha=where-left
            a=np.column_stack([f[c][left,:n] for c in ('dx','dy','dz')])
            z=np.column_stack([f[c][left+1,:n] for c in ('dx','dy','dz')])
            points=base+(1-alpha)*a+alpha*z
            speed=(z-a)/(period/steps)
            reset=f['reset'][left+1,:n].astype(bool)
            flat,inside=voxel_indices(points,shape,fov,offset)
            count=np.bincount(flat,minlength=size);counts.append(count.reshape(shape).astype(np.uint16))
            good=(~reset)&inside
            ids,_=voxel_indices(points[good],shape,fov,offset)
            clean=np.bincount(ids,minlength=size)
            mean=np.full((size,3),np.nan,np.float32);coherence=np.full((size,3),np.nan,np.float32)
            for c in range(3):
                velocity=speed[good,c]
                np.divide(np.bincount(ids,weights=velocity,minlength=size),clean,out=mean[:,c],where=clean>0)
                real=np.bincount(ids,weights=np.cos(np.pi*velocity/5),minlength=size)
                imag=np.bincount(ids,weights=np.sin(np.pi*velocity/5),minlength=size)
                np.divide(np.hypot(real,imag),clean,out=coherence[:,c],where=clean>0)
            means.append(mean.reshape(*shape,3));dephasing.append(coherence.reshape(*shape,3))
            excluded.append(dict(bin=b+1,reset_crossing_spins=int(reset.sum()),outside_FOV=int((~inside).sum())))
    return np.array(counts),np.array(means),np.array(dephasing),excluded


def diagnose(output):
    if output.exists():raise FileExistsError(output)
    stage=output.with_name(output.name+'.building-'+uuid.uuid4().hex[:8]);stage.mkdir(parents=True)
    infofile=E047/'phantom_inputs_v1/case_info.toml';info=tomllib.loads(infofile.read_text())
    inputs=[BASE/'PROTOCOL.md',Path(__file__).resolve(),REFERENCE,infofile,E047/'phantom_inputs_v1/case_inputs.h5',
            E047/'phantom_inputs_v1/legacy_timing.h5',E047/'phantom_analysis_v1/report.json',PROD/'simulation_report.toml']
    folders=dict(map20k=ROOT/'outputs/upstream_migration_e046/phantom_updated_map_v1',
                 legacy100k=E047/'phantom_inputs_v1',map100k=E047/'phantom_mapped_v1',legacy1m=PROD)
    for arm,folder in folders.items():
        inputs.extend(folder/(e.lower()+('_result.jld2' if arm=='legacy1m' else '_legacy_result.h5' if arm=='legacy100k' else '_result.h5')) for e in ENC)
    registration=dict(experiment='E-048',status='REGISTERED_BEFORE_DIAGNOSTICS',created_utc=datetime.now(timezone.utc).isoformat(),
                      inputs={str(p):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in inputs})
    dump(stage/'registration.json',registration)
    with h5py.File(REFERENCE) as f:
        fraction=f['volume_fraction_2'][:];area=f['plane_area_fraction_2'][:,:,70]*3.125
        ref=f['cfd_voxel_midpoint'][:];z=f['source_z_m'][:]
    prodreport=tomllib.loads((PROD/'simulation_report.toml').read_text())
    oldreport=tomllib.loads((Path(info['source_output'])/'simulation_report.toml').read_text())
    if prodreport['motion_timing']['seam_alignment']!=oldreport['motion_timing']['seam_alignment']:
        raise ValueError('Old1M timing differs; density comparison not qualified')
    results={};raw100k=None;mask100k=None;ratio100k=None; pass100k=None;legacy1m_mask=None
    for arm,folder in folders.items():
        print('E048 SIGNAL',arm,flush=True);images={}
        for e in ENC:
            name=e.lower()+('_result.jld2' if arm=='legacy1m' else '_legacy_result.h5' if arm=='legacy100k' else '_result.h5')
            with h5py.File(folder/name) as f:
                images[e]=read_complex(f['image'])
                if arm=='legacy1m':
                    if read_text(f['phantom_sha256'])!=prodreport['phantom_sha256'] or read_text(f['sequence_sha256'])!=info['sequence_sha256'][e]:
                        raise ValueError('Old1M source identity mismatch')
        result,mask,minimum,passbin=signal_decomposition(images,fraction,area)
        field=velocities(images,info['Venc_m_s'])
        result['vector_error_own']=weighted_metrics(ref,field,fraction,mask)
        distance=distance_transform_edt(fraction>0,sampling=(.0025,.00125,.00125))
        result['geometry_sensitivity']=[]
        for label,selection in [('all',fraction>0),('full_fraction',fraction>=1-1e-6),('voxelized_core_2_5mm',distance>.0025)]:
            keep=mask&selection
            result['geometry_sensitivity'].append(dict(mask=label,retained_fraction_of_eligible=float(fraction[keep].sum()/fraction[selection].sum()),
                     fraction_of_full_numerical_volume=float(fraction[keep].sum()/fraction.sum()),metrics=weighted_metrics(ref,field,fraction,keep)))
        results[arm]=result
        if arm=='map100k':raw100k=field;mask100k=mask;ratio100k=minimum;pass100k=passbin
        if arm=='legacy1m':legacy1m_mask=mask
        del images
    prior=json.loads((E047/'phantom_analysis_v1/report.json').read_text())
    if not np.isclose(results['map100k']['fixed_all_bin_all_encoding_volume_fraction'],prior['thresholds'][0]['volume']['own']['map100k']['coverage_fraction'],atol=1e-6):
        raise AssertionError('E047 coverage reproduction failed')
    fov=(.5,.08,.08);counts,particle,coherence,excluded=occupancy(E047/'phantom_inputs_v1/case_inputs.h5',fraction.shape,fov,info['period_s'])
    stats=[]
    for b,count in enumerate(counts):
        valid=fraction>0
        for name,selector in [('zero',count==0),('one',count==1),('2to3',(count>=2)&(count<=3)),('4plus',count>=4)]:
            m=valid&selector
            stats.append(dict(bin=b+1,occupancy=name,numerical_volume_fraction=float(fraction[m].sum()/fraction.sum()),
                              signal_pass_fraction_of_group=None if not m.any() else float(fraction[m&pass100k[b]].sum()/fraction[m].sum())))
    mean_count=counts.mean(axis=0);minimum_ratio=np.minimum.reduce(list(ratio100k.values()))
    eligible=(fraction>0)&np.isfinite(mean_count)&np.isfinite(minimum_ratio)
    corr=spearmanr(mean_count[eligible],minimum_ratio[eligible])
    summary=dict(status='COMPLETE_SAVED_SIGNAL_AND_OCCUPANCY_DIAGNOSIS',cases=results,
                 bin_center_occupancy=dict(mean_count_over_numerical_voxels=float(mean_count[eligible].mean()),
                    zero_count_volume_fraction_per_bin=[float(fraction[c==0].sum()/fraction.sum()) for c in counts],
                    occupancy_signal_spearman_r=float(corr.statistic),occupancy_bins=stats,reset_exclusions=excluded,
                    limitation='Instantaneous15-bin-center proxy, not RF/ADC-weighted occupancy; correlation is not causal proof.'),
                 no_new_simulation=True,old1m_is_not_remapped=True)
    dump(stage/'report.json',summary)
    with h5py.File(stage/'diagnostic_fields.h5','w') as f:
        for name,value in dict(volume_fraction=fraction,map100k_mask=mask100k,legacy1m_mask=legacy1m_mask,
            voxelized_distance_m=distance,bin_center_counts=counts,particle_mean_velocity=particle,
            particle_complex_coherence=coherence,minimum_signal_ratio=minimum_ratio,all_encoding_bin_pass=pass100k).items():
            f.create_dataset(name,data=value,compression='gzip',compression_opts=1)
    dump(stage/'MANIFEST.json',dict(inputs=registration['inputs'],outputs={p.name:sha(p) for p in stage.iterdir() if p.is_file()}))
    stage.rename(output)
    print(json.dumps(dict(status=summary['status'],coverage={k:dict(volume=v['fixed_all_bin_all_encoding_volume_fraction'],area=v['fixed_upstream_area_fraction'],REF=v['fixed_encoding_volume_fraction']['REF']) for k,v in results.items()},occupancy_r=summary['bin_center_occupancy']['occupancy_signal_spearman_r'])))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    diagnose(p.parse_args().output)
