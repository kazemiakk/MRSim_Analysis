"""E048 independent timing/coordinate reference sensitivities; preserve old results."""
import argparse
from dataclasses import dataclass
from datetime import datetime,timezone
import json
from pathlib import Path
import tomllib
import uuid
import xml.etree.ElementTree as ET

import h5py
import numpy as np

from scripts.build_aorta_inlet_picker import sha
from scripts.dp150_hemodynamic_reference import MRIGrid,read_e022_contract,load_periodic_reference,build_voxel_lookup
from scripts.analyze_remapped_confirmation import weighted_metrics
from scripts.diagnose_phantom_signal import BASE,E047,PROD,REFERENCE,ROOT,dump
from scripts.compare_dp150_hemodynamics import REGIONS
from scripts.aorta_hemodynamic_reference import read_pulseq


@dataclass(frozen=True)
class CenteredFFTGrid(MRIGrid):
    def axis_centers_m(self):
        return tuple((np.arange(n)-n//2)*d for n,d in zip(self.shape_spr[::-1],self.spacing_rps_m))


def catmull_weights(phases,frames=72):
    """Match io._interp_time, including its1e-3 near-knot snapping."""
    x=np.mod(np.asarray(phases,float),1)*frames
    left=np.floor(x).astype(int);s=x-left
    s=np.where(s<.001,0,np.where(s>.999,1,s));s2=s*s;s3=s2*s
    basis=[.5*(-s+2*s2-s3),.5*(2-5*s2+3*s3),.5*(s+4*s2-3*s3),.5*(-s2+s3)]
    weights=np.zeros(frames,float)
    for shift,values in zip((-1,0,1,2),basis):
        weights+=np.bincount((left+shift)%frames,weights=values,minlength=frames)
    return weights/len(x)


def timing_weights(path,period,seam):
    with h5py.File(path) as f:
        g=f['REF'];starts=g['adc_first_s'][:];dwell=g['dwell_s'][:];n=g['n'][:];labels=g['labels'][:]
        for e in ('X','Y','Z'):
            if not np.array_equal(labels,f[e]['labels'][:]) or not np.allclose(starts,f[e]['adc_first_s'][:],atol=1e-9,rtol=0):
                raise ValueError('Encoding timing differs')
    weights=[];weights32=[];metadata=[]
    for b in range(15):
        ids=np.flatnonzero(labels[:,2]==b)
        if len(ids)!=2048 or not np.all(n[ids]==400):raise ValueError('Missing15-bin encoding labels')
        # Saved Julia sorter uses the actual ADC first sample, not half-dwell shifted timestamps.
        times=(starts[ids,None]+np.arange(400)[None,:]*dwell[ids,None]).ravel()
        phase=np.mod(times-seam,period)/period
        # Precision sensitivity, not a claim to reproduce every Julia interpolation intermediate.
        phase32=(np.mod(times.astype(np.float32)-np.float32(seam),np.float32(period))/np.float32(period)).astype(float)
        weights.append(catmull_weights(phase));weights32.append(catmull_weights(phase32))
        midpoint=(b+.5)/15
        error=((phase-midpoint+.5)%1-.5)*period
        metadata.append(dict(bin=b+1,ADC_samples=len(times),nominal_source_phase_s=midpoint*.72,
                 mean_offset_from_midpoint_s=float(error.mean()),minimum_offset_s=float(error.min()),maximum_offset_s=float(error.max()),
                 float32_vs_float64_phase_rms_s=float(np.sqrt(np.mean((((phase32-phase+.5)%1-.5)*period)**2)))))
    return np.array(weights),np.array(weights32),metadata


def fft_controls(grid):
    rows=[]
    for axis,(n,fov) in enumerate(zip(grid.shape_spr[::-1],grid.fov_rps_m)):
        for shift in (-3,0,2):
            coordinate=shift*fov/n;k=(np.arange(n)-n//2)/fov
            image=np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(np.exp(-2j*np.pi*k*coordinate))))
            peak=int(np.argmax(abs(image)));wanted=n//2+shift
            if peak!=wanted or not np.isclose(abs(image[peak]),1,atol=1e-12):raise AssertionError('Analytic DFT control')
            rows.append(dict(axis=axis,shift=shift,peak_index=peak,coordinate_m=coordinate,
                             FFT_coordinate_m=float((peak-n//2)*fov/n),legacy_reference_coordinate_m=float((peak+.5-n//2)*fov/n)))
    return rows


def averaged_complex(lookup,nodes,venc=5.):
    """Instantaneous constant-velocity voxel proxy, NOT a full Bloch response."""
    samples=np.einsum('ni,nij->nj',lookup.barycentric,nodes[lookup.vertex_indices])
    count=len(lookup.flat_indices);coherence=[];circular=[]
    for c in range(3):
        real=np.bincount(lookup.sample_voxel,weights=np.cos(np.pi*samples[:,c]/venc),minlength=count)
        imag=np.bincount(lookup.sample_voxel,weights=np.sin(np.pi*samples[:,c]/venc),minlength=count)
        value=(real+1j*imag)/lookup.counts
        coherence.append(lookup.scatter(abs(value).astype(np.float32)))
        circular.append(lookup.scatter((np.angle(value)*venc/np.pi).astype(np.float32)))
    return np.stack(coherence,axis=-1),np.stack(circular,axis=-1)


def audit(output):
    if output.exists():raise FileExistsError(output)
    stage=output.with_name(output.name+'.building-'+uuid.uuid4().hex[:8]);stage.mkdir(parents=True)
    pipe=ROOT/'outputs/dp150_mri_pipeline'
    support=PROD/'final_image_analysis/support_constrained_phase/dp150_seam_safe_support_phase.h5'
    # Locate the exact accepted derivative from E025's declared input path if needed.
    if not support.exists():
        from scripts.compare_dp150_hemodynamics import H5
        support=H5
    grid,timing=read_e022_contract(pipe/'motion_1m_report.json',PROD/'simulation_report.toml',support)
    fftgrid=CenteredFFTGrid(grid.shape_spr,grid.fov_rps_m,grid.source_center_m,grid.source_axis_order)
    info=tomllib.loads((E047/'phantom_inputs_v1/case_info.toml').read_text())
    pvd=pipe/'flow_periodic/dp150_periodic.pvd'
    if sha(pvd)!='342c2cfd195048ee1a469df10d986a6219a8bbac6ea967c83cfff9b8436ae39a':raise ValueError('Wrong matched CFD')
    files=[pvd.parent/x.attrib['file'] for x in ET.parse(pvd).findall('.//DataSet')]
    sources=[pvd,*files,Path(__file__).resolve(),ROOT/'scripts/dp150_hemodynamic_reference.py',ROOT/'mrsimtracks/io.py',
             BASE/'PROTOCOL.md',REFERENCE,E047/'phantom_inputs_v1/legacy_timing.h5',
             E047/'phantom_analysis_v1/paired_fields.h5',E047/'phantom_analysis_v1/masks.npz']
    dump(stage/'registration.json',dict(experiment='E-048',created_utc=datetime.now(timezone.utc).isoformat(),
         status='REGISTERED_BEFORE_REFERENCE_SENSITIVITY',inputs={str(p):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sources}))
    adc,adc32,tmetadata=timing_weights(E047/'phantom_inputs_v1/legacy_timing.h5',info['period_s'],info['seam_time_s'])
    mid=np.array([catmull_weights(np.array([(b+.5)/15])) for b in range(15)])
    if not np.allclose(adc.sum(axis=1),1,atol=1e-12) or not np.allclose(mid.sum(axis=1),1,atol=1e-12):raise AssertionError('Weights not unity')
    print('E048 load original matched periodic CFD',flush=True)
    flow=load_periodic_reference(pvd)
    print('E048 build existing half-voxel geometry lookup',flush=True)
    half=build_voxel_lookup(flow.active_mesh,grid,subsamples=2)
    print('E048 build centered-FFT geometry lookup',flush=True)
    fft=build_voxel_lookup(flow.active_mesh,fftgrid,subsamples=2)
    with h5py.File(REFERENCE) as f:
        old_fraction=f['volume_fraction_2'][:];old_ref=f['cfd_voxel_midpoint'][:]
        old_q3=f['cfd_voxel_3x3x3'][:];old_three=f['cfd_voxel_three_sample_mean'][:]
    if not np.array_equal(half.fraction_grid(),old_fraction):raise AssertionError('Existing geometry not reproduced')
    allweights=np.concatenate([mid,adc,adc32]);nodes=np.zeros((45,flow.active_mesh.n_points,3),np.float64)
    for frame in range(72):
        velocity=grid.velocity_to_mri(np.asarray(flow._frame_vel(frame)))*timing['velocity_scale']
        nodes+=allweights[:,frame,None,None]*velocity[None]
    arrays={k:np.empty((15,*grid.shape_spr,3),np.float32) for k in ('half_mid','half_adc','fft_mid','fft_adc','fft_adc_f32')}
    coherence=np.empty_like(arrays['fft_mid']);circular=np.empty_like(coherence)
    for b in range(15):
        print('E048 evaluate bin',b+1,flush=True)
        for name,lookup,index in [('half_mid',half,b),('half_adc',half,b+15),('fft_mid',fft,b),('fft_adc',fft,b+15),('fft_adc_f32',fft,b+30)]:
            arrays[name][b]=np.moveaxis(lookup.average(nodes[index])['velocity_mri'],0,-1)
        coherence[b],circular[b]=averaged_complex(fft,nodes[b+15])
    match=np.isfinite(old_ref)&np.isfinite(arrays['half_mid'])
    delta=float(np.max(abs(old_ref[match]-arrays['half_mid'][match])))
    if delta>5e-5:raise AssertionError('Existing midpoint reference not reproduced: '+str(delta))
    arrays['half_q3_mid']=old_q3;arrays['half_3sample']=old_three
    with h5py.File(E047/'phantom_analysis_v1/paired_fields.h5') as f:
        fields={a:f[a][:] for a in ('legacy100k','map100k')}
    masks=np.load(E047/'phantom_analysis_v1/masks.npz',allow_pickle=False)
    fraction=np.minimum(half.fraction_grid(),fft.fraction_grid())
    common=masks['pair100k']&(fraction>0)
    source_z=grid.axis_centers_m()[0]+grid.source_center_m[2]
    regional=[]
    for region,(low,high) in REGIONS.items():
        mask=common&((source_z>=low)&(source_z<high))[None,None,:]
        for reference,ref in arrays.items():
            for arm,field in fields.items():
                metrics=weighted_metrics(ref,field,fraction,mask)
                regional.append(dict(region=region,reference=reference,arm=arm,
                     numerical_volume_weight=float(fraction[mask].sum()),metrics=metrics))
    q=circular-arrays['fft_adc'];valid=common&np.isfinite(q).all(axis=(0,4))
    proxy=dict(description='Idealized instantaneous uniform-density constant-velocity complex averaging of8subvoxels; not full Bloch or TKE',
               circular_vs_linear_vector_rmse_m_s=None if not valid.any() else float(np.sqrt(np.mean(np.sum(q[:,valid].astype(float)**2,axis=-1)))),
               mean_component_coherence=None if not valid.any() else np.mean(coherence[:,valid],axis=(0,1)).tolist(),
               minimum_component_coherence=None if not valid.any() else np.min(coherence[:,valid],axis=(0,1)).tolist())
    # Confirm the readout frequency increment independently from saved Pulseq amplitudes/dwell.
    sections=read_pulseq(Path(info['sequence_directory'])/'ref.seq')
    block=next(row for row in sections['BLOCKS'] if int(row[6])>0)
    trap={int(row[0]):row for row in sections['TRAP']};adcs={int(row[0]):row for row in sections['ADC']}
    dk=float(trap[int(block[3])][1])*float(adcs[int(block[6])][2])*1e-9
    if not np.isclose(abs(dk),1/grid.fov_rps_m[0],rtol=2e-6):raise AssertionError('Readout/FOV mismatch')
    report=dict(experiment='E-048',status='COMPLETE_DECLARED_REFERENCE_SENSITIVITIES',
         analytical_FFT_controls=fft_controls(grid),readout_delta_k_cycles_m=dk,expected_delta_k_cycles_m=1/grid.fov_rps_m[0],
         midpoint_reproduction_max_component_delta_m_s=delta,timing=tmetadata,
         physical_grid_offset_m=(grid.spacing_rps_m*.5).tolist(),
         common_volume_fraction_of_old_numerical_support=float(fraction[common].sum()/old_fraction.sum()),
         results=regional,complex_proxy=proxy,
         voxel_mean_component_outside_VENC5=int(np.sum(abs(arrays['fft_adc'][:,common])>5)),
         limitation='Declared reference sensitivities; no fitted alignment, changed acquisition, independent validation or old report overwrite.')
    dump(stage/'report.json',report)
    with h5py.File(stage/'reference_fields.h5','w') as f:
        for name,a in arrays.items():f.create_dataset(name,data=a,compression='gzip',compression_opts=1)
        for name,a in dict(fraction_half=half.fraction_grid(),fraction_fft=fft.fraction_grid(),common_mask=common,
                           common_fraction=fraction,complex_coherence=coherence,circular_proxy_velocity=circular,
                           adc_weights=adc,adc_weights_f32=adc32,midpoint_weights=mid).items():
            f.create_dataset(name,data=a,compression='gzip',compression_opts=1)
    dump(stage/'MANIFEST.json',dict(outputs={p.name:sha(p) for p in stage.iterdir() if p.is_file()}))
    stage.rename(output)
    print(json.dumps(dict(status=report['status'],global_map100k=[r for r in regional if r['region']=='whole' and r['arm']=='map100k'],proxy=proxy)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    audit(p.parse_args().output)
