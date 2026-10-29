"""E048 integrate saved trajectories against X-minus-REF gradients; no Bloch run."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import tomllib
import uuid
import h5py
import numpy as np

from scripts.aorta_hemodynamic_reference import read_pulseq
from scripts.build_aorta_inlet_picker import sha
from scripts.diagnose_phantom_signal import BASE,E047,ROOT,dump,voxel_indices
from scripts.compare_dp150_hemodynamics import REGIONS


def first_shot(path):
    data=read_pulseq(path);defs={r[0]:r[1:] for r in data['DEFINITIONS']}
    raster=float(defs['BlockDurationRaster'][0]);traps={int(r[0]):list(map(float,r[1:])) for r in data['TRAP']}
    rfs={int(r[0]):r for r in data['RF']};adcs={int(r[0]):r for r in data['ADC']}
    blocks=[];start=0;rf_center=None
    for row in data['BLOCKS']:
        row=list(map(int,row[:8]));end=start+row[1]*raster
        if row[2]:rf_center=start+(float(rfs[row[2]][5])+float(rfs[row[2]][6]))*1e-6;blocks=[]
        blocks.append((start,end,row[3:6]))
        if row[6]:
            adc=adcs[row[6]];echo=start+float(adc[3])*1e-6+(int(adc[1])//2)*float(adc[2])*1e-9
            break
        start=end
    if rf_center is None:raise ValueError('No RF before ADC')
    events=[];knots=[rf_center,echo]
    for start,end,ids in blocks:
        if ids[0]==0:continue
        amp,rise,flat,fall,delay=traps[ids[0]]
        times=start+np.array([delay,delay+rise,delay+rise+flat,delay+rise+flat+fall])*1e-6
        events.append((times,np.array([0,amp,amp,0])));knots.extend(times)
    def gradient(t):
        out=np.zeros_like(np.asarray(t,float))
        for times,values in events:out+=np.interp(t,times,values,left=0,right=0)
        return out
    return dict(gradient=gradient,knots=np.array(knots),echo=echo,rf_center=rf_center)


def quadrature(knots):
    knots=np.unique(np.sort(knots));left=knots[:-1];right=knots[1:]
    mid=(left+right)/2;half=(right-left)/2
    return np.column_stack((mid-half/np.sqrt(3),mid+half/np.sqrt(3))).ravel(),np.repeat(half,2)


def phase_kernel(seqdir):
    a=first_shot(seqdir/'ref.seq');b=first_shot(seqdir/'x.seq')
    if not np.isclose(a['echo'],b['echo'],atol=1e-9):raise ValueError('Echo mismatch')
    start=max(a['rf_center'],b['rf_center']);end=a['echo']
    knots=np.unique(np.r_[a['knots'],b['knots'],start,end]);knots=knots[(knots>=start)&(knots<=end)]
    points,weights=quadrature(knots);g=b['gradient'](points)-a['gradient'](points);tau=points-end
    moments=[float(np.sum(weights*g*tau**order)) for order in range(3)]
    if abs(moments[0])>1e-7 or not np.isclose(-10*moments[1],1,atol=2e-6):
        raise AssertionError('Gradient zero/first moment or sign not verified: '+str(moments))
    return a,b,knots,moments


def central_reset_exclusion(window_reset,reset_at_center,reset_after_center):
    """Central velocity spans BOTH neighboring intervals, unlike the phase integral."""
    return np.asarray(window_reset,bool)|np.asarray(reset_at_center,bool)|np.asarray(reset_after_center,bool)


def diagnose(output):
    if output.exists():raise FileExistsError(output)
    stage=output.with_name(output.name+'.building-'+uuid.uuid4().hex[:8]);stage.mkdir(parents=True)
    info=tomllib.loads((E047/'phantom_inputs_v1/case_info.toml').read_text());seqdir=Path(info['sequence_directory'])
    a,b,knots,moments=phase_kernel(seqdir)
    relative_knots=knots-a['echo'];period=info['period_s'];step=period/1440
    with h5py.File(BASE/'reference_v1/reference_fields.h5') as f:ref=f['fft_adc'][:];fraction=f['fraction_fft'][:]
    with h5py.File(BASE/'basic_v1/diagnostic_fields.h5') as f:mask=f['map100k_mask'][:].astype(bool)
    with h5py.File(E047/'phantom_analysis_v1/paired_fields.h5') as f:koma=f['map100k'][:,:,:,:,0]
    report=dict(experiment='E-048',status='COMPLETE_SAVED_TRAJECTORY_ENCODING_KERNEL_PROXY',
         created_utc=datetime.now(timezone.utc).isoformat(),gradient_M0_cycles_m=moments[0],
         gradient_M1_s_m=moments[1],gradient_M2_s2_m=moments[2],constant_velocity_gain=-10*moments[1],
         equivalent_acceleration_time_offset_s=moments[2]/(2*moments[1]),TE_s=a['echo']-a['rf_center'],
         regional=[],excluded_reset_spins_per_bin=[],
         limitation='Equal-spin instantaneous-RF trajectory/gradient proxy. No T1-history, full k-space or Bloch simulation; not a causal fraction or corrected MRI velocity.')
    summary={};proxy_mean=[];instant_mean=[];count_array=[]
    shape=fraction.shape;size=int(np.prod(shape));source_z=(np.arange(shape[2])-shape[2]//2)*.00125+.12699999287724495
    with h5py.File(E047/'phantom_inputs_v1/case_inputs.h5') as f:
        base=f['x'][:100000]
        basepoints=np.column_stack([f[c][:100000] for c in ('x','y','z')])
        for bin in range(15):
            center=(bin+.5)/15*period
            begin=center+relative_knots[0]
            trajectory_knots=np.arange(np.floor(begin/step),np.ceil(center/step)+1)*step-center
            cuts=np.unique(np.r_[relative_knots,trajectory_knots])
            cuts=cuts[(cuts>=relative_knots[0])&(cuts<=0)]
            points,weights=quadrature(cuts)
            gradient=b['gradient'](points+a['echo'])-a['gradient'](points+a['echo'])
            phase_velocity=np.zeros(100000,float);reset=np.zeros(100000,bool)
            for t,w,g in zip(points,weights,gradient):
                if g==0:continue
                u=(center+t)/period*1440;left=int(np.floor(u));alpha=u-left
                position=base+(1-alpha)*f['dx'][left,:100000]+alpha*f['dx'][left+1,:100000]
                phase_velocity+=-10*w*g*position
                reset|=f['reset'][left+1,:100000].astype(bool)
            index=int(round(center/period*1440))
            reset=central_reset_exclusion(reset,f['reset'][index,:100000],f['reset'][index+1,:100000])
            positions=basepoints+np.column_stack([f[c][index,:100000] for c in ('dx','dy','dz')])
            instant=(f['dx'][index+1,:100000]-f['dx'][index-1,:100000])/(2*step)
            ids,inside=voxel_indices(positions,shape,(.5,.08,.08),offset=0)
            good=inside&~reset
            ids,_=voxel_indices(positions[good],shape,(.5,.08,.08),offset=0)
            count=np.bincount(ids,minlength=size);count_array.append(count.reshape(shape).astype(np.uint16))
            mean=np.full(size,np.nan);original=np.full(size,np.nan)
            np.divide(np.bincount(ids,weights=phase_velocity[good],minlength=size),count,out=mean,where=count>0)
            np.divide(np.bincount(ids,weights=instant[good],minlength=size),count,out=original,where=count>0)
            proxy_mean.append(mean.reshape(shape));instant_mean.append(original.reshape(shape))
            report['excluded_reset_spins_per_bin'].append(int(reset.sum()))
    proxy=np.array(proxy_mean);instant=np.array(instant_mean);counts=np.array(count_array)
    valid=mask[None]&(fraction>0)[None]&np.isfinite(proxy)&np.isfinite(instant)
    for region,(low,high) in REGIONS.items():
        m=valid&((source_z>=low)&(source_z<high))[None,None,None,:]
        weights=np.broadcast_to(fraction,m.shape)[m].astype(float)
        if not weights.size:continue
        target=koma[m];cfd=ref[:,:,:,:,0][m];p=proxy[m];i=instant[m]
        rms=lambda x,y:float(np.sqrt(np.sum(weights*(x-y)**2)/weights.sum()))
        bias=lambda x,y:float(np.sum(weights*(x-y))/weights.sum())
        report['regional'].append(dict(region=region,space_time_samples=int(m.sum()),
             retained_space_time_fraction=float(weights.sum()/(15*fraction.sum())),
             Koma_vs_CFD_axial_rmse_m_s=rms(target,cfd),Koma_vs_instant_particles_axial_rmse_m_s=rms(target,i),
             Koma_vs_gradient_proxy_axial_rmse_m_s=rms(target,p),
             instantaneous_particle_bias_vs_CFD_m_s=bias(i,cfd),gradient_proxy_bias_vs_CFD_m_s=bias(p,cfd),
             Koma_bias_vs_CFD_m_s=bias(target,cfd),gradient_minus_instant_mean_m_s=bias(p,i)))
    dump(stage/'report.json',report)
    with h5py.File(stage/'kernel_fields.h5','w') as f:
        for name,value in dict(gradient_velocity_proxy=proxy,instantaneous_centered_velocity=instant,clean_counts=counts,
                               relative_gradient_knots_s=relative_knots).items():f.create_dataset(name,data=value,compression='gzip',compression_opts=1)
    sources=[Path(__file__).resolve(),seqdir/'ref.seq',seqdir/'x.seq',E047/'phantom_inputs_v1/case_inputs.h5',
             BASE/'reference_v1/reference_fields.h5',BASE/'basic_v1/diagnostic_fields.h5',E047/'phantom_analysis_v1/paired_fields.h5']
    dump(stage/'MANIFEST.json',dict(inputs={str(p):sha(p) for p in sources},outputs={p.name:sha(p) for p in stage.iterdir() if p.is_file()}))
    stage.rename(output)
    print(json.dumps(dict(status=report['status'],effective_time_s=report['equivalent_acceleration_time_offset_s'],regions=report['regional'])))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    diagnose(p.parse_args().output)
