"""E048 saved legacy1M/100k local controls and centered-FFT plane comparisons."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import tomllib
import uuid

import h5py
import numpy as np
from scipy.ndimage import distance_transform_edt

from scripts.diagnose_phantom_signal import BASE,E047,PROD,ROOT,read_complex,dump,occupancy,ENC
from scripts.audit_phantom_reference import CenteredFFTGrid
from scripts.dp150_hemodynamic_reference import read_e022_contract,load_periodic_reference,build_section_lookup
from scripts.compare_dp150_hemodynamics import REGIONS,PLANES,H5
from scripts.analyze_remapped_confirmation import weighted_metrics,velocities,flow_metrics
from scripts.build_aorta_inlet_picker import sha


def pooled_vector(reference,estimate,fraction,mask):
    keep=np.asarray(mask)&np.isfinite(reference).all(axis=-1)&np.isfinite(estimate).all(axis=-1)
    weights=np.broadcast_to(fraction,keep.shape)
    if not keep.any():return None
    w=weights[keep].astype(float);a=reference[keep].astype(float);b=estimate[keep].astype(float)
    error=float(np.sqrt(np.sum(w*np.sum((b-a)**2,axis=-1))/w.sum()))
    return dict(vector_rmse_m_s=error,retained_space_time_fraction=float(w.sum()/(len(reference)*fraction.sum())),
                samples=int(keep.sum()))


def analyze(output):
    if output.exists():raise FileExistsError(output)
    stage=output.with_name(output.name+'.building-'+uuid.uuid4().hex[:8]);stage.mkdir(parents=True)
    basic=BASE/'basic_v1';audit=BASE/'reference_v1'
    frozen=json.loads((basic/'registration.json').read_text())['inputs']
    for e in ENC:
        p=PROD/(e.lower()+'_result.jld2')
        if sha(p)!=frozen[str(p)]['sha256']:raise ValueError('Preserved1M changed')
    registration=dict(created_utc=datetime.now(timezone.utc).isoformat(),experiment='E-048',status='REGISTERED_BEFORE_LOCAL_DIAGNOSTICS',
        input_sha256={str(p):sha(p) for p in [Path(__file__).resolve(),BASE/'PROTOCOL.md',basic/'report.json',basic/'diagnostic_fields.h5',
                  audit/'report.json',audit/'reference_fields.h5',E047/'phantom_analysis_v1/paired_fields.h5']})
    dump(stage/'registration.json',registration)
    with h5py.File(audit/'reference_fields.h5') as f:
        ref=f['fft_adc'][:];fraction=f['fraction_fft'][:];weights=f['adc_weights'][:];old_fraction=f['fraction_half'][:]
    with h5py.File(basic/'diagnostic_fields.h5') as f:
        masks=dict(map100k=f['map100k_mask'][:].astype(bool),legacy1m=f['legacy1m_mask'][:].astype(bool))
    maskfile=np.load(E047/'phantom_analysis_v1/masks.npz',allow_pickle=False)
    masks['legacy100k']=maskfile['legacy100k']
    with h5py.File(E047/'phantom_analysis_v1/paired_fields.h5') as f:
        fields={a:f[a][:] for a in ('legacy100k','map100k')}
    images={}
    for e in ENC:
        with h5py.File(PROD/(e.lower()+'_result.jld2')) as f:images[e]=read_complex(f['image'])
    fields['legacy1m']=velocities(images,5.);del images
    grid,_=read_e022_contract(ROOT/'outputs/dp150_mri_pipeline/motion_1m_report.json',PROD/'simulation_report.toml',H5)
    fftgrid=CenteredFFTGrid(grid.shape_spr,grid.fov_rps_m,grid.source_center_m,grid.source_axis_order)
    z=fftgrid.axis_centers_m()[0]+grid.source_center_m[2]
    pair=np.logical_and.reduce(list(masks.values()))&(fraction>0)
    rows=[];core=distance_transform_edt(fraction>0,sampling=grid.spacing_rps_m[::-1])>.0025
    for name,(low,high) in REGIONS.items():
        region=((z>=low)&(z<high))[None,None,:]&(fraction>0)
        for arm,field in fields.items():
            own=masks[arm]&region;shared=pair&region
            rows.append(dict(region=name,arm=arm,
                own_coverage_fraction=float(fraction[own].sum()/fraction[region].sum()),
                own_metrics=weighted_metrics(ref,field,fraction,own),
                shared_coverage_fraction=float(fraction[shared].sum()/fraction[region].sum()),
                shared_metrics=weighted_metrics(ref,field,fraction,shared)))
    print('E048 particle occupancy on centered FFT grid',flush=True)
    info=tomllib.loads((E047/'phantom_inputs_v1/case_info.toml').read_text())
    counts,particle,coherence,excluded=occupancy(E047/'phantom_inputs_v1/case_inputs.h5',grid.shape_spr,grid.fov_rps_m,info['period_s'],offset=0)
    particle_valid=np.isfinite(particle).all(axis=-1)&masks['map100k'][None]&(fraction>0)[None]
    particle_comparison=dict(Koma_vs_CFD=pooled_vector(ref,fields['map100k'],fraction,particle_valid),
        particles_vs_CFD=pooled_vector(ref,particle,fraction,particle_valid),
        Koma_vs_particles=pooled_vector(particle,fields['map100k'],fraction,particle_valid),
        counts_groups=[],limitation='Instantaneous equal-spin bin-center means, not a full signal response; covariance is not a causal attribution')
    for region,(low,high) in {'whole':REGIONS['whole'],'upstream':REGIONS['upstream'],'throat':REGIONS['throat'],'near_downstream':REGIONS['near_downstream']}.items():
        eligible=((z>=low)&(z<high))[None,None,:]
        for name,group in [('zero',counts==0),('one',counts==1),('2to3',(counts>=2)&(counts<=3)),('4plus',counts>=4)]:
            m=masks['map100k'][None]&eligible[None]&group
            particle_comparison['counts_groups'].append(dict(region=region,occupancy=name,
                Koma_vs_CFD=pooled_vector(ref,fields['map100k'],fraction,m),
                particles_vs_CFD=pooled_vector(ref,particle,fraction,m)))
    # New section references use zero-thickness source planes on the derived FFT grid.
    print('E048 true-plane geometry lookup on centered FFT grid',flush=True)
    flow=load_periodic_reference(ROOT/'outputs/dp150_mri_pipeline/flow_periodic/dp150_periodic.pvd')
    plane=build_section_lookup(flow.active_mesh,fftgrid,transverse_subsamples=2)
    nodes=np.zeros((15,flow.active_mesh.n_points,3),float)
    for frame in range(72):
        values=grid.velocity_to_mri(np.asarray(flow._frame_vel(frame)))*(.72/info['period_s'])
        nodes+=weights[:,frame,None,None]*values[None]
    plane_ref=np.stack([np.moveaxis(plane.average(node)['velocity_mri'],0,-1) for node in nodes])
    plane_fraction=plane.fraction_grid();unit_area=float(grid.spacing_rps_m[1]*grid.spacing_rps_m[2]*1e6)
    sections=[]
    for name,target in PLANES.items():
        i=int(np.argmin(abs(z-target)));area=plane_fraction[:,:,i]*unit_area
        shared=area>0
        for mask in masks.values():shared&=mask[:,:,i]
        for arm,field in fields.items():
            for policy,m in [('own',(area>0)&masks[arm][:,:,i]),('shared_three_arm',shared)]:
                cfd=plane_ref[:,:,:,i,0][:,m]@area[m]
                koma=field[:,:,:,i,0][:,m]@area[m]
                sections.append(dict(section=name,index=i,source_z_m=float(z[i]),arm=arm,support=policy,
                    area_mm2=float(area[m].sum()),geometric_area_mm2=float(area.sum()),coverage_fraction=float(area[m].sum()/area.sum()),
                    CFD_ml_s=cfd.tolist(),Koma_ml_s=koma.tolist(),metrics=flow_metrics(cfd,koma,info['period_s'])))
    peak=int(np.nanargmax(np.nansum(np.linalg.norm(ref,axis=-1)*np.where(masks['map100k'],fraction,0)[None],axis=(1,2,3))))
    case_summary={a:next(r for r in rows if r['region']=='whole' and r['arm']==a) for a in fields}
    report=dict(experiment='E-048',status='COMPLETE_LOCAL_DENSITY_AND_PLANE_DIAGNOSTICS',created_utc=datetime.now(timezone.utc).isoformat(),
        reference='Centered inverse-DFT grid, actual saved ADC-time averaged retimed CFD, zero-thickness section quadrature',
        cases=case_summary,regions=rows,sections=sections,particle_comparison=particle_comparison,
        mapped_core_error=weighted_metrics(ref,fields['map100k'],fraction,masks['map100k']&core),peak_bin=peak+1,
        limitations=['Preserved1M is legacy, not remapped; density comparisons are descriptive, not convergence.',
                     'Own-area errors differ in denominator. Shared comparisons use identical support.',
                     'Phase filling, fitted shift/gain, smoothing and new MRI acquisition were not performed.',
                     'New reference sensitivities do not silently replace E025/E047 or establish independent validation.'])
    dump(stage/'report.json',report)
    with h5py.File(stage/'local_fields.h5','w') as f:
        for name,value in dict(plane_reference=plane_ref,plane_fraction=plane_fraction,legacy1m_velocity=fields['legacy1m'],
                 shared_mask=pair,FFT_bin_center_counts=counts,FFT_particle_velocity=particle,FFT_particle_coherence=coherence,source_z_fft_m=z).items():
            f.create_dataset(name,data=value,compression='gzip',compression_opts=1)
    dump(stage/'MANIFEST.json',dict(outputs={p.name:sha(p) for p in stage.iterdir() if p.is_file()}))
    stage.rename(output)
    print(json.dumps(dict(status=report['status'],cases=case_summary,particles=particle_comparison['particles_vs_CFD'])))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    analyze(p.parse_args().output)
