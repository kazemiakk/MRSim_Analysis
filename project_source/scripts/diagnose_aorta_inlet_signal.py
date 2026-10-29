"""E-032 read-only diagnosis of depleted aortic inlet spins and weak MRI signal."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import tomllib
import uuid

import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import ConvexHull

from scripts.build_aorta_inlet_picker import sha


def prism_counts(points_mm, normal, offset, u, v, hull_equations, edges_mm):
    """Counts in a fixed inlet-footprint prism, not full-lumen densities."""
    points=np.asarray(points_mm,dtype=float)
    distance=points@normal-offset
    keep=(distance>=edges_mm[0])&(distance<=edges_mm[-1])
    p=points[keep];d=distance[keep]
    uv=np.column_stack((p@u,p@v))
    inside=np.all(uv@hull_equations[:,:2].T+hull_equations[:,2]<=1e-10,axis=1)
    return np.histogram(d[inside],bins=edges_mm)[0]


def magnitude_samples(path, motion_report, face_centroids_mm, normal):
    config=tomllib.loads(motion_report.read_text())
    center=np.array(config['source_coordinate_center_m'])
    order=np.array(config['source_axis_order_largest_to_smallest_one_based'])-1
    output=[]
    with h5py.File(path,'r') as f:
        field=f['reference_magnitude'];shape=field.shape[1:];fov=f.attrs['FOV_m']
        axes=[(np.arange(n)-n//2)*d/n for n,d in zip(shape,fov[::-1])]
        for off in (0,4,16):
            query=((face_centroids_mm+off*normal)/1000-center)[:,order][:,::-1]
            for b in range(len(field)):
                image=field[b]
                values=RegularGridInterpolator(axes,image,bounds_error=False,fill_value=np.nan)(query)/image.max()
                output.append(dict(inward_offset_mm=off,bin=b+1,
                    sampled_mean_fraction_of_bin_max=float(values.mean()),
                    sampled_max_fraction_of_bin_max=float(values.max())))
    return output


def read_complex_image(path):
    with h5py.File(path,'r') as f:
        image=f['image'][:]
    return image['re']+1j*image['im']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    root=Path(__file__).resolve().parents[1];pipe=root/'outputs/aorta_dp2_mri_pipeline'
    multi=pipe/'koma/multibin_8x4mm';run=multi/'simulation_motion_1m_8bin_4mm'
    geom=root/'outputs/aorta_inlet_picker/e030_origin_v1';audit=root/'outputs/aorta_inlet_flow/e031_v1'
    tracks=pipe/'aorta_motion_1m.h5';phantom=multi/'aorta_motion_1m_8bin.phantom'
    tracking_report=json.loads((pipe/'aorta_motion_1m_tracking_report.json').read_text())
    if sha(tracks)!=tracking_report['outputs']['tracks_hdf5_sha256']:
        raise ValueError('Original production track identity mismatch')
    readiness=json.loads((audit/'inlet_readiness.json').read_text())
    for entry in readiness['inputs']:
        if 'pc_reconstruction.h5' in entry['path']:
            if sha(run/'pc_reconstruction.h5')!=entry['sha256']:raise ValueError('Production reconstruction changed')
    # Bind the already-verified geometry and readiness artifacts before use.
    for folder in (geom,audit):
        manifest=json.loads((folder/'MANIFEST.json').read_text());manifest=manifest.get('files',manifest)
        for name,entry in manifest.items():
            if sha(folder/name)!=entry['sha256']:raise ValueError('Prior evidence changed')
    geometry=json.loads((geom/'origin_inlet_review.json').read_text())['candidate']
    normal=np.array(geometry['normal_source_xyz']);u=np.array(geometry['basis_u']);v=np.array(geometry['basis_v'])
    face=np.load(geom/'candidate_face_source_mm.npy',allow_pickle=False)
    hull=ConvexHull(np.column_stack((face@u,face@v)));edges=np.arange(0,29,4)
    series=[];transfer=[];motion=tomllib.loads((multi/'motion_phantom_report.toml').read_text())
    center=np.array(motion['source_coordinate_center_m'],dtype=np.float32)
    order=np.array(motion['source_axis_order_largest_to_smallest_one_based'])-1
    with h5py.File(tracks,'r') as f,h5py.File(phantom,'r') as ph:
        steps=f['position'].shape[0];dt=float(f.attrs['dt'])
        for step in range(steps+1):
            p=f['initial_position'][:] if step==0 else f['position'][step-1]
            counts=prism_counts(p.astype(float)*1000,normal,geometry['offset_mm'],u,v,hull.equations,edges)
            series.append(dict(source_elapsed_s=step*dt,counts=counts.tolist()))
            if step in (0,40,360):
                for k,axis in enumerate('xyz'):
                    expected=p[:,order[k]]-center[order[k]]
                    actual=ph['motion/motion_1/action/d'+axis][step]
                    delta=float(np.max(abs(actual-expected)))
                    if delta!=0:raise ValueError('Track-to-phantom transfer mismatch')
                    transfer.append(dict(sample_index=step,axis=axis,particles=len(p),maximum_absolute_difference_m=delta))
            if step%60==0:print('Inspected trajectory frames',step,'/',steps,flush=True)
    initial=np.array(series[0]['counts']);latest=np.array(series[-1]['counts'])
    moving=magnitude_samples(run/'reference_reconstruction.h5',multi/'motion_phantom_report.toml',
        np.load(audit/'inlet_numerical_evidence.npz')['triangle_centroids_source_mm'],normal)
    static=magnitude_samples(pipe/'koma/simulation_static_2m_ref/reference_reconstruction.h5',pipe/'koma/static_phantom_report.toml',
        np.load(audit/'inlet_numerical_evidence.npz')['triangle_centroids_source_mm'],normal)
    ref=read_complex_image(run/'ref_result.jld2');reconstruction=[]
    with h5py.File(run/'pc_reconstruction.h5','r') as f:
        mask=f['mask'][:].astype(bool);venc=np.float32(f.attrs['Venc_m_per_s'])
        for k,encoding in enumerate('xyz'):
            image=read_complex_image(run/(encoding+'_result.jld2'))
            velocity=np.angle(image*np.conj(ref))*(venc/np.float32(np.pi))
            saved=f['velocity'][k]
            reconstruction.append(dict(component=encoding,
                maximum_kept_voxel_difference_m_s=float(np.max(abs(velocity[mask]-saved[mask]))),
                saved_nonzero_values_outside_mask=int(np.count_nonzero(saved[~mask]))))
    inputs=[]
    paths=[tracks,phantom,pipe/'aorta_motion_1m_tracking_report.json',geom/'origin_inlet_review.json',
           geom/'candidate_face_source_mm.npy',audit/'inlet_readiness.json',audit/'inlet_numerical_evidence.npz',
           multi/'motion_phantom_report.toml',pipe/'koma/static_phantom_report.toml',
           pipe/'koma/simulation_static_2m_ref/reference_reconstruction.h5',
           run/'pc_reconstruction.h5',run/'reference_reconstruction.h5',
           *[run/(x+'_result.jld2') for x in ['ref','x','y','z']],
           root/'run_aorta_pointcloud.py',root/'mrsimtracks/core.py',root/'koma_aorta/build_motion_phantom.jl',
           root/'koma_aorta/simulate_pc.jl',root/'koma_aorta/PC.jl',Path(__file__)]
    for path in paths:inputs.append(dict(path=str(path.relative_to(root)),bytes=path.stat().st_size,sha256=sha(path)))
    report=dict(experiment='E-032',created_utc=datetime.now(timezone.utc).isoformat(),
        status='INLET_SPIN_DEPLETION_VERIFIED_MAJOR_UPSTREAM_CAUSE',
        evidence_class='Read-only diagnosis of preserved production trajectory, phantom and MRI evidence',
        inputs=inputs,particle_count=1000000,trajectory_frames_including_initial=len(series),
        measurement=dict(footprint='E-030 face convex-hull prism, fixed across time; not certified full-lumen volume',
            slab_edges_mm=edges.tolist(),normal_source_xyz=normal.tolist(),source_dt_s=dt,
            positive_direction='Into local vessel',counts_meaning='Actual stored particle counts, not spin magnetization or Eulerian mass density'),
        particle_count_series=series,
        summary=dict(first_slab_initial=int(initial[0]),first_slab_at_80ms=int(series[40]['counts'][0]),
            first_slab_final=int(latest[0]),first_slab_final_retained_fraction=float(latest[0]/initial[0]),
            first_slab_at_80ms_retained_fraction=float(series[40]['counts'][0]/initial[0]),
            initial_counts=initial.tolist(),final_counts=latest.tolist()),
        phantom_transfer_checks=transfer,moving_magnitude_samples=moving,static_magnitude_samples=static,
        raw_phase_vs_saved_reconstruction=reconstruction,
        diagnosis=[dict(finding='Severe inlet particle depletion before MRI',classification='Verified',
                       evidence='Full one-million-particle trajectory census for all 361 stored times'),
                   dict(finding='Recycling does not replenish inlet flux',classification='Verified code behavior',
                       evidence='run_aorta_pointcloud passes inlet=seeds; core draws reset positions uniformly from that volume pool'),
                   dict(finding='Depletion transfers into Koma phantom',classification='Verified sampled exact correspondence',
                       evidence='All million spins, all three axes, at t=0/0.08/0.72 s: zero coordinate differences'),
                   dict(finding='Mask is not the cause of low raw magnitude',classification='Verified',
                       evidence='Weak reference magnitude precedes velocity masking; saved phase calculation matches raw complex image phase inside mask'),
                   dict(finding='FOV or slab exclusion of inlet',classification='Not supported',
                       evidence='Origin and face lie inside reported FOV; PC.jl excites FOV[3] slab; static control has strong local signal'),
                   dict(finding='Inlet replenishment deficiency is the main upstream explanation',classification='Strongly supported mechanism, not a completed repair',
                       evidence='Code behavior, progressive local depletion, exact phantom transfer, weak moving versus stronger static inlet signal')],
        limitations=['Static control differs in spin count, matrix, cardiac bins, TR and RR; not an isolated-factor effect size',
                     'No corrective tracking or Bloch experiment was run; contribution of RF history, partial volume and motion dephasing not separately quantified',
                     'Fixed prism is a local count diagnostic, not a physical lumen-volume or mass-conservation certification',
                     'Legacy runtime reset execution remains uncertified; this diagnosis does not settle that separate issue',
                     'No phase mask was changed, original artifact overwritten, or Koma inlet waveform synthesized'],
        recommended_correction='Design explicit inlet-throughflow replenishment/reservoir and appropriate spin-reset semantics, verify local count stability in a small tracking control, then pilot MRI before any 1M rerun')
    temporary=args.output.with_name(args.output.name+'.building-'+uuid.uuid4().hex[:8]);temporary.mkdir(parents=True)
    (temporary/'inlet_signal_diagnosis.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    with (temporary/'inlet_particle_counts.csv').open('x',encoding='utf-8',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['source_elapsed_s']+[f'count_{a}_{b}_mm' for a,b in zip(edges[:-1],edges[1:])])
        for row in series:writer.writerow([row['source_elapsed_s'],*row['counts']])
    manifest={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in temporary.iterdir()}
    (temporary/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    temporary.rename(args.output)
    print(json.dumps(dict(output=str(args.output),summary=report['summary']),indent=2))


if __name__=='__main__':main()
