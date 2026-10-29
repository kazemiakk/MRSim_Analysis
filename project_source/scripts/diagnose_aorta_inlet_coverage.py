"""E-035: explain excluded inlet area without changing MRI masks or data."""
import argparse
from datetime import datetime,timezone
import itertools
import json
from pathlib import Path
import tomllib
import uuid

import h5py
import numpy as np
from scipy.spatial import ConvexHull

from scripts.aorta_hemodynamic_reference import AortaGrid
from scripts.build_aorta_inlet_picker import sha


def rim_margin(points_uv,hull_equations):
    """Positive distance to closest convex-hull edge, not a native wall."""
    points=np.asarray(points_uv,dtype=float);eq=np.asarray(hull_equations,dtype=float)
    return -np.max((points@eq[:,:2].T+eq[:,2])/np.linalg.norm(eq[:,:2],axis=1),axis=1)


def trilinear_stencil(axes,points):
    points=np.asarray(points,dtype=float)
    index=np.column_stack([(points[:,k]-axis[0])/(axis[1]-axis[0]) for k,axis in enumerate(axes)])
    low=np.floor(index).astype(int);fraction=index-low
    if np.any(low<0) or any(np.any(low[:,k]+1>=len(a)) for k,a in enumerate(axes)):
        raise ValueError('Quadrature outside MRI grid')
    corners=np.array(list(itertools.product((0,1),repeat=3)))
    locations=low[:,None,:]+corners[None,:,:]
    weights=np.prod(np.where(corners[None,:,:],fraction[:,None,:],1-fraction[:,None,:]),axis=2)
    return locations,weights


def exclusion_categories(strict,point_signal_above):
    """Mutually exclusive area categories; no support policy is changed."""
    strict=np.asarray(strict,dtype=bool);point_signal_above=np.asarray(point_signal_above,dtype=bool)
    return dict(accepted_complete_stencil=strict,
        rejected_stencil_point_signal_at_least_05=(~strict)&point_signal_above,
        rejected_and_point_signal_below_05=(~strict)&(~point_signal_above))


def sampled_ring_counts(points_mm,normal,offset,u,v,equations):
    distance=points_mm@normal-offset
    points=points_mm[(distance>=0)&(distance<4)]
    margin=rim_margin(np.column_stack((points@u,points@v)),equations)
    return np.histogram(margin[margin>=0],bins=[0,2,4,6,np.inf])[0]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    root=Path(__file__).resolve().parents[1];base=root/'outputs/aorta_inlet_replenishment'
    geometry_path=root/'outputs/aorta_inlet_picker/e030_origin_v1/origin_inlet_review.json'
    numerical_path=root/'outputs/aorta_inlet_flow/e031_v1/inlet_numerical_evidence.npz'
    audit_path=base/'e034_100k_audit_v1/pilot_audit.json'
    reconstruction=base/'e034_100k_reconstruction_v1/pc_reconstruction.h5'
    evidence_path=base/'e034_100k_audit_v1/paired_partial_flow_evidence.npz'
    tracking=base/'e033_100k_v1/tracks.h5';warmup=base/'e033_100k_v1/warmup_and_recorded.h5'
    original_tracks=root/'outputs/aorta_dp2_mri_pipeline/aorta_motion_1m.h5'
    simulation_path=reconstruction.parent/'simulation_report.toml'
    simulation=tomllib.loads(simulation_path.read_text())
    audit=json.loads(audit_path.read_text());z=np.load(numerical_path,allow_pickle=False)
    if sha(numerical_path)!='11a76b06940052d0b13361560b8ec85581c9c0a701eaaf5e895afdea701ca5c0':
        raise ValueError('Native inlet evidence identity changed')
    if sha(reconstruction)!=audit['reconstruction_sha256']:raise ValueError('Pilot reconstruction changed')
    track_manifest=json.loads((tracking.parent/'MANIFEST.json').read_text())
    for path in (tracking,warmup):
        if sha(path)!=track_manifest[path.name]['sha256']:raise ValueError('Tracking evidence changed')
    g=json.loads(geometry_path.read_text())['candidate']
    normal=np.asarray(g['normal_source_xyz']);u=np.asarray(g['basis_u']);v=np.asarray(g['basis_v'])
    vertices=z['face_points_source_mm'];area=z['triangle_area_mm2'];points=z['triangle_centroids_source_mm']
    hull=ConvexHull(np.column_stack((vertices@u,vertices@v)));equations=hull.equations
    margin=rim_margin(np.column_stack((points@u,points@v)),equations)
    grid=AortaGrid();axes=grid.axes()[::-1]
    query=((points/1000-grid.center)[:,grid.order])[:,::-1]
    locations,weights=trilinear_stencil(axes,query);ix=tuple(np.moveaxis(locations,-1,0))
    with h5py.File(reconstruction,'r') as f:
        magnitude=f['reference_magnitude'][:];mask=f['mask'][:].astype(bool)
    ratios=magnitude/magnitude.max(axis=(1,2,3),keepdims=True)
    corner_mask=mask[(slice(None),)+ix]
    support=np.sum(np.all(corner_mask,axis=0)*weights,axis=1)
    strict=support>=1-1e-10
    saved=np.load(evidence_path,allow_pickle=False)
    if not np.array_equal(strict,saved['common']):raise ValueError('Original stencil membership not reproduced')
    point_ratio=np.sum(ratios[(slice(None),)+ix]*weights[None,:,:],axis=2)
    minimum=point_ratio.min(axis=0);above=np.all(point_ratio>=.05,axis=0)
    categories=exclusion_categories(strict,above)
    fraction=lambda selected:float(area[selected].sum()/area.sum())
    # Classify actual stencil corner positions using only the inferred inlet
    # prism. It is not a certified wall/inside-lumen test.
    corner_mri=np.stack([axes[k][locations[:,:,k]] for k in range(3)],axis=-1)[:,:,::-1]
    corner_source=grid.to_source(corner_mri.reshape(-1,3))*1000
    corner_margin=rim_margin(np.column_stack((corner_source@u,corner_source@v)),equations).reshape(len(area),8)
    corner_depth=(corner_source@normal-g['offset_mm']).reshape(len(area),8)
    upstream=corner_depth<-.01;lateral=corner_margin< -1e-6
    rejected=(~np.all(corner_mask,axis=0))&(weights>1e-12)
    inside_failure=rejected&(~upstream)&(~lateral)
    exclusively_exterior=(~strict)&(~np.any(inside_failure,axis=1))
    rings=[]
    for lower,upper in [(0,2),(2,4),(4,6),(6,np.inf)]:
        select=(margin>=lower)&(margin<upper)
        rings.append(dict(distance_from_inferred_rim_mm=[lower,None if np.isinf(upper) else upper],
            area_mm2=float(area[select].sum()),excluded_area_mm2=float(area[select&(~strict)].sum()),
            excluded_fraction_of_ring=float(area[select&(~strict)].sum()/area[select].sum()) if select.any() else None))
    # The streamed two-cycle file stores poststeps only. Reproduce the exact
    # registered pre-warmup seeds from the preserved source, never substitute
    # its first poststep or the already-warmed recorded initial frame.
    if sha(original_tracks)!='92095e6a7858f7860fa6d778972ff8c877aa8d1cf7f8670fa720b36f434dd59d':
        raise ValueError('Original initial-seed source changed')
    ids=np.sort(np.random.default_rng(20261021).choice(1000000,100000,replace=False))
    with h5py.File(original_tracks,'r') as f:
        initial=sampled_ring_counts(f['initial_position'][:][ids].astype(float)*1000,normal,g['offset_mm'],u,v,equations)
    registered=json.loads((tracking.parent/'tracking_report.json').read_text())
    if initial.sum()!=registered['baseline_counts'][0]:raise ValueError('Registered inlet baseline not reproduced')
    counts=[]
    with h5py.File(tracking,'r') as f:
        for step in (0,40,90,180,360):
            positions=f['initial_position'][:] if step==0 else f['position'][step-1]
            count=sampled_ring_counts(positions.astype(float)*1000,normal,g['offset_mm'],u,v,equations)
            counts.append(dict(recorded_time_s=step*.002,ring_counts=count.tolist(),
                relative_to_pre_warmup_initial=np.divide(count,initial,out=np.full(4,np.nan),where=initial>0).tolist()))
    bin_checks=[]
    for b in range(8):
        bin_support=np.sum(corner_mask[b]*weights,axis=1)>=1-1e-10
        bin_checks.append(dict(bin=b+1,complete_stencil_area_fraction=fraction(bin_support),
            point_sample_at_least_05_area_fraction=fraction(point_ratio[b]>=.05)))
    encoded_checks=[];raw_paths=[];complex_valid=np.ones((8,len(area)),dtype=bool);complex_ratios=[]
    for encoding in ('REF','X','Y','Z'):
        path=base/'e034_100k_mri_v2'/(encoding.lower()+'_result.jld2');raw_paths.append(path)
        if sha(path)!=simulation['raw_result_sha256'][encoding]:raise ValueError('Raw image identity changed')
        with h5py.File(path,'r') as f:data=f['image'][:]
        raw=data['re']+1j*data['im'];mag=abs(raw);mag/=mag.max(axis=(1,2,3),keepdims=True)
        sampled=np.sum(mag[(slice(None),)+ix]*weights[None,:,:],axis=2)
        # Magnitude interpolation and complex interpolation are not equivalent:
        # phase cancellation can lower the latter. Diagnose both, without yet
        # calculating a replacement full-face velocity or flow curve.
        sampled_complex=np.sum(raw[(slice(None),)+ix]*weights[None,:,:],axis=2)
        ratio_complex=abs(sampled_complex)/abs(raw).max(axis=(1,2,3),keepdims=False)[:,None]
        complex_ratios.append(ratio_complex);complex_valid &= ratio_complex>=.05
        encoded_checks.append(dict(encoding=encoding,
            point_sample_all_bin_at_least_05_area_fraction=fraction(np.all(sampled>=.05,axis=0)),
            complex_point_all_bin_at_least_05_area_fraction=fraction(np.all(ratio_complex>=.05,axis=0)),
            minimum_complex_point_magnitude_fraction=float(ratio_complex.min()),
            all_values_finite=bool(np.isfinite(raw).all())))
    face_vn=np.einsum('ncf,c->nf',z['face_velocities_source_m_s'],normal)
    tri_vn=face_vn[z['inferred_triangles']].mean(axis=1)
    cfd_full=z['frame_weights']@(area@tri_vn)*(.72/.7261599898338318)
    cfd_excluded=z['frame_weights']@(area[~strict]@tri_vn[~strict])*(.72/.7261599898338318)
    inputs=[geometry_path,numerical_path,audit_path,evidence_path,reconstruction,simulation_path,tracking,warmup,original_tracks,*raw_paths]
    report=dict(experiment='E-035',created_utc=datetime.now(timezone.utc).isoformat(),
        status='COMPLETE_COVERAGE_DIAGNOSIS_NOT_FULL_FLOW_RECOVERY',threshold_unchanged=.05,
        accepted_area_fraction=fraction(strict),excluded_area_fraction=fraction(~strict),
        diagnostic_area_categories={name:dict(area_fraction=fraction(select),area_mm2=float(area[select].sum())) for name,select in categories.items()},
        minimum_point_signal_fraction_range=[float(minimum.min()),float(minimum.max())],
        rejected_triangles_whose_only_failed_corners_are_outside_inferred_prism_fraction_of_face=fraction(exclusively_exterior),
        excluded_area_within_4mm_of_inferred_rim_fraction_of_excluded=float(area[(~strict)&(margin<4)].sum()/area[~strict].sum()),
        reference_bin_diagnostics=bin_checks,encoding_point_signal_diagnostics=encoded_checks,
        complex_point_all_encoding_all_bin_signal_coverage=fraction(np.all(complex_valid,axis=0)),
        complex_point_signal_classification='Experimental sampling-QC diagnostic only; no alternate flow estimate or acceptance promoted',
        inferred_rim_rings=rings,pre_warmup_inlet_prism_ring_counts=initial.tolist(),sampled_particle_ring_counts=counts,
        excluded_area_CFD_transport_fraction_by_bin=(cfd_excluded/cfd_full).tolist(),
        inputs={str(p):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in inputs},diagnostic_code_sha256=sha(__file__),
        scope='Diagnosis only; point-sampled coverage is not a replacement acceptance mask or recovered full-inlet flow',
        limitations=['Convex-hull rim/prism is inferred geometry, not native connected wall/caps',
            'Five sampled tracking times do not establish full-series near-wall density stability',
            'No new simulation, lower threshold, phase imputation, area scaling or full-flow estimate'])
    # Treat an empty ring baseline as unknown rather than emitting NaN JSON.
    for row in counts:
        row['relative_to_pre_warmup_initial']=[value if np.isfinite(value) else None for value in row['relative_to_pre_warmup_initial']]
    staging=args.output.with_name(args.output.name+'.building-'+uuid.uuid4().hex[:8]);staging.mkdir(parents=True)
    (staging/'coverage_diagnosis.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    np.savez_compressed(staging/'coverage_diagnostic_evidence.npz',strict=strict,point_ratio=point_ratio,
        corner_mask=corner_mask,corner_weights=weights,corner_depth_mm=corner_depth,corner_rim_margin_mm=corner_margin,
        area_mm2=area,centroid_rim_margin_mm=margin,complex_point_magnitude_ratios=np.array(complex_ratios))
    manifest={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in staging.iterdir()}
    (staging/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    staging.rename(args.output)
    print(json.dumps({key:report[key] for key in ('status','accepted_area_fraction','excluded_area_fraction',
        'diagnostic_area_categories','rejected_triangles_whose_only_failed_corners_are_outside_inferred_prism_fraction_of_face',
        'excluded_area_within_4mm_of_inferred_rim_fraction_of_excluded','inferred_rim_rings',
        'sampled_particle_ring_counts')},indent=2))


if __name__=='__main__':main()
