"""E-031: confirmed inlet CFD trace and preserved Koma coverage, without filling.

An inferred planar Delaunay face is an approximation, not native cap topology.
Unavailable Koma inlet flow is null, never a zero-valued measurement.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import Delaunay

from scripts.aorta_hemodynamic_reference import AortaGrid
from scripts.build_aorta_inlet_picker import SOURCE_SHA, sha, source_to_mri_mm


def face_quadrature(points_mm, u, v):
    points=np.asarray(points_mm, dtype=float)
    uv=np.column_stack((points@u,points@v))
    triangles=Delaunay(uv).simplices
    q=uv[triangles];d=q[:,1:]-q[:,:1]
    areas=abs(d[:,0,0]*d[:,1,1]-d[:,0,1]*d[:,1,0])/2
    if not np.isfinite(areas).all() or np.any(areas<=0):
        raise ValueError('Invalid inferred face triangulation')
    return triangles,areas,points[triangles].mean(axis=1)


def nodal_face_flow_ml_s(velocity, triangles, area_mm2, normal):
    """Linear nodal field triangle integral; mm2*m/s equals mL/s."""
    values=np.asarray(velocity,dtype=float)
    if values.ndim!=3 or values.shape[1]!=3 or not np.isfinite(values).all():
        raise ValueError('Finite node,component,frame velocities required')
    nodal_normal=np.einsum('ncf,c->nf',values,normal)
    triangle_means=nodal_normal[triangles].mean(axis=1)
    return np.einsum('q,qf->f',area_mm2,triangle_means)


def coverage(axes_zyx, mask, mri_points_m, areas):
    fractions=RegularGridInterpolator(axes_zyx,np.asarray(mask,dtype=float),
        bounds_error=False,fill_value=0)(mri_points_m[:,::-1])
    total=float(np.sum(areas))
    return dict(strict_stencil_area_fraction=float(areas[fractions>=1-1e-10].sum()/total),
                any_stencil_neighbor_area_fraction=float(areas[fractions>0].sum()/total))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    root=Path(__file__).resolve().parents[1]
    geom=root/'outputs/aorta_inlet_picker/e030_origin_v1'
    previous=root/'outputs/aorta_hemodynamic_comparison/e027_v1/results_v2'
    source=root.parent/'external_inputs/aorta_pointcloud/processed/dp2'
    recon=root/'outputs/aorta_dp2_mri_pipeline/koma/multibin_8x4mm/simulation_motion_1m_8bin_4mm/pc_reconstruction.h5'
    expected={source/'point_coordinates.npy':SOURCE_SHA,
      source/'velocities.npy':'0dde75ca9b94a1aa6b795912bfc16bb0238f206fdcb0bc9336aaf4a91c5e37f6',
      previous/'report.json':'a54cf57e0e1fa27cf6e0238df338bcb16c6292095ad4225540ec441bbd1c9660',
      previous/'matched_reference.h5':'ade84dcad431246d86d13f429824168700dc2c1e306c5e6fa83126bdd0208afe',
      recon:'90b88e5e60b3ea250f462b2ed40254192b703e3a64b48a0fc9b3c1c8ec62cae7'}
    inputs=[]
    for path,digest in expected.items():
        if sha(path)!=digest:raise ValueError('Input identity mismatch: '+str(path))
        inputs.append(dict(path=str(path.relative_to(root)) if path.is_relative_to(root) else '../external_inputs/aorta_pointcloud/processed/dp2/'+path.name,
                           bytes=path.stat().st_size,sha256=digest))
    manifest=json.loads((geom/'MANIFEST.json').read_text())
    for name,entry in manifest['files'].items():
        path=geom/name
        if sha(path)!=entry['sha256'] or path.stat().st_size!=entry['bytes']:
            raise ValueError('Geometry evidence identity mismatch')
        inputs.append(dict(path=str(path.relative_to(root)),**entry))
    review=json.loads((geom/'origin_inlet_review.json').read_text());c=review['candidate']
    ids=np.load(geom/'candidate_face_original_rows.npy',allow_pickle=False)
    points=np.load(geom/'candidate_face_source_mm.npy',allow_pickle=False)
    original=np.load(source/'point_coordinates.npy',mmap_mode='r',allow_pickle=False)
    if not np.array_equal(original[ids].astype(float)*1000,points):
        raise ValueError('Face original coordinate rows do not match')
    normal=np.array(c['normal_source_xyz']);tri,areas,centers=face_quadrature(points,c['basis_u'],c['basis_v'])
    velocities=np.load(source/'velocities.npy',mmap_mode='r',allow_pickle=False)
    face_velocities=np.array(velocities[ids],dtype=float)
    native_q=nodal_face_flow_ml_s(face_velocities,tri,areas,normal)
    old=json.loads((previous/'report.json').read_text());grid=AortaGrid()
    samples=[];offset_rows=[]
    with h5py.File(previous/'matched_reference.h5','r') as f,h5py.File(recon,'r') as raw:
        weights=f['frame_weights'][:8]
        if weights.shape!=(8,72) or not np.allclose(weights.sum(1),1):
            raise ValueError('Temporal weight contract mismatch')
        paired_q=(weights@native_q)*old['source_velocity_time_scale']
        masks={name:f['masks/'+name][:] for name in ('core05','core10','core15','core20')}
        masks['stored_all_bins']=np.all(raw['mask'][:].astype(bool),axis=0)
        for offset in (0,4,8,12,16,20):
            locations=source_to_mri_mm((centers+offset*normal)/1000)/1000
            for name,mask in masks.items():
                offset_rows.append(dict(inward_offset_mm=offset,support=name,
                    **coverage(grid.axes()[::-1],mask,locations,areas)))
        locations=source_to_mri_mm(centers/1000)/1000
        for b in range(8):
            magnitude=raw['reference_magnitude'][b]
            mag=RegularGridInterpolator(grid.axes()[::-1],magnitude,
                bounds_error=False,fill_value=np.nan)(locations[:,::-1])/magnitude.max()
            samples.append(dict(bin=b+1,nominal_time_s=old['time_s'][b],
                cfd_adc_average_ml_s=float(paired_q[b]),koma_inlet_ml_s=None,
                magnitude_fraction_of_global_max_min=float(mag.min()),
                magnitude_fraction_of_global_max_max=float(mag.max()),
                stored_per_bin_support=coverage(grid.axes()[::-1],raw['mask'][b],locations,areas)))
    report=dict(experiment='E-031',created_utc=datetime.now(timezone.utc).isoformat(),
       status='CFD_INLET_TRACE_COMPUTED_KOMA_INLET_COMPARISON_UNAVAILABLE',
       evidence_class='Approximate planar CFD flux and preserved acquisition coverage; no independent validation',
       user_confirmation='User confirmed E-030 origin-based face preview as correct in this session',
       inputs=inputs,script_sha256=sha(__file__),
       geometry=dict(face_nodes=len(points),inferred_triangles=len(tri),approximate_area_mm2=float(areas.sum()),
          topology='2D Delaunay of detected native planar nodes, convex-hull footprint; not certified native faces',
          normal_source_xyz=normal.tolist(),normal_sign='Into local vessel, +source Z',
          origin_source_mm=[0,0,0],previous_E029_plane_used=False),
       timing=dict(native_source_dt_s=.01,native_source_frames=72,native_source_period_s=.72,
          mri_period_s=old['period_s'],velocity_scale=old['source_velocity_time_scale'],
          comparator='E-027 first eight frame_weights: actual ADC average, start frame18, no fitted alignment'),
       native_cfd_waveform=[dict(source_time_s=float(t),flow_ml_s=float(q)) for t,q in zip(np.arange(72)*.01,native_q)],
       matched_bins=samples,coverage_by_offset=offset_rows,
       limitations=['No Koma inlet waveform: missing fixed-mask interpolation support is not zero flow',
          'Finite stored velocities alone are not evidence of usable phase',
          'Offsets are coverage-only diagnostics using translated face footprint, not measured inlet flow or actual downstream lumen shapes',
          'No phase filling, threshold lowering, amplitude/phase fitting or new acquisition performed',
          'CFD curve uses an approximate triangulated point-cloud face, not verified solver cap connectivity',
          'Same-source CFD and Koma would be dependent internal consistency, not independent validation'])
    temporary=args.output.with_name(args.output.name+'.building-'+uuid.uuid4().hex[:8]);temporary.mkdir(parents=True)
    np.savez_compressed(temporary/'inlet_numerical_evidence.npz',original_rows=ids,face_points_source_mm=points,
        inferred_triangles=tri,triangle_area_mm2=areas,triangle_centroids_source_mm=centers,
        normal_source=normal,face_velocities_source_m_s=face_velocities,frame_weights=weights,
        native_q_ml_s=native_q,adc_q_ml_s=paired_q)
    (temporary/'inlet_readiness.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    for name,rows in [('native_cfd_waveform',report['native_cfd_waveform']),('coverage_by_offset',offset_rows),
                      ('matched_cfd_bins',[{k:v for k,v in x.items() if k!='stored_per_bin_support'} for x in samples])]:
        with (temporary/(name+'.csv')).open('x',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    manifest={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in temporary.iterdir()}
    (temporary/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    temporary.rename(args.output)
    print(json.dumps(dict(output=str(args.output),status=report['status'],cfd_adc_q_ml_s=paired_q.tolist(),
                         exact_face_coverage=[x for x in offset_rows if x['inward_offset_mm']==0]),indent=2))


if __name__=='__main__':main()
