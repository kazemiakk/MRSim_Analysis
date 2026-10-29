"""E-033: fail-closed small inferred-inlet recycling control, no MRI launch."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import time
import uuid

import h5py
import numpy as np
from scipy.spatial import ConvexHull

import mrsimtracks as mt
from mrsimtracks.pointcloud import PointCloudFlow
from mrsimtracks.pointcloud_inlet import PointCloudInletReseeder
from scripts.build_aorta_inlet_picker import SOURCE_SHA,sha
from scripts.diagnose_aorta_inlet_signal import prism_counts

GATES={'first_slab_mean_ratio_min':.7,'first_slab_mean_ratio_max':1.3,
       'first_slab_min_ratio_min':.5,'first_slab_max_ratio_max':1.5,
       'inlet_flux_weighted_profile_expected':True,'all_finite':True}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--particles',type=int,default=20000);args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if args.particles not in (20000,100000):raise ValueError('Only preregistered pilot counts allowed')
    root=Path(__file__).resolve().parents[1];base=root/'outputs/aorta_inlet_flow/e031_v1'
    data=root.parent/'external_inputs/aorta_pointcloud/processed/dp2'
    for path,expected in [(data/'point_coordinates.npy',SOURCE_SHA),
        (data/'velocities.npy','0dde75ca9b94a1aa6b795912bfc16bb0238f206fdcb0bc9336aaf4a91c5e37f6')]:
        if sha(path)!=expected:raise ValueError('Source identity mismatch')
    z=np.load(base/'inlet_numerical_evidence.npz',allow_pickle=False)
    if sha(base/'inlet_numerical_evidence.npz')!='11a76b06940052d0b13361560b8ec85581c9c0a701eaaf5e895afdea701ca5c0':
        raise ValueError('Inlet evidence identity mismatch')
    vertices=z['face_points_source_mm']/1000;normal=z['normal_source'];tri=z['inferred_triangles']
    u=vertices[tri[0,1]]-vertices[tri[0,0]];u/=np.linalg.norm(u);v=np.cross(normal,u)
    hull=ConvexHull(np.c_[vertices@u,vertices@v]);hull_mm=hull.equations.copy();hull_mm[:,2]*=1000
    offset=float(np.mean(vertices@normal))*1000
    with h5py.File(root/'outputs/aorta_dp2_mri_pipeline/aorta_motion_1m.h5','r') as f:
        ids=np.sort(np.random.default_rng(20261021).choice(1000000,args.particles,replace=False))
        seeds=f['initial_position'][:][ids]
    coordinates=np.load(data/'point_coordinates.npy',mmap_mode='r',allow_pickle=False)
    velocities=np.load(data/'velocities.npy',mmap_mode='r',allow_pickle=False)
    flow=PointCloudFlow(coordinates,velocities,frame_dt=.01,start_frame=18,k=8,support_radius=.002322211721826266)
    reseeder=PointCloudInletReseeder(vertices,tri,z['face_velocities_source_m_s'],normal,
        frame_dt=.01,start_frame=18,dt=.002,rng=np.random.default_rng(20261022))
    tmp=args.output.with_name(args.output.name+'.building-'+uuid.uuid4().hex[:8]);tmp.mkdir(parents=True)
    registration=dict(experiment='E-033',created_utc=datetime.now(timezone.utc).isoformat(),particles=args.particles,
        warmup_cycles=1,recorded_cycles=1,dt_s=.002,source_frame_dt_s=.01,start_frame=18,
        gates=GATES,inlet_model='Inferred-face flux-weighted replacement of tracker exits; not native caps or rate-prescribed injection',
        code_sha256=sha(__file__),reseeder_sha256=sha(root/'mrsimtracks/pointcloud_inlet.py'))
    (tmp/'registration.json').write_text(json.dumps(registration,indent=2)+'\n',encoding='utf-8')
    start=time.perf_counter()
    _,metrics=mt.track(flow,seeds=seeds,reseeder=reseeder,dt=.002,tmax=1.44,
        output_path=tmp/'warmup_and_recorded.h5',return_metrics=True,pbar=False)
    elapsed=time.perf_counter()-start
    series=[];edges=np.arange(0,29,4)
    baseline=prism_counts(seeds.astype(float)*1000,normal,offset,u,v,hull_mm,edges)
    with h5py.File(tmp/'warmup_and_recorded.h5','r') as f:
        for i in range(360,720):
            points=f['position'][i]
            counts=prism_counts(points.astype(float)*1000,normal,offset,u,v,hull_mm,edges)
            series.append(counts.tolist())
        count=np.array(series);ratios=count[:,0]/baseline[0]
        tests=dict(first_slab_mean=(GATES['first_slab_mean_ratio_min']<=ratios.mean()<=GATES['first_slab_mean_ratio_max']),
                   first_slab_min=ratios.min()>=GATES['first_slab_min_ratio_min'],
                   first_slab_max=ratios.max()<=GATES['first_slab_max_ratio_max'],all_finite=bool(np.isfinite(f['position'][:]).all()))
        with h5py.File(tmp/'tracks.h5','x') as out:
            out.create_dataset('initial_position',data=f['position'][359])
            out.create_dataset('position',data=f['position'][360:],chunks=(1,min(args.particles,65536),3))
            out.create_dataset('reset',data=f['reset'][360:])
            out.attrs.update(dt=.002,n_steps=360,n_particles=args.particles)
    result=dict(**registration,status='PASS_TRACKING_CONTROL' if all(tests.values()) else 'FAIL_TRACKING_CONTROL_NO_MRI',
        elapsed_s=elapsed,metrics=metrics,checks={k:bool(v) for k,v in tests.items()},
        baseline_counts=baseline.tolist(),first_slab_mean_ratio=float(ratios.mean()),
        first_slab_min_ratio=float(ratios.min()),first_slab_max_ratio=float(ratios.max()),
        recorded_slab_counts=series,reset_events=reseeder.injected,
        limitations=['Exit count is tracker-determined, not prescribed by inlet volume flux',
                     'Surface is inferred from point cloud, not certified cap connectivity',
                     'MRI and reset semantics have not yet been validated'])
    (tmp/'tracking_report.json').write_text(json.dumps(result,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n',encoding='utf-8')
    manifest={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in tmp.iterdir()}
    (tmp/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8');tmp.rename(args.output)
    print(json.dumps({k:result[k] for k in ['status','elapsed_s','first_slab_mean_ratio','first_slab_min_ratio','first_slab_max_ratio','checks']},indent=2))


if __name__=='__main__':main()
