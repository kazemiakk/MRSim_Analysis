"""E-045: new frame0 tracking pilots; original E-033 stays immutable."""
import argparse
from datetime import datetime,timezone
import hashlib
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
from scripts.track_aorta_inlet_pilot import GATES


def main(output,particles):
    if output.exists():raise FileExistsError(output)
    if particles not in (20000,100000):raise ValueError('Pilot only')
    root=Path(__file__).resolve().parents[1];base=root/'outputs/aorta_inlet_flow/e031_v1';data=root.parent/'external_inputs/aorta_pointcloud/processed/dp2'
    for path,expected in [(data/'point_coordinates.npy',SOURCE_SHA),(data/'velocities.npy','0dde75ca9b94a1aa6b795912bfc16bb0238f206fdcb0bc9336aaf4a91c5e37f6'),(base/'inlet_numerical_evidence.npz','11a76b06940052d0b13361560b8ec85581c9c0a701eaaf5e895afdea701ca5c0')]:
        if sha(path)!=expected:raise ValueError('Original matched input changed')
    z=np.load(base/'inlet_numerical_evidence.npz',allow_pickle=False);vertices=z['face_points_source_mm']/1000;normal=z['normal_source'];tri=z['inferred_triangles']
    u=vertices[tri[0,1]]-vertices[tri[0,0]];u/=np.linalg.norm(u);v=np.cross(normal,u)
    hull=ConvexHull(np.c_[vertices@u,vertices@v]);hull_mm=hull.equations.copy();hull_mm[:,2]*=1000;offset=float(np.mean(vertices@normal))*1000
    with h5py.File(root/'outputs/aorta_dp2_mri_pipeline/aorta_motion_1m.h5','r') as f:
        ids=np.sort(np.random.default_rng(20261021).choice(1000000,particles,replace=False));seeds=f['initial_position'][:][ids]
    coordinates=np.load(data/'point_coordinates.npy',mmap_mode='r',allow_pickle=False);velocities=np.load(data/'velocities.npy',mmap_mode='r',allow_pickle=False)
    print('Building native-zero kNN flow; particles='+str(particles),flush=True)
    flow=PointCloudFlow(coordinates,velocities,frame_dt=.01,start_frame=0,k=8,support_radius=.002322211721826266)
    reseeder=PointCloudInletReseeder(vertices,tri,z['face_velocities_source_m_s'],normal,frame_dt=.01,start_frame=0,dt=.002,rng=np.random.default_rng(20261022))
    output.parent.mkdir(parents=True,exist_ok=True);tmp=output.with_name(output.name+'.building-'+uuid.uuid4().hex[:8]);tmp.mkdir()
    protocol=root/'outputs/aorta_inlet_replenishment/E045_NATIVE_ZERO_PROTOCOL.md'
    registration=dict(experiment='E-045',created_utc=datetime.now(timezone.utc).isoformat(),particles=particles,warmup_cycles=1,recorded_cycles=1,dt_s=.002,source_frame_dt_s=.01,start_frame=0,source_cycle_s=.72,gates=GATES,code_sha256=sha(__file__),protocol_sha256=sha(protocol),reseeder_sha256=sha(root/'mrsimtracks/pointcloud_inlet.py'),seed_subset_sha256=hashlib.sha256(seeds.tobytes()).hexdigest(),inlet_model='Same inferred-face flux-weighted tracker-exit replacement as E-033; not certified native caps')
    (tmp/'registration.json').write_text(json.dumps(registration,indent=2)+'\n',encoding='utf-8')
    start=time.perf_counter();_,metrics=mt.track(flow,seeds=seeds,reseeder=reseeder,dt=.002,tmax=1.44,output_path=tmp/'warmup_and_recorded.h5',return_metrics=True,pbar=False)
    elapsed=time.perf_counter()-start;series=[];edges=np.arange(0,29,4);baseline=prism_counts(seeds.astype(float)*1000,normal,offset,u,v,hull_mm,edges)
    with h5py.File(tmp/'warmup_and_recorded.h5','r') as f:
        finite=True
        for i in range(360,720):
            points=f['position'][i];finite=finite and bool(np.isfinite(points).all());series.append(prism_counts(points.astype(float)*1000,normal,offset,u,v,hull_mm,edges).tolist())
        count=np.array(series);ratios=count[:,0]/baseline[0]
        tests=dict(first_slab_mean=GATES['first_slab_mean_ratio_min']<=ratios.mean()<=GATES['first_slab_mean_ratio_max'],first_slab_min=ratios.min()>=GATES['first_slab_min_ratio_min'],first_slab_max=ratios.max()<=GATES['first_slab_max_ratio_max'],all_finite=finite)
        with h5py.File(tmp/'tracks.h5','x') as out:
            out.create_dataset('initial_position',data=f['position'][359]);out.create_dataset('position',data=f['position'][360:],chunks=(1,min(particles,65536),3));out.create_dataset('reset',data=f['reset'][360:]);out.attrs.update(dt=.002,n_steps=360,n_particles=particles,source_start_frame=0,source_period_s=.72)
    report=dict(**registration,status='PASS_TRACKING_CONTROL' if all(tests.values()) else 'FAIL_TRACKING_CONTROL_NO_MRI',elapsed_s=elapsed,metrics=metrics,checks={k:bool(v) for k,v in tests.items()},baseline_counts=baseline.tolist(),first_slab_mean_ratio=float(ratios.mean()),first_slab_min_ratio=float(ratios.min()),first_slab_max_ratio=float(ratios.max()),recorded_slab_counts=series,reset_events=reseeder.injected)
    (tmp/'tracking_report.json').write_text(json.dumps(report,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n',encoding='utf-8')
    (tmp/'MANIFEST.json').write_text(json.dumps({p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in tmp.iterdir()},indent=2)+'\n',encoding='utf-8');tmp.rename(output)
    print(json.dumps({k:report[k] for k in ['status','elapsed_s','first_slab_mean_ratio','first_slab_min_ratio','first_slab_max_ratio']},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--particles',type=int,required=True);a=p.parse_args();main(a.output,a.particles)
