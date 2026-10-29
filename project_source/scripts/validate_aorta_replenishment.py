"""Independent saved-artifact and local-count checks for E-033 pilot controls."""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
from matplotlib.path import Path as Polygon
from scipy.spatial import ConvexHull
from scripts.aorta_hemodynamic_reference import temporal_weights
from scripts.build_aorta_inlet_picker import sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    root=Path(__file__).resolve().parents[1];base=root/'outputs/aorta_inlet_replenishment'
    z=np.load(root/'outputs/aorta_inlet_flow/e031_v1/inlet_numerical_evidence.npz',allow_pickle=False)
    geometry=json.loads((root/'outputs/aorta_inlet_picker/e030_origin_v1/origin_inlet_review.json').read_text())['candidate']
    u=np.array(geometry['basis_u']);v=np.array(geometry['basis_v']);n=np.array(geometry['normal_source_xyz'])
    face=z['face_points_source_mm'];uv=np.c_[face@u,face@v];hull=ConvexHull(uv);poly=Polygon(uv[hull.vertices]);edges=np.arange(0,29,4)
    times=(np.arange(360)+.5)*.002
    sourceq=np.stack([temporal_weights([t],period_s=.72) for t in times])@z['native_q_ml_s']
    results=[];checks=0
    for name in ['e033_20k_v1','e033_100k_v1']:
        folder=base/name;r=json.loads((folder/'tracking_report.json').read_text());m=json.loads((folder/'MANIFEST.json').read_text())
        for artifact,entry in m.items():
            path=folder/artifact
            assert path.stat().st_size==entry['bytes'] and sha(path)==entry['sha256'];checks+=2
        assert r['status']=='PASS_TRACKING_CONTROL' and all(r['checks'].values());checks+=1
        with h5py.File(folder/'tracks.h5','r') as f:
            assert f['position'].shape==(360,r['particles'],3);checks+=1
            for i in [0,44,89,179,359]:
                points=f['position'][i].astype(float)*1000;distance=points@n-geometry['offset_mm']
                keep=(distance>=0)&(distance<=28);loc=np.c_[points[keep]@u,points[keep]@v]
                inside=poly.contains_points(loc,radius=1e-9)
                counts=np.histogram(distance[keep][inside],edges)[0]
                assert np.array_equal(counts,r['recorded_slab_counts'][i]);checks+=7
            events=f['reset'][:].sum(axis=1)
        density=r['baseline_counts'][0]/(z['triangle_area_mm2'].sum()*4/1000)
        expected=float(np.sum(density*sourceq*.002));observed=int(events.sum())
        results.append(dict(run=name,tracking_report_sha256=sha(folder/'tracking_report.json'),
             observed_recorded_replacements=observed,expected_from_initial_local_density_and_CFD=expected,
             replacement_ratio=observed/expected,replacement_waveform_correlation=float(np.corrcoef(sourceq,events)[0,1]),
             limitation='Initial local density and inferred-face CFD give an approximate rate comparator, not certified exact flux conservation'))
    result=dict(status='PASS_SAVED_TRACKING_CONTROL_CHECKS',check_count=checks,sampled_times_per_control=5,
        all_output_hashes_checked=True,independent_polygon_method=True,controls=results,validator_sha256=sha(__file__))
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
