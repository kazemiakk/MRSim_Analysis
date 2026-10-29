"""Cross-check E-035 coverage categories via SciPy and saved complex data."""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import ConvexHull
from scripts.aorta_hemodynamic_reference import AortaGrid
from scripts.build_aorta_inlet_picker import sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    root=Path(__file__).resolve().parents[1];report_path=args.run/'coverage_diagnosis.json'
    report=json.loads(report_path.read_text());e=np.load(args.run/'coverage_diagnostic_evidence.npz',allow_pickle=False)
    checks=[]
    def check(name,value):
        if not value:raise AssertionError(name)
        checks.append(name)
    for path,entry in report['inputs'].items():
        path=Path(path)
        check('input bytes '+path.name,path.stat().st_size==entry['bytes'])
        check('input hash '+str(path),sha(path)==entry['sha256'])
    for name,entry in json.loads((args.run/'MANIFEST.json').read_text()).items():
        check('output '+name,(args.run/name).stat().st_size==entry['bytes'] and sha(args.run/name)==entry['sha256'])
    check('diagnostic code binding',sha(root/'scripts/diagnose_aorta_inlet_coverage.py')==report['diagnostic_code_sha256'])
    z=np.load(root/'outputs/aorta_inlet_flow/e031_v1/inlet_numerical_evidence.npz',allow_pickle=False)
    grid=AortaGrid();point=((z['triangle_centroids_source_mm']/1000-grid.center)[:,grid.order])[:,::-1]
    area=z['triangle_area_mm2'];base=root/'outputs/aorta_inlet_replenishment'
    with h5py.File(base/'e034_100k_reconstruction_v1/pc_reconstruction.h5','r') as f:
        mag=f['reference_magnitude'][:];mask=f['mask'][:].astype(bool)
    selected=RegularGridInterpolator(grid.axes()[::-1],mask.all(axis=0).astype(float))(point)>=1-1e-10
    check('independent stencil support',np.array_equal(selected,e['strict']))
    # Preserve the author's Float32 normalization order, while independently
    # evaluating interpolation with SciPy instead of explicit corner weights.
    point_magnitude=np.array([RegularGridInterpolator(grid.axes()[::-1],im/im.max())(point) for im in mag])
    check('independent point magnitudes',np.allclose(point_magnitude,e['point_ratio'],rtol=1e-10,atol=1e-12))
    check('all point magnitudes above 5 percent',point_magnitude.min()>=.05)
    check('category area sums',np.isclose(sum(v['area_fraction'] for v in report['diagnostic_area_categories'].values()),1))
    valid=np.ones((8,len(area)),dtype=bool)
    for k,encoding in enumerate(('REF','X','Y','Z')):
        path=base/'e034_100k_mri_v2'/(encoding.lower()+'_result.jld2')
        with h5py.File(path,'r') as f:a=f['image'][:]
        a=a['re']+1j*a['im']
        samples=np.array([RegularGridInterpolator(grid.axes()[::-1],im)(point) for im in a])
        ratio=abs(samples)/abs(a).max(axis=(1,2,3))[:,None]
        check('independent complex interpolation '+encoding,np.allclose(ratio,e['complex_point_magnitude_ratios'][k],rtol=1e-10,atol=1e-12))
        valid &= ratio>=.05
    check('near-full signal-QC coverage',np.isclose(area[valid.all(axis=0)].sum()/area.sum(),report['complex_point_all_encoding_all_bin_signal_coverage'],rtol=1e-12))
    check('no full-coverage promotion',report['complex_point_all_encoding_all_bin_signal_coverage']<1)
    check('threshold unchanged',report['threshold_unchanged']==.05)
    # A second distance implementation uses bounded segments, not half-spaces.
    geometry=json.loads((root/'outputs/aorta_inlet_picker/e030_origin_v1/origin_inlet_review.json').read_text())['candidate']
    uv=np.column_stack((z['face_points_source_mm']@geometry['basis_u'],z['face_points_source_mm']@geometry['basis_v']))
    q=np.column_stack((z['triangle_centroids_source_mm']@geometry['basis_u'],z['triangle_centroids_source_mm']@geometry['basis_v']))
    polygon=uv[ConvexHull(uv).vertices];distances=[]
    for a,b in zip(polygon,np.roll(polygon,-1,axis=0)):
        d=b-a;t=np.clip((q-a)@d/(d@d),0,1)
        distances.append(np.linalg.norm(q-a-t[:,None]*d,axis=1))
    margin=np.min(distances,axis=0)
    check('independent rim distances',np.allclose(margin,e['centroid_rim_margin_mm'],atol=1e-10))
    fraction=area[(~selected)&(margin<4)].sum()/area[~selected].sum()
    check('edge-loss fraction',np.isclose(fraction,report['excluded_area_within_4mm_of_inferred_rim_fraction_of_excluded'],rtol=1e-12))
    result=dict(status='PASS_COVERAGE_DIAGNOSTIC_CROSSCHECKS',check_count=len(checks),checks=checks,
        report_sha256=sha(report_path),validator_sha256=sha(__file__),scope='Arithmetic/implementation cross-check, not physical validation')
    with args.output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(dict(status=result['status'],check_count=len(checks))))


if __name__=='__main__':main()
