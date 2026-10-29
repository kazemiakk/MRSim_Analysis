"""Build exact inlet quadrature input and audit the gated E-034 MRI pilot."""
import argparse
import json
from pathlib import Path
import uuid
import tomllib
import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scripts.aorta_hemodynamic_reference import AortaGrid
from scripts.build_aorta_inlet_picker import sha,source_to_mri_mm


def reconstruction_checks(magnitude, velocity, mask, speed, venc):
    """Check the explicit Python layout, threshold identity and saved norm."""
    if magnitude.shape != (8,32,32,72) or velocity.shape != (8,32,32,72,3):
        raise ValueError('Unexpected pilot matrix/bin/component layout')
    if mask.shape != magnitude.shape or speed.shape != magnitude.shape:
        raise ValueError('Mask/speed shape differs from magnitude')
    expected=magnitude >= np.float32(.05)*magnitude.max(axis=(1,2,3),keepdims=True)
    return dict(all_finite=bool(np.isfinite(magnitude).all() and np.isfinite(velocity).all() and np.isfinite(speed).all()),
        binary_mask=bool(np.isin(mask,[0,1]).all()),
        unchanged_05_threshold=bool(np.array_equal(mask,expected)),
        excluded_velocities_zero=bool(np.all(velocity[~expected]==0)),
        speed_is_vector_norm=bool(np.allclose(speed,np.linalg.norm(velocity,axis=-1),rtol=2e-6,atol=1e-8)),
        component_venc_bound_structural_only=bool(np.max(abs(velocity))<=venc*(1+1e-6)))


def flow_metrics(cfd,koma):
    if cfd is None or koma is None or not len(cfd):
        return dict(partial_flow_rmse_ml_s=None,partial_flow_correlation=None,cycle_volume_ratio=None)
    cfd=np.asarray(cfd);koma=np.asarray(koma)
    return dict(partial_flow_rmse_ml_s=float(np.sqrt(np.mean((koma-cfd)**2))),
        partial_flow_correlation=float(np.corrcoef(cfd,koma)[0,1]) if np.std(cfd)>0 and np.std(koma)>0 else None,
        cycle_volume_ratio=float(koma.sum()/cfd.sum()) if cfd.sum()!=0 else None)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--make-grid',type=Path)
    p.add_argument('--run',type=Path);p.add_argument('--output',type=Path);args=p.parse_args()
    root=Path(__file__).resolve().parents[1];e031=root/'outputs/aorta_inlet_flow/e031_v1'
    if sha(e031/'inlet_numerical_evidence.npz')!='11a76b06940052d0b13361560b8ec85581c9c0a701eaaf5e895afdea701ca5c0':
        raise ValueError('Inlet numerical evidence changed')
    z=np.load(e031/'inlet_numerical_evidence.npz',allow_pickle=False)
    if args.make_grid:
        with h5py.File(args.make_grid,'x') as f:
            f.create_dataset('centroids_source_m',data=z['triangle_centroids_source_mm']/1000)
            f.create_dataset('area_mm2',data=z['triangle_area_mm2'])
        print(args.make_grid);return
    if not args.run or not args.output:raise ValueError('Supply --run and --output')
    if args.output.exists():raise FileExistsError(args.output)
    simulation_path=args.run/'simulation_report.toml'
    simulation=tomllib.loads(simulation_path.read_text())
    if sha(args.run/'pc_reconstruction.h5') != simulation['reconstruction_sha256']:
        raise ValueError('Reconstruction does not match its publication report')
    acquisition=Path(simulation.get('acquisition_directory',str(args.run)))
    reset_path=acquisition/'reset_preflight.toml';reference_path=acquisition/'reference_viability.toml'
    reset=tomllib.loads(reset_path.read_text());reference=tomllib.loads(reference_path.read_text())
    if reset['status']!='PASS_RESET_PREFLIGHT' or reference['status']!='PASS_REFERENCE_VIABILITY':
        raise ValueError('Reset or reference preflight did not pass')
    for encoding,digest in simulation.get('raw_result_sha256',{}).items():
        if sha(acquisition/(encoding.lower()+'_result.jld2'))!=digest:
            raise ValueError('Raw result changed after verified reconstruction')
    grid=AortaGrid();area=z['triangle_area_mm2'];loc=source_to_mri_mm(z['triangle_centroids_source_mm']/1000)/1000
    with h5py.File(args.run/'pc_reconstruction.h5','r') as f:
        magnitude=f['reference_magnitude'][:];velocity=f['velocity'][:].transpose(1,2,3,4,0)
        mask=f['mask'][:];venc=float(f.attrs['Venc_m_per_s']);speed=f['speed'][:]
        if not np.allclose(f.attrs['FOV_m'],grid.fov,rtol=1e-6):raise ValueError('Unexpected FOV')
    checks=reconstruction_checks(magnitude,velocity,mask,speed,venc)
    mask=mask.astype(bool)
    core=np.all(mask,axis=0)
    support=RegularGridInterpolator(grid.axes()[::-1],core.astype(float),bounds_error=False,fill_value=0)(loc[:,::-1])
    common=support>=1-1e-10;fraction=float(area[common].sum()/area.sum())
    sensitivity={}
    for threshold in (.05,.15):
        selected=np.all(magnitude>=threshold*magnitude.max(axis=(1,2,3),keepdims=True),axis=0)
        stencils=RegularGridInterpolator(grid.axes()[::-1],selected.astype(float),bounds_error=False,fill_value=0)(loc[:,::-1])
        sensitivity[str(threshold)]=float(area[stencils>=1-1e-10].sum()/area.sum())
    n=z['normal_source'][[2,0,1]];means=[];kq=[]
    old=json.loads((e031/'inlet_readiness.json').read_text());w=z['frame_weights'];scale=old['timing']['velocity_scale']
    vn=np.einsum('ncf,c->nf',z['face_velocities_source_m_s'],z['normal_source'])
    tri_vn=vn[z['inferred_triangles']].mean(axis=1)
    cfd=(w@(area[common]@tri_vn[common]))*scale
    for b in range(8):
        mag=RegularGridInterpolator(grid.axes()[::-1],magnitude[b],bounds_error=False,fill_value=np.nan)(loc[:,::-1])/magnitude[b].max()
        means.append(float(area@mag/area.sum()))
        est=RegularGridInterpolator(grid.axes()[::-1],velocity[b],bounds_error=False,fill_value=np.nan)(loc[common,::-1])
        kq.append(float(area[common]@(est@n)))
    kq=np.array(kq)
    checks.update(inlet_each_bin_mean_magnitude_ge_05=bool(min(means)>=.05),fixed05_inlet_area_ge_25=bool(fraction>=.25),
        agrees_with_pre_raw_reference_gate=bool(np.allclose(means,reference['area_weighted_mean_magnitude_fraction'],rtol=1e-5,atol=1e-7)
            and np.isclose(fraction,reference['fixed05_complete_stencil_area_fraction'],rtol=1e-6)),
        actual_cpu_gpu_reset_checks=bool(reset['actual_cpu_magnetization_reset_passed'] and reset['actual_gpu_magnetization_reset_passed']))
    paired=bool(common.any())
    inputs={str(path):dict(bytes=path.stat().st_size,sha256=sha(path)) for path in
        [simulation_path,reset_path,reference_path,e031/'inlet_numerical_evidence.npz',e031/'inlet_readiness.json',
         root/'outputs/aorta_inlet_replenishment/PILOT_PROTOCOL_E033_E034.md']}
    report=dict(experiment='E-034',status='PASS_MRI_PILOT_VIABILITY' if all(checks.values()) else 'FAIL_MRI_PILOT_VIABILITY',
        checks=checks,fixed05_area_fraction=fraction,area_weighted_mean_magnitude_fraction=means,
        paired_partial_area_mm2=float(area[common].sum()),approximate_face_area_mm2=float(area.sum()),
        CFD_partial_ml_s=cfd.tolist() if paired else None,Koma_partial_ml_s=kq.tolist() if paired else None,
        **flow_metrics(cfd if paired else None,kq if paired else None),
        fixed_all_bin_threshold_coverage_sensitivity=sensitivity,inputs=inputs,
        timing='Original ADC averages/startframe18/velocity retiming; no fitted alignment',
        limitations=['Common fixed partial area only, not physical full-lumen inlet flow',
            'Inferred cap and kNN point-cloud boundaries; tracker-determined replacement count',
            'Pilot uses explicit Float32-safe reset partitioning with continuous RF state; reset-jump interpolation is still approximate',
            'Dependent internal comparison, not independent validation; no 1M acquisition authorized'],
        reconstruction_sha256=sha(args.run/'pc_reconstruction.h5'),audit_code_sha256=sha(__file__))
    tmp=args.output.with_name(args.output.name+'.building-'+uuid.uuid4().hex[:8]);tmp.mkdir(parents=True)
    (tmp/'pilot_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    np.savez_compressed(tmp/'paired_partial_flow_evidence.npz',common=common,area_mm2=area,cfd_ml_s=cfd,koma_ml_s=kq,
                        mean_magnitude=means)
    (tmp/'MANIFEST.json').write_text(json.dumps({p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in tmp.iterdir()},indent=2)+'\n',encoding='utf-8')
    tmp.rename(args.output);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
