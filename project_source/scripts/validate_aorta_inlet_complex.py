"""Independent saved-data/manual complex-quadrature cross-check for E-036."""
import argparse
import csv
import json
from pathlib import Path
import h5py
import numpy as np
from PIL import Image
from scripts.aorta_hemodynamic_reference import AortaGrid
from scripts.build_aorta_inlet_picker import sha
from scripts.validate_aorta_replenishment_mri import manual_trilinear


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True,type=Path)
    p.add_argument('--figures',required=True,type=Path);p.add_argument('--output',required=True,type=Path);args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    root=Path(__file__).resolve().parents[1];registration=json.loads((args.run/'registration.json').read_text())
    report_path=args.run/'complex_inlet_report.json';report=json.loads(report_path.read_text())
    e=np.load(args.run/'complex_inlet_evidence.npz',allow_pickle=False);checks=[]
    def check(name,value):
        if not value:raise AssertionError(name)
        checks.append(name)
    for path,entry in registration['inputs'].items():
        path=Path(path);check('input '+str(path),path.stat().st_size==entry['bytes'] and sha(path)==entry['sha256'])
    check('authoring code binding',sha(root/'scripts/integrate_aorta_inlet_complex.py')==registration['code_sha256'])
    check('registration binding',sha(args.run/'registration.json')==report['registration_sha256'])
    check('controls passed before real flow',registration['synthetic_controls']['status']=='PASS_SYNTHETIC_CONTROLS' and registration['CFD_flow_not_yet_evaluated'])
    grid=AortaGrid();samples=[];ratios=[];base=root/'outputs/aorta_inlet_replenishment'
    for name in ('REF','X','Y','Z'):
        with h5py.File(base/'e034_100k_mri_v2'/(name.lower()+'_result.jld2'),'r') as f:a=f['image'][:]
        a=a['re']+1j*a['im'];values=[]
        for image in a:
            values.append(manual_trilinear(image.real,grid.axes()[::-1],e['query_mri_zyx_m'])
                          +1j*manual_trilinear(image.imag,grid.axes()[::-1],e['query_mri_zyx_m']))
        values=np.array(values);samples.append(values);ratios.append(abs(values)/abs(a).max(axis=(1,2,3))[:,None])
    ratio=np.stack(ratios,axis=-1);valid=np.all(ratio>=.05,axis=-1);common=np.all(valid,axis=0)
    check('manual complex signal samples',np.allclose(ratio,e['relative_complex_signal'],rtol=1e-9,atol=1e-11))
    check('manual common-area membership',np.array_equal(common,e['fixed']))
    check('per-bin validity identity',np.array_equal(valid,e['valid']))
    with h5py.File(base/'e034_100k_reconstruction_v1/pc_reconstruction.h5','r') as f:venc=float(f.attrs['Venc_m_per_s'])
    velocity=np.stack([np.arctan2((a*samples[0].conj()).imag,(a*samples[0].conj()).real)*(venc/np.pi) for a in samples[1:]],axis=-1)
    check('missing velocities remain unavailable',np.isnan(e['velocity_mri_m_s'][~valid]).all())
    check('phase velocity identity',np.allclose(velocity[valid],e['velocity_mri_m_s'][valid],rtol=1e-9,atol=1e-10))
    area=e['area_mm2'];normal=e['normal_mri']
    qk=np.sum(np.sum(velocity[:,common]*normal,axis=-1)*area[common],axis=1)
    check('independent Koma area quadrature',np.allclose(qk,report['Koma_physical_ml_s'],rtol=1e-10,atol=1e-9))
    z=np.load(root/'outputs/aorta_inlet_flow/e031_v1/inlet_numerical_evidence.npz',allow_pickle=False)
    triangles=z['inferred_triangles'][common]
    integrated=np.sum(z['face_velocities_source_m_s'][triangles].mean(axis=1)*area[common,None,None],axis=0)
    native_q=np.sum(integrated*z['normal_source'][:,None],axis=0)
    qc=e['frame_weights']@native_q*report['known_source_time_velocity_scale']
    check('independent CFD quadrature',np.allclose(qc,report['CFD_physical_ml_s'],rtol=1e-12))
    m=report['physical_units_metrics'];err=qk-qc
    check('RMSE',np.isclose(np.sqrt(np.mean(err**2)),m['rmse_ml_s'],rtol=1e-10))
    check('range NRMSE',np.isclose(np.sqrt(np.mean(err**2))/np.ptp(qc),m['range_normalized_rmse'],rtol=1e-10))
    check('correlation',np.isclose(np.corrcoef(qc,qk)[0,1],m['correlation'],rtol=1e-10))
    check('cycle volume ratio',np.isclose(qk.sum()/qc.sum(),m['nominal_bin_cycle_volume_ratio'],rtol=1e-10))
    check('area coverage',np.isclose(area[common].sum()/area.sum(),report['coverage_fraction'],rtol=1e-12))
    check('excluded area',np.isclose(area[~common].sum(),report['excluded_area_mm2'],rtol=1e-12))
    check('not full-face promoted',not report['full_face_supported'] and report['coverage_fraction']<1)
    scale=report['known_source_time_velocity_scale']
    check('known retiming not fitted gain',scale==.72/report['physical_RR_s'])
    check('native-time Koma units',np.allclose(qk/scale,report['Koma_native_time_ml_s'],rtol=1e-10))
    check('native-time CFD units',np.allclose(qc/scale,report['CFD_native_time_ml_s'],rtol=1e-12))
    centers=(np.angle(e['frame_weights']@np.exp(2j*np.pi*np.arange(72)/72))%(2*np.pi))*.72/(2*np.pi)
    check('known source-phase centers',np.allclose(centers,report['native_source_phase_centers_s'],atol=1e-12))
    check('source ordering not fitted',report['native_phase_bin_order_one_based']==[7,8,1,2,3,4,5,6])
    rows=list(csv.DictReader((args.run/'inlet_flow_bins.csv').open(newline='')))
    check('CSV eight actual bins',len(rows)==8)
    check('CSV flow values',np.allclose([float(r['Koma_physical_ml_s']) for r in rows],qk,rtol=1e-10))
    for folder in (args.run,args.figures):
        manifest=json.loads((folder/'MANIFEST.json').read_text())
        for name,entry in manifest.get('files',manifest).items():
            path=folder/name;check('artifact '+name,path.stat().st_size==entry['bytes'] and sha(path)==entry['sha256'])
            if path.suffix=='.png':
                with Image.open(path) as image:image.verify()
                check('PNG decode '+name,True)
            if path.suffix=='.pdf':check('PDF signature '+name,path.read_bytes().startswith(b'%PDF-'))
    figure_manifest=json.loads((args.figures/'MANIFEST.json').read_text())
    check('figure report binding',figure_manifest['report_sha256']==sha(report_path))
    check('figure code binding',figure_manifest['plotter_sha256']==sha(root/'scripts/plot_aorta_inlet_complex.py'))
    result=dict(status='PASS_COMPLEX_INTEGRATION_SAVED_DATA_CHECKS',check_count=len(checks),checks=checks,
        report_sha256=sha(report_path),validator_sha256=sha(__file__),
        scope='Independent implementation/arithmetic checks of the same source, not independent physical validation',
        PDF_verification='Hash/signature checked; separate rendered PDF inspection not performed')
    with args.output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(dict(status=result['status'],check_count=len(checks))))


if __name__=='__main__':main()
