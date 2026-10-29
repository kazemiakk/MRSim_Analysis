"""Independent NumPy/raw-phase/manual-stencil checks for the E-034 pilot."""
import argparse
import itertools
import json
from pathlib import Path
import tomllib
import h5py
import numpy as np
from PIL import Image
from scripts.aorta_hemodynamic_reference import AortaGrid,sequence_timing
from scripts.build_aorta_inlet_picker import sha


def manual_trilinear(values,axes,points):
    """Explicit eight-corner interpolation, separate from SciPy's audit path."""
    index=np.column_stack([(points[:,k]-axis[0])/(axis[1]-axis[0]) for k,axis in enumerate(axes)])
    low=np.floor(index).astype(int);fraction=index-low
    if np.any(low<0) or any(np.any(low[:,k]+1>=len(axis)) for k,axis in enumerate(axes)):
        raise ValueError('Quadrature leaves the grid')
    result=np.zeros((len(points),)+values.shape[3:],dtype=float)
    for corner in itertools.product((0,1),repeat=3):
        pos=low+corner
        weight=np.prod(np.where(np.array(corner),fraction,1-fraction),axis=1)
        sampled=values[tuple(pos.T)]
        result+=sampled*weight.reshape((-1,)+(1,)*(sampled.ndim-1))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--figures',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    root=Path(__file__).resolve().parents[1]
    audit=json.loads(args.audit.read_text());checks=[];metrics={}
    def check(name,condition):
        if not condition:raise AssertionError(name)
        checks.append(name)
    for path,entry in audit['inputs'].items():
        source=Path(path)
        check('source '+str(source),source.stat().st_size==entry['bytes'] and sha(source)==entry['sha256'])
    recon=root/'outputs/aorta_inlet_replenishment/e034_100k_reconstruction_v1'
    acquisition=root/'outputs/aorta_inlet_replenishment/e034_100k_mri_v2'
    simulation=tomllib.loads((recon/'simulation_report.toml').read_text())
    reset=tomllib.loads((acquisition/'reset_preflight.toml').read_text())
    with h5py.File(recon/'pc_reconstruction.h5','r') as f:
        mag=f['reference_magnitude'][:];mask=f['mask'][:].astype(bool)
        velocity=f['velocity'][:].transpose(1,2,3,4,0);speed=f['speed'][:]
        venc=float(f.attrs['Venc_m_per_s']);rr=float(f.attrs['RR_s'])
    check('reconstruction digest',sha(recon/'pc_reconstruction.h5')==audit['reconstruction_sha256']==simulation['reconstruction_sha256'])
    check('saved speed norm',np.allclose(speed,np.sqrt((velocity**2).sum(axis=-1)),rtol=2e-6,atol=1e-8))
    images={}
    seqdir=root/'outputs/aorta_dp2_mri_pipeline/koma/multibin_8x4mm/sequences'
    z=np.load(root/'outputs/aorta_inlet_flow/e031_v1/inlet_numerical_evidence.npz',allow_pickle=False)
    for encoding in ('REF','X','Y','Z'):
        rawpath=acquisition/(encoding.lower()+'_result.jld2')
        check(encoding+' raw hash',sha(rawpath)==simulation['raw_result_sha256'][encoding])
        with h5py.File(rawpath,'r') as f:
            data=f['image'][:];image=data['re']+1j*data['im']
            raw=f['rawsort'][:];raw=raw['re']+1j*raw['im']
            check(encoding+' label',f['encoding'][()].decode()==encoding)
            check(encoding+' phantom binding',f['phantom_sha256'][()].decode()==simulation['phantom_sha256'])
            check(encoding+' sequence binding',f['sequence_sha256'][()].decode()==sha(seqdir/(encoding.lower()+'.seq'))==simulation['sequence_sha256'][encoding])
        rebuilt=np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(raw,axes=(1,2,3)),axes=(1,2,3)),axes=(1,2,3))
        metrics[encoding+'_fft_relative_max_error']=float(np.max(abs(rebuilt-image))/np.max(abs(image)))
        check(encoding+' NumPy FFT',metrics[encoding+'_fft_relative_max_error']<2e-6)
        check(encoding+' finite shape',image.shape==(8,32,32,72) and np.isfinite(image).all())
        images[encoding]=image
        timing,weights,records=sequence_timing(seqdir/(encoding.lower()+'.seq'),rr)
        check(encoding+' ADC weights',np.allclose(weights,z['frame_weights'],rtol=0,atol=2e-12))
        check(encoding+' complete ADC labels',len(records)==8192 and len({(r['bin'],r['line'],r['partition']) for r in records})==8192)
        check(encoding+' reset windows',reset['encodings'][encoding]['repeated_reset_windows_checked']==12651)
    check('raw REF magnitude',np.allclose(abs(images['REF']),mag,rtol=1e-6,atol=1e-8))
    expected=mag>=np.float32(.05)*mag.max(axis=(1,2,3),keepdims=True)
    check('original threshold identity',np.array_equal(expected,mask))
    for k,encoding in enumerate(('X','Y','Z')):
        phase=np.angle(images[encoding]*images['REF'].conj())*np.float32(venc/np.float32(np.pi))
        error=float(np.max(abs(phase[mask]-velocity[...,k][mask])))
        metrics[encoding+'_raw_phase_velocity_max_error_m_s']=error
        check(encoding+' raw phase',error<1e-6)
        check(encoding+' excluded zeros',np.all(velocity[...,k][~mask]==0))
    grid=AortaGrid();axis=grid.axes()[::-1];center=np.asarray(grid.center)
    loc=((z['triangle_centroids_source_mm']/1000-center)[:,grid.order])[:,::-1]
    area=z['triangle_area_mm2'];normal=z['normal_source'][list(grid.order)]
    core=np.all(mask,axis=0)
    common=manual_trilinear(core.astype(float),axis,loc)>=1-1e-10
    evidence=np.load(args.audit.parent/'paired_partial_flow_evidence.npz',allow_pickle=False)
    check('manual fixed-area triangle membership',np.array_equal(common,evidence['common']))
    check('manual partial area',np.isclose(area[common].sum(),audit['paired_partial_area_mm2'],rtol=1e-12))
    qk=[];means=[]
    for b in range(8):
        estimate=manual_trilinear(velocity[b],axis,loc[common])
        qk.append(sum(area[common]*np.sum(estimate*normal,axis=1)))
        means.append(sum(area*manual_trilinear(mag[b],axis,loc))/area.sum()/mag[b].max())
    check('manual Koma quadrature',np.allclose(qk,audit['Koma_partial_ml_s'],rtol=1e-11,atol=1e-10))
    check('manual inlet magnitude',np.allclose(means,audit['area_weighted_mean_magnitude_fraction'],rtol=1e-11))
    # Reorder the sums relative to the authoring audit: area-weighted nodal
    # velocity vectors first, then dot normal and apply the actual ADC weights.
    triangles=z['inferred_triangles'][common]
    integrated=np.sum(z['face_velocities_source_m_s'][triangles].mean(axis=1)*area[common,None,None],axis=0)
    source_flow=np.sum(integrated*z['normal_source'][:,None],axis=0)
    qc=z['frame_weights']@source_flow*(.72/rr)
    check('alternative CFD quadrature',np.allclose(qc,audit['CFD_partial_ml_s'],rtol=1e-12))
    qk=np.array(qk)
    rmse=np.sqrt(sum((qk-qc)**2)/8)
    check('RMSE arithmetic',np.isclose(rmse,audit['partial_flow_rmse_ml_s'],rtol=1e-12))
    check('correlation arithmetic',np.isclose(np.corrcoef(qk,qc)[0,1],audit['partial_flow_correlation'],rtol=1e-12))
    check('cycle-volume arithmetic',np.isclose(qk.sum()/qc.sum(),audit['cycle_volume_ratio'],rtol=1e-12))
    metrics.update(CFD_partial_nominal_bin_cycle_volume_ml=float(qc.sum()*rr/8),
        Koma_partial_nominal_bin_cycle_volume_ml=float(qk.sum()*rr/8),
        maximum_absolute_bin_flow_error_ml_s=float(np.max(abs(qk-qc))),
        range_normalized_flow_RMSE=float(rmse/np.ptp(qc)))
    sensitivity=manual_trilinear(np.all(mag>=.15*mag.max(axis=(1,2,3),keepdims=True),axis=0).astype(float),axis,loc)>=1-1e-10
    check('15 percent coverage sensitivity',np.isclose(area[sensitivity].sum()/area.sum(),audit['fixed_all_bin_threshold_coverage_sensitivity']['0.15'],rtol=1e-12))
    for folder in (args.audit.parent,args.figures):
        manifest=json.loads((folder/'MANIFEST.json').read_text())
        files=manifest.get('files',manifest)
        for name,entry in files.items():
            path=folder/name
            check('artifact '+name,path.stat().st_size==entry['bytes'] and sha(path)==entry['sha256'])
            if path.suffix=='.png':
                with Image.open(path) as image:
                    image.verify()
                check('PNG decoded '+name,True)
            if path.suffix=='.pdf':check('PDF signature '+name,path.read_bytes().startswith(b'%PDF-'))
    figures=json.loads((args.figures/'MANIFEST.json').read_text())
    check('figure audit binding',figures['audit_sha256']==sha(args.audit))
    report=dict(status='PASS_INDEPENDENT_SAVED_PILOT_CHECKS',check_count=len(checks),checks=checks,metrics=metrics,
        audit_sha256=sha(args.audit),validator_sha256=sha(__file__),
        scope='Independent arithmetic/implementation cross-checks of the same simulated source, not independent physical validation',
        PDF_visual_verification='PNG exports visually inspected; PDFs checked by signature/hash only')
    with args.output.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(status=report['status'],check_count=len(checks),metrics=metrics),indent=2))


if __name__=='__main__':main()
