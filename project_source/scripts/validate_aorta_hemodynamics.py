"""Independent arithmetic reread of E-027 saved fields; no pipeline imports."""
import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime,timezone
import h5py
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);args=p.parse_args()
    folder=args.results.resolve(); report=json.loads((folder/'report.json').read_text(encoding='utf-8'))
    root=Path(__file__).resolve().parents[1]; checks=0
    def close(a,b):
        nonlocal checks
        np.testing.assert_allclose(a,b,rtol=2e-6,atol=1e-10);checks+=1
    with h5py.File(folder/'matched_reference.h5','r') as f:
        a=f['adc_time_average'][...].astype(float); b=f['Koma'][...].astype(float)
        mask=f['masks/core15'][...].astype(bool); fraction=f['fraction'][...].astype(float)
        weight=np.broadcast_to(fraction,a.shape[:-1]); valid=np.broadcast_to(mask,a.shape[:-1])
        difference=b[valid]-a[valid]; w=weight[valid]
        rmse=np.sqrt(np.sum(w*np.sum(difference**2,axis=-1))/w.sum())
        nrmse=np.sqrt(np.sum(w*np.sum(difference**2,axis=-1))/np.sum(w*np.sum(a[valid]**2,axis=-1)))
        close(rmse,report['primary_pooled']['vector_rmse_m_s'])
        close(nrmse,report['primary_pooled']['vector_nrmse_reference_rms'])
        for component,name in enumerate(('X','Y','Z','speed')):
            ar=a[...,component] if component<3 else np.linalg.norm(a,axis=-1)
            br=b[...,component] if component<3 else np.linalg.norm(b,axis=-1)
            delta=br[valid]-ar[valid]
            for key,value in [('bias',np.average(delta,weights=w)),('mae',np.average(abs(delta),weights=w)),('rmse',np.sqrt(np.average(delta**2,weights=w)))]:
                close(value,report['primary_pooled'][name][key])
        spacing=np.asarray(report['grid']['fov'])/np.asarray(report['grid']['shape'][::-1])
        area=f['plane_fraction'][...]*spacing[1]*spacing[2]
        for row in report['section_series']:
            section=next(s for s in report['section_summary'] if s['section']==row['section'])
            roi=f['sections/'+row['section']][...]; i=int(np.argmin(abs((np.arange(72)-36)*spacing[0]-section['x_m'])))
            paired=roi&mask[:,:,i]&(area[:,:,i]>0)
            u=(a if row['method']=='CFD' else b)[row['bin']-1,:,:,i,0]
            flux=u[paired]*area[:,:,i][paired]
            close(flux.sum()*1e6,row['net_ml_s'])
            close(np.maximum(flux,0).sum()*1e6,row['forward_ml_s'])
            close(np.maximum(-flux,0).sum()*1e6,row['reverse_ml_s'])
            close(area[:,:,i][paired].sum()*1e6,row['sampled_area_mm2'])
        for summary in report['section_summary']:
            for method in ['CFD','Koma']:
                q=np.array([r['net_m3_s'] for r in report['section_series'] if r['section']==summary['section'] and r['method']==method])
                close(q.sum()*report['period_s']/8,summary[method+'_cycle']['net_m3'])
        x=(np.arange(72)-36)*spacing[0]
        regions={'whole':mask,'readout_below_minus45':mask&(x[None,None,:]<-.045),
                 'readout_middle':mask&(x[None,None,:]>=-.045)&(x[None,None,:]<=.045),
                 'readout_above45':mask&(x[None,None,:]>.045)}
        for row in report['regional_series']:
            m=regions[row['region']];u=(a if row['method']=='CFD' else b)[row['bin']-1]
            volume=fraction[m]*np.prod(spacing)
            close(.5*np.sum(np.sum(u[m]**2,axis=-1)*volume),row['energy_per_density_m5_s2'])
            close(volume.sum(),row['sampled_volume_m3'])
        close(fraction[mask].sum()*np.prod(spacing)*1e6,report['coverage']['common_numerical_volume_ml'])
        close(np.sum(mask),report['coverage']['primary_voxels'])
        close(f['frame_weights'][...].sum(axis=1),np.ones(24))
    for row in report['inputs']:
        path=root/row['path'];close(path.stat().st_size,row['bytes'])
        with path.open('rb') as stream: actual=hashlib.file_digest(stream,'sha256').hexdigest()
        if actual!=row['sha256']:raise ValueError('Bound input changed: '+row['path'])
        checks+=1
    for name,item in json.loads((folder/'MANIFEST.json').read_text()).items():
        path=folder/name;close(path.stat().st_size,item['bytes'])
        with path.open('rb') as stream: actual=hashlib.file_digest(stream,'sha256').hexdigest()
        if actual!=item['sha256']: raise ValueError('Output manifest mismatch: '+name)
        checks+=1
    result={'status':'PASS','checks':checks,'failed_checks':0,'created_utc':datetime.now(timezone.utc).isoformat(),
            'report_sha256':hashlib.sha256((folder/'report.json').read_bytes()).hexdigest(),
            'vector_rmse_m_s':float(rmse),'vector_nrmse_reference_rms':float(nrmse),
            'scope':'Independent saved-field arithmetic, identity and output-hash checks; not independent physical validation or a reimplementation of source interpolation/ADC parsing'}
    destination=folder.parent/'validation';destination.mkdir(exist_ok=True)
    with (destination/'numeric_validation.json').open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
