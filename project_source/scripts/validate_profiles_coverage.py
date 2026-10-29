"""Independent saved-source checks for E-039 (does not alter figure bundle)."""
import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import binary_erosion, generate_binary_structure

from scripts.complete_advanced_comparisons import load_inputs, EXT
from scripts.extend_hemodynamic_comparisons import DP, AO, DP_H5, PILOT, ROOT, sha
from scripts.compare_aorta_hemodynamics import dump


def validate(run, receipt):
    report=json.loads((run/'report.json').read_text())
    checks=[]
    def check(name, value):
        checks.append(dict(name=name,passed=bool(value)))
        if not value:raise AssertionError(name)
    def close(name,a,b,rtol=1e-10,atol=1e-12):
        check(name,np.allclose(a,b,rtol=rtol,atol=atol,equal_nan=True))
    for filename,record in json.loads((run/'MANIFEST.json').read_text()).items():
        check('output hash '+filename,sha(run/filename)==record['sha256'])
        check('output size '+filename,(run/filename).stat().st_size==record['bytes'])
        if filename.endswith('.pdf'):check('PDF signature '+filename,(run/filename).read_bytes()[:5]==b'%PDF-')
        if filename.endswith('.png'):check('PNG signature '+filename,(run/filename).read_bytes()[:8]==b'\x89PNG\r\n\x1a\n')
    for item in report['inputs']:check('source hash '+item['path'],sha(ROOT/item['path'])==item['sha256'])
    _,cases,dp,ao,ext,references,_=load_inputs()
    selections=report['phase_selection']
    check('Twelve PNG/PDF pairs',len(list(run.glob('*.png')))==12 and len(list(run.glob('*.pdf')))==12)
    with h5py.File(run/'distribution_ecdfs.h5') as curves:
        for case in cases:
            name=case['name'];volume=case['volume'];whole=case['regions']['whole']
            if name=='phantom_1m':source=[r for r in dp['section_values'] if r['dataset']=='cfd_voxel_common']
            elif name=='aorta_1m':source=[r for r in ao['section_series'] if r['method']=='CFD']
            else:source=[r for r in ext['aorta_pilot_sections']['series'] if r['method']=='CFD']
            for selected in [s for s in selections if s['case']==name]:
                rows=sorted([r for r in source if r['section']==selected['section']],key=lambda r:r['bin'])
                q=np.array([r['net_m3_s'] for r in rows]);index=int(np.argmax(np.abs(q)) if selected['phase']=='peak' else np.argmin(np.abs(q)))
                check(name+' selection '+selected['section']+selected['phase'],selected['bin']==rows[index]['bin'])
                close('selected flow',selected['CFD_flow_ml_s'],q[index]*1e6)
                close('selected time',selected['time_s'],case['times'][index])
                check('tail flag',selected['synthetic_tail']==(name=='phantom_1m' and selected['bin']>=14))
            for selected in report['cases'][name]['regional_selections']:
                b=selected['bin']-1;phase=selected['phase']
                for region,mask in case['regions'].items():
                    pair=mask & np.isfinite(case['reference'][b]).all(axis=-1) & np.isfinite(case['estimate'][b]).all(axis=-1) & (volume>0)
                    weights=volume[pair]
                    values={'CFD_speed':np.linalg.norm(case['reference'][b,pair],axis=-1),
                        'Koma_speed':np.linalg.norm(case['estimate'][b,pair],axis=-1),
                        'vector_error':np.linalg.norm(case['estimate'][b,pair]-case['reference'][b,pair],axis=-1)}
                    for metric,v in values.items():
                        order=np.argsort(v,kind='stable');sv=v[order];sw=weights[order]
                        last=np.r_[np.where(np.diff(sv)!=0)[0],len(sv)-1] if len(sv) else np.array([],int)
                        expected=np.column_stack((sv[last],np.cumsum(sw)[last]/sw.sum())) if len(sv) else np.empty((0,2))
                        key=f'{name}/{region}/{phase}/{metric}'
                        actual=curves[key][:]
                        close('ECDF '+key,actual,expected,rtol=1e-10)
                        if len(actual):close('ECDF closes '+key,actual[-1,1],1)
            if name=='phantom_1m':
                with h5py.File(DP/'matched_reference.h5') as f:
                    direct=f['direct_common_mask'][:];fraction=f['volume_fraction_2'][:]
                    refs={'Midpoint (primary)':case['reference'],'Three-sample box':f['cfd_voxel_three_sample_mean'][:].astype(float),'3-cubed reference':f['cfd_voxel_3x3x3'][:].astype(float)}
                with h5py.File(DP_H5) as f:raw=np.moveaxis(f['velocity_raw_m_per_s'][:],0,-1).astype(float)
                inner=whole&(fraction>=.999999);eroded=binary_erosion(inner)
                masks={'Hybrid common':whole,'Raw direct only':direct,'Regularized only':whole&~direct,
                    'Full-voxel interior':inner,'Eroded interior':eroded,'Exclude synthetic-tail bins':whole}
                gmasks={'Primary interior':inner,'Direct-only stencil':inner&direct,'Extra erosion':eroded,'Matched 1 mm smoothing':inner}
            else:
                if name=='aorta_1m':
                    with h5py.File(AO/'matched_reference.h5') as f:masks={k:f['masks/'+k][:] for k in f['masks']}
                else:
                    with h5py.File(PILOT/'pc_reconstruction.h5') as f:mag=f['reference_magnitude'][:];stored=f['mask'][:].astype(bool)
                    maximum=mag.reshape(len(mag),-1).max(axis=1)
                    masks={f'core{int(t*100):02}':np.all(stored&(mag>=t*maximum[:,None,None,None]),axis=0)&(volume>0) for t in [.05,.10,.15,.20]}
                    masks['eroded15']=binary_erosion(masks['core15'],structure=np.ones((3,3,3)))
                refs=references;gmasks={'Primary core15':whole,'Extra erosion':masks['eroded15'],'Matched 4 mm smoothing':whole}
            for row in [r for r in report['analysis_choices'] if r['case']==name]:
                bins=np.arange(13) if row['choice']=='Exclude synthetic-tail bins' else np.arange(len(case['reference']))
                if row['family']=='derivative':
                    mask=binary_erosion(gmasks[row['choice']],structure=generate_binary_structure(3,1))
                else:
                    mask=masks[row['choice']] if row['family']=='support' else whole
                    ref=refs[row['choice']] if row['family']=='reference' else case['reference']
                    est=raw if name=='phantom_1m' and row['choice']=='Raw direct only' else case['estimate']
                    mask=mask&np.isfinite(ref[bins]).all(axis=(0,4))&np.isfinite(est[bins]).all(axis=(0,4))&(volume>0)
                    w=volume[mask];a=ref[bins][:,mask];c=est[bins][:,mask]
                    mse=np.average(np.sum((c-a)**2,axis=-1),weights=np.broadcast_to(w,(len(bins),len(w))))
                    mean_square=np.average(np.sum(a*a,axis=-1),weights=np.broadcast_to(w,(len(bins),len(w))))
                    close('vector RMSE '+row['choice'],row['rmse'],np.sqrt(mse))
                    close('vector NRMSE '+row['choice'],row['nrmse_pct'],100*np.sqrt(mse/mean_square))
                check(name+' voxel count '+row['choice'],row['voxels']==int(mask.sum()))
                close('retained volume '+row['choice'],row['retained_volume_ml'],volume[mask].sum()*1e6)
                close('coverage '+row['choice'],row['retained_coverage_pct'],100*volume[mask].sum()/volume.sum())
                check('retained bins '+row['choice'],row['retained_bins']==len(bins))
                if not mask.any():check('empty is null, not zero',row['rmse'] is None and row['nrmse_pct'] is None)
    with (run/'selected_profiles.csv').open(newline='',encoding='utf-8') as stream:profiles=list(csv.DictReader(stream))
    with (run/'regional_distribution_summary.csv').open(newline='',encoding='utf-8') as stream:summaries=list(csv.DictReader(stream))
    by_case={case['name']:case for case in cases}
    with h5py.File(run/'distribution_ecdfs.h5') as curves:
        for row in summaries:
            case=by_case[row['case']];mask=case['regions'][row['region']]
            metric='vector_error' if row['quantity']=='vector_error_m_s' else row['method']+'_speed'
            values=curves[f"{row['case']}/{row['region']}/{row['phase']}/{metric}"][:]
            for q in [.05,.25,.5,.75,.95,.99]:
                quantile=values[min(np.searchsorted(values[:,1],q),len(values)-1),0]
                close('summary quantile',float(row[f'p{int(q*100):02}']),quantile)
            close('summary paired volume',float(row['volume_ml']),case['volume'][mask].sum()*1e6)
            check('summary voxel count',int(row['voxels'])==int(mask.sum()))
    with (DP/'profiles.csv').open(newline='',encoding='utf-8') as stream:original_dp=list(csv.DictReader(stream))
    original={('phantom_1m',r['section'],int(r['bin']),{'cfd':'CFD','koma':'Koma'}[r['dataset']],r['profile_axis'][-1],float(r['coordinate_mm'])):float(r['axial_velocity_m_s']) if r['axial_velocity_m_s'] else None for r in original_dp}
    for name,rows in [('aorta_1m',ao['profile_series']),('aorta_100k_corrected',ext['aorta_pilot_sections']['profiles'])]:
        for r in rows:original[(name,r['section'],r['bin'],r['method'],(r.get('axis') or r.get('transverse_axis'))[-1],r['coordinate_mm'])]=r['normal_velocity_m_s']
    for row in profiles:
        key=(row['case'],row['section'],int(row['bin']),row['method'],row['axis'],float(row['coordinate_mm']))
        expected=original[key]
        if expected is None or not np.isfinite(expected):check('preserved missing profile',row['velocity']=='')
        else:close('profile source equality',float(row['velocity']),expected)
    dump(receipt,dict(experiment='E-039',status='PASS',passed=len(checks),total=len(checks),checks=checks,
        scope='All exported profiles, phase selections, ECDF arrays, velocity-choice errors, support/stencil coverage, source/output hashes and file signatures; primary curl independently reproduced E038 during generation. No physical validation or rendered-PDF review implied.'))
    print(f'{len(checks)}/{len(checks)} independent source/export checks passed')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True)
    a=p.parse_args();validate(a.run,a.receipt)
