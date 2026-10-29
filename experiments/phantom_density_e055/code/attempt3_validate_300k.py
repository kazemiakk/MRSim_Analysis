"""Independently reproduce E055 retained-support errors and section fluxes."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np


def digest(path):
    with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def validate(repo,run,analysis=None):
    analysis=analysis or run/'analysis_v1'
    report_path=analysis/'report.json';report=json.loads(report_path.read_text())
    refpath=repo/'outputs/phantom_diagnosis_e048/reference_v1/reference_fields.h5'
    localpath=repo/'outputs/phantom_diagnosis_e048/local_v1/local_fields.h5'
    assert digest(refpath)=='f4c4ec3153d30a2a9dfd360d8ff1c21811e14e6f942697cdd990b669fec75694'
    assert digest(localpath)=='6c57a8708ea4a5aff09f61db232b0ba1528ae4d000bf60d9ca00524516a27885'
    checks=2
    with h5py.File(refpath) as f:reference=f['fft_adc'][:];fraction=f['fraction_fft'][:].astype(float)
    with h5py.File(localpath) as f:source_z=f['source_z_fft_m'][:]
    with h5py.File(analysis/'paired_fields.h5') as f:
        fields={a:f[a][:] for a in ['baseline100k','candidate300k']}
        saved_masks={a:f[a+'_fixed5_mask'][:].astype(bool) for a in fields}
        saved_pair=f['paired_fixed5_mask'][:].astype(bool)
    masks={};folders={'baseline100k':repo/'outputs/phantom_precision_e050/mapped_f64_v1','candidate300k':run/'mapped_f64_v1'}
    for a,folder in folders.items():
        for threshold in [.05,.1,.15,.2]:masks[a,threshold]=np.ones(fraction.shape,bool)
        for e in ['ref','x','y','z']:
            with h5py.File(folder/(e+'_result.h5')) as f:magnitude=abs(f['image'][:])
            maxima=magnitude.max(axis=(1,2,3),keepdims=True)
            for threshold in [.05,.1,.15,.2]:masks[a,threshold]&=((magnitude>=threshold*maxima)&(magnitude>0)).all(axis=0)
        assert np.array_equal(masks[a,.05],saved_masks[a]);checks+=1
    assert np.array_equal(saved_pair,masks['baseline100k',.05]&masks['candidate300k',.05]);checks+=1
    bounds={'whole':(-np.inf,np.inf),'upstream':(-.080,-.020),'throat':(-.010,.010),
            'near_downstream':(.010,.060),'mid_downstream':(.060,.140),'far_downstream':(.140,.300)}
    for row in report['regions']:
        a=row['arm'];threshold=row['threshold'];low,high=bounds[row['region']]
        eligible=(fraction>0)&((source_z>=low)&(source_z<high))[None,None,:]
        support=masks[a,threshold] if row['support']=='own' else masks['baseline100k',threshold]&masks['candidate300k',threshold]
        keep=eligible&support
        cov=float(fraction[keep].sum()/fraction[eligible].sum()) if eligible.any() else None
        if cov is None:assert row['coverage_fraction'] is None
        else:np.testing.assert_allclose(cov,row['coverage_fraction'],atol=1e-12,rtol=0)
        checks+=1
        if not keep.any():assert row['metrics'] is None;continue
        delta=fields[a][:,keep].astype(float)-reference[:,keep].astype(float)
        squared=np.sum(delta*delta,axis=2);w=fraction[keep]
        rmse=float(np.sqrt(np.sum(squared*w[None,:])/(15*w.sum())))
        np.testing.assert_allclose(rmse,row['metrics']['vector_rmse_m_s'],atol=1e-10,rtol=0);checks+=1
        assert int(keep.sum())==row['metrics']['voxels'];checks+=1
    with h5py.File(localpath) as f:
        for row in report['sections']:
            idx=row['index'];a=row['arm'];threshold=row['threshold']
            support=masks[a,threshold] if row['support']=='own' else masks['baseline100k',threshold]&masks['candidate300k',threshold]
            area=f['plane_fraction'][:,:,idx].astype(float)*3.125;keep=support[:,:,idx]&(area>0)
            if not keep.any():assert row['CFD_ml_s'] is None and row['Koma_ml_s'] is None;checks+=1;continue
            cfd=f['plane_reference'][:,:,:,idx,0].astype(float)
            cfd_q=np.sum(cfd[:,keep]*area[keep],axis=1)
            koma_q=np.sum(fields[a][:,:,:,idx,0][:,keep].astype(float)*area[keep],axis=1)
            np.testing.assert_allclose(cfd_q,row['CFD_ml_s'],atol=3e-5,rtol=1e-6)
            np.testing.assert_allclose(koma_q,row['Koma_ml_s'],atol=3e-5,rtol=1e-6);checks+=2
            rmse=float(np.sqrt(np.mean((koma_q-cfd_q)**2)))
            np.testing.assert_allclose(rmse,row['metrics']['rmse_ml_s'],atol=3e-5,rtol=1e-6);checks+=1
    for audit in report['reconstruction_audits'].values():
        for row in audit:
            assert row['raw_signal_sort_exact_after_f32_cast'] and row['independent_fft_relative_max']<=3e-6;checks+=1
    decision=report['one_million_decision']
    assert decision['new_1m_started'] is False and decision['independent_seed_variability_established'] is False;checks+=1
    assert decision['status'].startswith('HOLD_1M_');checks+=1
    receipt=dict(status='PASS_INDEPENDENT_300K_MASK_ERROR_AND_FLOW_CHECKS',checks=checks,
                 utc=datetime.now(timezone.utc).isoformat(),report_sha256=digest(report_path),new_1m_started=False)
    target=analysis/'validation.json'
    if target.exists():raise FileExistsError(target)
    target.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt),flush=True)
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();validate(a.repo,a.run)
