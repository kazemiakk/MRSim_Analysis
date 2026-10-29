"""Independent saved-field, arithmetic and provenance QA for E056."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,tomllib
import h5py
import numpy as np

root=Path('D:/MRSim/outputs/acquisition_physics_e056')
target=root/'verification_v1';stage=target.with_name(target.name+'.building')
if target.exists() or stage.exists():raise FileExistsError(target)
stage.mkdir();checks=0
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()
def array(d):
    a=d[:]
    if a.dtype.names:return a['r'].astype(float)+1j*a['i'].astype(float)
    return a
def close(a,b):
    global checks
    np.testing.assert_allclose(a,b,rtol=3e-10,atol=2e-12);checks+=1
r=json.loads((root/'analysis_v1/report.json').read_text(encoding='utf-8'))
for dirname in ['analysis_v1']:
    m=json.loads((root/dirname/'MANIFEST.json').read_text(encoding='utf-8'))
    for path,record in m['inputs'].items():
        assert Path(path).stat().st_size==record['bytes'] and sha(path)==record['sha256'],path;checks+=1
    for path,record in m['outputs'].items():
        p=root/dirname/path
        assert p.stat().st_size==record['bytes'] and sha(p)==record['sha256'],str(p);checks+=1
assert len(r['comparisons'])==9 and len(r['numerical_checks'])==36;checks+=2
assert r['new_1m_started'] is False;checks+=1
acq=tomllib.loads((root/'acquisition_v1/report.toml').read_text(encoding='utf-8'))
assert acq['status']=='COMPLETE_E056_SYNTHETIC_3D_ACQUISITIONS' and len(acq['result_sha256'])==36;checks+=2
arms=['baseline','bw700_te274','bw700_te210'];numerical=True;max_phase_identity=0.
with h5py.File(root/'analysis_v1/independent_fields.h5') as fields:
    for model in ['static','uniform_osc','parabolic_osc']:
        data={}
        for arm in arms:
            actual=[];predicted=[]
            for enc in ['REF','X','Y','Z']:
                p=root/'acquisition_v1'/f'{model}_{arm}_{enc.lower()}.h5'
                with h5py.File(p) as f:
                    raw=array(f['raw_signal_f64']).ravel();image=array(f['image'])
                    assert raw.shape==(614400,) and image.shape==(15,32,16,80);checks+=2
                    assert np.isfinite(raw).all() and np.isfinite(image).all();checks+=2
                g=fields[f'{model}/{arm}/{enc}'];reference=array(g['raw_signal_independent']).ravel();pi=array(g['image_independent'])
                row=next(x for x in r['numerical_checks'] if (x['model'],x['arm'],x['encoding'])==(model,arm,enc))
                # Dot products independently recompute complex L2 ratios.
                residual=raw-reference;image_residual=(image-pi).ravel()
                raw_l2=float(np.sqrt(np.vdot(residual,residual).real/np.vdot(reference,reference).real))
                image_l2=float(np.sqrt(np.vdot(image_residual,image_residual).real/np.vdot(pi.ravel(),pi.ravel()).real))
                close(raw_l2,row['raw_relative_l2']);close(image_l2,row['image_relative_l2'])
                passed=raw_l2<=.01 and image_l2<=.01
                assert passed==row['passed_numerical_gate'];checks+=1;numerical=numerical and passed
                actual.append(image);predicted.append(pi)
            g=fields[f'{model}/{arm}/analysis']
            a=np.stack(actual);p=np.stack(predicted)
            v=np.moveaxis(np.arctan2((a[1:]*a[0].conj()).imag,(a[1:]*a[0].conj()).real)*5/np.pi,0,-1)
            pv=np.moveaxis(np.arctan2((p[1:]*p[0].conj()).imag,(p[1:]*p[0].conj()).real)*5/np.pi,0,-1)
            # Retain the full spatial and bin field identity, not just means.
            max_phase_identity=max(max_phase_identity,float(np.max(abs(v-g['velocity_Koma'][:]))))
            close(v,g['velocity_Koma'][:]);close(pv,g['velocity_independent'][:])
            mag=abs(a);support=np.logical_and.reduce(np.logical_and.reduce(mag>=.05*mag.max(axis=(2,3,4),keepdims=True),axis=0),axis=0)
            np.testing.assert_array_equal(support,g['support'][:].astype(bool));checks+=1
            data[arm]=dict(v=v,pv=pv,ideal=g['velocity_instantaneous_analytic'][:],support=support,
                           core=g['core'][:].astype(bool),geometry=g['geometry'][:].astype(bool))
        common=np.logical_and.reduce([data[a]['core'] for a in arms])
        paired=common&np.logical_and.reduce([data[a]['support'] for a in arms])
        for arm in arms:
            d=data[arm];row=next(x for x in r['comparisons'] if x['model']==model and x['arm']==arm)
            np.testing.assert_array_equal(paired,fields[f'{model}/{arm}/analysis/paired_core'][:].astype(bool));checks+=1
            close((d['geometry']&d['support']).sum()/d['geometry'].sum(),row['own_geometry_coverage'])
            close((d['core']&d['support']).sum()/d['core'].sum(),row['own_core_coverage'])
            close(paired.sum()/common.sum(),row['identical_paired_core_coverage'])
            for field,label in [('ideal','Koma_vs_instantaneous_analytic'),('pv','Koma_vs_independent_complex_model')]:
                delta=d['v'][:,paired]-d[field][:,paired]
                vector=float(np.sqrt(np.mean(np.linalg.norm(delta,axis=-1)**2))) if paired.any() else None
                axial=float(np.sqrt(np.mean(delta[...,0]**2))) if paired.any() else None
                bias=float(delta[...,0].mean()) if paired.any() else None
                for val,name in [(vector,'vector_rmse_m_s'),(axial,'axial_rmse_m_s'),(bias,'axial_bias_m_s')]:
                    if val is None:assert row[label][name] is None;checks+=1
                    else:close(val,row[label][name])
                if field=='pv':numerical=numerical and vector is not None and vector<=.01
                if model=='static' and field=='ideal':numerical=numerical and vector is not None and vector<=.001
            flow=row['paired_partial_section_flow'];section=paired[:,:,40]
            if section.any():
                q=(d['v'][:,:,:,40,0][:,section]*3.125).sum(axis=1)
                truth=(d['ideal'][:,:,:,40,0][:,section]*3.125).sum(axis=1)
                close(q,flow['Koma_ml_s']);close(truth,flow['analytic_ml_s'])
                close(float(np.sqrt(np.mean(np.square(q-truth)))),flow['RMSE_ml_s'])
                close(float(section.sum()*3.125),flow['retained_area_mm2'])
                if abs(truth.mean())<1e-12:assert flow['relative_mean_bias'] is None;checks+=1
                else:close(float(q.mean()/truth.mean()-1),flow['relative_mean_bias'])
            else:assert flow is None;checks+=1
assert numerical==(r['status']=='PASS_INDEPENDENT_REPEATED_RF_COMPLEX_MODEL');checks+=1
for row in r['mechanism_tests']:
    b=next(x for x in r['comparisons'] if x['model']==row['model'] and x['arm']=='baseline')
    a=next(x for x in r['comparisons'] if x['model']==row['model'] and x['arm']==row['arm'])
    before=b['Koma_vs_instantaneous_analytic']['axial_rmse_m_s'];after=a['Koma_vs_instantaneous_analytic']['axial_rmse_m_s']
    reduction=None if before in (None,0) or after is None else 1-after/before
    coverage=a['own_geometry_coverage']-b['own_geometry_coverage']
    if reduction is None:assert row['paired_axial_RMSE_reduction_fraction'] is None;checks+=1
    else:close(reduction,row['paired_axial_RMSE_reduction_fraction'])
    close(coverage,row['own_coverage_change_fraction'])
    assert row['predefined_improvement_target_pass']==bool(numerical and reduction is not None and reduction>=.1 and coverage>=-.01);checks+=1
receipt=dict(experiment='E-056',utc=datetime.now(timezone.utc).isoformat(),status='PASS_SAVED_FIELD_AND_RESULT_QA',
             primary_numerical_status=r['status'],checks=checks,max_full_phase_field_recomputation_error=max_phase_identity,
             report_sha256=sha(root/'analysis_v1/report.json'),verification_source_sha256=sha(Path(__file__)),new_1m=False,
             scope='All36 raw/image L2 values, nine phase fields/support masks/paired errors/partial flows, four primary decisions and bound artifact identities. Numerical correctness is not independent physical validation.')
(stage/'report.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n',encoding='utf-8')
stage.rename(target)
print(json.dumps(receipt),flush=True)
