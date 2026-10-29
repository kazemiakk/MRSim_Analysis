"""Registered 5/10/15/20% support sensitivities; never optimize a mask."""
from pathlib import Path
from datetime import datetime, timezone
import json
import hashlib
import h5py
import numpy as np

root=Path('D:/MRSim/outputs/acquisition_physics_e056')
source=root/'analysis_v1';target=root/'analysis_sensitivity_v1'
stage=target.with_name(target.name+'.building')
if target.exists() or stage.exists():raise FileExistsError(target)
stage.mkdir()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()
def complex_values(d):
    a=d[:]
    if np.iscomplexobj(a):return a
    names=a.dtype.names
    if names and {'r','i'}<=set(names):return a['r']+1j*a['i']
    raise TypeError(d.name)
def errors(a,b,mask):
    if not mask.any():return dict(voxels=0,vector_rmse_m_s=None,axial_rmse_m_s=None,axial_bias_m_s=None)
    d=(a-b)[:,mask,:]
    return dict(voxels=int(mask.sum()),vector_rmse_m_s=float(np.sqrt(np.sum(d*d)/d.shape[0]/d.shape[1])),
                axial_rmse_m_s=float(np.sqrt(np.mean(d[...,0]*d[...,0]))),axial_bias_m_s=float(np.mean(d[...,0])))
base=json.loads((source/'report.json').read_text(encoding='utf-8'))
manifest=json.loads((source/'MANIFEST.json').read_text(encoding='utf-8'))
assert sha(source/'report.json')==manifest['outputs']['report.json']['sha256']
assert sha(source/'independent_fields.h5')==manifest['outputs']['independent_fields.h5']['sha256']
acq_report=__import__('tomllib').loads((root/'acquisition_v1/report.toml').read_text(encoding='utf-8'))
inputs={str(source/'report.json'):sha(source/'report.json'),str(source/'independent_fields.h5'):sha(source/'independent_fields.h5'),
        str(Path(__file__)):sha(Path(__file__)),str(root/'acquisition_v1/report.toml'):sha(root/'acquisition_v1/report.toml')}
arms=['baseline','bw700_te274','bw700_te210'];rows=[];mechanisms=[];checks=0
with h5py.File(source/'independent_fields.h5') as f:
    for model in ['static','uniform_osc','parabolic_osc']:
        fields={}
        for arm in arms:
            g=f[f'{model}/{arm}/analysis']
            data={name:g[name][:].astype(bool) for name in ['geometry','core']}
            for name in ['velocity_Koma','velocity_independent','velocity_instantaneous_analytic']:
                data[name]=g[name][:]
            magnitude=[]
            for enc in ['ref','x','y','z']:
                p=root/'acquisition_v1'/f'{model}_{arm}_{enc}.h5'
                digest=sha(p);assert digest==acq_report['result_sha256'][f'{model}_{arm}_{enc}'];checks+=1
                inputs[str(p)]=digest
                with h5py.File(p) as result:magnitude.append(abs(complex_values(result['image'])))
            data['magnitude']=np.stack(magnitude)
            fields[arm]=data
        common_core=np.logical_and.reduce([fields[a]['core'] for a in arms])
        common_geometry=np.logical_and.reduce([fields[a]['geometry'] for a in arms])
        for threshold in [.05,.10,.15,.20]:
            support={a:np.all(fields[a]['magnitude']>=threshold*fields[a]['magnitude'].max(axis=(2,3,4),keepdims=True),axis=(0,1)) for a in arms}
            paired=common_core&np.logical_and.reduce([support[a] for a in arms])
            threshold_rows=[]
            for arm in arms:
                d=fields[arm];geom=d['geometry'];core=d['core']
                row=dict(model=model,arm=arm,threshold_fraction=threshold,
                         own_geometry_coverage=float((support[arm]&geom).sum()/geom.sum()),
                         own_core_coverage=float((support[arm]&core).sum()/core.sum()),
                         identical_paired_core_coverage=float(paired.sum()/common_core.sum()),
                         Koma_vs_instantaneous_analytic=errors(d['velocity_Koma'],d['velocity_instantaneous_analytic'],paired),
                         Koma_vs_independent_complex_model=errors(d['velocity_Koma'],d['velocity_independent'],paired))
                section=paired[:,:,40];area=3.125
                if section.any():
                    q=d['velocity_Koma'][:,:,:,40,0][:,section].sum(axis=1)*area
                    truth=d['velocity_instantaneous_analytic'][:,:,:,40,0][:,section].sum(axis=1)*area
                    row['paired_partial_section_flow']=dict(retained_area_mm2=float(section.sum()*area),
                        coverage_fraction=float(section.sum()/common_geometry[:,:,40].sum()),
                        RMSE_ml_s=float(np.sqrt(np.mean((q-truth)**2))),
                        relative_mean_bias=None if abs(truth.mean())<1e-12 else float(q.mean()/truth.mean()-1))
                else:row['paired_partial_section_flow']=None
                if threshold==.05:
                    primary=next(x for x in base['comparisons'] if x['model']==model and x['arm']==arm)
                    for name in ['own_geometry_coverage','own_core_coverage','identical_paired_core_coverage']:
                        np.testing.assert_allclose(row[name],primary[name],atol=1e-14,rtol=1e-13);checks+=1
                    for comparison in ['Koma_vs_instantaneous_analytic','Koma_vs_independent_complex_model']:
                        for name in ['vector_rmse_m_s','axial_rmse_m_s','axial_bias_m_s']:
                            actual,expected=row[comparison][name],primary[comparison][name]
                            if actual is None or expected is None:assert actual is expected
                            else:np.testing.assert_allclose(actual,expected,atol=1e-13,rtol=1e-11)
                            checks+=1
                threshold_rows.append(row);rows.append(row)
            baseline=threshold_rows[0]
            for row in threshold_rows[1:]:
                before=baseline['Koma_vs_instantaneous_analytic']['axial_rmse_m_s'];after=row['Koma_vs_instantaneous_analytic']['axial_rmse_m_s']
                mechanisms.append(dict(model=model,arm=row['arm'],threshold_fraction=threshold,
                    paired_axial_RMSE_reduction_fraction=None if before in (None,0) or after is None else 1-after/before,
                    own_geometry_coverage_change_fraction=row['own_geometry_coverage']-baseline['own_geometry_coverage']))
report=dict(experiment='E-056',utc=datetime.now(timezone.utc).isoformat(),status='COMPLETE_REGISTERED_MASK_SENSITIVITIES',
            primary_numerical_status=base['status'],rows=rows,mechanisms=mechanisms,checks=checks,new_1m=False,
            definition='All-bin, all-encoding fixed relative magnitude support; same all-arm paired core within each threshold. Thresholds are sensitivity conditions, not optimized.',
            limitation='Synthetic stroboscopic mechanism controls, not physical CFD validation. Own retained coverage and paired error populations differ.')
(stage/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
(stage/'MANIFEST.json').write_text(json.dumps(dict(inputs_sha256=inputs,report_sha256=sha(stage/'report.json')),indent=2)+'\n',encoding='utf-8')
stage.rename(target)
print(json.dumps(dict(status=report['status'],rows=len(rows),checks=checks,new_1m=False)),flush=True)
