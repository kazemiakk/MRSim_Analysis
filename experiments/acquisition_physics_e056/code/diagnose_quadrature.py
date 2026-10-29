"""Saved-data quadrature sensitivity, not a replacement acceptance reference."""
from pathlib import Path
import json,hashlib
from datetime import datetime,timezone
import h5py
import numpy as np
import analyze_physics as reference

root=Path('D:/MRSim/outputs/acquisition_physics_e056')
target=root/'quadrature_diagnostic_v1';stage=target.with_name(target.name+'.building')
if target.exists() or stage.exists():raise FileExistsError(target)
def candidate_trapezoid(template,rf_start,rf_end,path_time,path_factor,period):
    adc=template['adc'];finish=float(adc[-1])
    cuts=np.unique(np.r_[rf_end,finish,adc,template['t']])
    cuts=cuts[(cuts>=rf_end-1e-15)&(cuts<=finish+1e-15)]
    g=reference.interpolate(cuts,template['t'],template['g'])
    motion=np.interp(np.remainder(rf_start+cuts,period),path_time,path_factor)
    dt=np.diff(cuts)
    ordinary=np.vstack([np.zeros(3),np.cumsum((g[:-1]+g[1:])*dt[:,None]/2,axis=0)])
    product=g[:,0]*motion
    moving=np.r_[0.,np.cumsum((product[:-1]+product[1:])*dt/2)]
    idx=np.searchsorted(cuts,adc)
    np.testing.assert_allclose(cuts[idx],adc,atol=1e-12,rtol=0)
    return ordinary[idx],moving[idx]
with h5py.File(root/'inputs_v1/controls.h5') as f:
    positions=f['positions'][:];rho=f['rho'][:];u=f['u_uniform_osc'][:]
    t1=float(f['T1'][0]);t2=float(f['T2'][0]);times=f['path_time_s'][:];factor=f['path_basis_s'][:];period=float(f['motion_period_s'][()])
primary=json.loads((root/'analysis_v1/report.json').read_text(encoding='utf-8'))
rows=[];images={};arm='bw700_te274'
with h5py.File(root/'reference_contract_v2/waveforms.h5') as f:
    for enc in ['REF','X']:
        meta=reference.load_group(f[arm][enc],80,16,32);meta['gamma']=float(f['gamma_Hz_T'][()])
        original=reference.phase_moments
        try:
            reference.phase_moments=candidate_trapezoid
            predicted,image=reference.predict(meta,positions,rho,u,t1,t2,times,factor,period,(80,16,32,15))
        finally:reference.phase_moments=original
        with h5py.File(root/'acquisition_v1'/f'uniform_osc_{arm}_{enc.lower()}.h5') as f_actual:
            actual=reference.complex_array(f_actual['raw_signal_f64']).ravel();actual_image=reference.complex_array(f_actual['image'])
        baseline=next(r for r in primary['numerical_checks'] if r['model']=='uniform_osc' and r['arm']==arm and r['encoding']==enc)
        rows.append(dict(encoding=enc,primary_exact_integration_relative_L2=baseline['raw_relative_l2'],
                         candidate_waveform_knot_trapezoid_relative_L2=float(np.linalg.norm(actual-predicted)/np.linalg.norm(predicted))))
        images[enc]=(actual_image,image)
with h5py.File(root/'analysis_v1/independent_fields.h5') as f:
    paired=f[f'uniform_osc/{arm}/analysis/paired_core'][:].astype(bool)
actual_velocity=np.angle(images['X'][0]*images['REF'][0].conj())*5/np.pi
candidate_velocity=np.angle(images['X'][1]*images['REF'][1].conj())*5/np.pi
report=dict(experiment='E-056',utc=datetime.now(timezone.utc).isoformat(),
    status='COMPLETE_SAVED_DATA_QUADRATURE_SENSITIVITY',rows=rows,
    candidate_vs_Koma_paired_axial_RMSE_m_s=float(np.sqrt(np.mean((actual_velocity[:,paired]-candidate_velocity[:,paired])**2))),
    primary_acceptance_reference_changed=False,primary_numerical_status=primary['status'],new_1m=False,
    interpretation='Only the independent free-precession quadrature is changed: exact integration versus trapezoids on exported waveform/ADC knots. No measured gain or phase is fitted. This is a numerical mechanism diagnostic, not a passing physical reference.',
    limitation='The complete internal simulator sampling grid has not been independently exported and certified identical to this candidate knot grid. Confirm by a small numerical-step convergence test before claiming the cause resolved.')
stage.mkdir();(stage/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
paths=[root/'analysis_v1/report.json',root/'reference_contract_v2/waveforms.h5',root/'inputs_v1/controls.h5',
       root/'code/analyze_physics.py',Path(__file__),
       root/'acquisition_v1/uniform_osc_bw700_te274_ref.h5',root/'acquisition_v1/uniform_osc_bw700_te274_x.h5',
       Path('D:/MRSim/MRSIM/MRSimTracks/outputs/upstream_migration_e046/depot/packages/KomaMRICore/PMbeb/src/simulation/SimMethods/Bloch/gpu/PrecessionKernel.jl')]
(stage/'MANIFEST.json').write_text(json.dumps(dict(inputs_sha256={str(p):sha(p) for p in paths},report_sha256=sha(stage/'report.json')),indent=2)+'\n',encoding='utf-8')
stage.rename(target)
print(json.dumps(report),flush=True)
