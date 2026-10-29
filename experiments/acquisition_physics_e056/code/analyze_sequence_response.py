"""Unfitted constant-acceleration timing sensitivity of differential gradients."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import h5py
import numpy as np
root=Path('D:/MRSim/outputs/acquisition_physics_e056')
target=root/'sequence_response_v1';stage=target.with_name(target.name+'.building')
if target.exists() or stage.exists():raise FileExistsError(target)
rows=[];checks=0
with h5py.File(root/'reference_contract_v2/waveforms.h5') as f:
    gamma=float(f['gamma_Hz_T'][()])
    for arm in ['baseline','bw700_te274','bw700_te210']:
        ref=f[arm]['REF']['shots']['0264']
        for axis,enc in enumerate(['X','Y','Z']):
            shot=f[arm][enc]['shots']['0264'];center=float(ref['rf_center_s'][()])
            close=lambda a,b:np.testing.assert_allclose(a,b,rtol=0,atol=1e-12)
            close(center,shot['rf_center_s'][()]);checks+=1
            echo=float(ref['adc_time_s'][40]);close(echo,shot['adc_time_s'][40]);checks+=1
            # Reintegrate piecewise-linear differential gradients independently.
            cuts=np.unique(np.r_[center,echo,ref['t_s'][:],shot['t_s'][:]])
            cuts=cuts[(cuts>=center)&(cuts<=echo)]
            mid=(cuts[1:]+cuts[:-1])/2;half=np.diff(cuts)/2
            t=np.column_stack([mid-half/np.sqrt(3),mid+half/np.sqrt(3)]).ravel()
            dg=np.interp(t,shot['t_s'][:],shot['g_T_m'][:,axis])-np.interp(t,ref['t_s'][:],ref['g_T_m'][:,axis])
            moments=[float(np.sum((dg*(t-center)**order).reshape(-1,2).sum(axis=1)*half)) for order in [0,1,2]]
            for order,value in enumerate(moments):
                exported=float(shot[f'moment{order}_adc_T_s'+('_m' if order==0 else f'{order+1}_m')][40,axis]-ref[f'moment{order}_adc_T_s'+('_m' if order==0 else f'{order+1}_m')][40,axis])
                np.testing.assert_allclose(value,exported,rtol=1e-8,atol=1e-18);checks+=1
            assert abs(moments[0])<1e-12;checks+=1
            np.testing.assert_allclose(moments[1],-1/(2*gamma*5),rtol=1e-7,atol=1e-18);checks+=1
            delay=echo-center-.5*moments[2]/moments[1]
            rows.append(dict(arm=arm,encoding=enc,ADC_index0=40,differential_moments=moments,
                             constant_acceleration_sensitivity_before_echo_s=delay,
                             velocity_error_per_unit_constant_acceleration_s=-delay))
report=dict(experiment='E-056',utc=datetime.now(timezone.utc).isoformat(),status='PASS_UNFITTED_DIFFERENTIAL_GRADIENT_RESPONSE',
            rows=rows,checks=checks,new_1m=False,
            definition='For a single spin with locally constant acceleration and zero differential M0, the differential-gradient phase reports v_echo - acceleration*delay. Delay is calculated from M2/(2*M1), not fitted.',
            limitations=['Not a universal time shift or correction for oscillatory, sheared, voxel-mixed complex images.',
                         'Does not replace actual finite-RF or full independent Bloch checks, and does not establish a cause fraction for historical CFD phantom errors.'])
stage.mkdir();(stage/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
(stage/'MANIFEST.json').write_text(json.dumps(dict(inputs_sha256={str(p):sha(p) for p in [root/'reference_contract_v2/waveforms.h5',Path(__file__)]},report_sha256=sha(stage/'report.json')),indent=2)+'\n',encoding='utf-8')
stage.rename(target)
print(json.dumps(report),flush=True)
