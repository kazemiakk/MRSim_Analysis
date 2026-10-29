"""Repair derived RF-phase metadata from Pulseq tables; never use measured gains."""
from pathlib import Path
import json,shutil,hashlib
from datetime import datetime,timezone
import numpy as np,h5py

run=Path('D:/MRSim/outputs/acquisition_physics_e056')
output=run/'reference_contract_v2';stage=run/'reference_contract_v2.building'
if output.exists() or stage.exists():raise FileExistsError(output)
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()

def parse_rf(path):
    section=None;blocks=[];rf={}
    for line in path.read_text(encoding='utf-8').splitlines():
        line=line.split('#')[0].strip()
        if not line:continue
        if line.startswith('['):section=line[1:-1];continue
        row=line.split()
        if section=='RF':
            if len(row)!=12:raise ValueError('Expected Pulseq1.5 RF phase columns')
            rf[int(row[0])]=float(row[10])
        elif section=='BLOCKS':blocks.append([int(x) for x in row[:8]])
    return np.asarray([rf[b[2]] for b in blocks if b[2]],float)

stage.mkdir();original=run/'sequences_v1/waveforms.h5';derived=stage/'waveforms.h5'
shutil.copyfile(original,derived)
checks=[];sources={str(original):sha(original),str(Path(__file__)):sha(Path(__file__))}
with h5py.File(derived,'r+') as f:
    for arm in [a for a in f if isinstance(f[a],h5py.Group)]:
        for encoding in ['REF','X','Y','Z']:
            path=run/'sequences_v1'/arm/(encoding.lower()+'.seq');sources[str(path)]=sha(path)
            phases=parse_rf(path);g=f[arm][encoding]
            assert len(phases)==len(g['rf_phase_rad'])
            g['rf_phase_rad'][...]=phases
            labels=g['labels0'][:];mapping=g['profile_rf_index0'][:].astype(int)
            comp=g['adc_phase_compensation'][:]
            # Receiver compensation is stored ComplexF32, whereas RF table
            # phases are Float64. Check its representation at its own precision;
            # do not change the 1% independent signal acceptance threshold.
            phase_tolerance=4*np.finfo(comp.real.dtype).eps
            error=np.max(abs(np.angle(np.exp(1j*(phases[mapping]+np.angle(comp[:,0].astype(np.complex128)))))))
            assert error<phase_tolerance,(arm,encoding,error,phase_tolerance)
            maximum=0.
            for key in range(512):
                p=int(np.flatnonzero((labels[:,0]+16*labels[:,1]==key)&(labels[:,2]==0))[0])
                phase=phases[mapping[p]];shot=g['shots'][f'{key:04d}'];b=shot['b1_T'][:]
                measured=float(np.angle(b[np.argmax(abs(b))]))
                mismatch=abs(np.angle(np.exp(1j*(measured-phase))))
                assert mismatch<2e-9
                # Exported B1 already contains RF.ϕ; remove that known table
                # phase once, then the independent predictor reapplies each RF.
                shot['b1_T'][...]=b*np.exp(-1j*phase)
                maximum=max(maximum,mismatch)
            checks.append(dict(arm=arm,encoding=encoding,profiles=len(mapping),
                receiver_compensation_dtype=str(comp.dtype),
                RF_ADC_representation_tolerance_rad=float(phase_tolerance),
                maximum_RF_ADC_phase_identity_error_rad=float(error),maximum_template_phase_error_rad=maximum))
receipt=dict(experiment='E-056',status='PASS_PULSEQ_BOUND_RF_PHASE_REFERENCE_CORRECTION',
    utc=datetime.now(timezone.utc).isoformat(),inputs_sha256=sources,waveforms_sha256=sha(derived),checks=checks,
    reason='RF.ϕ is stored separately from normalized RF.A. The first export inspected RF.A only and failed to neutralize the template phase.',
    acquisition_data_changed=False,empirical_gain_or_phase_fit=False,new_1m=False)
(stage/'registration.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
stage.rename(output)
print(json.dumps(receipt),flush=True)
