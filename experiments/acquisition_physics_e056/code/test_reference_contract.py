"""Read-only regression checks for E056's Pulseq-bound RF-phase repair."""
from pathlib import Path
import hashlib
import json
import numpy as np
import h5py

run=Path('D:/MRSim/outputs/acquisition_physics_e056')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
receipt=json.loads((run/'reference_contract_v2/registration.json').read_text(encoding='utf-8'))
checks=0
assert receipt['acquisition_data_changed'] is False
assert receipt['empirical_gain_or_phase_fit'] is False
assert receipt['new_1m'] is False
checks+=3
assert sha(run/'reference_contract_v2/waveforms.h5')==receipt['waveforms_sha256'];checks+=1
for path,digest in receipt['inputs_sha256'].items():
    assert sha(path)==digest,path
    checks+=1
with h5py.File(run/'sequences_v1/waveforms.h5') as old, h5py.File(run/'reference_contract_v2/waveforms.h5') as new:
    for arm in [k for k in old if isinstance(old[k],h5py.Group)]:
        for enc in ['REF','X','Y','Z']:
            a,b=old[arm][enc],new[arm][enc]
            # Reparse the source text without importing the repair's writer.
            section=None;rf={};block_ids=[]
            for line in (run/'sequences_v1'/arm/(enc.lower()+'.seq')).read_text(encoding='utf-8').splitlines():
                s=line.split('#')[0].strip()
                if not s:continue
                if s.startswith('['):section=s[1:-1];continue
                r=s.split()
                if section=='RF':rf[int(r[0])]=float(r[10])
                elif section=='BLOCKS' and int(r[2]):block_ids.append(int(r[2]))
            expected=np.array([rf[i] for i in block_ids])
            np.testing.assert_array_equal(expected,b['rf_phase_rad'][:]);checks+=1
            assert np.max(abs(np.angle(np.exp(1j*(expected-a['rf_phase_rad'][:])))))>1.;checks+=1
            for key in ['rf_start_s','rf_scale','labels0','profile_rf_index0','adc_first_s','dwell_s','adc_phase_compensation']:
                np.testing.assert_array_equal(a[key][:],b[key][:]);checks+=1
            labels=b['labels0'][:];idx=b['profile_rf_index0'][:].astype(int)
            for key in range(512):
                p=np.flatnonzero((labels[:,0]+16*labels[:,1]==key)&(labels[:,2]==0))[0]
                phase=expected[idx[p]]
                ao,bo=a['shots'][f'{key:04d}'],b['shots'][f'{key:04d}']
                np.testing.assert_allclose(bo['b1_T'][:]*np.exp(1j*phase),ao['b1_T'][:],rtol=2e-15,atol=1e-20)
                assert np.max(abs(bo['b1_T'][:].imag))<1e-14
                for name in ['t_s','g_T_m','rf_frequency_Hz','adc_time_s','rf_phase_rad_at_knots']:
                    np.testing.assert_array_equal(ao[name][:],bo[name][:])
                checks+=7
print(json.dumps(dict(status='PASS_RF_PHASE_REFERENCE_REGRESSION',checks=checks,
                      source_waveforms_unchanged=True,empirical_fit=False)),flush=True)
