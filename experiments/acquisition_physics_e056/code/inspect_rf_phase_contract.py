"""Read-only check of source RF and receiver phase representations."""
from pathlib import Path
import h5py
import numpy as np

root = Path('D:/MRSim/outputs/acquisition_physics_e056')
def phases_from_pulseq(path):
    section = None
    blocks, rf = [], {}
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.split('#')[0].strip()
        if not line:
            continue
        if line.startswith('['):
            section = line[1:-1]
            continue
        row = line.split()
        if section == 'RF':
            rf[int(row[0])] = float(row[10])
        elif section == 'BLOCKS' and int(row[2]):
            blocks.append(int(row[2]))
    return np.array([rf[i] for i in blocks])

with h5py.File(root/'sequences_v1/waveforms.h5') as f:
    for arm in [k for k in f if isinstance(f[k], h5py.Group)]:
        for enc in ['REF','X','Y','Z']:
            g = f[arm][enc]
            phases = phases_from_pulseq(root/'sequences_v1'/arm/(enc.lower()+'.seq'))
            comp = g['adc_phase_compensation'][:]
            idx = g['profile_rf_index0'][:].astype(int)
            delta = np.angle(np.exp(1j*(phases[idx]+np.angle(comp[:,0]))))
            worst = np.argmax(abs(delta))
            print(arm, enc, 'dtype',comp.dtype,'max_rad',abs(delta[worst]),
                  'worst_profile',worst,'RF_rad',phases[idx[worst]],
                  'receiver_rad',np.angle(comp[worst,0]), flush=True)
