"""E049 equal-spin superposition controls; not a full phantom reconstruction."""
import argparse
from itertools import combinations
import json
from pathlib import Path
import uuid

import h5py
import numpy as np
from scripts.analyze_phantom_encoding import centered_ifft
from scripts.build_aorta_inlet_picker import sha
from scripts.prepare_phantom_encoding import BASE
from scripts.diagnose_phantom_signal import read_complex


def mixing(output):
    if output.exists():raise FileExistsError(output)
    stage=output.with_name(output.name+'.building-'+uuid.uuid4().hex[:8]);stage.mkdir(parents=True)
    source=BASE/'analysis_v1/analysis_fields.h5'
    controls=BASE/'inputs_v3/controls.json'
    rows=json.loads(controls.read_text())['cases']
    names=['static_origin','constant_axis1_0.5','constant_axis1_1.5','constant_axis1_-0.75']
    selected=[next(i for i,r in enumerate(rows) if r['name']==name) for name in names]
    result=[]
    with h5py.File(source) as f,h5py.File(stage/'mixing_fields.h5','w') as out:
        signals={e:f[e+'_Koma'][:] for e in ('REF','X','Y','Z')}
        for i,j in combinations(selected,2):
            images={e:centered_ifft(signals[e][i]+signals[e][j]) for e in signals}
            velocity=np.angle(images['X']*images['REF'].conj())*5/np.pi
            mask=np.ones(400,bool)
            for e,im in images.items():mask&=(abs(im)>=.05*max(abs(im)))&(abs(im)>0)
            arithmetic=(rows[i]['echo_velocity'][0]+rows[j]['echo_velocity'][0])/2
            raw_echo=np.angle((signals['X'][i,200]+signals['X'][j,200])*np.conj(signals['REF'][i,200]+signals['REF'][j,200]))*5/np.pi
            name=f'pair_{i:03}_{j:03}'
            group=out.create_group(name)
            for e,image in images.items():group[e]=image
            group['velocity']=velocity;group['mask']=mask
            result.append(dict(case_1=rows[i]['name'],case_2=rows[j]['name'],
                 equal_spin_echo_velocity_mean_m_s=arithmetic,raw_echo_phase_velocity_m_s=float(raw_echo),
                 center_pixel_phase_velocity_m_s=float(velocity[200]) if mask[200] else None,
                 center_pixel_signal_pass=bool(mask[200]),retained_readout_pixels=int(mask.sum()),
                 interpretation='Two point populations coincide at echo, move at different known velocities. Complex signals add, not velocity images. 1D point-response only.'))
    report=dict(experiment='E-049',status='COMPLETE_SAVED_SIGNAL_SUPERPOSITION_CONTROL',pairs=result,
                limitation='Linear Bloch superposition of separately simulated noninteracting spins; not a fresh two-spin run, 3D image, full phantom explanation or quantitative no-aliasing proof.')
    (stage/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (stage/'MANIFEST.json').write_text(json.dumps(dict(inputs={str(p):sha(p) for p in [Path(__file__).resolve(),source,controls]},
        outputs={p.name:sha(p) for p in stage.iterdir() if p.is_file()}),indent=2)+'\n')
    stage.rename(output);print(json.dumps(report))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    mixing(p.parse_args().output)
