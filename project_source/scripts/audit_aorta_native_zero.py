"""E-045: exact saved sequence timing and frame0 weights before acquisition."""
import argparse
import json
from pathlib import Path
import tomllib
import numpy as np
from scripts.aorta_hemodynamic_reference import sequence_timing,temporal_weights,read_pulseq
from scripts.build_aorta_inlet_picker import sha


def audit(sequence_dir,output):
    if output.exists():raise FileExistsError(output)
    reports={};hashes={};weights=[]
    for encoding in ['REF','X','Y','Z']:
        path=sequence_dir/(encoding.lower()+'.seq');defs={r[0]:r[1:] for r in read_pulseq(path)['DEFINITIONS']}
        if int(defs['NativeSourceStartFrame'][0])!=0 or int(defs['CardiacBins'][0])!=16:raise ValueError('Frame0/16bin contract')
        if abs(float(defs['RR'][0])-.72)>1e-12:raise ValueError('No deliberate motion retiming allowed')
        if abs(float(defs['TR'][0])-.00313)>1e-10:raise ValueError('Original shot TR changed')
        info,_,records=sequence_timing(path,float(np.float32(.72)))
        if info['RF_to_ADC_seam_crossings']:raise ValueError('Global coordinate seam during coherent acquisition')
        native=[];windows=[]
        for b in range(1,17):
            rows=[r for r in records if r['bin']==b]
            samples=np.concatenate([r['adc_first_s']+np.arange(r['n'])*r['dwell_s'] for r in rows])
            w=temporal_weights(samples,period_s=float(np.float32(.72)),start_frame=0)
            phase=(np.angle(w@np.exp(2j*np.pi*np.arange(72)/72))%(2*np.pi))*.72/(2*np.pi)
            native.append(w)
            windows.append(dict(bin=b,source_center_s=float(phase),source_phase_min_s=float(np.mod(samples,float(np.float32(.72))).min()),source_phase_max_s=float(np.mod(samples,float(np.float32(.72))).max()),profiles=len(rows),samples=len(samples)))
        native=np.array(native)
        if encoding=='REF':weights=native
        elif not np.allclose(weights,native,atol=2e-12,rtol=0):raise ValueError('Encoding source-time weights differ')
        hashes[encoding]=sha(path);reports[encoding]=dict(profiles=len(records),seam_crossings=0,source_bins=windows)
    output.parent.mkdir(parents=True,exist_ok=True)
    result=dict(experiment='E-045',status='PASS_NATIVE_ZERO_TIMING',sequence_sha256=hashes,native_source_start_frame=0,nominal_period_s=.72,realized_Float32_motion_period_s=float(np.float32(.72)),source_time_velocity_scale=.72/float(np.float32(.72)),cardiac_bins=16,first_source_phase_center_s=reports['REF']['source_bins'][0]['source_center_s'],last_source_phase_center_s=reports['REF']['source_bins'][-1]['source_center_s'],first_point_is_not_an_instantaneous_t0_measurement=True,encodings=reports)
    with output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    np.savez_compressed(output.with_suffix('.npz'),native_frame_weights=weights)
    # Minimal machine-readable Julia launch gate with exact sequence hashes.
    lines=['status = "PASS_NATIVE_ZERO_TIMING"','[sequence_sha256]']+[f'{e} = "{h}"' for e,h in hashes.items()]
    output.with_suffix('.toml').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['status','cardiac_bins','first_source_phase_center_s','last_source_phase_center_s']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sequence-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();audit(a.sequence_dir,a.output)
