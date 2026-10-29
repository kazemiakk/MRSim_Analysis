"""Analytic saved-gradient moment and sign audit; no Bloch simulation."""
from pathlib import Path
import numpy as np
from scripts.aorta_hemodynamic_reference import read_pulseq,sequence_timing


def segment_moments(start,length,g0,slope,until):
    length=min(length,max(0.,until-start))
    area=g0*length+.5*slope*length**2
    moment=start*area+.5*g0*length**2+slope*length**3/3
    return area,moment


def trapezoid_moments(amplitude,rise,flat,fall,start,until):
    segments=((start,rise,0.,amplitude/rise if rise else 0.),
              (start+rise,flat,amplitude,0.),
              (start+rise+flat,fall,amplitude,-amplitude/fall if fall else 0.))
    return np.sum([segment_moments(*s,until) for s in segments],axis=0)


def first_profile_moments(path,period):
    s=read_pulseq(path); defs={r[0]:r[1:] for r in s['DEFINITIONS']}
    first=sequence_timing(path,period)[2][0]
    until=(first['adc_first_s']+first['adc_last_s'])/2
    traps={int(r[0]):[float(v) for v in r[1:6]] for r in s['TRAP']}
    raster=float(defs['BlockDurationRaster'][0]); ticks=0; moments=np.zeros((3,2))
    for row in s['BLOCKS']:
        block=list(map(int,row[:8])); start=ticks*raster
        if start>until: break
        for axis,gid in enumerate(block[3:6]):
            if gid:
                amp,rise,flat,fall,delay=traps[gid]
                moments[axis]+=trapezoid_moments(amp,rise*1e-6,flat*1e-6,fall*1e-6,start+delay*1e-6,until)
        ticks+=block[1]
    moments[:,1]-=first['rf_start_s']*moments[:,0]
    return moments


def audit(directory,period):
    directory=Path(directory)
    ref=first_profile_moments(directory/'ref.seq',period); output={}
    for axis,e in enumerate(('X','Y','Z')):
        moments=first_profile_moments(directory/(e.lower()+'.seq'),period)
        delta=moments-ref
        # Pulseq gradient amplitudes are cycles/s/m. Bloch phase = -2pi*M1*v.
        # Saved reconstruction multiplies phase by Venc/pi.
        gain=-2*2.5*delta[:,1]
        expected=np.eye(3)[axis]
        if not np.allclose(gain,expected,rtol=1e-7,atol=1e-8):
            raise ValueError('Saved encoding moment/sign differs from documented formula')
        output[e]={'delta_zero_moment_cycles_m':delta[:,0].tolist(),
                   'delta_first_moment_cycles_s_m':delta[:,1].tolist(),
                   'predicted_constant_velocity_reconstruction_gain':gain.tolist()}
    return {'method':'Analytic trapezoid integration through first ADC center, encoded-minus-REF',
            'passed':True,'encodings':output,
            'qualification':'Checks recorded gradient scale/sign under the documented Bloch convention; no new simulator accuracy claim'}
