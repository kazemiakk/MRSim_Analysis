"""E049 independent finite-RF/phase/FFT checks on single-spin CPU controls."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import tomllib
import uuid

import h5py
import numpy as np

from scripts.build_aorta_inlet_picker import sha
from scripts.diagnose_phantom_signal import read_complex, ROOT
from scripts.prepare_phantom_encoding import BASE, shot_contract, gradient


def rhs(m,field,t1,t2,gamma):
    """Laboratory rotating-frame Bloch ODE: gamma M cross B + relaxation."""
    result = 2*np.pi*gamma*np.cross(m,field)
    result[:,:2] -= m[:,:2]/t2[:,None]
    result[:,2] += (1-m[:,2])/t1
    return result


def independent_bloch(time,positions,b1,gradients,adc_time,t1,t2,gamma):
    # Finite-RF RK4, followed by analytic free-precession with independently
    # trapezoid-integrated Bz. No Koma simulation/interpolation helpers used.
    last = int(np.flatnonzero(abs(b1)>0)[-1])+1
    m = np.zeros((len(positions),3));m[:,2]=1
    for j in range(last):
        dt = time[j+1]-time[j]
        bz0 = positions[:,j,:]@gradients[j]
        bz1 = positions[:,j+1,:]@gradients[j+1]
        p_mid = (positions[:,j,:]+positions[:,j+1,:])/2
        bz_mid = p_mid@((gradients[j]+gradients[j+1])/2)
        def field(rf,bz):
            out=np.empty_like(m);out[:,0]=rf.real;out[:,1]=rf.imag;out[:,2]=bz
            return out
        f0=field(b1[j],bz0);f1=field(b1[j+1],bz1);fm=field((b1[j]+b1[j+1])/2,bz_mid)
        k1=rhs(m,f0,t1,t2,gamma)
        k2=rhs(m+dt*k1/2,fm,t1,t2,gamma)
        k3=rhs(m+dt*k2/2,fm,t1,t2,gamma)
        k4=rhs(m+dt*k3,f1,t1,t2,gamma)
        m += dt*(k1+2*k2+2*k3+k4)/6
    t=time[last:]
    bz = np.einsum('nti,ti->nt',positions[:,last:,:],gradients[last:])
    integral=np.c_[np.zeros(len(positions)),np.cumsum((bz[:,:-1]+bz[:,1:])*np.diff(t)[None]/2,axis=1)]
    phase=np.stack([np.interp(adc_time,t,r) for r in integral])*(-2*np.pi*gamma)
    signal=(m[:,0]+1j*m[:,1])[:,None]*np.exp(1j*phase-(adc_time-time[last])[None]/t2[:,None])
    return signal


def centered_ifft(k):
    return np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(k,axes=-1),axis=-1),axes=-1)


def numerical_point_width(signal,spacing_mm=1.25):
    image=centered_ifft(signal)
    energy=abs(image)**2;energy/=energy.sum()
    x=(np.arange(len(image))-len(image)//2)*spacing_mm
    center=float(np.sum(x*energy))
    rms=float(np.sqrt(np.sum((x-center)**2*energy)))
    order=np.argsort(abs(x-center))
    radius=float(abs(x[order[np.searchsorted(np.cumsum(energy[order]),.95)]]-center))
    return dict(peak_index=int(abs(image).argmax()),energy_centroid_mm=center,
                rms_width_mm=rms,radius_95_energy_mm=radius)


def analyze(inputs,cpu,output):
    if output.exists():raise FileExistsError(output)
    stage=output.with_name(output.name+'.building-'+uuid.uuid4().hex[:8]);stage.mkdir(parents=True)
    contract=json.loads((inputs/'controls.json').read_text())
    runtime=tomllib.loads((cpu/'report.toml').read_text())
    if runtime['signals_sha256']!=sha(cpu/'signals.h5') or runtime['input_sha256']!=sha(inputs/'controls.h5'):
        raise ValueError('Control identity mismatch')
    if runtime['status']!='COMPLETE_SINGLE_SPIN_CPU_BLOCH_CONTROLS' or not runtime['CPU_only']:
        raise ValueError('Unexpected control runtime')
    cases=contract['cases'];n=len(cases)
    with h5py.File(inputs/'controls.h5') as f:
        time=f['time_s'][:];adc=f['adc_time_s'][:]
        positions=np.stack([np.column_stack([f[f'case_{i:03}/'+c][:] for c in 'xyz']) for i in range(n)])
        t1=np.array([f[f'case_{i:03}/T1'][()] for i in range(n)])
        t2=np.array([f[f'case_{i:03}/T2'][()] for i in range(n)])
    signal={};independent={};wave_checks=[]
    # Gamma is obtained from saved physical gradients versus independent
    # Pulseq Hz/m amplitudes, not fitted to phase or velocity observations.
    with h5py.File(cpu/'signals.h5') as f:
        for enc in ('REF','X','Y','Z'):
            signal[enc]=np.stack([read_complex(f[f'case_{i:03}/'+enc]) for i in range(n)])
            wave=f[enc+'_waveform']
            grad=np.column_stack([wave[c+'_T_m'][:] for c in ('gx','gy','gz')])
            b1=read_complex(wave['b1_T'])
            parser_grad=gradient(shot_contract(Path(contract['sequence_directory'])/(enc.lower()+'.seq')),time)
            nonzero=abs(grad)>1e-6
            gamma=float(np.median(parser_grad[nonzero]/grad[nonzero]))
            grad_delta=float(np.max(abs(parser_grad/gamma-grad)))
            if grad_delta>1e-9 or not 42e6<gamma<43e6:raise ValueError('Waveform/gamma mismatch')
            if np.max(abs(wave['rf_frequency_Hz'][:]))>1e-9 or np.max(abs(wave['rf_phase_rad'][:]))>1e-9:
                raise ValueError('Nonzero RF rotating-frame modulation requires explicit ODE treatment')
            print('E049 independent finite-RF Bloch '+enc,flush=True)
            independent[enc]=independent_bloch(time,positions,b1,grad,adc,t1,t2,gamma)
            # ADC demodulation does not affect REF/encoded differences, but is
            # applied for signal-level comparison to the simulator output.
            independent[enc]*=read_complex(wave['adc_phase_compensation'])[None]
            wave_checks.append(dict(encoding=enc,gamma_Hz_T=gamma,gradient_max_delta_T_m=grad_delta,
                                    maximum_signal_absolute_delta=float(np.max(abs(signal[enc]-independent[enc]))),
                                    maximum_signal_relative_delta=float(np.max(abs(signal[enc]-independent[enc]))/np.max(abs(signal[enc])))))
    recovered=np.stack([np.angle(signal[e]*signal['REF'].conj())*5/np.pi for e in ('X','Y','Z')],axis=-1)
    independent_velocity=np.stack([np.angle(independent[e]*independent['REF'].conj())*5/np.pi for e in ('X','Y','Z')],axis=-1)
    center=len(adc)//2
    rows=[];gates=[]
    for i,case in enumerate(cases):
        history=np.array(case['expected_gradient_velocity_m_s'])
        instant=np.array(case['echo_velocity'])
        obs=recovered[i,center]
        reference=independent_velocity[i,center]
        history_error=float(np.max(abs(obs-history)))
        independent_error=float(np.max(abs(obs-reference)))
        constant=case['kind']=='analytic' and 'accelerated' not in case['name']
        gate_error=float(np.max(abs(obs-instant))) if constant else history_error
        tolerance=.002 if constant else .005
        passed=gate_error<=tolerance
        gates.append(dict(case=case['name'],passed=passed,error_m_s=gate_error,tolerance_m_s=tolerance))
        rows.append(dict(case=case['name'],kind=case['kind'],region=case.get('region'),
                         echo_velocity_m_s=instant.tolist(),gradient_history_velocity_m_s=history.tolist(),
                         Koma_velocity_m_s=obs.tolist(),independent_finite_RF_velocity_m_s=reference.tolist(),
                         maximum_gradient_history_error_m_s=history_error,
                         maximum_independent_Bloch_error_m_s=independent_error,
                         echo_vector_discrepancy_m_s=float(np.linalg.norm(obs-instant)),
                         readout_point_response=numerical_point_width(signal['REF'][i])))
    spatial=[]
    for axis,(length,spacing) in enumerate(((400,.00125),(64,.00125),(32,.0025))):
        k=(np.arange(length)-length//2)/(length*spacing)
        for shift in (-2,0,1):
            coordinate=shift*spacing
            image=centered_ifft(np.exp(-2j*np.pi*k*coordinate))
            peak=int(abs(image).argmax())
            if peak!=length//2+shift:raise AssertionError('DFT localization failed')
            spatial.append(dict(axis=axis,coordinate_m=coordinate,peak_index=peak,
                                centered_coordinate_m=(peak-length//2)*spacing,
                                historical_coordinate_m=(peak+.5-length//2)*spacing))
    # Actual Koma readout point controls establish the physical readout pixel
    # convention independently of the synthetic Fourier algebra alone.
    point_checks=[]
    for name,expected in [('static_origin',200),('static_axis1',201)]:
        r=next(row for row in rows if row['case']==name)
        point_checks.append(dict(case=name,expected_peak_index=expected,
                                 observed_peak_index=r['readout_point_response']['peak_index']))
    if any(r['expected_peak_index']!=r['observed_peak_index'] for r in point_checks):
        raise AssertionError('Actual Koma static-point localization failed')
    report=dict(experiment='E-049',created_utc=datetime.now(timezone.utc).isoformat(),
                status='PASS_SMALL_ENCODING_CONTROLS' if all(g['passed'] for g in gates) else 'FAIL_SMALL_ENCODING_GATE',
                cases=rows,gates=gates,waveform_checks=wave_checks,spatial_controls=spatial,
                Koma_point_localization=point_checks,
                maximum_gradient_history_error_m_s=max(r['maximum_gradient_history_error_m_s'] for r in rows),
                maximum_independent_Bloch_velocity_error_m_s=max(r['maximum_independent_Bloch_error_m_s'] for r in rows),
                limitation='Single-shot single-spin numerical controls, not sparse full-volume MRI or independent physical validation. No full-case accuracy recovery or1M authorization.')
    (stage/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    with h5py.File(stage/'analysis_fields.h5','w') as f:
        f['recovered_velocity']=recovered;f['independent_Bloch_velocity']=independent_velocity
        for e in signal:f[e+'_Koma']=signal[e];f[e+'_independent']=independent[e]
    sources=[Path(__file__).resolve(),inputs/'controls.json',inputs/'controls.h5',cpu/'report.toml',cpu/'signals.h5']
    (stage/'MANIFEST.json').write_text(json.dumps(dict(inputs={str(p):sha(p) for p in sources},
        outputs={p.name:sha(p) for p in stage.iterdir() if p.is_file()}),indent=2)+'\n',encoding='utf-8')
    stage.rename(output)
    print(json.dumps({k:report[k] for k in ('status','maximum_gradient_history_error_m_s','maximum_independent_Bloch_velocity_error_m_s','Koma_point_localization')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--cpu',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();analyze(a.inputs,a.cpu,a.output)
