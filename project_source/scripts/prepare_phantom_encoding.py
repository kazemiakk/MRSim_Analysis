"""E049 freeze single-spin analytic and saved-trajectory controls."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import tomllib
import uuid

import h5py
import numpy as np

from scripts.aorta_hemodynamic_reference import read_pulseq
from scripts.build_aorta_inlet_picker import sha
from scripts.diagnose_phantom_signal import ROOT, E047

BASE = ROOT/'outputs/phantom_encoding_e049'


def shot_contract(path):
    """First acquired shot: physical RF block start through ADC block end."""
    data = read_pulseq(path)
    definitions = {r[0]:r[1:] for r in data['DEFINITIONS']}
    raster = float(definitions['BlockDurationRaster'][0])
    adcs = {int(r[0]):r for r in data['ADC']}
    traps = {int(r[0]):np.array(r[1:],float) for r in data['TRAP']}
    start = 0.
    blocks = []
    for row in data['BLOCKS']:
        row = list(map(int,row[:8]))
        if row[2]:
            origin = start
            blocks = []
        blocks.append((start,row))
        end = start+row[1]*raster
        if row[6]:
            adc = adcs[row[6]]
            count, dwell = int(adc[1]), float(adc[2])*1e-9
            # Pulseq stores the ADC gate delay; Koma's Pulseq reader centers
            # samples in their dwells. Exported starts+ADC.delay is NOT the
            # first actual sampling time. Guarded against get_adc_sampling_times.
            adc_times = start-origin+float(adc[3])*1e-6+(np.arange(count)+.5)*dwell
            events = []
            for block_start, entry in blocks:
                for axis, identifier in enumerate(entry[3:6]):
                    if not identifier:
                        continue
                    amplitude,rise,flat,fall,delay = traps[identifier]
                    t = block_start-origin+np.array([delay,delay+rise,delay+rise+flat,delay+rise+flat+fall])*1e-6
                    events.append((axis,t,np.array([0.,amplitude,amplitude,0.])))
            return dict(duration=end-origin,echo=adc_times[count//2],adc_times=adc_times,events=events,
                        rf_block_start_absolute=origin)
        start = end
    raise ValueError('No acquired shot')


def gradient(shot,times):
    values = np.zeros((len(times),3))
    for axis,t,a in shot['events']:
        values[:,axis] += np.interp(times,t,a,left=0,right=0)
    return values


def phase_expectation(ref,enc,time,positions,echo,venc=5.):
    # Split at every path and gradient knot. Two-point Gauss is exact for the
    # product of each piecewise-linear trajectory and gradient segment.
    cuts = np.unique(np.r_[time,*[r[1] for s in (ref,enc) for r in s['events']],0.,echo])
    cuts = cuts[(cuts>=0)&(cuts<=echo)]
    mid = (cuts[:-1]+cuts[1:])/2
    half = np.diff(cuts)/2
    points = np.column_stack((mid-half/np.sqrt(3),mid+half/np.sqrt(3))).ravel()
    weights = np.repeat(half,2)
    position = np.column_stack([np.interp(points,time,positions[:,c]) for c in range(3)])
    difference = gradient(enc,points)-gradient(ref,points)
    return float(-2*venc*np.sum(weights*np.sum(difference*position,axis=1)))


def prepare(output):
    if output.exists():
        raise FileExistsError(output)
    stage = output.with_name(output.name+'.building-'+uuid.uuid4().hex[:8])
    stage.mkdir(parents=True)
    info = tomllib.loads((E047/'phantom_inputs_v1/case_info.toml').read_text())
    seqdir = Path(info['sequence_directory'])
    shots = {e:shot_contract(seqdir/(e.lower()+'.seq')) for e in ('REF','X','Y','Z')}
    ref = shots['REF']
    for shot in shots.values():
        np.testing.assert_allclose(shot['adc_times'],ref['adc_times'],rtol=0,atol=1e-9)
    t = np.linspace(0,ref['duration'],int(np.ceil(ref['duration']/.25e-6))+1)
    tau = t-ref['echo']
    cases = []

    def analytic(name,position,velocity,acceleration):
        p,v,a = [np.asarray(x,float) for x in (position,velocity,acceleration)]
        trajectory = p+tau[:,None]*v+.5*tau[:,None]**2*a
        cases.append(dict(name=name,kind='analytic',positions=trajectory,echo_velocity=v,
                          acceleration=a,T1=1.3,T2=.1))

    analytic('static_origin',[0,0,0],[0,0,0],[0,0,0])
    for c,d in enumerate((.00125,.00125,.0025)):
        p = np.zeros(3);p[c]=d
        analytic('static_axis'+str(c+1),p,[0,0,0],[0,0,0])
    for c,speeds in enumerate(((.5,1.5,-.75),(.4,),(-.3,))):
        for speed in speeds:
            v = np.zeros(3);v[c]=speed
            analytic('constant_axis'+str(c+1)+'_'+str(speed),[0,0,0],v,[0,0,0])
    for c,acceleration,velocity in ((0,200.,1.5),(0,-200.,1.5),(1,100.,.4),(2,-100.,-.3)):
        v = np.zeros(3);a=np.zeros(3);v[c]=velocity;a[c]=acceleration
        analytic('accelerated_axis'+str(c+1)+'_'+str(acceleration),[0,0,0],v,a)
    sources = [Path(__file__).resolve(),ROOT/'koma_migration_e046/encoding_controls_e049.jl',
               BASE/'PROTOCOL.md',ROOT/'koma_migration_e046/Project.toml',ROOT/'koma_migration_e046/Manifest.toml',
               E047/'phantom_inputs_v1/case_inputs.h5',E047/'phantom_inputs_v1/case_info.toml',
               ROOT/'outputs/phantom_diagnosis_e048/FINAL_MANIFEST_v2.json']
    sources += [seqdir/(e.lower()+'.seq') for e in ('REF','X','Y','Z')]
    # Samples fixed by E048's independently checked controls, not selected for
    # better Koma agreement. Both throat and upstream paths, two source phases.
    with h5py.File(E047/'phantom_inputs_v1/case_inputs.h5') as f:
        track_time = np.arange(1441)*info['period_s']/1440
        for bin_index,spin,region in ((3,11455,'throat'),(3,31556,'throat'),(3,53157,'upstream'),
                                      (9,22576,'throat'),(9,44685,'throat'),(9,88456,'throat')):
            center = (bin_index+.5)/15*info['period_s']
            sample_time = center+tau
            left = np.searchsorted(track_time,sample_time[0],side='right')-1
            right = np.searchsorted(track_time,sample_time[-1],side='left')
            if np.any(f['reset'][left:right+2,spin]):
                raise ValueError('Selected source control crosses a reset')
            positions = np.column_stack([np.interp(sample_time,track_time,f[d][:,spin].astype(float))+float(f[c][spin])
                                         for c,d in zip(('x','y','z'),('dx','dy','dz'))])
            index = 48+96*bin_index
            velocity = np.array([(float(f[d][index+1,spin])-float(f[d][index-1,spin]))/(2*info['period_s']/1440)
                                 for d in ('dx','dy','dz')])
            cases.append(dict(name=f'saved_bin{bin_index+1}_spin{spin}',kind='saved_path',region=region,
                              positions=positions,echo_velocity=velocity,source_bin=bin_index+1,source_spin=spin,
                              T1=float(f['T1'][spin]),T2=float(f['T2'][spin])))
    registration = dict(experiment='E-049',created_utc=datetime.now(timezone.utc).isoformat(),
                        status='REGISTERED_BEFORE_SMALL_CPU_CONTROLS',
                        inputs={str(p):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sources},
                        sources_preserved=True,no_1m=True)
    (stage/'registration.json').write_text(json.dumps(registration,indent=2)+'\n',encoding='utf-8')
    rows = []
    with h5py.File(stage/'controls.h5','w') as f:
        f['time_s']=t
        f['echo_s']=ref['echo']
        f['adc_time_s']=ref['adc_times']
        for i,case in enumerate(cases):
            group=f.create_group(f'case_{i:03}')
            p=case['positions']
            for c in range(3):group['xyz'[c]]=p[:,c]
            group['T1']=case['T1'];group['T2']=case['T2']
            expected = [phase_expectation(ref,shots[e],t,p,ref['echo']) for e in ('X','Y','Z')]
            row={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in case.items() if k!='positions'}
            row.update(index=i,expected_gradient_velocity_m_s=expected,
                       displacement_RFstart_to_echo_mm=((p[np.argmin(abs(t-ref['echo']))]-p[0])*1000).tolist())
            rows.append(row)
    report=dict(experiment='E-049',status='PREPARED_KNOWN_TRAJECTORY_CONTROLS',cases=rows,
                echo_s=ref['echo'],shot_duration_s=ref['duration'],sequence_directory=str(seqdir),
                phantom_T1_T2_preserved_for_source_controls=True,
                gates=dict(constant_velocity_absolute_m_s=.002,phase_history_absolute_m_s=.005))
    (stage/'controls.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    (stage/'MANIFEST.json').write_text(json.dumps({p.name:sha(p) for p in stage.iterdir() if p.is_file()},indent=2)+'\n')
    stage.rename(output)
    print(json.dumps(dict(status=report['status'],cases=len(rows),echo_s=ref['echo'],shot_duration_s=ref['duration'])))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    prepare(p.parse_args().output)
