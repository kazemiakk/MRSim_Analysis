"""E051 deterministic 3D control and independent native-Pulseq moment audit."""
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

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'outputs/known_flow_3d_e051'
INPUTS = ROOT/'outputs/remapped_confirmation_e047/phantom_inputs_v1'
ENC = ('REF', 'X', 'Y', 'Z')
SHAPE = (15, 32, 64, 400)
SPACING = np.array([.00125, .00125, .0025])


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def geometry():
    landmarks = np.array([[12, 21, 124], [18, 40, 263], [13, 43, 217], [21, 27, 179], [15, 34, 203]], int)
    index = np.array([(z, y, x) for z in range(12, 18) for y in range(21, 29) for x in range(400)], int)
    convert = lambda a:(a[:, ::-1]-np.array([200, 32, 16]))*SPACING
    return dict(static_landmarks=dict(positions=convert(landmarks), indices=landmarks,
                    rho=np.array([1., .8, .65, .5, .35]), velocity=np.zeros(3)),
                plug_positive=dict(positions=convert(index), indices=index, rho=np.ones(len(index)), velocity=np.array([1.5, 0., 0.])),
                plug_negative=dict(positions=convert(index), indices=index, rho=np.ones(len(index)), velocity=np.array([-.75, 0., 0.])))


def native_profiles(path):
    parsed = read_pulseq(path)
    definitions = {r[0]:r[1:] for r in parsed['DEFINITIONS']}
    raster = float(definitions['BlockDurationRaster'][0])
    traps = {int(r[0]):np.array(r[1:], float) for r in parsed['TRAP']}
    adc = {int(r[0]):r for r in parsed['ADC']}
    origin = 0; clock = 0; blocks = []; records = []; templates = {}
    for line in parsed['BLOCKS']:
        row = list(map(int, line[:8]))
        if row[2]:
            origin = clock; blocks = []; rf_id = row[2]
        if any(row[3:6]):
            blocks.append((clock-origin, *row[3:6]))
        if row[6]:
            a = adc[row[6]]
            sample = (clock-origin)*raster+float(a[3])*1e-6+(np.arange(int(a[1]))+.5)*float(a[2])*1e-9
            key = (tuple(blocks), tuple(a[1:4]))
            if key not in templates:
                templates[key] = dict(blocks=tuple(blocks), adc=sample)
            records.append(dict(key=key, rf_start_s=origin*raster, rf_id=rf_id,
                                first_s=sample[0]+origin*raster, count=int(a[1])))
        clock += row[1]
    if len(records) != 30720:
        raise ValueError('Expected 30720 acquired profiles')
    return parsed, traps, raster, templates, records, clock*raster


def trapezoid_moments(samples, amplitude, rise, flat, fall, start):
    """Exact integrals of G(t) and t*G(t), G in Pulseq Hz/m."""
    m0 = np.zeros_like(samples, dtype=float); m1 = m0.copy()
    pieces = [(start, rise, 0., amplitude/rise if rise else 0.),
              (start+rise, flat, amplitude, 0.),
              (start+rise+flat, fall, amplitude, -amplitude/fall if fall else 0.)]
    for t0, length, intercept, slope in pieces:
        if length == 0:
            continue
        q = np.clip(np.asarray(samples)-t0, 0, length)
        area = intercept*q+slope*q*q/2
        m0 += area
        m1 += t0*area+intercept*q*q/2+slope*q*q*q/3
    return m0, m1


def moments(template, traps, raster):
    result0 = np.zeros((400, 3)); result1 = result0.copy()
    for offset, *identifiers in template['blocks']:
        for axis, identifier in enumerate(identifiers):
            if not identifier:
                continue
            amp, rise, flat, fall, delay = traps[identifier]
            m0, m1 = trapezoid_moments(template['adc'], amp, rise*1e-6, flat*1e-6, fall*1e-6,
                                      offset*raster+delay*1e-6)
            result0[:, axis] += m0; result1[:, axis] += m1
    return result0, result1


def prepare(output):
    if output.exists():
        raise FileExistsError(output)
    stage = output.with_name(output.name+'.building-'+uuid.uuid4().hex[:8]); stage.mkdir(parents=True)
    info = tomllib.loads((INPUTS/'case_info.toml').read_text())
    seqdir = Path(info['sequence_directory'])
    sources = [Path(__file__).resolve(), BASE/'PROTOCOL.md', ROOT/'koma_migration_e046/known_flow_3d_e051.jl',
               ROOT/'koma_migration_e046/Manifest.toml', INPUTS/'case_info.toml', INPUTS/'legacy_timing.h5',
               ROOT/'scripts/aorta_hemodynamic_reference.py', ROOT/'scripts/build_aorta_inlet_picker.py']
    sources += [seqdir/(e.lower()+'.seq') for e in ENC]
    for e in ENC:
        if sha(seqdir/(e.lower()+'.seq')) != info['sequence_sha256'][e]:
            raise ValueError('Frozen sequence changed')
    dump(stage/'registration.json', dict(experiment='E-051', status='REGISTERED_BEFORE_CONTROL_INPUTS',
        created_utc=datetime.now(timezone.utc).isoformat(), inputs={str(p):sha(p) for p in sources}))
    parsed, traps, raster, templates, records, duration = native_profiles(seqdir/'ref.seq')
    ref_moments = {key:moments(t, traps, raster) for key, t in templates.items()}
    model = geometry()
    audit = []
    with h5py.File(stage/'controls.h5', 'w') as f, h5py.File(INPUTS/'legacy_timing.h5') as old:
        f['rf_start_s'] = np.array([r['rf_start_s'] for r in records])
        f['labels'] = old['REF/labels'][:]
        f['duration_s'] = duration
        for name, g in model.items():
            target = f.create_group(name)
            for key in ('positions', 'indices', 'rho', 'velocity'):
                target[key] = g[key]
        for e in ENC:
            print('E051_MOMENTS', e, flush=True)
            data, tr, step, ts, rs, length = native_profiles(seqdir/(e.lower()+'.seq'))
            if data['RF'] != parsed['RF'] or data['ADC'] != parsed['ADC'] or length != duration or step != raster:
                raise ValueError('RF/ADC/sequence clock differs across encodings')
            first = np.array([r['first_s'] for r in rs])
            if np.max(abs(first-old[e+'/adc_first_s'][:])) > 1e-7:
                raise ValueError('Native Pulseq ADC timing differs from saved actual ADC centers')
            if [r['rf_id'] for r in rs] != [r['rf_id'] for r in records]:
                raise ValueError('Physical RF ordering differs')
            mm = {key:moments(t, tr, step) for key, t in ts.items()}
            pairs = set((a['key'], b['key']) for a, b in zip(records, rs))
            base0 = mm[rs[0]['key']][0]-ref_moments[records[0]['key']][0]
            base1 = mm[rs[0]['key']][1]-ref_moments[records[0]['key']][1]
            spread0 = spread1 = 0.
            for a, b in pairs:
                spread0 = max(spread0, float(np.max(abs(mm[b][0]-ref_moments[a][0]-base0))))
                spread1 = max(spread1, float(np.max(abs(mm[b][1]-ref_moments[a][1]-base1))))
            # Uniform velocity trajectories have r(t)=r_initial+v*t. All
            # profile variations are bounded, never fitted to Koma data.
            worst = 2*np.pi*(spread0*(.25+1.5*duration)+spread1*1.5)
            initial_position_bound = 2*np.pi*float(abs(base0).sum(axis=1).max())*.25
            if worst+initial_position_bound > 1e-4:
                raise ValueError('Profile-invariance phase bound fails')
            encoded = -10*base1
            expected = np.zeros(3)
            if e != 'REF':
                expected[ENC.index(e)-1] = 1.
            if np.max(abs(encoded-expected[None])) > 2e-5:
                raise ValueError('Gradient moments do not implement the declared VENC/sign/axis')
            f[e+'_delta_m0'] = base0; f[e+'_delta_m1'] = base1
            audit.append(dict(encoding=e, profiles=len(rs), unique_profile_pairs=len(pairs),
                maximum_ADC_timestamp_delta_s=float(np.max(abs(first-old[e+'/adc_first_s'][:]))),
                maximum_m0_profile_spread=spread0, maximum_m1_profile_spread=spread1,
                profile_phase_bound_rad=worst, initial_position_phase_bound_rad=initial_position_bound,
                maximum_VENC_axis_coefficient_error=float(np.max(abs(encoded-expected[None])))))
    contract = dict(experiment='E-051', status='PREPARED_ANALYTIC_3D_CONTROLS', shape=list(SHAPE),
        sequence_directory=str(seqdir), duration_s=duration, scanner_B0_T=3., venc_m_s=5.,
        models={name:dict(spins=len(g['positions']), velocity_mri_m_s=g['velocity'].tolist(),
            geometry='five off-axis weighted points' if name == 'static_landmarks' else 'uniform readout tube: y21:29,z12:18',
            T1_s=1.3, T2_s=.1) for name, g in model.items()},
        moment_audit=audit, input_h5_sha256=sha(stage/'controls.h5'),
        sequence_sha256=info['sequence_sha256'], source_case_info_sha256=sha(INPUTS/'case_info.toml'),
        manifest_sha256=sha(ROOT/'koma_migration_e046/Manifest.toml'),
        prediction='Known-velocity gradient phase conditional on REF; no independent magnitude/Bloch amplitude claim',
        limitation='Periodic-FOV axial throughflow; no through-slab flow, inflow reset, wall, shear, acceleration or CFD validation.')
    dump(stage/'contract.json', contract)
    stage.rename(output)
    print(json.dumps(dict(status=contract['status'], models=contract['models'], moment_audit=audit)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    prepare(parser.parse_args().output)
