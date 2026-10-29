"""E051 independent source, raw phase, spatial localization and voxel-error checks."""
import argparse
import json
from pathlib import Path
import tomllib

import h5py
import numpy as np

from scripts.prepare_3d_flow_control import ENC, SHAPE, dump
from scripts.build_aorta_inlet_picker import sha


def validate(inputs, acquisition, analysis):
    target = analysis/'validation.json'
    if target.exists():
        raise FileExistsError(target)
    checks = []
    def require(test, name):
        if not test:
            raise AssertionError(name)
        checks.append(name)
    for name, digest in json.loads((analysis/'MANIFEST.json').read_text())['outputs'].items():
        require(sha(analysis/name) == digest, 'output hash '+name)
    for name, digest in json.loads((analysis/'registration.json').read_text())['inputs'].items():
        require(sha(Path(name)) == digest, 'input hash '+name)
    report = json.loads((analysis/'report.json').read_text())
    runtime = tomllib.loads((acquisition/'control_report.toml').read_text())
    require(runtime['status'] == 'COMPLETE_E051_3D_ACQUISITIONS', 'twelve encodings complete')
    require(len(runtime['result_sha256']) == 12, 'twelve distinct encoding identities')
    with h5py.File(inputs/'controls.h5') as source:
        for model in ('static_landmarks', 'plug_positive', 'plug_negative'):
            summary = report['models'][model]
            coordinates = source[model+'/indices'][:]
            truth = source[model+'/velocity'][:]
            if model == 'static_landmarks':
                core_coords = coordinates
            else:
                core_coords = np.array([(z, y, x) for z in range(13, 17) for y in range(22, 28) for x in range(400)])
            core = tuple(core_coords.T)
            signals = {}; sample_ref = None; valid = np.ones((15, len(core_coords)), bool)
            for e in ENC:
                path = acquisition/(model+'_'+e.lower()+'.h5')
                require(sha(path) == runtime['result_sha256'][model+'_'+e.lower()], 'runtime result hash '+model+e)
                with h5py.File(path) as f:
                    image = f['image'][:]
                    # Only a deterministic subset for this independent raw
                    # check; the main analysis checks every ADC sample.
                    raw = f['raw_signal_f64'][:].ravel()
                require(image.shape == SHAPE and np.isfinite(image).all(), 'finite full3D image '+model+e)
                sample = raw[np.arange(0, 30720, 7)*400+200]
                if e == 'REF':
                    sample_ref = sample
                else:
                    phase = np.pi*truth[ENC.index(e)-1]/5
                    rel = np.linalg.norm(sample-sample_ref*np.exp(1j*phase))/np.linalg.norm(sample_ref)
                    require(np.isfinite(rel) and rel <= .003, 'sampled analytic VENC phase '+model+e)
                values = image[(slice(None), *core)]
                signals[e] = values
                maxima = abs(image).max(axis=(1, 2, 3))
                valid &= (abs(values) >= .05*maxima[:, None])&(abs(values) > 0)
                if model == 'static_landmarks' and e == 'REF':
                    for b in range(15):
                        for point in coordinates:
                            low = point-3; high = point+4
                            box = image[b, low[0]:high[0], low[1]:high[1], low[2]:high[2]]
                            observed = np.array(np.unravel_index(abs(box).argmax(), box.shape))+low
                            require(np.array_equal(observed, point), 'independent exact3D landmark')
                del raw, image
            keep = valid.all(axis=0)
            cov = float(keep.sum()/len(keep))
            require(np.isclose(cov, summary['fixed_core_coverage_fraction'], atol=1e-12), 'independent fixed core coverage '+model)
            if keep.any():
                # Preserve the declared primary Float32 phase conversion's
                # operation order; independent aggregation/masks follow below.
                recovered = np.stack([np.angle(signals[e][:, keep]*signals['REF'][:, keep].conj())*(5/np.pi) for e in ENC[1:]], axis=-1)
                delta = recovered.astype(float)-truth
                rmse = float(np.sqrt(np.sum(delta*delta)/(delta.shape[0]*delta.shape[1])))
                require(np.isclose(rmse, summary['metrics']['vector_rmse_m_s'], atol=1e-10), 'independent known-vector RMSE '+model)
                require(np.isclose(abs(delta).max(), summary['metrics']['maximum_component_error_m_s'], atol=1e-10), 'independent maximum component error '+model)
            else:
                require(summary['metrics'] is None, 'unavailable metrics on empty support')
    require(all(g['passed'] for g in report['gates']) == (report['status'] == 'PASS_3D_KNOWN_FLOW_IMAGE_GATES'), 'gates and status agree')
    value = dict(status='PASS_INDEPENDENT_E051_REPRODUCTION_CHECKS', checks=len(checks),
        scientific_gates_pass=all(g['passed'] for g in report['gates']), report_sha256=sha(analysis/'report.json'),
        limitation='Numerical verification only. Conditional signal law and sampled raw check do not independently validate Bloch magnitude or CFD accuracy.',
        check_names=checks)
    dump(target, value)
    print(json.dumps({k:value[k] for k in ('status', 'checks', 'scientific_gates_pass', 'report_sha256')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--acquisition', type=Path, required=True)
    parser.add_argument('--analysis', type=Path, required=True)
    args = parser.parse_args(); validate(args.inputs, args.acquisition, args.analysis)
