"""Bounded metadata-only inventory of the separately discovered native CFD.

Never reads velocity arrays or assumes these files are the Koma driving data.
No phase averaging, turbulence calculation or mesh remapping is performed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

import h5py
import numpy as np


def frame_metadata(path):
    with h5py.File(path) as f:
        raw = f['settings/Data Variables'][0]
        text = raw.decode('utf-8', errors='strict')
        time = re.search(r'\(flow-time\s+([^)]+)\)', text)
        step = re.search(r'\(time-step\s+([^)]+)\)', text)
        if not time or not step:
            raise ValueError('Missing native timing: '+str(path))
        cells = f['results/1/phase-1/cells']
        return dict(path=str(path), bytes=path.stat().st_size,
            settings_payload_sha256=hashlib.sha256(raw).hexdigest(),
            full_file_hash_recomputed=False,
            native_time_s=float(time.group(1)), native_step=int(float(step.group(1))),
            velocity_component_shape=list(cells['SV_U/1'].shape), fields=sorted(cells.keys()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    base = Path('C:/ansys/CFD new phantom_cycle2_files/dp0/FFF/Fluent')
    paths = sorted(base.glob('FFF-14-*.dat.h5'))
    initial = base/'FFF-13-00000.dat.h5'
    rows = [frame_metadata(path) for path in [initial, *paths]]
    rows.sort(key=lambda r: r['native_time_s'])
    times = np.array([r['native_time_s'] for r in rows])
    if len(rows) != 2101 or not np.allclose(np.diff(times), .01, rtol=0, atol=2e-5):
        raise ValueError('Unexpected source cadence or series; inspect before interpretation')
    source_evidence = []
    for name in ['new_phantom_Q120_4cycle.c', 'new_phantom_Q120_v2.c']:
        path = base/'libudf/src'/name
        raw = path.read_bytes(); text = raw.decode('utf-8')
        period = re.search(r'#define\s+CYCLE_PERIOD\s+([\d.]+)', text)
        source_evidence.append(dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
            declared_period_s=float(period.group(1)) if period else None,
            uses_fmod='fmod(t, CYCLE_PERIOD)' in text,
            compiled_active_hook_verified=False))
    report = dict(audit='E-037-supplemental-native-metadata-inventory',
        status='DIFFERENT_MESH_DATASET_NOT_QUALIFIED_AS_MATCHED_KOMA_REFERENCE',
        evidence_class='Metadata inventory only; not turbulence quantification',
        source_directory=str(base), primary_series_files=len(rows),
        native_start_s=float(times.min()), native_end_s=float(times.max()),
        stored_interval_min_s=float(np.diff(times).min()), stored_interval_max_s=float(np.diff(times).max()),
        nominal_stored_interval_s=.01, confirmed_1ms_saved_output=False,
        native_cells=int(rows[0]['velocity_component_shape'][0]),
        matched_dp150_reference_cells=268052,
        source_series_bytes=sum(r['bytes'] for r in rows),
        source_evidence=source_evidence, frames=rows,
        limitations=[
            'No velocity arrays read; no startup/transient, stationary turbulence or phase ensemble assessed.',
            'Source code declares0.7s periodicity in one candidate; active compiled solver hook is not verified.',
            'Do not use the0.72s transformed MRI-input cycle for this separate dataset without qualification.',
            'Ten-ms saved output does not establish the internal solver step or resolve sub10ms fluctuations.',
            'No k/Reynolds-stress/SGS-energy fields identified in the listed cells; MU_T and velocity time-history M1/M2 storage are not automatically turbulence moments.',
            'Metadata payloads are hashed, not the entire approximately808GB series. Full hashes remain required before a quantitative new reference study.',
            'A CFD-only phase-ensemble study and any later Koma comparison require a separate declared dataset/method decision.'
        ])
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(report, f, indent=2); f.write('\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ['frames', 'source_evidence', 'limitations']}, indent=2))


if __name__ == '__main__':
    main()
