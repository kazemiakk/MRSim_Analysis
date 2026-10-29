"""E051 early/late 3D-volume central-profile known-velocity preflight gate."""
import argparse
from pathlib import Path
import tomllib

import h5py
import numpy as np

from scripts.build_aorta_inlet_picker import sha


def check(folder):
    target = folder/'numerical_gate.toml'
    if target.exists():
        raise FileExistsError(target)
    report = tomllib.loads((folder/'report.toml').read_text())
    if report['status'] != 'COMPLETE_3D_CONTROL_RESOURCE_PREFIX' or sha(folder/'prefix_signals.h5') != report['signal_sha256']:
        raise ValueError('Incomplete/mismatched preflight')
    rows = []
    with h5py.File(folder/'prefix_signals.h5') as f:
        for clock in ('early', 'late'):
            ref, enc = f['REF_'+clock][:].ravel(), f['X_'+clock][:].ravel()
            expected = ref*np.exp(1j*np.pi*1.5/5)
            rel = float(np.linalg.norm(enc-expected)/np.linalg.norm(ref))
            velocity = float(np.angle(enc[200]*ref[200].conjugate())*5/np.pi)
            if not np.isfinite(rel) or rel > .003 or abs(velocity-1.5) > .005:
                raise AssertionError(f'{clock} known-flow gate fails: {velocity}, relative L2={rel}')
            rows.append(dict(clock=clock, echo_velocity_m_s=velocity, relative_complex_L2=rel))
    content = ['status = "PASS_3D_PREFIX_KNOWN_VELOCITY"',
               f'prefix_report_sha256 = "{sha(folder/"report.toml")}"',
               f'checker_sha256 = "{sha(Path(__file__).resolve())}"']
    for row in rows:
        content += ['', '[[rows]]', f'clock = "{row["clock"]}"',
                    f'echo_velocity_m_s = {row["echo_velocity_m_s"]!r}',
                    f'relative_complex_L2 = {row["relative_complex_L2"]!r}']
    target.write_text('\n'.join(content)+'\n', encoding='utf-8')
    print('\n'.join(content), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', type=Path, required=True)
    check(parser.parse_args().preflight)
