"""Compare saved full phantom REF acquisitions before/after exact lookup acceleration."""
import argparse
import json
from pathlib import Path
import tomllib
import h5py
import numpy as np
from scripts.build_aorta_inlet_picker import sha


def validate(base, output):
    if output.exists():
        raise FileExistsError(output)
    before, after = base/'phantom_updated_nomap_v1', base/'phantom_updated_nomap_v2'
    old = tomllib.loads((before/'registration.toml').read_text())
    new = tomllib.loads((after/'registration.toml').read_text())
    for key in ['arm', 'case_info_sha256', 'input_sha256', 'map_sha256', 'runtime_manifest_sha256', 'script_sha256']:
        if old[key] != new[key]:
            raise ValueError('Unmatched sampling equivalence inputs: '+key)
    evidence = {}; exact = True
    with h5py.File(before/'ref_result.h5') as f, h5py.File(after/'ref_result.h5') as g:
        for key in ['rawsort', 'image']:
            a, b = f[key][:], g[key][:]
            identical = bool(np.array_equal(a, b)); exact &= identical
            error = float(abs(a-b).max()/max(abs(a).max(), 1e-30))
            if error > 3e-6:
                raise ValueError('Full acquisition changed beyond fixed FFT/single-precision tolerance')
            evidence[key] = dict(shape=list(a.shape), bitwise_equal=identical,
                                 different_values=int(np.count_nonzero(a != b)), maximum_relative_error=error)
    result = dict(status='PASS_BITWISE_FULL_REF_SAMPLING_EQUIVALENCE' if exact else 'PASS_FLOAT32_FULL_REF_SAMPLING_EQUIVALENCE',
        numerical_gate_relative=3e-6, evidence=evidence, code_sha256=sha(__file__),
        input_sha256=old['input_sha256'],
        before_file_sha256=sha(before/'ref_result.h5'), after_file_sha256=sha(after/'ref_result.h5'))
    output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); validate(a.base, a.output)
