"""Independent full-array readback of the display-only SNR12 export.

No import or execution of the plotting source. Preserve after first execution.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

ROOT = Path('D:/MRSim/outputs/aorta_presentation_2026-10-25/magnitude_orthoviews_1m_snr12_v1')
REC = Path('D:/MRSim/outputs/aorta_20bin_e062/combined/simulation_motion_1m_20bin_4mm/pc_reconstruction.h5')
MATCH = Path('D:/MRSim/outputs/aorta_20bin_e062/analysis_v1/matched_reference_20bin.h5')
PARENT = ROOT.parent/'magnitude_orthoviews_1m_v2'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


checks = []


def check(condition, name):
    if not condition:
        raise AssertionError(name)
    checks.append(name)


destination = ROOT/'INDEPENDENT_VALIDATION.json'
if destination.exists():
    raise FileExistsError(destination)
manifest = json.loads((ROOT/'MANIFEST.json').read_text(encoding='utf-8'))
report = json.loads((ROOT/'REPORT.json').read_text(encoding='utf-8'))
parent = json.loads((PARENT/'REPORT.json').read_text(encoding='utf-8'))
for category in ['input_sha256', 'additional_source_sha256']:
    for path, digest in manifest[category].items():
        check(sha(path) == digest, 'Input/source identity '+str(path))
for name, record in manifest['outputs'].items():
    check((ROOT/name).stat().st_size == record['bytes'], 'Export size '+name)
    check(sha(ROOT/name) == record['sha256'], 'Export hash '+name)
with h5py.File(REC, 'r') as source:
    original = source['reference_magnitude'][:]
with h5py.File(MATCH, 'r') as source:
    core = source['masks/core15'][:].astype(bool)
amplitude = float(np.median(original.mean(axis=0)[core]))
sigma = amplitude/12.0
rng = np.random.Generator(np.random.PCG64(20261025))
nr = rng.standard_normal(original.shape)[1]
ni = rng.standard_normal(original.shape)[1]
reconstructed = np.hypot(original[1]+sigma*nr, sigma*ni)
with np.load(ROOT/'displayed_arrays.npz', allow_pickle=False) as saved:
    check(np.array_equal(saved['calibration_core15'], core), 'Unchanged fixed calibration core15')
    check(np.array_equal(saved['clean_volume'], original[1]), 'Clean source volume unchanged')
    check(np.array_equal(saved['noise_real_standard_normal'], nr), 'Full saved real RNG draw reproducible')
    check(np.array_equal(saved['noise_imag_standard_normal'], ni), 'Full saved imaginary RNG draw reproducible')
    check(float(saved['reference_amplitude']) == amplitude, 'Independent full core amplitude calibration')
    check(float(saved['sigma_per_component']) == sigma, 'Independent per-component SD calibration')
    check(float(saved['target_snr']) == 12.0 and int(saved['seed']) == 20261025, 'SNR12 and seed')
    check(np.allclose(saved['noisy_magnitude_volume'], reconstructed,
                      rtol=8*np.finfo(float).eps, atol=0), 'Independent hypot from full noise arrays')
    volume = saved['noisy_magnitude_volume']
    check(np.all(volume > 0), 'Noise retained in every voxel; no zero background mask')
    expected = dict(projection_xy=volume.max(axis=0), projection_xz=volume.max(axis=1),
        projection_yz=volume.max(axis=2), slice_xy=volume[16],
        slice_xz=volume[:, 16, :], slice_yz=volume[:, :, 36])
    for key, array in expected.items():
        check(np.array_equal(saved[key], array), 'Complete noisy volume/panel binding '+key)
    for key, n, fov in [('x_mm', 72, .28), ('y_mm', 32, .128), ('z_mm', 32, .128)]:
        check(np.array_equal(saved[key], (np.arange(n)-n//2)*fov/n*1000),
              'Native centered-FFT coordinate '+key)
check(report['noise']['reference_amplitude'] == amplitude and
      report['noise']['sigma_per_component'] == sigma, 'Report noise calibration identity')
check(report['noise']['calibration_voxels'] == int(core.sum()), 'Report calibration voxel count')
check(report['display']['vmax'] == parent['display']['vmax'] and report['display']['vmin'] == 0,
      'Clean/noisy shared grayscale window')
check(report['actual_adc_center_ms'] == parent['actual_adc_center_ms'], 'Actual ADC time preserved')
check(report['spins'] == 1000000 and report['original_bin'] == 2 and report['total_bins'] == 20,
      '1M and original bin2/20 provenance')
check(report['display']['mask'] is False and report['display']['smoothing'] is False,
      'No display mask or smoothing')
check(report['new_acquisition'] is False and report['new_registered_metric'] is False,
      'No acquisition or scientific metric registration')
name = 'aorta_1m_bin02_magnitude_orthoviews_snr12'
with Image.open(ROOT/(name+'.png')) as image:
    check(image.size == (5280, 2832), 'PNG dimensions')
    image.verify()
check((ROOT/(name+'.pdf')).read_bytes().startswith(b'%PDF-'), 'PDF signature')
out = dict(status='PASS_INDEPENDENT_SNR12_ORTHOVIEW_READBACK',
    created_utc=datetime.now(timezone.utc).isoformat(),
    checks=len(checks), check_names=checks, source_script_sha256=sha(__file__),
    scope='Full saved bin2 arrays; full20-bin calibration and exact seeded noise draws; input/output identities',
    limitations='Numerical reproducibility of an assumed display-noise model; not scanner SNR validation. PNG visual review recorded separately; PDF not independently rasterized.',
    new_acquisition=False, new_registered_metric=False)
destination.write_text(json.dumps(out, indent=2)+'\n', encoding='utf-8')
print(json.dumps(dict(status=out['status'], checks=len(checks), report=str(destination))))
