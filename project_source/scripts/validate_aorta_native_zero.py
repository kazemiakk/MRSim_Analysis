"""Independent saved-array arithmetic/flux checks for E-045 (not physical validation)."""
import argparse
import csv
import json
from pathlib import Path
import tomllib
import h5py
import numpy as np
from PIL import Image
from scripts.aorta_hemodynamic_reference import AortaGrid, sequence_timing, temporal_weights
from scripts.build_aorta_inlet_picker import sha
from scripts.validate_aorta_replenishment_mri import manual_trilinear


def validate(run, output):
    if output.exists():
        raise FileExistsError(output)
    registration = json.loads((run/'registration.json').read_text())
    report_path = run/'native_zero_report.json'
    report = json.loads(report_path.read_text())
    e = np.load(run/'native_zero_evidence.npz', allow_pickle=False)
    root = Path(__file__).resolve().parents[1]; checks = []
    def check(name, value):
        if not value:
            raise AssertionError(name)
        checks.append(name)
    for path, entry in registration['inputs'].items():
        path = Path(path)
        check('input '+str(path), path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'])
    check('analyzer code binding', sha(root/'scripts/analyze_aorta_native_zero.py') == registration['code_sha256'])
    check('unchanged E036 estimator binding', sha(root/'scripts/integrate_aorta_inlet_complex.py') == registration['estimator_code_sha256'])
    check('registration binding', sha(run/'registration.json') == report['registration_sha256'])
    check('pre-flow controls', registration['CFD_flow_not_yet_evaluated'] and registration['synthetic_controls']['status'] == 'PASS_SYNTHETIC_CONTROLS')
    check('native-zero contract', report['particle_count'] == 100000 and report['source_start_frame'] == 0 and report['cardiac_bins'] == 16)
    mri = Path(registration['mri_directory'])
    sim = tomllib.loads((mri/'simulation_report.toml').read_text())
    samples = []; ratios = []; grid = AortaGrid()
    for name in ('REF', 'X', 'Y', 'Z'):
        with h5py.File(mri/(name.lower()+'_result.jld2'), 'r') as f:
            image = f['image'][:]; raw = f['rawsort'][:]
        image = image['re']+1j*image['im']; raw = raw['re']+1j*raw['im']
        reconstructed = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(raw, axes=(1, 2, 3)), axes=(1, 2, 3)), axes=(1, 2, 3))
        fft_relative_error = np.max(abs(image-reconstructed))/np.max(abs(reconstructed))
        check('saved raw-to-image FFT '+name, fft_relative_error < 2e-6)
        value = np.array([manual_trilinear(a.real, grid.axes()[::-1], e['query_mri_zyx_m'])
            +1j*manual_trilinear(a.imag, grid.axes()[::-1], e['query_mri_zyx_m']) for a in image])
        samples.append(value); ratios.append(abs(value)/abs(image).max(axis=(1, 2, 3))[:, None])
        path = Path(sim['sequence_directory'])/(name.lower()+'.seq')
        timing, _, records = sequence_timing(path, report['physical_RR_s'])
        check('no RF-to-ADC global seam '+name, timing['RF_to_ADC_seam_crossings'] == 0)
        weights = []
        for b in range(1, 17):
            records_bin = [r for r in records if r['bin'] == b]
            check(f'{name} bin{b} 1024 acquired profiles', len(records_bin) == 1024)
            adc = np.concatenate([r['adc_first_s']+np.arange(r['n'])*r['dwell_s'] for r in records_bin])
            weights.append(temporal_weights(adc, period_s=report['physical_RR_s'], start_frame=0))
        check('frame0 ADC weights '+name, np.allclose(weights, e['frame_weights'], atol=2e-12, rtol=0))
    ratio = np.stack(ratios, axis=-1); valid = np.all(ratio >= .05, axis=-1); common = valid.all(axis=0)
    check('manual complex interpolation', np.allclose(ratio, e['relative_complex_signal'], rtol=1e-9, atol=1e-11))
    check('per-bin signal support', np.array_equal(valid, e['valid']))
    check('fixed all16/all-encoding support', np.array_equal(common, e['fixed']))
    with h5py.File(mri/'pc_reconstruction.h5', 'r') as f:
        venc = float(f.attrs['Venc_m_per_s'])
    velocity = np.stack([np.arctan2((a*samples[0].conj()).imag, (a*samples[0].conj()).real)*venc/np.pi for a in samples[1:]], axis=-1)
    check('independent phase velocity', np.allclose(velocity[valid], e['velocity_mri_m_s'][valid], rtol=1e-9, atol=1e-10))
    check('missing phase remains unavailable', np.isnan(e['velocity_mri_m_s'][~valid]).all())
    area = e['area_mm2']; normal = e['normal_mri']
    qk = np.sum(np.sum(velocity[:, common]*normal, axis=-1)*area[common], axis=1)
    check('independent Koma quadrature', np.allclose(qk, report['Koma_physical_ml_s'], rtol=1e-10, atol=1e-9))
    z = np.load(root/'outputs/aorta_inlet_flow/e031_v1/inlet_numerical_evidence.npz', allow_pickle=False)
    triangles = z['inferred_triangles'][common]
    integrated = np.sum(z['face_velocities_source_m_s'][triangles].mean(axis=1)*area[common, None, None], axis=0)
    native = np.sum(integrated*z['normal_source'][:, None], axis=0)
    scale = report['known_source_time_velocity_scale']
    qc = e['frame_weights']@native*scale
    check('independent CFD quadrature', np.allclose(qc, report['CFD_physical_ml_s'], rtol=1e-12))
    check('no deliberate legacy retiming', abs(report['physical_RR_s']-.72) < 1e-7 and scale == .72/report['physical_RR_s'])
    centers = (np.angle(e['frame_weights']@np.exp(2j*np.pi*np.arange(72)/72)) % (2*np.pi))*.72/(2*np.pi)
    check('unfitted frame0 phase centers', np.allclose(centers, report['native_source_phase_centers_s'], atol=1e-12, rtol=0))
    check('chronological native order', np.all(np.diff(centers) > 0))
    check('no invented instantaneous zero MRI', centers[0] > 0 and not report['MRI_measurement_at_exact_time_zero'])
    error = qk-qc; metrics = report['physical_units_metrics']
    check('RMSE', np.isclose(np.sqrt(np.mean(error**2)), metrics['rmse_ml_s'], rtol=1e-10))
    check('correlation', np.isclose(np.corrcoef(qk, qc)[0, 1], metrics['correlation'], rtol=1e-10))
    check('nominal cycle transport ratio', np.isclose(qk.sum()/qc.sum(), metrics['nominal_bin_cycle_volume_ratio'], rtol=1e-10))
    check('area coverage', np.isclose(area[common].sum()/area.sum(), report['coverage_fraction'], rtol=1e-12))
    check('full-face statement matches membership', report['full_face_supported'] == bool(common.all()))
    rows = list(csv.DictReader((run/'inlet_flow_bins.csv').open(newline='')))
    check('CSV16 actual bins', len(rows) == 16)
    check('CSV native-time flow', np.allclose([float(r['Koma_ml_s']) for r in rows], qk/scale, rtol=1e-10))
    manifest = json.loads((run/'MANIFEST.json').read_text())
    for name, entry in manifest['files'].items():
        path = run/name
        check('artifact '+name, path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'])
        if path.suffix == '.png':
            with Image.open(path) as image:
                image.verify()
            check('PNG decode '+name, True)
        if path.suffix == '.pdf':
            check('PDF signature '+name, path.read_bytes().startswith(b'%PDF-'))
    result = dict(status='PASS_NATIVE_ZERO_SAVED_DATA_CHECKS', check_count=len(checks), checks=checks,
        validator_sha256=sha(__file__), report_sha256=sha(report_path),
        scope='Independent same-data implementation/arithmetic checks, not independent physical validation',
        PDF_verification='Signature/hash only; no rendered PDF review')
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(dict(status=result['status'], check_count=len(checks))))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); validate(a.run, a.output)
