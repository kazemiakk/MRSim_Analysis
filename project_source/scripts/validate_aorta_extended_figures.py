"""Verify E-028 static files and independently recompute section flow metrics."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--figures', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    if args.receipt.exists():
        raise FileExistsError(args.receipt)
    manifest = json.loads((args.figures/'MANIFEST.json').read_text(encoding='utf-8'))
    report = json.loads((args.results/'report.json').read_text(encoding='utf-8'))
    metrics = json.loads((args.figures/'flow_waveform_comparison.json').read_text(encoding='utf-8'))
    checks = 0

    def close(a, b):
        nonlocal checks
        np.testing.assert_allclose(a, b, rtol=2e-6, atol=1e-10)
        checks += 1

    assert manifest['source_report_sha256'] == sha(args.results/'report.json')
    assert manifest['source_h5_sha256'] == sha(args.results/'matched_reference.h5')
    assert manifest['script_sha256'] == sha(Path(__file__).with_name('plot_aorta_extended_comparison.py'))
    checks += 3
    expected = set()
    for b in range(1, 9):
        expected.update(f'speed_projections/aorta_speed_{v}_bin{b:02}' for v in ('XY', 'XZ', 'YZ'))
        expected.add(f'velocity_components/aorta_components_bin{b:02}')
        expected.add(f'velocity_profiles/aorta_profiles_bin{b:02}')
        expected.update(f'cross_sections/aorta_sections_{q}_bin{b:02}' for q in ('speed', 'normal_velocity'))
    expected.update(f'all_bin_overviews/aorta_speed_{v}_all_bins' for v in ('XY', 'XZ', 'YZ'))
    expected.update(f'flow_waveforms/aorta_flow_detail_{s["section"]}' for s in report['section_summary'])
    expected.update(['flow_waveforms/aorta_flow_waveforms_all_sections', 'flow_waveforms/aorta_flow_agreement'])
    expected = {name+'.'+ext for name in expected for ext in ('png', 'pdf')}
    actual_figures = {name for name in manifest['files'] if name.endswith(('.png', '.pdf'))}
    assert expected == actual_figures
    close(len(expected), 130)
    for name, item in manifest['files'].items():
        path = args.figures/name
        close(path.stat().st_size, item['bytes'])
        assert sha(path) == item['sha256'], name
        checks += 1
        if name.endswith('.png'):
            with Image.open(path) as image:
                assert min(image.size) >= 600, (name, image.size)
                image.verify()
            checks += 1
        elif name.endswith('.pdf'):
            with path.open('rb') as stream:
                assert stream.read(5) == b'%PDF-'
            checks += 1
    with h5py.File(args.results/'matched_reference.h5', 'r') as f:
        fields = {'CFD': f['adc_time_average'][...].astype(float), 'Koma': f['Koma'][...].astype(float)}
        mask = f['masks/core15'][...].astype(bool)
        shape = report['grid']['shape']; fov = report['grid']['fov']
        dx, dy, dz = np.asarray(fov)/np.asarray(shape[::-1])
        coordinates = (np.arange(shape[2])-shape[2]/2)*dx
        for saved in metrics:
            section = next(s for s in report['section_summary'] if s['section'] == saved['section'])
            i = int(np.argmin(abs(coordinates-section['x_m'])))
            region = f['sections/'+saved['section']][...].astype(bool) & mask[:, :, i]
            area = f['plane_fraction'][..., i]*dy*dz
            waveforms = {}
            for method, data in fields.items():
                waveforms[method] = np.sum(data[:, :, :, i, 0][:, region]*area[region], axis=1)*1e6
                for b, value in enumerate(waveforms[method]):
                    source = next(x for x in report['section_series'] if x['section'] == saved['section'] and x['method'] == method and x['bin'] == b+1)
                    close(value, source['net_ml_s'])
                close(saved[method+'_net_cycle_ml'], waveforms[method].sum()*report['period_s']/8)
                close(saved[method+'_abs_peak_ml_s'], abs(waveforms[method]).max())
                close(saved[method+'_abs_peak_bin'], int(np.argmax(abs(waveforms[method]))+1))
            cfd, koma = waveforms['CFD'], waveforms['Koma']
            delta = koma-cfd
            close(saved['bias_ml_s'], delta.mean())
            close(saved['mae_ml_s'], abs(delta).mean())
            close(saved['rmse_ml_s'], np.sqrt(np.mean(delta**2)))
            close(saved['correlation'], np.corrcoef(cfd, koma)[0, 1])
            close(saved['nrmse_rms_pct'], np.sqrt(np.mean(delta**2)/np.mean(cfd**2))*100)
            close(saved['nrmse_range_pct'], np.sqrt(np.mean(delta**2))/np.ptp(cfd)*100)
            close(saved['cycle_difference_pct'], (koma.sum()/cfd.sum()-1)*100)
    receipt = dict(status='PASS', experiment='E-028', created_utc=datetime.now(timezone.utc).isoformat(),
        checks=checks, failed_checks=0, png_count=65, pdf_count=65,
        source_report_sha256=sha(args.results/'report.json'), figure_manifest_sha256=sha(args.figures/'MANIFEST.json'),
        scope='All artifact identities, expected view/bin coverage, PNG decoding, PDF headers and four section metrics independently recomputed from saved fields; no physical validation or claim of reviewing every image visually')
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt.open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
