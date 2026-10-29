"""Layout derivative: original magnitude projections + exact user-attached phase views.

No reconstruction, velocity calculation, noise generation, or scientific metrics.
The phase-derived grayscale displays are copied from hash-bound rendered PNGs.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.util
import json

import h5py
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

ROOT = Path('D:/MRSim/outputs/magnitude_layout_2026-10-27')
OUT = ROOT/'attached_phase_v2'
AORTA_PNG = Path('D:/MRSim/outputs/aorta_aligned_e064/figure_v3/aorta_candycane_acquired_v3.png')
PHANTOM_PNG = Path('D:/MRSim/outputs/static_background_e063/phantom_candycane_v3_flow_only/phantom_candycane_peak_systole_flow_only.png')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def main():
    if OUT.exists() or OUT.with_name(OUT.name+'.building').exists():
        raise FileExistsError(OUT)
    original = module(ROOT/'code/organize_magnitude_v4.py', 'original_magnitude')
    expected = {
        original.AORTA/'displayed_arrays.npz': 'e5a7169a5ad1e63bdc224c0e34726f948475530b18700e9bcc5f45de3b091d0c',
        original.PHANTOM: '079712cfc44ef1909cf80eae798f4bdd41cc61a435a376288fd36ae460ec913b',
        AORTA_PNG: '683e354f7173b9a33a7a9b79fa387648cec41c23cd549c4014182a9449aa7a7b',
        PHANTOM_PNG: '4204eab7629a62c2e6a8a343beac3cd4f601143d50fc329ab926abd4300d3211',
        Path('D:/MRSim/outputs/aorta_aligned_e064/code/attempt4_plot_e064.py'): 'bc8ec8a5247067075cb8732556a6a1b539b702f55026b2f4ba43553206143093',
        Path('D:/MRSim/outputs/static_background_e063/code/plot_phantom_flow_only.py'): 'ec51baf8b8c128aee68a2e95bfbb42e361dc25b894489d5af25323f6298991bf',
    }
    for p, digest in expected.items():
        assert sha(p) == digest, str(p)

    with np.load(original.AORTA/'displayed_arrays.npz', allow_pickle=False) as saved:
        aorta = [saved[k].copy() for k in original.KEYS[:3]]
        coordinates = [saved[k].copy() for k in ('x_mm', 'y_mm', 'z_mm')]
    bounds = []
    for axis in coordinates:
        step = axis[1]-axis[0]
        bounds.append([float(axis[0]-step/2), float(axis[-1]+step/2)])
    bx, by, bz = bounds
    mag_extents = {'aorta': [bx+by, bx+bz, by+bz]}
    limits = {'aorta': json.loads((original.AORTA/'REPORT.json').read_text(encoding='utf-8'))['display']['vmax']}
    helper = module(original.HELPER, 'static_phantom_helper')
    with h5py.File(original.PHANTOM, 'r') as f:
        volume = np.asarray(f['reference_magnitude'][1], dtype=np.float32)
    p_specs = helper.panel_data(volume)
    magnitudes = {'aorta': aorta, 'phantom': [p[1].copy() for p in p_specs[:3]]}
    mag_extents['phantom'] = [p[4] for p in p_specs[:3]]
    limits['phantom'] = helper.robust_limit(volume)

    # Read original renderer metadata, not CFD clouds, scan DICOM or new images.
    frame = json.loads(Path('D:/MRSim/outputs/aorta_aligned_e064/frame_v1/frame.json').read_text())
    dims = np.asarray(frame['matrix'])
    vox = np.asarray(frame['voxel_m'])*1000
    xs = (np.arange(dims[0])-dims[0]//2)*vox[0]
    ys = (np.arange(dims[1])-dims[1]//2)*vox[1]
    # Exact figure_v3 uses cr(a)=a[2:-2,2:-2], NOT the earlier one-voxel crop.
    a_extent = [ys[2]-2, ys[-3]+2, xs[2]-2, xs[-3]+2]
    assert np.allclose(a_extent, [-74, 70, -138, 134])
    with h5py.File('D:/MRSim/outputs/scanner_geometry_e060/references_v2/references.h5', 'r') as f:
        zz = f['partition_z_m'][:]*1000
    pr = (np.arange(64)-32)*1.25
    vv = np.sort(-pr)  # The attachment's vertical direction is -P / original -Y.
    p_extent = [float(zz[0]-1.25), float(zz[-1]+1.25), float(vv[0]-.625), float(vv[-1]+.625)]
    assert np.allclose(p_extent, [-38.75, 36.25, -39.375, 40.625])

    # Inner plot rectangles in the exact attachments; only spines/ticks/titles
    # are excluded. All displayed anatomy AND surrounding noise are retained.
    # Do not mask, flip, rotate, smooth, adjust contrast or regenerate noise.
    boxes = {
        'aorta': [(743, 663, 1308, 1732), (1401, 663, 1967, 1732), (2060, 663, 2625, 1732)],
        'phantom': [(774, 728, 1250, 1235), (1435, 728, 1911, 1235), (2096, 728, 2572, 1235)],
    }
    info = {
        'aorta': dict(path=AORTA_PNG, size=(2860, 1870), extent=a_extent,
            titles=['Through-plane', 'In-plane horizontal', 'In-plane vertical'],
            components=["Zprime (slab normal)", "Yprime (horizontal)", "Xprime (vertical)"],
            windows=[30, 30, 30], horizontal='Horizontal (mm)', vertical='Vertical (mm)'),
        'phantom': dict(path=PHANTOM_PNG, size=(2640, 1485), extent=p_extent,
            titles=['Along-vessel', 'Through-plane', 'In-plane vertical'],
            components=['S / original CFD Z', 'R / original CFD X', '-P / original CFD -Y'],
            windows=[300, 150, 150], horizontal='Z (mm)', vertical='Vertical (mm)'),
    }
    phases = {}
    phase_checks = []
    for case, meta in info.items():
        with Image.open(meta['path']) as image:
            assert image.size == meta['size']
            rgb = np.asarray(image.convert('RGB'))
        phases[case] = []
        for component, box in zip(meta['components'], boxes[case]):
            left, top, right, bottom = box
            crop = rgb[top:bottom, left:right].copy()
            assert crop.shape[0] > 400 and crop.shape[1] > 400
            assert np.array_equal(crop[:, :, 0], crop[:, :, 1])
            assert np.array_equal(crop[:, :, 1], crop[:, :, 2])
            phases[case].append(crop)
            phase_checks.append(dict(case=case, component=component, source_png=str(meta['path']),
                source_box_xyxy=list(box), crop_rgb_sha256=hashlib.sha256(crop.tobytes()).hexdigest(),
                extent_mm=[float(n) for n in meta['extent']], pixel_values_unchanged=True,
                inherited_noise_retained=True, operation='Crop plot rectangle only; no spatial transform'))

    stage = OUT.with_name(OUT.name+'.building')
    stage.mkdir(parents=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 17, 'axes.labelsize': 16,
        'xtick.labelsize': 14, 'ytick.labelsize': 14, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})
    fig = plt.figure(figsize=(18, 24), facecolor='white')
    mag_columns = [(.08, .34), (.455, .34), (.83, .15)]
    phase_columns = [(.085, .25), (.395, .25), (.705, .25)]
    mag_checks = []
    for case, y, h, title_y in [('aorta', .824, .13, .960), ('phantom', .267, .095, .372)]:
        titles = ('XY', 'XZ', 'YZ') if case == 'aorta' else ('XZ', 'YZ', 'XY')
        horizontal = ('X', 'X', 'Y') if case == 'aorta' else ('Z', 'Z', 'X')
        vertical = ('Y', 'Z', 'Z') if case == 'aorta' else ('X', 'Y', 'Y')
        for col, (x, w) in enumerate(mag_columns):
            ax = fig.add_axes([x, y, w, h])
            artist = ax.imshow(magnitudes[case][col], origin='lower', extent=mag_extents[case][col],
                cmap='gray', vmin=0, vmax=limits[case], interpolation='nearest', aspect='equal')
            ax.set_xlabel(horizontal[col], labelpad=3)
            ax.set_ylabel(vertical[col], labelpad=3)
            ax.tick_params(length=3, width=.7, pad=3)
            fig.text(x+w/2, title_y, titles[col], ha='center', va='bottom', fontsize=20)
            assert np.array_equal(np.asarray(artist.get_array()), magnitudes[case][col])
            assert artist.get_clim() == (0, limits[case])
            mag_checks.append(dict(case=case, plane=titles[col], horizontal=horizontal[col], vertical=vertical[col],
                array_sha256=hashlib.sha256(magnitudes[case][col].tobytes()).hexdigest(), original_array_and_contrast=True))
        fig.text(.02, y+h/2, 'Magnitude projection', rotation=90, ha='center', va='center', fontsize=18)

    for case, y, h, title_y in [('aorta', .442, .316, .765), ('phantom', .060, .164, .232)]:
        meta = info[case]
        for col, (x, w) in enumerate(phase_columns):
            ax = fig.add_axes([x, y, w, h])
            data = phases[case][col]
            artist = ax.imshow(data, origin='upper', extent=meta['extent'], interpolation='nearest', aspect='equal')
            ax.set_xlabel(meta['horizontal'], labelpad=3)
            ax.set_ylabel(meta['vertical'], labelpad=3)
            ax.tick_params(length=3, width=.7, pad=3)
            fig.text(x+w/2, title_y, meta['titles'][col], ha='center', va='bottom', fontsize=18)
            assert np.array_equal(np.asarray(artist.get_array()), data)
            if case == 'phantom':
                bar = fig.colorbar(ScalarMappable(norm=Normalize(-meta['windows'][col], meta['windows'][col]), cmap='gray'),
                    cax=fig.add_axes([x+.04, .025, w-.08, .006]), orientation='horizontal',
                    ticks=[-meta['windows'][col], 0, meta['windows'][col]])
                bar.ax.tick_params(labelsize=13, pad=2, length=2)
                bar.set_label('cm/s', fontsize=13, labelpad=1)
        fig.text(.02, y+h/2, 'Flow-encoded phase', rotation=90, ha='center', va='center', fontsize=18)
    bar = fig.colorbar(ScalarMappable(norm=Normalize(-30, 30), cmap='gray'),
        cax=fig.add_axes([.953, .468, .010, .26]), ticks=[-30, -15, 0, 15, 30])
    bar.ax.tick_params(labelsize=14, pad=3, length=3)
    bar.set_label('cm/s', fontsize=15, labelpad=5)

    fig.text(.035, .987, 'A  Aorta', va='top', fontsize=26, fontweight='bold')
    fig.text(.98, .987, 'Magnitude: 1M spins, bin 2/20', ha='right', va='top', fontsize=18)
    fig.text(.50, .800, 'Phase: E-064, bin 20/20  |  Candy-cane slab, 24 mm  |  SNR 12', ha='center', fontsize=17)
    fig.text(.50, .412, 'Phase display VENC 30 cm/s; acquired at 250 cm/s. Synthetic stationary tissue retained.',
        ha='center', fontsize=15, color='#333333')
    fig.text(.035, .396, 'B  Stenosis phantom', va='top', fontsize=26, fontweight='bold')
    fig.text(.98, .396, 'Magnitude: static bin 2', ha='right', va='top', fontsize=18)
    fig.text(.50, .247, 'Phase: E-060/E-061, bin 8/30 (253 ms)  |  Long-axis slice  |  SNR 84', ha='center', fontsize=17)

    name = 'Aorta_Phantom_Magnitude_Attached_Phase'
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(stage/(name+'.'+ext), dpi=300, facecolor='white')
    fig.savefig(stage/'preview.png', dpi=80, facecolor='white')
    plt.close(fig)
    caption = ('Original magnitude projections and grayscale flow-encoded phase views from the supplied figures. '
        'Aorta: magnitude bin 2/20; E064 candy-cane phase bin 20/20, SNR12, display VENC30 cm/s (acquired250). '
        'Phantom: static magnitude bin2; E060/E061 long-axis phase bin8/30, SNR84. '
        'Phase scales are cm/s; original noise and contrast retained. Phantom magnitude axes use CFD coordinates (Z along vessel); '
        'phase vertical=-Y. Rows differ in acquisition, view and cardiac bin.')
    assert len(caption) <= 500
    (stage/'caption.txt').write_text(caption+'\n', encoding='utf-8')
    (stage/'README.md').write_text('# Original magnitude and user-attached phase panels\n\n'
        'This supersedes magnitude_phase_v1 for the current figure request. The two supplied phase-figure versions '
        'were visually matched to the exact saved PNGs listed in REPORT.json. Their bottom-row three directional '
        'views were selected: whole candy-cane slab for aorta, longitudinal slice for phantom. All surrounding noise '
        'inside each original plot rectangle is retained. Raster plot interiors are copied as RGB arrays without '
        'rotation, flipping, masking, color/contrast changes, smoothing or a new noise realization. Export rescales '
        'the raster for layout using nearest-neighbor display; no new quantitative velocity values are inferred.\n\n'
        'The six magnitude projections and contrast limits are identical to layout_v4. Phantom long-axis labels '
        'remain original CFD Z; its circular magnitude view is XY. Aorta magnitude labels retain its MRI axes. '
        'The E064 phase views use rotated acquisition horizontal/vertical/depth axes, NOT the earlier magnitude axes.\n\n'
        'PHASE UNITS: These are flow-encoded, phase-derived grayscale velocity displays, not raw phase angles in radians. '
        'Aorta components are slab-normal, horizontal, vertical, all +/-30 cm/s display window; the flow signal was '
        'previously re-encoded from the actual 2.5 m/s acquisition to display VENC0.3 m/s, with synthetic tissue, '
        'complex noise SNR12, Tukey filtering and a 24-mm complex-sum slab. These inherited presentation operations '
        'were not rerun here. The matched original renderer is attempt4_plot_e064.py (hash verified).\n\n'
        'Phantom components retain original attachment order: along-vessel S/CFD Z (+/-300 cm/s), through-plane '
        'R/CFD X (+/-150), and vertical -P/CFD -Y (+/-150). Acquired VENC is 1.5/1.5/4 m/s, '
        'while displayed windows are 1.5/1.5/3 m/s. SNR84.4 noise and Tukey filter are inherited. '
        'No bottles or stationary tissue were added to this phantom source. Random air phase is retained.\n\n'
        'PROVENANCE LIMIT: Magnitude and phase rows use DIFFERENT acquisitions, views and bins. Aorta magnitude '
        'is E062 bin2, phase E064 bin20. Phantom magnitude is historical static dp150 bin2; phase is moving '
        'E060/E061 dp150 bin8, NOT dp122 E065 or E066. This is an appearance montage, not a same-frame comparison. '
        'No raw scan DICOM, private scan arrays or CFD cloud was read. No simulation, GPU job, gate change or '
        'new registered metric. Original files and previous figure versions remain unchanged.\n', encoding='utf-8')
    for p, digest in expected.items():
        assert sha(p) == digest, 'Input changed during assembly: '+str(p)
    with Image.open(stage/(name+'.png')) as im:
        assert im.size == (5400, 7200)
        im.verify()
    assert (stage/(name+'.pdf')).read_bytes().startswith(b'%PDF-')
    report = dict(status='PASS_ORIGINAL_MAGNITUDE_AND_ATTACHED_PHASE_IDENTITIES',
        created_utc=datetime.now(timezone.utc).isoformat(), script_sha256=sha(__file__),
        input_sha256={str(p): digest for p, digest in expected.items()},
        magnitude_panels=mag_checks, phase_panels=phase_checks, source_phase_rows='bottom',
        phase_units='cm/s (phase-derived grayscale velocity display)',
        phase_display_windows_cm_s={'aorta': [30,30,30], 'phantom': [300,150,150]},
        new_noise=False, new_reconstruction=False, new_acquisition=False, new_metrics=False,
        caption_characters=len(caption), visual_review='Pending preview inspection')
    (stage/'REPORT.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    (stage/'MANIFEST.json').write_text(json.dumps({p.name: dict(bytes=p.stat().st_size, sha256=sha(p))
        for p in stage.iterdir() if p.is_file()}, indent=2)+'\n', encoding='utf-8')
    stage.rename(OUT)
    print(json.dumps(dict(status=report['status'], png=str(OUT/(name+'.png')), caption_characters=len(caption)), indent=2))


if __name__ == '__main__':
    main()
