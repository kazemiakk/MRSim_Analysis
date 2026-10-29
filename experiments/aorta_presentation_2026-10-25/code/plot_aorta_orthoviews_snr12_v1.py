"""Display-only SNR-12 derivative of completed E062 magnitude orthoviews.

Reuses the existing aorta_common.py REF noise convention, not a scanner fit.
No acquisition, velocity analysis, new registered metric or input modification.
Preserve these exact source bytes after the first execution.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

BASE = Path('D:/MRSim/outputs/aorta_presentation_2026-10-25')
RECON = Path('D:/MRSim/outputs/aorta_20bin_e062/combined/simulation_motion_1m_20bin_4mm/pc_reconstruction.h5')
MATCH = Path('D:/MRSim/outputs/aorta_20bin_e062/analysis_v1/matched_reference_20bin.h5')
AUDIT = Path('D:/MRSim/outputs/aorta_20bin_e062/sequence_audit_v1/report.json')
PARENT = BASE/'magnitude_orthoviews_1m_v2'
COMMON = BASE/'code/aorta_common.py'
GRID = Path('D:/MRSim/MRSIM/MRSimTracks/scripts/aorta_hemodynamic_reference.py')
EXPECTED = {
    RECON: '7b3072819a723508e5bd8b753dffb76680ea29fee33800123e32f6c5682a44a8',
    MATCH: '74915302fbd20039caea010d72f337ff7faf37d65d38ca3ec160326da308cde8',
    AUDIT: '04548d001bca5bc54f29c205d645ecf68ba84ea0af9aa47beeef5c43c17b8b29',
    PARENT/'REPORT.json': '2ed0cbb677c81e40b581d31981d0ccfbead744dbadc67623d8b1f7dceb3df3a9',
    PARENT/'aorta_1m_bin02_magnitude_orthoviews.png': '8784778343d6d2d9277876325792ac936f1877bc9c3ed0613ac67f2f94ef10f8',
    COMMON: 'c22b07eebf4358e94005340065e2a057c08281880500c710b0b439f08b799479',
}
SNR, SEED, BIN_INDEX = 12.0, 20261025, 1


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    final = parser.parse_args().output.resolve()
    stage = final.with_name(final.name+'.building')
    if final.exists() or stage.exists():
        raise FileExistsError(final)
    checks = []

    def check(condition, name):
        if not condition:
            raise AssertionError(name)
        checks.append(name)

    for path, digest in EXPECTED.items():
        check(sha(path) == digest, 'Pinned input identity '+str(path))
    with h5py.File(RECON, 'r') as source:
        mag = source['reference_magnitude'][:]
        fov = np.asarray(source.attrs['FOV_m'])
        matrix = tuple(np.asarray(source.attrs['matrix'], dtype=int))
        phantom = source.attrs['source_phantom_sha256']
        if isinstance(phantom, bytes):
            phantom = phantom.decode()
    with h5py.File(MATCH, 'r') as source:
        core = source['masks/core15'][:].astype(bool)
    check(mag.shape == (20, 32, 32, 72), 'Original E062 20-bin magnitude shape')
    check(matrix == (72, 32, 32), 'Native readout-phase-slice matrix')
    check(str(phantom) == 'd23241331c8ec57e0e0396ef5468e1902583e7d2095c6ece83e3c71854242720',
          'Original 1M-spin phantom identity')
    check(core.shape == mag.shape[1:] and core.any(), 'Fixed core15 calibration support')
    check(np.isfinite(mag).all() and (mag >= 0).all(), 'Source magnitude finite and nonnegative')
    check(np.array_equal(fov, np.asarray([.28, .128, .128], dtype=fov.dtype)),
          'Nominal FOV exactly represented at stored precision')
    from scripts.aorta_hemodynamic_reference import AortaGrid
    grid = AortaGrid()
    check(tuple(grid.shape) == mag.shape[1:], 'Authoritative centered-FFT grid shape')
    check(np.array_equal(np.asarray(grid.fov, dtype=fov.dtype), fov),
          'Authoritative grid and stored FOV agree')
    x, y, z = [a*1000 for a in grid.axes()]
    spacing = grid.spacing*1000
    iz, iy, ix = (n//2 for n in mag.shape[1:])
    check((z[iz], y[iy], x[ix]) == (0, 0, 0), 'True central planes through zero')

    # Match the existing aorta presentation convention and RNG call order.
    # The saved REF is magnitude-only. A zero-phase complex representative is
    # valid for this magnitude derivative: circular noise makes its distribution
    # independent of the unavailable original REF phase. No phase is recovered.
    reference_amplitude = float(np.median(mag.mean(axis=0)[core]))
    sigma = reference_amplitude/SNR
    check(np.isfinite(sigma) and sigma > 0, 'Positive per-component Gaussian noise SD')
    rng = np.random.default_rng(SEED)
    noise_real = rng.standard_normal(mag.shape)
    noise_imag = rng.standard_normal(mag.shape)
    noisy_complex = mag.astype(complex)+sigma*(noise_real+1j*noise_imag)
    vol = np.abs(noisy_complex[BIN_INDEX])
    clean = mag[BIN_INDEX]
    check(np.isfinite(vol).all() and (vol > 0).all(), 'Unmasked noisy magnitude at every voxel')
    check(np.allclose(vol, np.hypot(clean+sigma*noise_real[BIN_INDEX],
                                  sigma*noise_imag[BIN_INDEX]),
                      rtol=8*np.finfo(float).eps, atol=0),
          'Complex addition precedes magnitude; independent hypot identity')
    arrays = dict(projection_xy=vol.max(axis=0), projection_xz=vol.max(axis=1),
                  projection_yz=vol.max(axis=2), slice_xy=vol[iz],
                  slice_xz=vol[:, iy, :], slice_yz=vol[:, :, ix])
    for axis, key in [(0, 'projection_xy'), (1, 'projection_xz'), (2, 'projection_yz')]:
        reduced = np.take(vol, 0, axis=axis).copy()
        for j in range(1, vol.shape[axis]):
            np.maximum(reduced, np.take(vol, j, axis=axis), out=reduced)
        check(np.array_equal(reduced, arrays[key]), 'Post-noise 3D maximum projection '+key)
    parent = json.loads((PARENT/'REPORT.json').read_text(encoding='utf-8'))
    vmax = float(parent['display']['vmax'])
    check(vmax == float(np.quantile(mag[mag > 0], .995)), 'Unchanged clean all-bin grayscale window')
    timing = json.loads(AUDIT.read_text(encoding='utf-8'))['encodings']['REF']['bin_timing']
    selected = [r for r in timing if r['bin'] == BIN_INDEX+1]
    check(len(selected) == 1, 'Original bin2 actual ADC timing exists')
    phase_ms = float(selected[0]['phase_circular_mean_s'])*1000
    check(phase_ms == parent['actual_adc_center_ms'], 'Unshifted clean/noisy ADC time identity')

    def bounds(axis, d):
        return [float(axis[0]-d/2), float(axis[-1]+d/2)]

    bx, by, bz = bounds(x, spacing[0]), bounds(y, spacing[1]), bounds(z, spacing[2])
    specs = [('Projection - XY', 'projection_xy', 'X (mm)', 'Y (mm)', bx+by),
             ('Projection - XZ', 'projection_xz', 'X (mm)', 'Z (mm)', bx+bz),
             ('Projection - YZ', 'projection_yz', 'Y (mm)', 'Z (mm)', by+bz),
             ('Center slice - XY (Z = 0 mm)', 'slice_xy', 'X (mm)', 'Y (mm)', bx+by),
             ('Center slice - XZ (Y = 0 mm)', 'slice_xz', 'X (mm)', 'Z (mm)', bx+bz),
             ('Center slice - YZ (X = 0 mm)', 'slice_yz', 'Y (mm)', 'Z (mm)', by+bz)]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 15,
        'axes.titlesize': 18, 'axes.labelsize': 15, 'xtick.labelsize': 13.5,
        'ytick.labelsize': 13.5, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    stage.mkdir(parents=True)
    fig, axs = plt.subplots(2, 3, figsize=(22, 11.8),
        gridspec_kw={'width_ratios': [1.5, 1.5, 1.]}, layout='constrained')
    fig.get_layout_engine().set(rect=(0, .07, 1, .88), hspace=.10, wspace=.04)
    for ax, (title, key, xlabel, ylabel, extent) in zip(axs.ravel(), specs):
        artist = ax.imshow(arrays[key], cmap='gray', origin='lower', extent=extent,
            vmin=0, vmax=vmax, interpolation='nearest', aspect='equal')
        ax.set_title(title, pad=12)
        ax.set_xlabel(xlabel, labelpad=7)
        ax.set_ylabel(ylabel, labelpad=7)
        ax.tick_params(length=4, width=.8)
        ax.grid(False)
        check(np.array_equal(np.asarray(artist.get_array()), arrays[key]), 'Displayed noisy data '+key)
        check(artist.get_clim() == (0, vmax), 'Fixed clean-image window '+key)
    fig.suptitle('Aorta 1M-spin KomaMRI magnitude | cardiac bin 2 / 20', fontsize=25, y=.995)
    fig.text(.5, .945, f'SNR 12 (assumed) | display-only complex noise | actual ADC center {phase_ms:.1f} ms',
        ha='center', va='center', fontsize=17, color='#333333')
    fig.text(.5, .038, 'SNR = median time-mean core signal / per-component noise SD. '
        'Fixed clean-image window; no smoothing or display mask.',
        ha='center', va='center', fontsize=13, color='#333333')
    fig.text(.5, .016, 'Simulation axes: X = source Z, Y = source X, Z = source Y. '
        'Native voxels: 3.89 x 4 x 4 mm.', ha='center', va='center', fontsize=13, color='#333333')
    name = 'aorta_1m_bin02_magnitude_orthoviews_snr12'
    png, pdf = stage/(name+'.png'), stage/(name+'.pdf')
    fig.savefig(png, dpi=240, facecolor='white')
    fig.savefig(pdf, facecolor='white')
    plt.close(fig)
    np.savez_compressed(stage/'displayed_arrays.npz', **arrays, x_mm=x, y_mm=y, z_mm=z,
        clean_volume=clean, noisy_magnitude_volume=vol,
        noise_real_standard_normal=noise_real[BIN_INDEX],
        noise_imag_standard_normal=noise_imag[BIN_INDEX], calibration_core15=core,
        reference_amplitude=reference_amplitude, sigma_per_component=sigma, seed=SEED, target_snr=SNR)
    with Image.open(png) as image:
        check(image.size == (5280, 2832), 'PNG dimensions')
        image.verify()
    check(pdf.read_bytes().startswith(b'%PDF-'), 'PDF file signature')
    for path, digest in EXPECTED.items():
        check(sha(path) == digest, 'Input unchanged after export '+str(path))
    report = dict(status='PASS_SNR12_MAGNITUDE_PRESENTATION',
        created_utc=datetime.now(timezone.utc).isoformat(),
        evidence_class='Display-only noise derivative of completed E062; not a noisy acquisition or new registered metric',
        spins=1000000, original_bin=2, total_bins=20, actual_adc_center_ms=phase_ms,
        source_dataset='reference_magnitude', volume_axis_order='bin,sliceZ,phaseY,readoutX',
        spacing_mm=spacing.tolist(), slice_indices_zyx=[iz, iy, ix],
        noise=dict(target_snr=SNR, seed=SEED, generator='numpy.random.default_rng / PCG64',
            definition='median(time-mean REF magnitude in fixed core15) / sigma of each real or imaginary component',
            calibration_mask='masks/core15 from E062 matched_reference_20bin.h5; calibration only, no display masking',
            calibration_voxels=int(core.sum()), reference_amplitude=reference_amplitude,
            sigma_per_component=sigma, complex_noise_rms=float(np.sqrt(2)*sigma),
            formula='abs(reference_magnitude + sigma*(N_real + 1j*N_imag)); independent N(0,1)',
            draw_order='full20-bin real array, then full20-bin imaginary array, then select original bin2',
            spatial_support='all voxels including background; same sigma everywhere',
            phase_assumption='Zero-phase complex representative of saved magnitude; circular noise makes marginal magnitude independent of original REF phase; no phase recovered',
            background='Rayleigh for zero true signal; Rician where the original reconstruction has nonzero signal',
            existing_convention_source=str(COMMON), matched_to_existing_REF_noise=True),
        display=dict(cmap='gray', vmin=0, vmax=vmax,
            window='Exact same all20-bin clean-image 99.5th-percentile window as clean v2',
            interpolation='nearest', smoothing=False, mask=False,
            projections='Maxima of noisy3D magnitude; not independent2D noise overlays',
            slices='True center planes from the same noisy3D magnitude'),
        checks=checks, new_acquisition=False, new_registered_metric=False,
        limitations=['Assumed image-domain iid circular Gaussian noise; not measured scanner noise.',
            'Does not model coil combination, spatially correlated/parallel-imaging noise, g-factor or anatomical tissue.',
            'SNR12 is an amplitude/per-component-noise definition, not signal divided by magnitude-background SD.',
            'MIP backgrounds are brighter due to maxima across depth; slices retain native noise.',
            'Approximate point-cloud phantom; unchanged quantitative E062 velocity and flow evidence.'])
    (stage/'REPORT.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    (stage/'README.md').write_text(
        '# Aorta 1M-spin magnitude: assumed SNR12\n\n'
        'Display-only derivative of E062 bin2; no new acquisition or velocity analysis.\n'
        f'Actual REF ADC center: {phase_ms:.6f}ms. Native voxels:3.89x4x4mm.\n\n'
        'Independent complex Gaussian noise is added before taking magnitude, in\n'
        'every voxel including the background. SNR12 means median time-mean REF\n'
        'signal in the existing fixed core15 divided by the noise SD of EACH\n'
        'complex component. This reuses aorta_common.py and seed20261025.\n'
        f'Reference amplitude:{reference_amplitude:.9g}; component SD:{sigma:.9g}.\n\n'
        'Only saved REF magnitude is available: a zero-phase complex representative\n'
        'is used. Circular noise gives the same marginal magnitude distribution\n'
        'regardless of unknown REF phase. No original complex phase is recovered.\n'
        'Zero-signal background becomes Rayleigh; nonzero signal is Rician.\n\n'
        'Top: maximum-intensity projections AFTER noise is added in3D. Therefore\n'
        'projection background is brighter than a single slice. Bottom: true\n'
        'central planes from the same noisy volume. The clean-image grayscale\n'
        'window, native grid, no smoothing and no display masking are preserved.\n\n'
        'Assumed noise, not scanner calibration. No coil-combination, g-factor,\n'
        'correlated-noise or anatomical tissue model. Clean data and prior\n'
        'quantitative velocity/flow results are unchanged.\n', encoding='utf-8')
    manifest = dict(input_sha256={str(p): h for p, h in EXPECTED.items()},
        additional_source_sha256={str(GRID): sha(GRID), str(Path(__file__).resolve()): sha(__file__)},
        outputs={p.name: dict(bytes=p.stat().st_size, sha256=sha(p)) for p in stage.iterdir() if p.is_file()})
    (stage/'MANIFEST.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    stage.rename(final)
    print(json.dumps(dict(status=report['status'], checks=len(checks), sigma=sigma,
        reference_amplitude=reference_amplitude, png=str(final/png.name), pdf=str(final/pdf.name))))


if __name__ == '__main__':
    main()
