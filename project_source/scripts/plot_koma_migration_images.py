"""Fixed true-plane grayscale E046 sheets; no normalization per image or phase filling."""
import argparse
import json
from pathlib import Path
import tomllib
import numpy as np
from scripts.analyze_koma_migration import ARMS, LABEL, ROOT, load_arm, image_quantities
from scripts.build_aorta_inlet_picker import sha


def plot(analysis, case_dir, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if output.exists():
        raise FileExistsError(output)
    analysis = analysis.resolve(); case_dir = case_dir.resolve()
    report = json.loads((analysis/'report.json').read_text())
    info = tomllib.loads((case_dir/'case_info.toml').read_text())
    case = report['case']; quantities = {}; paths = [analysis/'report.json', case_dir/'case_info.toml']
    for arm in ARMS:
        images, _ = load_arm(case, arm, case_dir, paths)
        quantities[arm] = image_quantities(images, info['Venc_m_s'])
    with np.load(analysis/'support_evidence.npz') as f:
        joint = f['joint']
    shape = quantities['legacy'][0].shape
    plane = shape[2]//2; chosen_bins = [0, shape[0]-1]
    fov_x, fov_z = (.28, .128) if case == 'aorta' else (.5, .08)
    extent = [-fov_x*500, fov_x*500, -fov_z*500, fov_z*500]
    output.mkdir(parents=True)
    phase_limit = max(float(np.max(np.abs(quantities[a][1][chosen_bins, :, plane, :, 0][:, joint[:, plane, :]])))
                      for a in ARMS)*np.pi/info['Venc_m_s']
    phase_limit = max(phase_limit, 1e-12)
    for kind in ['magnitude', 'phase_RO']:
        fig, axes = plt.subplots(2, 3, figsize=(12, 5.5), sharex=True, sharey=True)
        maximum = max(float(quantities[a][0][chosen_bins, :, plane, :].max()) for a in ARMS)
        cmap = plt.get_cmap('gray').copy(); cmap.set_bad('#aab0b8')
        for row, b in enumerate(chosen_bins):
            for col, arm in enumerate(ARMS):
                if kind == 'magnitude':
                    values = quantities[arm][0][b, :, plane, :]
                    limits = [0, maximum]
                else:
                    values = quantities[arm][1][b, :, plane, :, 0]*np.pi/info['Venc_m_s']
                    values = np.where(joint[:, plane, :], values, np.nan)
                    limits = [-phase_limit, phase_limit]
                im = axes[row, col].imshow(values, origin='lower', extent=extent, cmap=cmap,
                    vmin=limits[0], vmax=limits[1], interpolation='nearest', aspect='equal')
                axes[row, col].set_title(f'{LABEL[arm]} | bin {b+1}', fontsize=10)
                if col == 0:
                    axes[row, col].set_ylabel('MRI slice axis (mm)')
                if row == 1:
                    axes[row, col].set_xlabel('MRI readout axis (mm)')
        fig.suptitle(f'{case.capitalize()} | {kind.replace("_", " ")} | identical true plane', x=.07, ha='left')
        fig.subplots_adjust(left=.07, right=.88, bottom=.18, top=.87, wspace=.13, hspace=.3)
        cax = fig.add_axes([.91, .27, .015, .48])
        bar = fig.colorbar(im, cax=cax)
        bar.set_label('Magnitude (a.u.)' if kind == 'magnitude' else 'Wrapped RO phase (rad)')
        fig.text(.07, .035, f'20k physical spins; MRI phase-axis index {plane} (zero based). Same absolute scale for all panels.\n'
                 + ('All raw magnitudes shown; no image-wise gain fitting.' if kind == 'magnitude'
                    else f'Gray = unavailable; no phase filling or simulated wall. Shared phase range ±{phase_limit:.3g} rad; no clipping.'),
                 fontsize=8, color='#555c65')
        for ext in ['png', 'pdf']:
            fig.savefig(output/f'{case}_{kind}_first_last.{ext}', dpi=400)
        plt.close(fig)
    (output/'registration.json').write_text(json.dumps(dict(code_sha256=sha(__file__),
        analysis_report_sha256=sha(analysis/'report.json'), true_plane_zero_based=plane,
        selected_bins_one_based=[x+1 for x in chosen_bins], fitted_gain=False, phase_imputation=False,
        shared_phase_limit_rad=phase_limit, phase_display_not_a_wrapping_test=True,
        inputs={str(p.relative_to(ROOT)): sha(p) for p in paths}), indent=2)+'\n', encoding='utf-8')
    (output/'MANIFEST.json').write_text(json.dumps({p.name: sha(p) for p in output.iterdir()}, indent=2)+'\n', encoding='utf-8')
    print('PASS_FIXED_PLANE_IMAGE_EXPORT', case)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis', type=Path, required=True); p.add_argument('--case-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); plot(a.analysis, a.case_dir, a.output)
