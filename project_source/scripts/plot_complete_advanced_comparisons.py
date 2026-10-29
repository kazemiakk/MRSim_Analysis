"""Static scientific presentation derivative of E-038, never altered fields."""
import argparse
import json
from pathlib import Path
import tempfile

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scripts.extend_hemodynamic_comparisons import bind, sha
from scripts.compare_aorta_hemodynamics import dump


COLORS = {'CFD': '#176489', 'Koma': '#bf691b'}


def save(fig, path):
    fig.savefig(path.with_suffix('.png'), dpi=180)
    fig.savefig(path.with_suffix('.pdf'))
    plt.close(fig)


def plot_all(run, target):
    if target.exists():
        raise FileExistsError(target)
    manifest = json.loads((run/'MANIFEST.json').read_text())
    for name in ['report.json', 'advanced_fields.h5']:
        if sha(run/name) != manifest[name]['sha256']:
            raise ValueError('Unbound numerical inputs')
    r = json.loads((run/'report.json').read_text())
    target.parent.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix=target.name+'.partial-', dir=target.parent))
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    presentation = dict(inputs=[bind(run/'report.json'), bind(run/'advanced_fields.h5'), bind(Path(__file__))], cases={})
    with h5py.File(run/'advanced_fields.h5') as f:
        for case, result in r['cases'].items():
            times = result['times_s']; gradients = result['gradients']
            fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
            observables = [('vorticity_magnitude_s_inv', 'Mean vorticity magnitude', '1/s'),
                           ('helicity_m_s_inv2', 'Mean signed helicity', 'm/s²'),
                           ('strain_frobenius_s_inv', 'Mean strain-rate norm', '1/s')]
            for ax, (metric, title, unit) in zip(axes.ravel(), observables):
                rows = [x for x in gradients['per_bin_errors'] if x['metric'] == metric]
                for method, key in [('CFD', 'reference_mean'), ('Koma', 'estimate_mean')]:
                    ax.plot(times, [row[key] for row in rows], '-o' if method == 'CFD' else '--s',
                            color=COLORS[method], label=method, ms=4)
                ax.set(title=title, xlabel='Nominal MRI bin time (s)', ylabel=unit)
            for method in ['CFD', 'Koma']:
                rows = [x for x in gradients['region_integrals'] if x['region'] == 'whole' and x['method'] == method]
                physical = result['viscosity_pa_s'] is not None
                y = [row['resolved_dissipation_w']*1000 if physical else row['dissipation_per_viscosity_m3_s2'] for row in rows]
                axes[1, 1].plot(times, y, '-o' if method == 'CFD' else '--s', color=COLORS[method], label=method, ms=4)
            axes[1, 1].set(title='Resolved interior molecular dissipation', xlabel='Nominal MRI bin time (s)',
                           ylabel='mW' if result['viscosity_pa_s'] is not None else 'Dissipation / viscosity (m³/s²)')
            for ax in axes.ravel():
                ax.grid(alpha=.18); ax.legend(frameon=False)
                if case.startswith('phantom'):
                    ax.axvspan(.64/.72*result['period_s'], result['period_s'], color='#dddddd', alpha=.5)
            fig.suptitle(case.replace('_', ' ')+' | CFD versus Koma')
            note = 'Fixed interior stencils; discrete saved bins; not wall shear or turbulent loss.'
            if case.startswith('phantom'):
                note += '\nGray time band: synthetic CFD-input closure; bins14/15 overlap this interval.'
            fig.supxlabel(note, fontsize=10)
            save(fig, out/(case+'_advanced_waveforms'))
            group = f[case]; indices = group['flat_indices'][:]
            shape = tuple(result['shape']); spacing = group.attrs['spacing_xyz_m']
            mask = np.zeros(np.prod(shape), bool); mask[indices] = True; mask = mask.reshape(shape)
            phase_index = int(np.argmax(mask.sum(axis=(0, 2))))
            # Select a representative acquisition bin by CFD mean vorticity,
            # never by smallest reconstruction error or a tuned mask.
            rows = [x for x in gradients['per_bin_errors'] if x['metric'] == 'vorticity_magnitude_s_inv']
            b = int(np.argmax([x['reference_mean'] for x in rows]))
            axes_xyz = [(np.arange(n)+(.5 if case.startswith('phantom') else 0)-n/2)*d*1000
                        for n, d in zip(shape[::-1], spacing)]
            extent = [axes_xyz[0][0]-spacing[0]*500, axes_xyz[0][-1]+spacing[0]*500,
                      axes_xyz[2][0]-spacing[2]*500, axes_xyz[2][-1]+spacing[2]*500]
            fig, axs = plt.subplots(3, 3, figsize=(15, 10), layout='constrained')
            for row_index, (metric, label, unit, signed) in enumerate([
                ('vorticity_magnitude_s_inv', 'Vorticity magnitude', '1/s', False),
                ('helicity_m_s_inv2', 'Signed helicity', 'm/s²', True),
                ('dissipation_per_viscosity_s_inv2', 'Resolved dissipation / viscosity', '1/s²', False)]):
                maps = []
                for method in ['CFD', 'Koma']:
                    values = np.full(np.prod(shape), np.nan); values[indices] = group[method+'/'+metric][b]
                    maps.append(values.reshape(shape)[:, phase_index, :])
                difference = maps[1]-maps[0]
                finite = np.concatenate([x[np.isfinite(x)] for x in maps])
                bound = max(float(abs(finite).max()), 1e-15)
                delta = max(float(np.nanmax(abs(difference))), 1e-15)
                for col, (values, title) in enumerate(zip([*maps, difference], ['CFD', 'Koma', 'Koma − CFD'])):
                    limits = (-delta, delta) if col == 2 else (-bound, bound) if signed else (0, bound)
                    cmap = plt.get_cmap('RdBu_r' if col == 2 or signed else 'viridis').copy()
                    cmap.set_bad('#eeeeee')
                    im = axs[row_index, col].imshow(values, origin='lower', extent=extent, aspect='equal',
                        interpolation='nearest', vmin=limits[0], vmax=limits[1], cmap=cmap)
                    axs[row_index, col].set(title=label+'\n'+title, xlabel='MRI readout X (mm)', ylabel='MRI slice Z (mm)')
                    fig.colorbar(im, ax=axs[row_index, col], label=unit, shrink=.8)
            fig.suptitle(f"{case.replace('_', ' ')} | bin {b+1} | fixed phase index {phase_index}")
            fig.supxlabel('True XZ slice; gray = unavailable interior stencil; shared CFD/Koma scale; no smoothing or clipping.', fontsize=10)
            save(fig, out/(case+'_advanced_maps'))
            presentation['cases'][case] = dict(map_bin=b+1, phase_index=phase_index,
                selection='CFD maximum mean-vorticity bin; phase plane with most fixed stencil pixels',
                map_view='XZ true slice', bins_in_numeric_fields=len(times))
    dump(out/'presentation.json', presentation)
    dump(out/'MANIFEST.json', {p.name: dict(bytes=p.stat().st_size, sha256=sha(p)) for p in out.iterdir() if p.is_file()})
    if target.exists():
        raise FileExistsError(target)
    out.rename(target)
    print(str(target))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); a = p.parse_args()
    plot_all(a.run, a.output)
