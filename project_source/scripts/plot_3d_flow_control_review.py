"""E051 presentation-only layout repair; preserve the validated analysis_v1."""
import argparse
import json
from pathlib import Path
import uuid

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import FixedLocator

from scripts.prepare_3d_flow_control import ROOT, BASE, SHAPE, dump
from scripts.analyze_remapped_confirmation import fixed_mask
from scripts.build_aorta_inlet_picker import sha


def plot(output):
    if output.exists():
        raise FileExistsError(output)
    report_path = BASE/'analysis_v1/report.json'
    validation = json.loads((BASE/'analysis_v1/validation.json').read_text())
    if validation['report_sha256'] != sha(report_path) or not validation['scientific_gates_pass']:
        raise ValueError('Validated scientific result required')
    stage = output.with_name(output.name+'.building-'+uuid.uuid4().hex[:8]); stage.mkdir(parents=True)
    sources = [Path(__file__).resolve(), report_path, BASE/'analysis_v1/validation.json', BASE/'inputs_v1/controls.h5',
               ROOT/'scripts/analyze_remapped_confirmation.py']
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    models = ('static_landmarks', 'plug_positive', 'plug_negative')
    data = {}
    with h5py.File(BASE/'inputs_v1/controls.h5') as f:
        for model in models:
            images = {}
            for e in ('REF', 'X', 'Y', 'Z'):
                path = BASE/'acquisition_v2'/(model+'_'+e.lower()+'.h5'); sources.append(path)
                with h5py.File(path) as raw:
                    images[e] = raw['image'][:]
            mask = fixed_mask(images, .05)
            velocity = np.angle(images['X'][0]*images['REF'][0].conj())*(5/np.pi)
            points = f[model+'/positions'][:]; indices = f[model+'/indices'][:]
            geometry = np.zeros(SHAPE[1:], bool); geometry[tuple(indices.T)] = True
            data[model] = dict(points=points*1000, indices=indices,
                velocity=np.ma.array(velocity, mask=~(mask&geometry)),
                values=np.where(mask[tuple(indices.T)], velocity[tuple(indices.T)], np.nan))
    norm = Normalize(-1.6, 1.6)
    cmap = plt.get_cmap('coolwarm').copy(); cmap.set_bad('#eeeeee')
    fig = plt.figure(figsize=(17, 6))
    grid = fig.add_gridspec(1, 3, left=.025, right=.86, bottom=.15, top=.82, wspace=.34)
    titles = ('Static landmarks', '+1.50 m/s plug flow', '−0.75 m/s plug flow')
    for i, (model, title) in enumerate(zip(models, titles)):
        ax = fig.add_subplot(grid[0, i], projection='3d')
        points = data[model]['points']; values = data[model]['values']
        stride = 1 if model == 'static_landmarks' else 16
        points, values = points[::stride], values[::stride]
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=values, cmap=cmap, norm=norm,
                   s=65 if model == 'static_landmarks' else 4, alpha=1.,
                   edgecolors='#39404a' if model == 'static_landmarks' else 'none', linewidths=.6)
        ax.set(title=title, xlabel='MRI x (mm)', ylabel='MRI y (mm)', zlabel='MRI z (mm)')
        ax.xaxis.labelpad=10; ax.yaxis.labelpad=10; ax.zaxis.labelpad=8
        ax.tick_params(labelsize=9, pad=1)
        if model != 'static_landmarks':
            ax.set_box_aspect((4.2, 1.7, 1.7))
            ax.xaxis.set_major_locator(FixedLocator([-250, 0, 250]))
            ax.yaxis.set_major_locator(FixedLocator([-15, -10, -5]))
            ax.zaxis.set_major_locator(FixedLocator([-10, -5, 0, 5]))
        else:
            ax.set_box_aspect((1.4, 1., 1.))
            ax.xaxis.set_major_locator(FixedLocator([-100, 0, 100]))
            ax.yaxis.set_major_locator(FixedLocator([-15, 0, 15]))
            ax.zaxis.set_major_locator(FixedLocator([-10, 0, 10]))
        ax.view_init(elev=24, azim=-60)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor((1, 1, 1, 0)); axis._axinfo['grid']['color'] = (.6, .6, .6, .25)
    bar = fig.add_axes([.925, .23, .012, .49])
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=bar, label='Reconstructed axial velocity (m/s)')
    fig.suptitle('3D control geometry | reconstructed axial velocity, cardiac bin 1', y=.96, fontsize=14)
    fig.text(.04, .045, 'Prescribed sample coordinates; flow samples displayed at 1-in-16 stride. Axis aspect adjusted for visibility. No wall surface is inferred.', color='.35', fontsize=9)
    fig.savefig(stage/'known_flow_3d.png', dpi=300); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), gridspec_kw={'width_ratios':[1, 4]}, layout='constrained')
    for row, model in enumerate(models[1:]):
        volume = data[model]['velocity']
        for col, (plane, extent, labels) in enumerate([
            (volume[:, :, 200], [-40.625, 39.375, -41.25, 38.75], ('MRI y (mm)', 'MRI z (mm)')),
            (volume[14], [-250.625, 249.375, -40.625, 39.375], ('MRI x (mm)', 'MRI y (mm)'))]):
            ax = axes[row, col]
            ax.imshow(plane, cmap=cmap, norm=norm, origin='lower', extent=extent, interpolation='nearest', aspect='equal')
            ax.set(xlabel=labels[0], ylabel=labels[1],
                title=('Positive' if row == 0 else 'Negative')+' flow | '+('YZ slice, x = 0 mm' if col == 0 else 'XY slice, z = −5 mm'))
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, fraction=.025, pad=.02, label='Axial velocity (m/s)')
    fig.suptitle('Known-flow 2D slices | cardiac bin 1; prescribed geometry and fixed 5% support', fontsize=13)
    fig.savefig(stage/'known_flow_slices.png', dpi=300); plt.close(fig)
    dump(stage/'MANIFEST.json', dict(evidence_class='Presentation derivative; scientific report/arrays unchanged',
        inputs={str(p):sha(p) for p in sources}, outputs={p.name:sha(p) for p in stage.iterdir() if p.is_file()},
        masking='Prescribed geometry AND original fixed all-bin/all-encoding5% mask; unavailable values remain gray',
        view='Unchanged physical MRI coordinates; 3D display aspect/stride stated; no image interpolation'))
    stage.rename(output)
    print(str(output), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    plot(parser.parse_args().output)
