"""E-028: source-bound static aortic maps and partial-section waveforms."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import uuid

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


COLORS = {'CFD': '#176489', 'Koma': '#bf691b'}
VIEWS = {'XY': (0, 'MRI X (mm)', 'MRI Y (mm)'),
         'XZ': (1, 'MRI X (mm)', 'MRI Z (mm)'),
         'YZ': (2, 'MRI Y (mm)', 'MRI Z (mm)')}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def axes_and_edges(shape, fov):
    """Array order Z,Y,X; physical axes and voxel edges returned X,Y,Z."""
    lengths = tuple(reversed(shape))
    axes = [(np.arange(n) - n / 2) * length / n * 1000
            for n, length in zip(lengths, fov)]
    edges = [np.r_[a - length / n * 500, a[-1] + length / n * 500]
             for a, n, length in zip(axes, lengths, fov)]
    return axes, edges


def project(values, mask, fraction, axis):
    weights = np.where(mask, fraction, 0)
    denominator = weights.sum(axis=axis)
    numerator = (np.where(mask, values, 0) * weights).sum(axis=axis)
    return np.divide(numerator, denominator, out=np.full(denominator.shape, np.nan), where=denominator > 0)


def waveform_metrics(cfd, koma, period):
    cfd, koma = np.asarray(cfd, float), np.asarray(koma, float)
    if cfd.shape != (8,) or koma.shape != (8,) or not np.isfinite([cfd, koma]).all():
        raise ValueError('Exactly eight finite paired waveform samples required')
    difference = koma - cfd
    ref_rms = np.sqrt(np.mean(cfd ** 2))
    ref_range = np.ptp(cfd)
    correlation = None if np.std(cfd) == 0 or np.std(koma) == 0 else float(np.corrcoef(cfd, koma)[0, 1])
    net = np.array([cfd.sum(), koma.sum()]) * period / 8
    return dict(bias_ml_s=float(difference.mean()), mae_ml_s=float(abs(difference).mean()),
                rmse_ml_s=float(np.sqrt(np.mean(difference ** 2))), correlation=correlation,
                nrmse_rms_pct=None if ref_rms == 0 else float(np.sqrt(np.mean(difference ** 2)) / ref_rms * 100),
                nrmse_range_pct=None if ref_range == 0 else float(np.sqrt(np.mean(difference ** 2)) / ref_range * 100),
                CFD_net_cycle_ml=float(net[0]), Koma_net_cycle_ml=float(net[1]),
                cycle_difference_pct=None if net[0] == 0 else float((net[1] / net[0] - 1) * 100),
                CFD_abs_peak_ml_s=float(abs(cfd).max()), Koma_abs_peak_ml_s=float(abs(koma).max()),
                CFD_abs_peak_bin=int(np.argmax(abs(cfd)) + 1), Koma_abs_peak_bin=int(np.argmax(abs(koma)) + 1))


def save_pair(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ('png', 'pdf'):
        fig.savefig(path.with_suffix('.' + suffix), dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_map(ax, data, extent, title, limits, signed=False, difference=False):
    cmap = plt.get_cmap('RdBu_r' if difference else 'gray' if signed else 'viridis').copy()
    cmap.set_bad('#eeeeee')
    image = ax.imshow(data, origin='lower', extent=extent, interpolation='nearest',
                      cmap=cmap, vmin=limits[0], vmax=limits[1], aspect='equal')
    ax.set_title(title, fontsize=10)
    ax.set_facecolor('#eeeeee')
    return image


def triptych(values, extent, title, labels, limits, difference_limit, path, transverse=False, signed=False):
    fig, axes = plt.subplots(1 if transverse else 3, 3 if transverse else 1,
                             figsize=(13, 4.8) if transverse else (11, 12), layout='constrained')
    for index, (ax, data, label) in enumerate(zip(axes.ravel(), values, ('CFD', 'Koma', 'Koma minus CFD'))):
        bound = (-difference_limit, difference_limit) if index == 2 else limits
        image = plot_map(ax, data, extent, label, bound, signed, index == 2)
        ax.set(xlabel=labels[0], ylabel=labels[1])
        fig.colorbar(image, ax=ax, label='m/s', shrink=.85)
    fig.suptitle(title, fontsize=13)
    fig.supxlabel('Fixed 15% core; numerical support only. Gray = excluded. Native pixels; no spatial smoothing.', fontsize=9)
    save_pair(fig, path)


def flow_rows(report, section, method):
    values = sorted([x for x in report['section_series'] if x['section'] == section and x['method'] == method], key=lambda x: x['bin'])
    if [x['bin'] for x in values] != list(range(1, 9)):
        raise ValueError('Missing or duplicate section bins')
    return values


def render(results, output):
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((results / 'MANIFEST.json').read_text(encoding='utf-8'))
    for name in ('report.json', 'matched_reference.h5'):
        if digest(results / name) != manifest[name]['sha256']:
            raise ValueError('Source hash mismatch: ' + name)
    report = json.loads((results / 'report.json').read_text(encoding='utf-8'))
    if report['experiment'] != 'E-027':
        raise ValueError('Expected E-027')
    with h5py.File(results / 'matched_reference.h5', 'r') as f:
        ref, koma = f['adc_time_average'][...], f['Koma'][...]
        mask, fraction = f['masks/core15'][...].astype(bool), f['fraction'][...]
        plane_fraction = f['plane_fraction'][...]
        sections = {key: f['sections'][key][...].astype(bool) for key in f['sections']}
    if ref.shape != (8, *mask.shape, 3) or ref.shape != koma.shape:
        raise ValueError('Unexpected field shape')
    if not np.isfinite(ref[:, mask]).all() or not np.isfinite(koma[:, mask]).all():
        raise ValueError('Invalid paired velocities')
    temporary = output.with_name(output.name + '.building-' + uuid.uuid4().hex[:8])
    temporary.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    coords, edges = axes_and_edges(mask.shape, report['grid']['fov'])
    speed_ref, speed_koma = np.linalg.norm(ref, axis=-1), np.linalg.norm(koma, axis=-1)
    speed_max = float(max(speed_ref[:, mask].max(), speed_koma[:, mask].max()))
    difference_max = max(float(abs(speed_koma[:, mask] - speed_ref[:, mask]).max()), 1e-6)
    component_max = [float(max(abs(ref[:, mask, c]).max(), abs(koma[:, mask, c]).max())) for c in range(3)]
    component_error = [max(float(abs(koma[:, mask, c] - ref[:, mask, c]).max()), 1e-6) for c in range(3)]
    projected = {}
    for view, (axis, xlabel, ylabel) in VIEWS.items():
        horizontal, vertical = (0, 1) if view == 'XY' else (0, 2) if view == 'XZ' else (1, 2)
        extent = [edges[horizontal][0], edges[horizontal][-1], edges[vertical][0], edges[vertical][-1]]
        projected[view] = []
        for b in range(8):
            a, k = [project(field[b], mask, fraction, axis) for field in (speed_ref, speed_koma)]
            projected[view].append((a, k))
            triptych([a, k, k-a], extent, f'Aorta speed | {view} support-weighted projection | bin {b+1}/8',
                     (xlabel, ylabel), (0, speed_max), difference_max,
                     temporary / 'speed_projections' / f'aorta_speed_{view}_bin{b+1:02}', transverse=view == 'YZ')
            print(f'Projection {view}, bin {b+1}/8', flush=True)
        fig, axes = plt.subplots(4, 4, figsize=(18, 14 if view != 'YZ' else 19), layout='constrained')
        for b, (a, k) in enumerate(projected[view]):
            row, col = divmod(b, 2)
            for method_index, data in enumerate((a, k)):
                ax = axes[row, col*2 + method_index]
                image = plot_map(ax, data, extent, f'{("CFD", "Koma")[method_index]} | bin {b+1}', (0, speed_max))
                ax.set(xlabel=xlabel, ylabel=ylabel)
        fig.colorbar(image, ax=axes.ravel().tolist(), label='Speed (m/s)', shrink=.6)
        fig.suptitle(f'Aorta | {view} support-weighted speed projections | all eight discrete bins')
        fig.supxlabel('Common all-bin scale; gray = excluded numerical support; no interpolated cardiac phases.', fontsize=10)
        save_pair(fig, temporary / 'all_bin_overviews' / f'aorta_speed_{view}_all_bins')
    xy_extent = [edges[0][0], edges[0][-1], edges[1][0], edges[1][-1]]
    for b in range(8):
        fig, axes = plt.subplots(3, 3, figsize=(16, 10), layout='constrained')
        for c in range(3):
            a, k = [project(field[b, ..., c], mask, fraction, 0) for field in (ref, koma)]
            for j, data in enumerate((a, k, k-a)):
                bound = component_error[c] if j == 2 else component_max[c]
                image = plot_map(axes[c, j], data, xy_extent,
                                 f'{("CFD", "Koma", "Koma minus CFD")[j]} | V{"XYZ"[c]}', (-bound, bound), signed=True, difference=j == 2)
                axes[c, j].set(xlabel='MRI X (mm)', ylabel='MRI Y (mm)')
                fig.colorbar(image, ax=axes[c, j], label='m/s', shrink=.75)
        fig.suptitle(f'Aorta signed component velocities | XY support-weighted projections | bin {b+1}/8')
        fig.supxlabel('Velocity: black negative, gray zero, white positive. Excluded pixels light gray. All-bin scales; differences use separate scales.', fontsize=9)
        save_pair(fig, temporary / 'velocity_components' / f'aorta_components_bin{b+1:02}')
    for quantity in ('speed', 'normal_velocity'):
        for b in range(8):
            fig, axes = plt.subplots(4, 3, figsize=(13, 15), layout='constrained')
            for row, section in enumerate(report['section_summary']):
                name = section['section']; i = int(np.argmin(abs(coords[0] - section['x_m'] * 1000)))
                roi = sections[name]; paired = roi & mask[:, :, i] & (plane_fraction[:, :, i] > 0)
                location = np.argwhere(roi)
                lo = np.maximum(location.min(axis=0) - 2, 0)
                hi = np.minimum(location.max(axis=0) + 3, roi.shape)
                selection = (slice(lo[0], hi[0]), slice(lo[1], hi[1]))
                extent = [edges[1][lo[1]], edges[1][hi[1]], edges[2][lo[0]], edges[2][hi[0]]]
                fields = (speed_ref, speed_koma) if quantity == 'speed' else (ref[..., 0], koma[..., 0])
                a, k = [np.where(paired, field[b, :, :, i], np.nan)[selection] for field in fields]
                for j, data in enumerate((a, k, k-a)):
                    bound = (difference_max if quantity == 'speed' else component_error[0]) if j == 2 else (speed_max if quantity == 'speed' else component_max[0])
                    limits = (-bound, bound) if j == 2 or quantity == 'normal_velocity' else (0, bound)
                    image = plot_map(axes[row, j], data, extent,
                        f'{name} | {("CFD", "Koma", "Koma minus CFD")[j]}', limits,
                        signed=quantity == 'normal_velocity', difference=j == 2)
                    axes[row, j].set(xlabel='MRI Y (mm)', ylabel='MRI Z (mm)')
                    fig.colorbar(image, ax=axes[row, j], label='m/s', shrink=.8)
            fig.suptitle(f'Aorta section {quantity.replace("_", " ")} | bin {b+1}/8 | +MRI-X normal')
            fig.supxlabel('Partial fixed-core sections; gray excluded. Native pixels, fixed crops/scales; not physical wall/cap geometry.', fontsize=9)
            save_pair(fig, temporary / 'cross_sections' / f'aorta_sections_{quantity}_bin{b+1:02}')
    for b in range(8):
        fig, axes = plt.subplots(4, 2, figsize=(12, 14), layout='constrained')
        for row, section in enumerate(report['section_summary']):
            for column, transverse in enumerate(('phase_Y', 'slice_Z')):
                ax = axes[row, column]
                for method in ('CFD', 'Koma'):
                    data = [x for x in report['profile_series'] if x['section'] == section['section'] and x['bin'] == b+1 and x['method'] == method and x['transverse_axis'] == transverse]
                    ax.plot([x['coordinate_mm'] for x in data], [x['normal_velocity_m_s'] for x in data],
                            '-o' if method == 'CFD' else '--s', color=COLORS[method], ms=3, label=method)
                ax.set(title=section['section'], xlabel=f'MRI {"Y" if column == 0 else "Z"} (mm)', ylabel='+X normal velocity (m/s)')
                ax.set_ylim(-component_max[0]*1.05, component_max[0]*1.05)
                ax.axhline(0, color='.5', lw=.7); ax.legend(frameon=False); ax.grid(alpha=.2)
        fig.suptitle(f'Aorta partial-section normal-velocity profiles | bin {b+1}/8')
        fig.supxlabel('Null profile samples remain gaps. Common all-bin velocity scale; no profile interpolation.', fontsize=9)
        save_pair(fig, temporary / 'velocity_profiles' / f'aorta_profiles_bin{b+1:02}')
    times = np.asarray(report['time_s']) * 1000; period = report['period_s']; dt = period / 8
    summary = []
    overview, overview_axes = plt.subplots(2, 2, figsize=(13, 9), layout='constrained')
    agreement, agreement_axes = plt.subplots(2, 2, figsize=(11, 10), layout='constrained')
    for index, section in enumerate(report['section_summary']):
        name = section['section']; series = {m: flow_rows(report, name, m) for m in ('CFD', 'Koma')}
        q = {m: np.array([x['net_ml_s'] for x in values]) for m, values in series.items()}
        metrics = waveform_metrics(q['CFD'], q['Koma'], period)
        for measured, saved in [('rmse_ml_s', section['flow_error']['rmse'] * 1e6), ('correlation', section['flow_error']['pearson_r']), ('CFD_net_cycle_ml', section['CFD_cycle']['net_m3'] * 1e6), ('Koma_net_cycle_ml', section['Koma_cycle']['net_m3'] * 1e6)]:
            np.testing.assert_allclose(metrics[measured], saved, rtol=2e-6, atol=1e-10)
        metrics.update(section=name, x_mm=section['x_m'] * 1000, paired_pixels=section['paired_pixels'],
                       numerical_area_retained_pct=100 * series['CFD'][0]['sampled_area_mm2'] / series['CFD'][0]['numerical_section_area_mm2'])
        summary.append(metrics)
        fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout='constrained')
        for method in ('CFD', 'Koma'):
            style = '-o' if method == 'CFD' else '--s'
            axes[0, 0].plot(times, q[method], style, color=COLORS[method], label=method)
            overview_axes.ravel()[index].plot(times, q[method], style, color=COLORS[method], label=method)
            for field, linestyle, marker, direction in [('forward_ml_s', '-', 'o', '+X'), ('reverse_ml_s', '--', 's', '-X')]:
                axes[0, 1].plot(times, [x[field] for x in series[method]], color=COLORS[method], linestyle=linestyle,
                               marker=marker, ms=4, label=f'{method} {direction}')
            axes[1, 0].step(np.arange(9)*dt*1000, np.r_[0, np.cumsum(q[method]*dt)], where='post',
                            color=COLORS[method], linestyle='-' if method == 'CFD' else '--', label=method)
        axes[1, 1].plot(times, q['Koma']-q['CFD'], '-o', color='#653c83', label='Koma minus CFD')
        for ax, title, unit in zip(axes.ravel(), ['Signed section flow', 'Simultaneous directional transport magnitudes', 'Cumulative signed bin-integrated volume', 'Paired flow residual'], ['mL/s', 'mL/s', 'mL', 'mL/s']):
            ax.set(title=title, xlabel='Nominal MRI cycle time (ms)', ylabel=unit, xlim=(0, period*1000))
            ax.axhline(0, color='.5', lw=.7); ax.grid(alpha=.2); ax.legend(frameon=False, fontsize=9)
        fig.suptitle(f'{name} | r={metrics["correlation"]:.4f} | RMSE={metrics["rmse_ml_s"]:.2f} mL/s | cycle difference={metrics["cycle_difference_pct"]:+.2f}%')
        fig.supxlabel('Eight original bins; CFD uses actual ADC times. Partial core support; -X is a spatial direction, not clinical regurgitation.', fontsize=9)
        save_pair(fig, temporary / 'flow_waveforms' / f'aorta_flow_detail_{name}')
        ax = overview_axes.ravel()[index]
        ax.set(title=f'{name} | r={metrics["correlation"]:.4f}; RMSE={metrics["rmse_ml_s"]:.2f} mL/s',
               xlabel='Nominal MRI cycle time (ms)', ylabel='Signed +X flow (mL/s)', xlim=(0, period*1000))
        ax.axhline(0, color='.5', lw=.7); ax.grid(alpha=.2); ax.legend(frameon=False)
        ax = agreement_axes.ravel()[index]
        limits = [min(q['CFD'].min(), q['Koma'].min(), 0), max(q['CFD'].max(), q['Koma'].max(), 0)]
        padding = max(np.ptp(limits)*.08, 1); limits = (limits[0]-padding, limits[1]+padding)
        ax.plot(limits, limits, '--', color='.45', label='Equality')
        ax.scatter(q['CFD'], q['Koma'], color=COLORS['CFD'], s=30)
        # Separate the recurring cluster of bins 2/4/5 without moving data.
        label_offsets = {2: (-20, -16), 4: (-22, 13), 5: (13, 12)}
        for b, (a, k) in enumerate(zip(q['CFD'], q['Koma'])):
            ax.annotate(str(b+1), (a, k), xytext=label_offsets.get(b+1, (7, 7)),
                        textcoords='offset points', fontsize=9,
                        bbox=dict(facecolor='white', edgecolor='none', alpha=.9, pad=.25),
                        arrowprops=dict(arrowstyle='-', color='.5', lw=.6))
        ax.set(title=f'{name} | labels = cardiac bins', xlabel='CFD flow (mL/s)', ylabel='Koma flow (mL/s)', xlim=limits, ylim=limits)
        ax.set_aspect('equal'); ax.grid(alpha=.2); ax.legend(frameon=False)
    overview.suptitle('Aorta CFD versus Koma | four separate partial-section waveforms')
    overview.supxlabel('Markers are eight original bins; straight lines are visual guides. CFD is ADC-time averaged; no fitted shift or amplitude.', fontsize=9)
    save_pair(overview, temporary / 'flow_waveforms' / 'aorta_flow_waveforms_all_sections')
    agreement.suptitle('Paired flow agreement | per-section eight-bin samples, not independent measurements')
    save_pair(agreement, temporary / 'flow_waveforms' / 'aorta_flow_agreement')
    with (temporary / 'flow_waveform_comparison.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    (temporary / 'flow_waveform_comparison.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    (temporary / 'README.md').write_text(
        '# E-028: expanded aortic contour maps and flow waveforms\n\n'
        'Presentation derivative of immutable E-027 results_v2; original 1M-spin/eight-bin Koma versus original CFD.\n\n'
        '- `speed_projections/`: CFD, Koma and signed differences in XY/XZ/YZ, every bin. These are weighted projections, not single slices.\n'
        '- `all_bin_overviews/`: all eight bins, matched scales.\n'
        '- `velocity_components/`: signed grayscale X/Y/Z velocities and signed residuals, every bin.\n'
        '- `cross_sections/`: true MRI YZ slices at the four separate +X numerical sections; speed and normal velocity, every bin.\n'
        '- `velocity_profiles/`: both transverse directions, all four sections, every bin.\n'
        '- `flow_waveforms/`: overview, paired agreement and four detail sheets (net, directional, cumulative and residual).\n'
        '- `flow_waveform_comparison.csv` / `.json`: per-section metrics and numerical area retained.\n\n'
        'All figures are PNG/PDF pairs. Scales are fixed across all bins; difference scales are separate. Full retained ranges are shown without clipping. '
        'Gray excluded pixels are unavailable, not zero or repaired. Map pixels remain native resolution. No spatial smoothing or new cardiac phases. '
        'Section crops are fixed from numerical ROIs, not velocity-selected. Projections may combine overlapping vessel segments.\n\n'
        'Waveform markers use nominal bin centers; the saved CFD field averages actual ADC sample times. Lines are straight guides, not new temporal information. '
        'Cumulative volume is a bin-sum estimate displayed as steps at bin boundaries, not a continuous measured waveform. '
        'The +X normal is spatial: negative flow is not clinical regurgitation. No labeled inlet/outlet or branch assignment is inferred. '
        'These are partial fixed-core/numerical-support integrals, not full-lumen flow, wall validation or independent experimental accuracy.\n', encoding='utf-8')
    files = {p.relative_to(temporary).as_posix(): {'bytes': p.stat().st_size, 'sha256': digest(p)}
             for p in sorted(temporary.rglob('*')) if p.is_file()}
    metadata = dict(experiment='E-028', evidence_class='Presentation and internal arithmetic derivative of E-027',
        created_utc=datetime.now(timezone.utc).isoformat(), status='COMPLETE',
        source_report_sha256=digest(results/'report.json'), source_h5_sha256=digest(results/'matched_reference.h5'),
        script_sha256=digest(__file__), bins=8, mask='fixed all-bin core15',
        coordinates_mm=[x.tolist() for x in coords], speed_limits=[0, speed_max], speed_difference_limits=[-difference_max, difference_max],
        component_symmetric_limits=component_max, component_difference_symmetric_limits=component_error,
        png_count=sum(name.endswith('.png') for name in files), pdf_count=sum(name.endswith('.pdf') for name in files),
        section_arithmetic_comparisons=16, files=files)
    (temporary/'MANIFEST.json').write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding='utf-8')
    temporary.rename(output)
    print(json.dumps({'output': str(output), 'png': metadata['png_count'], 'pdf': metadata['pdf_count'],
                      'flow_summary': summary}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    render(args.results, args.output)


if __name__ == '__main__':
    main()
