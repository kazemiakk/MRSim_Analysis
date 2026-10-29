"""Render 3-case comparison videos and summary figures for Virtual Contrast Angiography."""
from pathlib import Path
import argparse, csv, json, os, sys, subprocess, time
import numpy as np, pyvista as pv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.colors import ListedColormap, PowerNorm

sys.path.insert(0, r'D:\FluentAgent\automation\20261029-extended-hemodynamics-v1')
from extended_common import PROJECT, KINDS, load, save, sha, CENTER_MM, RADII_MM

GRID = PROJECT / 'visualizations/20261002-measured-ica-velocity-context-v6/original-fluid-tetrahedra-mm.vtu'
FFMPEG = Path(r'D:\ANSYS Inc\v261\tp\ffmpeg\8.0.0\winx64\ffmpeg.exe')

CASE_DIRS = {
    'patient_specific': PROJECT / 'results/20261029-virtual-angiography-connected-250um-v1',
    'individualized': PROJECT / 'results/20261029-virtual-angiography-connected-250um-individualized-v1',
    'generic': PROJECT / 'results/20261029-virtual-angiography-connected-250um-generic-v1',
}
CASE_TITLES = {
    'patient_specific': 'Patient-specific MRI',
    'individualized': 'Individualized generic',
    'generic': 'Population average (generic)',
}
CASE_COLORS = {
    'patient_specific': '#1d4ed8',
    'individualized': '#d97706',
    'generic': '#059669',
}


def verify_and_summarize_case(out):
    result = load(out / 'result.json')
    assert result['status'] == 'available_first_cycle_transport_complete' and result['error'] is None, f"Failed: {out}"
    frames = load(out / 'frames.json')
    rows = list(csv.DictReader((out / 'transport-history.csv').open()))
    if not (out / 'injection-summary.csv').exists():
        summary = []
        for side in ['right', 'left']:
            subset = [r for r in rows if r['injection'] == side]
            last = subset[-1]
            peak = max(subset, key=lambda r: float(r['ellipsoidal_ROI_mean_C']))
            summary.append({
                'injection': side,
                'inlet_zone': 'inlet_b1' if side == 'right' else 'inlet_b2',
                'injected_virtual_label_volume_ml': float(last['injected_label_volume_ml']),
                'max_raw_relative_mass_defect': max(float(r['relative_mass_balance_error']) for r in subset),
                'max_corrected_ledger_relative_error': max(float(r['corrected_ledger_relative_error']) for r in subset),
                'peak_ellipsoidal_ROI_mean_C': float(peak['ellipsoidal_ROI_mean_C']),
                'peak_time_after_injection_s': float(peak['time_since_injection_s']),
                'end_ellipsoidal_ROI_mean_C': float(last['ellipsoidal_ROI_mean_C']),
                'scope': 'Qualitative first-cycle virtual dye, not validated clinical clearance or residence time'
            })
        with (out / 'injection-summary.csv').open('x', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0]))
            w.writeheader()
            w.writerows(summary)
    return frames, rows


def main(out):
    out.mkdir(exist_ok=False)
    vis = out / 'visualization'
    vis.mkdir(exist_ok=False)

    # Reuse the verified voxel map and boundary surface from the patient_specific run
    ps_vis = CASE_DIRS['patient_specific'] / 'visualization'
    mapping = np.load(ps_vis / 'voxel-native-cell-map.npz')
    ids = mapping['cell']
    dims = mapping['dimensions']
    origin = mapping['origin']
    spacing = float(mapping['spacing'])
    mask = ids >= 0
    surface = pv.read(ps_vis / 'original-fluid-boundary.vtp')

    def sample(c):
        v = np.zeros(len(ids), np.float32)
        v[mask] = c[ids[mask]]
        return v

    all_frames = {}
    all_rows = {}
    summary_rows = []
    for k in KINDS:
        cdir = CASE_DIRS[k]
        frames, rows = verify_and_summarize_case(cdir)
        all_frames[k] = frames
        all_rows[k] = rows
        for side in ['right', 'left']:
            subset = [r for r in rows if r['injection'] == side]
            last = subset[-1]
            peak = max(subset, key=lambda r: float(r['ellipsoidal_ROI_mean_C']))
            arrival = next((float(r['time_since_injection_s']) for r in subset if float(r['ellipsoidal_ROI_mean_C']) >= 0.01), None)
            auc_c_ms = float(np.trapezoid(
                [0.0] + [float(r['ellipsoidal_ROI_mean_C']) for r in subset],
                [0.0] + [float(r['time_since_injection_s']) * 1000.0 for r in subset]
            ))
            summary_rows.append({
                'case': k,
                'case_label': CASE_TITLES[k],
                'injection': side,
                'inlet_zone': 'inlet_b1' if side == 'right' else 'inlet_b2',
                'injected_virtual_label_volume_ml': float(last['injected_label_volume_ml']),
                'arrival_time_1pct_ms': round(arrival * 1000.0, 1) if arrival is not None else None,
                'peak_ellipsoidal_ROI_mean_C_pct': round(100.0 * float(peak['ellipsoidal_ROI_mean_C']), 3),
                'peak_time_after_injection_ms': round(1000.0 * float(peak['time_since_injection_s']), 1),
                'end_863ms_ROI_mean_C_pct': round(100.0 * float(last['ellipsoidal_ROI_mean_C']), 3),
                'retained_fraction_of_peak_at_863ms': round(float(last['ellipsoidal_ROI_mean_C']) / max(float(peak['ellipsoidal_ROI_mean_C']), 1e-12), 4),
                'sac_ROI_AUC_C_ms': round(auc_c_ms, 3),
                'max_raw_relative_mass_defect': max(float(r['relative_mass_balance_error']) for r in subset),
                'max_corrected_ledger_relative_error': max(float(r['corrected_ledger_relative_error']) for r in subset),
            })

    with (out / 'three-case-virtual-angiography-summary.csv').open('x', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        w.writeheader()
        w.writerows(summary_rows)

    times = np.asarray([x['time_s'] for x in all_frames['patient_specific']])
    start_t = times[0]

    # Compute global max projection across all 3 cases and both sides for fair comparison
    max_projection = 0.0
    for k in KINDS:
        cdir = CASE_DIRS[k]
        for f in all_frames[k]:
            with np.load(cdir / f['file']) as d:
                for side in ['right', 'left']:
                    p = sample(d[side]).reshape(tuple(dims), order='F').sum(axis=1) * spacing
                    max_projection = max(max_projection, float(p.max()))
    assert max_projection > 0
    print('GLOBAL_MAX_PROJECTION', max_projection, flush=True)

    # Save static comparison figure of sac ROI concentration curves
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
    fig_s, axes_s = plt.subplots(1, 2, figsize=(14, 5.2), dpi=150, sharey=True, layout='constrained')
    max_curve_y = 0.0
    for idx, side in enumerate(['right', 'left']):
        ax = axes_s[idx]
        for k in KINDS:
            r = [q for q in all_rows[k] if q['injection'] == side]
            x = [0.0] + [(float(q['solver_time_s']) - start_t) * 1000.0 for q in r]
            y = [0.0] + [100.0 * float(q['ellipsoidal_ROI_mean_C']) for q in r]
            max_curve_y = max(max_curve_y, max(y))
            ax.plot(x, y, color=CASE_COLORS[k], lw=2.3, label=CASE_TITLES[k])
        ax.axvspan(0, 100, color='#e2e8f0', alpha=0.55, label='100 ms bolus pulse' if idx == 0 else None)
        ax.set_title(f"{side.capitalize()} ICA Injection ({'inlet_b1' if side=='right' else 'inlet_b2'})", fontsize=13, fontweight='bold')
        ax.set_xlabel('Time since injection (ms)')
        if idx == 0:
            ax.set_ylabel('Ellipsoidal Sac ROI Mean C (% of inlet)')
        ax.set_xlim(0, (times[-1] - start_t) * 1000.0)
        ax.grid(True, color='#e2e8f0')
        ax.spines[['top', 'right']].set_visible(False)
    axes_s[0].set_ylim(0, max_curve_y * 1.15)
    axes_s[0].legend(frameon=False, loc='upper right')
    fig_s.suptitle('Virtual Contrast Angiography: Aneurysm Sac Filling & Washout Across Inflow Conditions', fontsize=14, fontweight='bold')
    fig_s.savefig(out / 'three-case-sac-concentration-curves.png', dpi=150)
    plt.close(fig_s)

    # Also render individual-case videos for individualized and generic using render_dye_bolus logic if not present
    dye_cmap = ListedColormap(plt.get_cmap('Oranges')(np.linspace(0.35, 1.0, 256)))
    opacity = (np.sqrt(np.linspace(0, 1, 256)) * 0.40 * 255).astype(np.uint8)
    opacity[0] = 0
    center = np.asarray(surface.center)
    extent = [origin[0], origin[0] + (dims[0] - 1) * spacing, origin[2], origin[2] + (dims[2] - 1) * spacing]

    videos_manifest = []
    for side in ['right', 'left']:
        video_path = vis / f'Three-cases-{side}-ICA-virtual-angiography.mp4'
        log_path = vis / f'Three-cases-{side}-ICA-encoding.log'

        # Set up 3 off-screen PyVista plotters (one per case) for fast frame updates
        plotters = {}
        grids = {}
        for k in KINDS:
            g = pv.ImageData(dimensions=dims, spacing=[spacing] * 3, origin=origin)
            g.point_data['Dye_C'] = np.zeros(len(ids), np.float32)
            p = pv.Plotter(off_screen=True, window_size=(560, 520))
            p.set_background('white')
            p.add_mesh(surface, color='#64748b', opacity=0.055)
            p.add_volume(g, scalars='Dye_C', clim=[0, 1], cmap=dye_cmap, opacity=opacity, shade=False, show_scalar_bar=False)
            p.camera_position = [center + np.array([45, -120, 25]), center, [0, 0, 1]]
            p.enable_parallel_projection()
            p.camera.parallel_scale = 32
            plotters[k] = p
            grids[k] = g

        # Build composite 1920x1080 Matplotlib layout:
        # Row 0: 3 columns of 3D volumetric dye (Patient-specific, Individualized, Generic)
        # Row 1: 3 columns of 2D Angiography-style projection (Patient-specific, Individualized, Generic)
        # Row 2: Sac ROI concentration curves comparing all 3 cases + colorbar summary
        fig = plt.figure(figsize=(19.2, 10.8), dpi=100, layout='constrained')
        spec = fig.add_gridspec(3, 3, height_ratios=[1.25, 1.05, 0.75])

        marks3 = {}
        dsas = {}
        for col, k in enumerate(KINDS):
            ax3 = fig.add_subplot(spec[0, col])
            ax3.axis('off')
            ax3.set_title(f"{CASE_TITLES[k]} — 3D Virtual Dye", fontsize=13, fontweight='bold', color=CASE_COLORS[k])
            marks3[k] = ax3.imshow(np.full((520, 560, 3), 255, np.uint8))

            ax2 = fig.add_subplot(spec[1, col])
            dsa = ax2.imshow(
                np.zeros((dims[2], dims[0])),
                origin='lower',
                extent=extent,
                cmap='Greys',
                norm=PowerNorm(0.5, vmin=0, vmax=max_projection),
                aspect='equal',
                interpolation='nearest'
            )
            ax2.set_title(f"{CASE_TITLES[k]} — 2D Projection", fontsize=12)
            ax2.set_xlabel('X (mm)', fontsize=10)
            if col == 0:
                ax2.set_ylabel('Z (mm)', fontsize=10)
            ax2.add_patch(Ellipse((CENTER_MM[0], CENTER_MM[2]), 2 * RADII_MM[0], 2 * RADII_MM[2],
                                  fill=False, edgecolor=CASE_COLORS[k], lw=1.2, ls='--'))
            ax2.text(CENTER_MM[0] + RADII_MM[0], CENTER_MM[2] + RADII_MM[2], 'Sac ROI', fontsize=8.5, color=CASE_COLORS[k])
            if col == 2:
                fig.colorbar(dsa, ax=ax2, label='Integrated C (mm)', fraction=0.04, pad=0.02)
            dsas[k] = dsa

        axc = fig.add_subplot(spec[2, :2])
        side_max_y = 0.0
        for k in KINDS:
            r = [q for q in all_rows[k] if q['injection'] == side]
            x = [0.0] + [(float(q['solver_time_s']) - start_t) * 1000.0 for q in r]
            y = [0.0] + [100.0 * float(q['ellipsoidal_ROI_mean_C']) for q in r]
            side_max_y = max(side_max_y, max(y))
            axc.plot(x, y, color=CASE_COLORS[k], lw=2.4, label=CASE_TITLES[k])
        axc.set(
            xlim=(0, (times[-1] - start_t) * 1000.0),
            ylim=(0, max(0.01, side_max_y * 1.18)),
            xlabel='Time since injection (ms)',
            ylabel='Sac ROI Mean C (% of inlet)',
            title=f'Ellipsoidal Sac ROI Concentration — {side.capitalize()} ICA 100 ms Bolus'
        )
        axc.grid(True, color='#e2e8f0')
        axc.spines[['top', 'right']].set_visible(False)
        axc.legend(frameon=False, ncol=3, loc='upper right', fontsize=10.5)
        cursor = axc.axvline(0, color='#0f172a', lw=1.5)
        axc.axvspan(0, 100, color='#e2e8f0', alpha=0.55)

        # Bottom right panel: quantitative metrics table for the selected ICA injection
        axt = fig.add_subplot(spec[2, 2])
        axt.axis('off')
        side_rows = [r for r in summary_rows if r['injection'] == side]
        table_lines = [f"{side.capitalize()} ICA Injection Summary (First Cycle, 5–868 ms):"]
        for sr in side_rows:
            table_lines.append(
                f"• {sr['case_label']}:\n"
                f"   Injected bolus: {sr['injected_virtual_label_volume_ml']:.3f} mL | "
                f"Peak Sac C: {sr['peak_ellipsoidal_ROI_mean_C_pct']:.1f}% at {sr['peak_time_after_injection_ms']:.0f} ms\n"
                f"   End (863 ms) Sac C: {sr['end_863ms_ROI_mean_C_pct']:.2f}% | "
                f"AUC: {sr['sac_ROI_AUC_C_ms']:.1f} C·ms"
            )
        axt.text(0.02, 0.95, "\n".join(table_lines), va='top', ha='left', fontsize=9.8,
                 color='#1e293b', bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8fafc', edgecolor='#cbd5e1'))

        title = fig.suptitle('', fontsize=16, fontweight='bold')
        fig.text(0.01, 0.004,
                 'Virtual Contrast Angiography Comparison | 0.25-mm connected scalar groups on unmodified CFD fluxes | Shared concentration & projection scales',
                 fontsize=9, color='#475569')

        command = [
            str(FFMPEG), '-hide_banner', '-loglevel', 'warning', '-f', 'rawvideo',
            '-pixel_format', 'rgb24', '-video_size', '1920x1080', '-framerate', '24', '-i', '-',
            '-an', '-c:v', 'libopenh264', '-b:v', '7M', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
            str(video_path)
        ]

        cached = {k: {} for k in KINDS}
        video_start = time.monotonic()
        with log_path.open('x') as log:
            proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log, creationflags=subprocess.CREATE_NO_WINDOW)
            try:
                for index, t in enumerate(np.linspace(times[0], times[-1], 240)):
                    j = min(len(times) - 2, max(0, int(np.searchsorted(times, t, side='right') - 1)))
                    alpha = (t - times[j]) / (times[j + 1] - times[j])
                    for k in KINDS:
                        cdir = CASE_DIRS[k]
                        for n in [j, j + 1]:
                            if n not in cached[k]:
                                with np.load(cdir / all_frames[k][n]['file']) as d:
                                    cached[k][n] = d[side].copy()
                        c = (1.0 - alpha) * cached[k][j] + alpha * cached[k][j + 1]
                        for n in list(cached[k]):
                            if n not in [j, j + 1]:
                                del cached[k][n]
                        v = sample(c)
                        g = grids[k]
                        g.point_data['Dye_C'][:] = v
                        g.GetPointData().GetArray('Dye_C').Modified()
                        plotters[k].render()
                        marks3[k].set_data(plotters[k].screenshot(return_img=True))
                        proj = v.reshape(tuple(dims), order='F').sum(axis=1) * spacing
                        dsas[k].set_data(proj.T)

                    elapsed = (t - start_t) * 1000.0
                    cursor.set_xdata([elapsed, elapsed])
                    phase = 'Injection (0–100 ms)' if elapsed < 100 else 'Washout (>100 ms)'
                    title.set_text(f"Virtual Contrast Angiography Comparison — {side.capitalize()} ICA Injection | {phase} | t = {elapsed:.0f} ms")
                    fig.canvas.draw()
                    rgb = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
                    proc.stdin.write(rgb.tobytes())
                    if index in [0, 28, 55, 110, 239]:
                        fig.savefig(vis / f'Three-cases-{side}-ICA-preview-{index:03d}.png', dpi=100)
                    if index % 30 == 0:
                        print('COMPARE_RENDER', side, index, '/240', round(time.monotonic() - video_start, 1), flush=True)
                proc.stdin.close()
                assert proc.wait(timeout=120) == 0
            except BaseException:
                proc.stdin.close()
                if proc.poll() is None:
                    proc.terminate()
                    proc.wait(timeout=30)
                raise
        plt.close(fig)
        for p in plotters.values():
            p.close()

        # Decode-verify the encoded MP4
        chk = subprocess.run([str(FFMPEG), '-v', 'error', '-xerror', '-i', str(video_path), '-f', 'null', '-'],
                             capture_output=True, text=True, timeout=120, creationflags=subprocess.CREATE_NO_WINDOW)
        assert chk.returncode == 0, chk.stderr
        videos_manifest.append({
            'injection': side,
            'file': str(video_path),
            'sha256': sha(video_path),
            'bytes': video_path.stat().st_size,
            'frames': 240,
            'fps': 24,
            'duration_s': 10,
            'resolution': [1920, 1080],
            'full_decode_passed': True
        })

    save(vis / 'comparison-video-manifest.json', {
        'status': 'completed_and_verified',
        'shared_concentration_scale': [0, 1],
        'shared_integrated_projection_scale_mm': [0, max_projection],
        'videos': videos_manifest
    })
    print('COMPARISON_VIDEOS_COMPLETE', str(out), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=PROJECT / 'results/20261029-virtual-angiography-3case-comparison-v1')
    a = p.parse_args()
    main(a.out)
