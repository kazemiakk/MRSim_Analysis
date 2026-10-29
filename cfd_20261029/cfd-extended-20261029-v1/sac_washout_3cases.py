"""Pre-filled aneurysm sac virtual-dye washout (C=1 in sac ROI at t=0, C=0 inflow) across all 3 cases."""
from pathlib import Path
import argparse, csv, json, os, sys, subprocess, time
import h5py, numpy as np, pyvista as pv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.colors import ListedColormap, PowerNorm

sys.path.insert(0, r'D:\FluentAgent\automation\20261029-extended-hemodynamics-v1')
from extended_common import RUN, PREVIOUS, PROJECT, KINDS, load, save, sha, CENTER_MM, RADII_MM
from dye_bolus import Transport, unit_checks

FFMPEG = Path(r'D:\ANSYS Inc\v261\tp\ffmpeg\8.0.0\winx64\ffmpeg.exe')
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


def run_washout_transport(out, dtmax=0.005, agglomerate_mm=0.25):
    started = time.monotonic()
    out.mkdir(exist_ok=False)
    (out / 'fields').mkdir()
    save(out / 'unit-checks.json', unit_checks())

    mapping = np.load(PREVIOUS / 'comparisons/fluid-ROI-mapping.npz')
    vol_native = mapping['volume_m3']
    roi_native = mapping['ellipsoid_mask'].astype(float)
    assert len(vol_native) == 1169992

    # Reuse the exact connected scalar agglomeration from the patient_specific virtual angiography run
    agg_path = PROJECT / 'results/20261029-virtual-angiography-connected-250um-v1/connected-scalar-agglomeration.npz'
    agg = np.load(agg_path)
    labels = agg['native_cell_to_scalar_cell']
    vol = agg['scalar_volume_m3']
    roi_volume = agg['scalar_ROI_volume_m3']
    count = len(vol)
    mass = 1060.0 * vol

    # Initial condition: volume-weighted fraction of each connected scalar cell inside the ellipsoidal sac ROI
    c_init = (roi_volume / vol).astype(float)
    initial_sac_mass = float(np.dot(1060.0 * roi_volume, c_init))
    initial_total_mass = float(np.dot(mass, c_init))
    initial_roi_mean_c = float(np.dot(roi_volume, c_init) / roi_volume.sum())

    cfg_ps = load(RUN / 'patient_specific/input-configuration.json')
    with h5py.File(cfg_ps['source_case'], 'r') as h:
        mesh = h['meshes/1']
        c0_raw = mesh['faces/c0/1'][:].astype(int) - 1
        c1_raw = mesh['faces/c1/1'][:].astype(int) - 1
    native_ni = len(c1_raw)
    coarse0 = labels[c0_raw[:native_ni]]
    coarse1 = labels[c1_raw]
    native_internal_keep = coarse0 != coarse1
    c0 = np.r_[coarse0[native_internal_keep], labels[c0_raw[native_ni:]]]
    c1 = coarse1[native_internal_keep]
    ni = len(c1)

    steps_by_case = {k: load(RUN / k / 'progress.json')['accepted_steps'] for k in KINDS}
    steps_ref = steps_by_case['patient_specific']
    start_t = float(steps_ref[0]['time_s'])
    end_t = float(steps_ref[-1]['time_s'])

    protocol = {
        'experiment': 'Pre-filled aneurysm sac virtual-dye washout',
        'cases': KINDS,
        'initial_condition': 'C = volume-fraction inside approved ellipsoidal sac ROI at t = 5 ms, C = 0 outside',
        'inlet_condition': 'C = 0 at all incoming boundaries (tracer-free blood inflow)',
        'pulse_start_solver_time_s': start_t,
        'end_solver_time_s': end_t,
        'maximum_scalar_step_s': dtmax,
        'carrier_continuity_correction': True,
        'linear_solver': 'Flow-graph SCC-ordered ILU and GMRES',
        'native_flow_cells': 1169992,
        'scalar_transport_cells': count,
        'connected_scalar_agglomeration_mm': agglomerate_mm,
        'initial_sac_ROI_mean_C': initial_roi_mean_c,
        'initial_total_tracer_volume_ml': initial_total_mass / 1060.0 * 1e6,
        'initial_sac_ROI_tracer_volume_ml': initial_sac_mass / 1060.0 * 1e6,
        'source_flow_unchanged': True,
        'new_Fluent_sessions': 0,
        'periodicity_validated': False,
    }
    save(out / 'protocol.json', protocol)

    def get_flux(step):
        with h5py.File(step['output_stem'] + '.dat.h5', 'r') as h:
            f = h['results/1/phase-1/faces/SV_FLUX/1'][:]
        return np.r_[f[:native_ni][native_internal_keep], f[native_ni:]]

    concentrations = {k: c_init.copy() for k in KINDS}
    exited = {k: 0.0 for k in KINDS}
    inherited_defect = {k: 0.0 for k in KINDS}
    history = []
    frames = []
    t = start_t

    def record_frame(curr_t):
        name = 'fields/W-t%07dus.npz' % round(curr_t * 1e6)
        np.savez_compressed(
            out / name,
            patient_specific=concentrations['patient_specific'][labels].astype('float32'),
            individualized=concentrations['individualized'][labels].astype('float32'),
            generic=concentrations['generic'][labels].astype('float32'),
            time_s=curr_t
        )
        frames.append({'time_s': curr_t, 'file': name})

    record_frame(t)
    for k in KINDS:
        c = concentrations[k]
        roi_rem = float(np.dot(1060.0 * roi_volume, c)) / initial_sac_mass
        tot_rem = float(np.dot(mass, c)) / initial_total_mass
        history.append({
            'case': k,
            'solver_time_s': t,
            'elapsed_washout_s': 0.0,
            'dt_s': 0.0,
            'C_min': float(c.min()),
            'C_max': float(c.max()),
            'sac_ROI_mean_C': float(np.dot(roi_volume, c) / roi_volume.sum()),
            'sac_ROI_tracer_remaining_fraction': roi_rem,
            'whole_domain_tracer_remaining_fraction': tot_rem,
            'exited_domain_fraction': 0.0,
            'relative_mass_balance_error': 0.0,
            'corrected_ledger_relative_error': 0.0,
            'linear_relative_residual': 0.0,
            'solver_iterations': 0,
        })

    fa = {k: get_flux(steps_by_case[k][0]) for k in KINDS}
    zeros_in = np.zeros(count, dtype=float)

    for interval in range(len(steps_ref) - 1):
        lo = float(steps_ref[interval]['time_s'])
        hi = float(steps_ref[interval + 1]['time_s'])
        sub_count = int(np.ceil((hi - lo) / dtmax - 1e-9))
        dt = (hi - lo) / sub_count

        for k in KINDS:
            fb = get_flux(steps_by_case[k][interval + 1])
            fmid = 0.5 * (fa[k] + fb)
            fa[k] = fb
            op = Transport(mass, c0, c1, fmid, dt, continuity_correction=True, flow_ordered=True)
            for sub in range(sub_count):
                right_t = lo + (sub + 1) * dt
                candidate, audit = op.advance(concentrations[k], zeros_in)
                minimum = float(candidate.min())
                maximum = float(candidate.max())
                if not np.isfinite(candidate).all() or minimum < -1e-6 or maximum > 1.001:
                    raise RuntimeError(f'Washout bounds failed for {k}: {(minimum, maximum)}')
                total_exited = exited[k] + audit['exited_kg_label']
                signed_balance = float(np.dot(mass, candidate)) + total_exited - initial_total_mass
                balance = abs(signed_balance) / initial_total_mass
                corrected_defect = inherited_defect[k] + audit['inherited_continuity_defect_kg_label']
                accounting_error = abs(signed_balance - corrected_defect) / initial_total_mass
                if accounting_error > 1e-5:
                    raise RuntimeError(f'Washout accounting error failed for {k}: {accounting_error}')
                concentrations[k] = candidate
                exited[k] = total_exited
                inherited_defect[k] = corrected_defect

                roi_rem = float(np.dot(1060.0 * roi_volume, candidate)) / initial_sac_mass
                tot_rem = float(np.dot(mass, candidate)) / initial_total_mass
                history.append({
                    'case': k,
                    'solver_time_s': right_t,
                    'elapsed_washout_s': right_t - start_t,
                    'dt_s': dt,
                    'C_min': minimum,
                    'C_max': maximum,
                    'sac_ROI_mean_C': float(np.dot(roi_volume, candidate) / roi_volume.sum()),
                    'sac_ROI_tracer_remaining_fraction': roi_rem,
                    'whole_domain_tracer_remaining_fraction': tot_rem,
                    'exited_domain_fraction': total_exited / initial_total_mass,
                    'relative_mass_balance_error': balance,
                    'corrected_ledger_relative_error': accounting_error,
                    'linear_relative_residual': audit['linear_relative_residual'],
                    'solver_iterations': audit['solver_iterations'],
                })
        t = hi
        record_frame(t)
        if interval % 5 == 0 or interval == len(steps_ref) - 2:
            rem_str = ", ".join(f"{k}={np.dot(1060.0*roi_volume, concentrations[k])/initial_sac_mass*100:.1f}%" for k in KINDS)
            print(f'WASHOUT t={t:.3f}s wall={time.monotonic()-started:.1f}s sac_rem: {rem_str}', flush=True)

    with (out / 'washout-history.csv').open('x', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)
    save(out / 'frames.json', frames)

    # Summarize washout metrics (t50, t90, remaining at end of cycle, retention AUC)
    summary_rows = []
    for k in KINDS:
        sub = [r for r in history if r['case'] == k]
        last = sub[-1]
        t_arr = np.array([float(r['elapsed_washout_s']) for r in sub])
        y_arr = np.array([float(r['sac_ROI_tracer_remaining_fraction']) for r in sub])
        t50 = next((float(r['elapsed_washout_s']) for r in sub if float(r['sac_ROI_tracer_remaining_fraction']) <= 0.50), None)
        t90 = next((float(r['elapsed_washout_s']) for r in sub if float(r['sac_ROI_tracer_remaining_fraction']) <= 0.10), None)
        mean_transit_est_s = float(np.trapezoid(y_arr, t_arr))
        summary_rows.append({
            'case': k,
            'case_label': CASE_TITLES[k],
            'initial_sac_tracer_volume_ml': round(initial_sac_mass / 1060.0 * 1e6, 4),
            't50_clearance_ms': round(t50 * 1000.0, 1) if t50 is not None else '>863 (not reached in 1 cycle)',
            't90_clearance_ms': round(t90 * 1000.0, 1) if t90 is not None else '>863 (not reached in 1 cycle)',
            'sac_tracer_remaining_at_863ms_pct': round(100.0 * float(last['sac_ROI_tracer_remaining_fraction']), 2),
            'sac_cleared_at_863ms_pct': round(100.0 * (1.0 - float(last['sac_ROI_tracer_remaining_fraction'])), 2),
            'whole_domain_remaining_at_863ms_pct': round(100.0 * float(last['whole_domain_tracer_remaining_fraction']), 2),
            'exited_domain_at_863ms_pct': round(100.0 * float(last['exited_domain_fraction']), 2),
            'truncated_1cycle_retention_integral_ms': round(mean_transit_est_s * 1000.0, 1),
            'max_raw_relative_mass_defect': max(float(r['relative_mass_balance_error']) for r in sub),
            'max_corrected_ledger_relative_error': max(float(r['corrected_ledger_relative_error']) for r in sub),
        })

    with (out / 'washout-summary.csv').open('x', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        w.writeheader()
        w.writerows(summary_rows)

    save(out / 'result.json', {
        'status': 'available_first_cycle_washout_complete',
        'last_time_s': t,
        'duration_s': t - start_t,
        'wall_seconds': time.monotonic() - started,
        'summary': summary_rows
    })
    return frames, history, summary_rows


def render_washout(out, frames, history, summary_rows):
    vis = out / 'visualization'
    vis.mkdir(exist_ok=False)

    ps_vis = PROJECT / 'results/20261029-virtual-angiography-connected-250um-v1/visualization'
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

    times = np.asarray([x['time_s'] for x in frames])
    start_t = times[0]

    max_projection = 0.0
    for f in frames:
        with np.load(out / f['file']) as d:
            for k in KINDS:
                p = sample(d[k]).reshape(tuple(dims), order='F').sum(axis=1) * spacing
                max_projection = max(max_projection, float(p.max()))
    assert max_projection > 0

    # Static figure comparing sac washout curves and domain clearance
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
    fig_s, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), dpi=150, layout='constrained')
    for k in KINDS:
        sub = [r for r in history if r['case'] == k]
        x = [float(r['elapsed_washout_s']) * 1000.0 for r in sub]
        y_sac = [100.0 * float(r['sac_ROI_tracer_remaining_fraction']) for r in sub]
        y_dom = [100.0 * float(r['whole_domain_tracer_remaining_fraction']) for r in sub]
        ax1.plot(x, y_sac, color=CASE_COLORS[k], lw=2.4, label=CASE_TITLES[k])
        ax2.plot(x, y_dom, color=CASE_COLORS[k], lw=2.4, label=CASE_TITLES[k])
    ax1.axhline(50, color='#64748b', ls='--', lw=1.1, label='50% clearance threshold')
    ax1.set(title='Aneurysm Sac ROI Tracer Remaining (%)', xlabel='Washout time (ms)', ylabel='Remaining in Sac ROI (%)', xlim=(0, 863), ylim=(0, 105))
    ax2.set(title='Whole Fluid Domain Tracer Remaining (%)', xlabel='Washout time (ms)', ylabel='Remaining in Domain (%)', xlim=(0, 863), ylim=(0, 105))
    for ax in (ax1, ax2):
        ax.grid(True, color='#e2e8f0')
        ax.spines[['top', 'right']].set_visible(False)
    ax1.legend(frameon=False, loc='lower left')
    fig_s.suptitle('Pre-Filled Virtual-Dye Sac Washout Comparison (First Cycle, 5–868 ms)', fontsize=14, fontweight='bold')
    fig_s.savefig(out / 'three-case-sac-washout-curves.png', dpi=150)
    plt.close(fig_s)

    # Render 3-case washout comparison video
    dye_cmap = ListedColormap(plt.get_cmap('PuBuGn')(np.linspace(0.30, 1.0, 256)))
    opacity = (np.sqrt(np.linspace(0, 1, 256)) * 0.42 * 255).astype(np.uint8)
    opacity[0] = 0
    center = np.asarray(surface.center)
    extent = [origin[0], origin[0] + (dims[0] - 1) * spacing, origin[2], origin[2] + (dims[2] - 1) * spacing]

    video_path = vis / 'Three-cases-sac-virtual-dye-washout.mp4'
    log_path = vis / 'Three-cases-sac-washout-encoding.log'

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

    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, layout='constrained')
    spec = fig.add_gridspec(3, 3, height_ratios=[1.25, 1.05, 0.75])

    marks3 = {}
    dsas = {}
    for col, k in enumerate(KINDS):
        ax3 = fig.add_subplot(spec[0, col])
        ax3.axis('off')
        ax3.set_title(f"{CASE_TITLES[k]} — 3D Sac Dye Washout", fontsize=13, fontweight='bold', color=CASE_COLORS[k])
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
    for k in KINDS:
        sub = [r for r in history if r['case'] == k]
        x = [float(r['elapsed_washout_s']) * 1000.0 for r in sub]
        y = [100.0 * float(r['sac_ROI_tracer_remaining_fraction']) for r in sub]
        axc.plot(x, y, color=CASE_COLORS[k], lw=2.4, label=CASE_TITLES[k])
    axc.axhline(50, color='#64748b', ls='--', lw=1.1, label='50% clearance')
    axc.set(
        xlim=(0, (times[-1] - start_t) * 1000.0),
        ylim=(0, 105),
        xlabel='Washout time (ms)',
        ylabel='Sac ROI Tracer Remaining (%)',
        title='Pre-Filled Aneurysm Sac Tracer Retention & Clearance (C = 1 in Sac ROI at t = 0)'
    )
    axc.grid(True, color='#e2e8f0')
    axc.spines[['top', 'right']].set_visible(False)
    axc.legend(frameon=False, ncol=4, loc='lower left', fontsize=10)
    cursor = axc.axvline(0, color='#0f172a', lw=1.5)

    axt = fig.add_subplot(spec[2, 2])
    axt.axis('off')
    table_lines = ["Pre-Filled Sac Washout Summary (First Cycle, 863 ms):"]
    for sr in summary_rows:
        table_lines.append(
            f"• {sr['case_label']}:\n"
            f"   t50 clearance: {sr['t50_clearance_ms']} ms | "
            f"Cleared at 863 ms: {sr['sac_cleared_at_863ms_pct']:.1f}%\n"
            f"   Remaining in Sac ROI at 863 ms: {sr['sac_tracer_remaining_at_863ms_pct']:.1f}%"
        )
    axt.text(0.02, 0.95, "\n".join(table_lines), va='top', ha='left', fontsize=9.8,
             color='#1e293b', bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8fafc', edgecolor='#cbd5e1'))

    title = fig.suptitle('', fontsize=16, fontweight='bold')
    fig.text(0.01, 0.004,
             'Pre-Filled Virtual-Dye Sac Washout | C=1 in ellipsoidal sac ROI at t=0, tracer-free inflow | 0.25-mm connected scalar groups',
             fontsize=9, color='#475569')

    command = [
        str(FFMPEG), '-hide_banner', '-loglevel', 'warning', '-f', 'rawvideo',
        '-pixel_format', 'rgb24', '-video_size', '1920x1080', '-framerate', '24', '-i', '-',
        '-an', '-c:v', 'libopenh264', '-b:v', '7M', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
        str(video_path)
    ]

    cached = {}
    video_start = time.monotonic()
    with log_path.open('x') as log:
        proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log, creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            for index, t in enumerate(np.linspace(times[0], times[-1], 240)):
                j = min(len(times) - 2, max(0, int(np.searchsorted(times, t, side='right') - 1)))
                alpha = (t - times[j]) / (times[j + 1] - times[j])
                for n in [j, j + 1]:
                    if n not in cached:
                        with np.load(out / frames[n]['file']) as d:
                            cached[n] = {k: d[k].copy() for k in KINDS}
                for n in list(cached):
                    if n not in [j, j + 1]:
                        del cached[n]
                for k in KINDS:
                    c = (1.0 - alpha) * cached[j][k] + alpha * cached[j + 1][k]
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
                title.set_text(f"Virtual-Dye Sac Washout Comparison — Pre-Filled Sac Clearance | t = {elapsed:.0f} ms")
                fig.canvas.draw()
                rgb = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
                proc.stdin.write(rgb.tobytes())
                if index in [0, 28, 55, 110, 239]:
                    fig.savefig(vis / f'Three-cases-sac-washout-preview-{index:03d}.png', dpi=100)
                if index % 30 == 0:
                    print('WASHOUT_RENDER', index, '/240', round(time.monotonic() - video_start, 1), flush=True)
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

    chk = subprocess.run([str(FFMPEG), '-v', 'error', '-xerror', '-i', str(video_path), '-f', 'null', '-'],
                         capture_output=True, text=True, timeout=120, creationflags=subprocess.CREATE_NO_WINDOW)
    assert chk.returncode == 0, chk.stderr
    save(vis / 'washout-video-manifest.json', {
        'status': 'completed_and_verified',
        'file': str(video_path),
        'sha256': sha(video_path),
        'bytes': video_path.stat().st_size,
        'frames': 240,
        'fps': 24,
        'duration_s': 10,
        'resolution': [1920, 1080],
        'full_decode_passed': True
    })
    print('WASHOUT_COMPLETE', str(out), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=PROJECT / 'results/20261029-virtual-dye-sac-washout-3cases-v1')
    a = p.parse_args()
    frames, history, summary_rows = run_washout_transport(a.out)
    render_washout(a.out, frames, history, summary_rows)

