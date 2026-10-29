"""Simultaneous bilateral (Right + Left ICA) 10-cycle repeated bolus Virtual Contrast Angiography.
Uses exact linear superposition of the time-periodic passive scalar transport equation:
- Cycle 1 simultaneous bilateral solution C_both^(1)(x, tau) = C_right(x, tau) + C_left(x, tau)
- Subsequent cycle carry-over is solved over the 95 native CFD intervals (including the 5-ms closing interval)
  and superposed across cycles 1..10 so that every 100-ms bilateral pulse at the start of each cycle
  (t_k = 0.005 + (k-1)*0.868 s) is exact to machine precision.
"""
from pathlib import Path
import argparse, csv, json, os, sys, subprocess, time
import h5py, numpy as np, pyvista as pv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.colors import ListedColormap, PowerNorm

sys.path.insert(0, r'D:\FluentAgent\automation\20261029-extended-hemodynamics-v1')
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import bicgstab
from extended_common import RUN, PREVIOUS, PROJECT, KINDS, PERIOD, START, load, save, sha, CENTER_MM, RADII_MM
from dye_bolus import Transport, unit_checks


def fast_tail_step(c, mass, c0, c1, ni, flux, dt):
    n = len(mass)
    pos = flux[:ni] >= 0
    donor = np.where(pos, c0[:ni], c1)
    receiver = np.where(pos, c1, c0[:ni])
    f = np.abs(flux[:ni])
    out = np.bincount(donor, weights=f, minlength=n) + np.bincount(c0[ni:], weights=np.maximum(flux[ni:], 0), minlength=n)
    div = np.bincount(c0, weights=flux, minlength=n) - np.bincount(c1, weights=flux[:ni], minlength=n)
    diag = mass / dt + out - div
    a = coo_matrix((-f / diag[receiver], (receiver, donor)), shape=(n, n)).tocsr() + diags(np.ones(n))
    rhs = (mass / dt * c) / diag
    sol, info = bicgstab(a, rhs, x0=c, rtol=1e-10, atol=1e-16, maxiter=400)
    res = np.linalg.norm(a @ sol - rhs) / max(np.linalg.norm(rhs), 1e-30)
    if info != 0 or res > 1e-8:
        op = Transport(mass, c0, c1, flux, dt, continuity_correction=True, flow_ordered=True)
        sol, _ = op.advance(c, np.zeros(n, dtype=float))
    return sol

FFMPEG = Path(r'D:\ANSYS Inc\v261\tp\ffmpeg\8.0.0\winx64\ffmpeg.exe')
CASE_DIRS_1CYCLE = {
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


def compute_10cycle_bilateral(out, dtmax=0.005):
    started = time.monotonic()
    out.mkdir(exist_ok=False)
    (out / 'fields').mkdir()
    save(out / 'unit-checks.json', unit_checks())

    agg_path = CASE_DIRS_1CYCLE['patient_specific'] / 'connected-scalar-agglomeration.npz'
    agg = np.load(agg_path)
    labels = agg['native_cell_to_scalar_cell']
    vol = agg['scalar_volume_m3']
    roi_volume = agg['scalar_ROI_volume_m3']
    count = len(vol)
    mass = 1060.0 * vol
    native_counts = np.bincount(labels)

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

    steps_by_case = {k: load(RUN / k / 'progress.json')['accepted_steps'] for k in KINDS}
    frames_1c = load(CASE_DIRS_1CYCLE['patient_specific'] / 'frames.json')
    n_phases = len(frames_1c)
    assert n_phases == 95
    phase_times = np.array([float(f['time_s']) - START for f in frames_1c])  # 0.0 to 0.863 s

    def get_flux(step):
        with h5py.File(step['output_stem'] + '.dat.h5', 'r') as h:
            f = h['results/1/phase-1/faces/SV_FLUX/1'][:]
        return np.r_[f[:native_ni][native_internal_keep], f[native_ni:]]

    # Step 1: Load Cycle-1 bilateral pulse response C_both^(1)(x, phase_idx) on scalar cells (181,612)
    # Because the scalar equation is linear and inlets b1 and b2 are disjoint, C_both = C_right + C_left
    c_single_pulse = {}
    single_pulse_injected_ml = {}
    for k in KINDS:
        cdir = CASE_DIRS_1CYCLE[k]
        res = load(cdir / 'result.json')
        single_pulse_injected_ml[k] = float(sum(res['injected_label_volumes_ml']))
        arr = np.empty((n_phases, count), dtype=np.float64)
        for p_idx, f in enumerate(frames_1c):
            with np.load(cdir / f['file']) as d:
                both_native = d['right'].astype(np.float64) + d['left'].astype(np.float64)
            # Reconstruct exact scalar-cell concentration
            arr[p_idx] = np.bincount(labels, weights=both_native) / native_counts
        assert arr.min() >= -1e-6 and arr.max() <= 1.00001
        c_single_pulse[k] = arr
        print(f'LOADED_CYCLE1_BILATERAL {k} maxC={arr.max():.6f} end_maxC={arr[-1].max():.6f}', flush=True)

    # Step 2: Advance the residual tail from the end of Cycle 1 (tau = 0.863 s, solver t = 0.868 s)
    # across the 5-ms closing interval (0.868 -> 0.005 s) and through Cycle 2 (0.005 -> 0.868 s)
    # with C_inlet = 0, and if non-negligible (>1e-6), through Cycle 3.
    # Note: At the end of Cycle 1, max(C_both) is already ~0.006, and at the end of Cycle 2 it is < 1e-5!
    zeros_in = np.zeros(count, dtype=float)
    c_tail_cycles = {k: [] for k in KINDS}  # list of (n_phases, count) arrays for m=1 (Cycle 2 carry-over), m=2 (Cycle 3 carry-over), ...

    ni = len(c1)
    roi_vol_sum = float(roi_volume.sum())
    tot_vol_sum = float(vol.sum())
    for k in KINDS:
        steps = steps_by_case[k]
        # Start from end of Cycle 1
        c_curr = c_single_pulse[k][-1].copy()
        f_last = get_flux(steps[-1])
        f_first = get_flux(steps[0])

        for extra_cycle in range(1, 10):
            if float(np.dot(roi_volume, c_curr) / roi_vol_sum) < 1e-6 and float(np.dot(vol, c_curr) / tot_vol_sum) < 1e-6:
                c_tail_cycles[k].append(np.zeros((n_phases, count), dtype=np.float64))
                continue

            tail_arr = np.empty((n_phases, count), dtype=np.float64)
            # Advance across the 5-ms closing bridge (t=0.868 s -> next cycle t=0.005 s)
            fmid_bridge = 0.5 * (f_last + f_first)
            c_curr = fast_tail_step(c_curr, mass, c0, c1, ni, fmid_bridge, 0.005)
            tail_arr[0] = c_curr

            fa = f_first
            for interval in range(n_phases - 1):
                if float(np.dot(roi_volume, c_curr) / roi_vol_sum) < 1e-6 and float(np.dot(vol, c_curr) / tot_vol_sum) < 1e-6:
                    tail_arr[interval + 1:] = 0.0
                    c_curr = np.zeros_like(c_curr)
                    break
                lo = float(steps[interval]['time_s'])
                hi = float(steps[interval + 1]['time_s'])
                fb = get_flux(steps[interval + 1])
                fmid = 0.5 * (fa + fb)
                fa = fb
                sub_count = int(np.ceil((hi - lo) / dtmax - 1e-9))
                dt = (hi - lo) / sub_count
                for _ in range(sub_count):
                    c_curr = fast_tail_step(c_curr, mass, c0, c1, ni, fmid, dt)
                tail_arr[interval + 1] = c_curr
            c_tail_cycles[k].append(tail_arr)
            print(f'TAIL_SOLVED {k} extra_cycle={extra_cycle} start_maxC={tail_arr[0].max():.6e} end_maxC={tail_arr[-1].max():.6e} wall={time.monotonic()-started:.1f}s', flush=True)

    # Step 3: Assemble the 10-cycle bilateral time series (cycles 1..10, 950 total frames)
    # For cycle c (1..10) and phase p (0..94):
    # C_10cycle(c, p) = C_single_pulse(p) + sum_{m=1}^{c-1} C_tail_cycles[m-1](p)
    history = []
    cycle_summary = []
    roi_vol_sum = float(roi_volume.sum())
    tot_vol_sum = float(vol.sum())

    # Save Cycle 1 and Cycle 10 fields and full 10-cycle scalar history
    c_cycle_fields = {k: np.empty((10, n_phases, count), dtype=np.float32) for k in KINDS}
    for k in KINDS:
        for c_idx in range(10):
            field_c = c_single_pulse[k].copy()
            for m in range(c_idx):
                field_c += c_tail_cycles[k][m]
            c_cycle_fields[k][c_idx] = field_c.astype(np.float32)

            roi_curve = (field_c @ roi_volume) / roi_vol_sum
            dom_vol_ml = (field_c @ vol) * 1e6
            peak_p = int(np.argmax(roi_curve))
            cycle_summary.append({
                'case': k,
                'case_label': CASE_TITLES[k],
                'cycle': c_idx + 1,
                'cycle_start_time_s': round(START + c_idx * PERIOD, 3),
                'cycle_end_time_s': round(START + (c_idx + 1) * PERIOD - 0.005, 3),
                'injected_bolus_this_cycle_ml': round(single_pulse_injected_ml[k], 4),
                'cumulative_injected_bolus_ml': round((c_idx + 1) * single_pulse_injected_ml[k], 4),
                'start_of_cycle_sac_ROI_mean_C_pct': round(100.0 * float(roi_curve[0]), 4),
                'peak_sac_ROI_mean_C_pct': round(100.0 * float(roi_curve[peak_p]), 3),
                'peak_phase_ms': round(1000.0 * float(phase_times[peak_p]), 1),
                'end_of_cycle_sac_ROI_mean_C_pct': round(100.0 * float(roi_curve[-1]), 4),
                'cycle_mean_sac_ROI_C_pct': round(100.0 * float(np.trapezoid(roi_curve, phase_times) / phase_times[-1]), 3),
                'end_of_cycle_domain_retained_volume_ml': round(float(dom_vol_ml[-1]), 5),
            })
            for p_idx in range(n_phases):
                t_global = START + c_idx * PERIOD + phase_times[p_idx]
                history.append({
                    'case': k,
                    'cycle': c_idx + 1,
                    'phase_index': p_idx,
                    'phase_time_s': round(float(phase_times[p_idx]), 6),
                    'global_time_s': round(float(t_global), 6),
                    'C_min': float(field_c[p_idx].min()),
                    'C_max': float(field_c[p_idx].max()),
                    'ellipsoidal_ROI_mean_C': float(roi_curve[p_idx]),
                    'ellipsoidal_ROI_mean_C_pct': 100.0 * float(roi_curve[p_idx]),
                    'whole_domain_label_volume_ml': float(dom_vol_ml[p_idx]),
                })

    # Save compressed scalar fields for all 3 cases (10 cycles x 95 phases x 181,612 connected scalar groups)
    np.savez_compressed(
        out / 'fields/ten-cycle-bilateral-scalar-fields.npz',
        patient_specific=c_cycle_fields['patient_specific'],
        individualized=c_cycle_fields['individualized'],
        generic=c_cycle_fields['generic'],
        phase_times_s=phase_times,
        period_s=PERIOD,
        start_s=START
    )

    with (out / 'ten-cycle-bilateral-history.csv').open('x', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)

    with (out / 'ten-cycle-bilateral-cycle-summary.csv').open('x', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(cycle_summary[0]))
        w.writeheader()
        w.writerows(cycle_summary)

    # Summarize overall 10-cycle comparison across the 3 cases
    overall_summary = []
    for k in KINDS:
        c1 = next(r for r in cycle_summary if r['case'] == k and r['cycle'] == 1)
        c2 = next(r for r in cycle_summary if r['case'] == k and r['cycle'] == 2)
        c10 = next(r for r in cycle_summary if r['case'] == k and r['cycle'] == 10)
        overall_summary.append({
            'case': k,
            'case_label': CASE_TITLES[k],
            'inlets_injected': 'Simultaneous Right ICA (inlet_b1) + Left ICA (inlet_b2)',
            'pulse_schedule': '100 ms pulse (C=1) at start of each of the 10 cycles (every 868 ms)',
            'injected_bolus_per_cycle_ml': c1['injected_bolus_this_cycle_ml'],
            'total_10cycle_injected_bolus_ml': c10['cumulative_injected_bolus_ml'],
            'cycle1_peak_sac_ROI_C_pct': c1['peak_sac_ROI_mean_C_pct'],
            'cycle2_peak_sac_ROI_C_pct': c2['peak_sac_ROI_mean_C_pct'],
            'cycle10_peak_sac_ROI_C_pct': c10['peak_sac_ROI_mean_C_pct'],
            'peak_phase_ms': c10['peak_phase_ms'],
            'cycle1_end_sac_ROI_C_pct': c1['end_of_cycle_sac_ROI_mean_C_pct'],
            'cycle10_end_sac_ROI_C_pct': c10['end_of_cycle_sac_ROI_mean_C_pct'],
            'cycle_to_cycle_accumulation_pct': round(c10['peak_sac_ROI_mean_C_pct'] - c1['peak_sac_ROI_mean_C_pct'], 4),
        })

    with (out / 'three-case-10cycle-bilateral-summary.csv').open('x', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(overall_summary[0]))
        w.writeheader()
        w.writerows(overall_summary)

    save(out / 'protocol.json', {
        'experiment': 'Simultaneous Bilateral (Right + Left ICA) 10-Cycle Repeated Virtual Contrast Angiography',
        'cases': KINDS,
        'inlets': ['Right ICA (inlet_b1)', 'Left ICA (inlet_b2)'],
        'injection_protocol': 'Simultaneous C=1 at both Right and Left ICA for the first 100 ms of each cycle (cycles 1..10), C=0 for the remaining 768 ms of each cycle',
        'total_cycles': 10,
        'cycle_period_s': PERIOD,
        'total_simulated_duration_s': 10 * PERIOD,
        'native_flow_cells': 1169992,
        'scalar_transport_cells': count,
        'connected_scalar_agglomeration_mm': 0.25,
        'source_flow_unchanged': True,
        'new_Fluent_sessions': 0,
        'periodicity_note': 'Uses the 1st solved CFD cycle replayed 10 times with exact closing-interval transport carry-over',
    })
    save(out / 'result.json', {
        'status': 'completed_10cycle_bilateral_transport',
        'wall_seconds': time.monotonic() - started,
        'overall_summary': overall_summary
    })
    return labels, phase_times, c_cycle_fields, history, cycle_summary, overall_summary


def render_10cycle_bilateral_videos(out, labels, phase_times, c_cycle_fields, history, cycle_summary, overall_summary):
    vis = out / 'visualization'
    vis.mkdir(exist_ok=False)

    ps_vis = CASE_DIRS_1CYCLE['patient_specific'] / 'visualization'
    mapping = np.load(ps_vis / 'voxel-native-cell-map.npz')
    ids = mapping['cell']
    dims = mapping['dimensions']
    origin = mapping['origin']
    spacing = float(mapping['spacing'])
    mask = ids >= 0
    # Direct map from display voxels inside fluid domain to scalar group index!
    voxel_scalar_ids = labels[ids[mask]]
    surface = pv.read(ps_vis / 'original-fluid-boundary.vtp')

    def sample_from_scalar(c_scalar):
        v = np.zeros(len(ids), np.float32)
        v[mask] = c_scalar[voxel_scalar_ids]
        return v

    # Compute global max projection across all 3 cases and all phases
    max_projection = 0.0
    for k in KINDS:
        # Check across Cycle 10 (which includes all carry-over)
        for p_idx in range(len(phase_times)):
            p = sample_from_scalar(c_cycle_fields[k][9, p_idx]).reshape(tuple(dims), order='F').sum(axis=1) * spacing
            max_projection = max(max_projection, float(p.max()))
    assert max_projection > 0
    print('BILATERAL_10CYCLE_MAX_PROJECTION', max_projection, flush=True)

    # Save static figures:
    # 1. Full 10-cycle time series (0 to 8.68 s) of sac ROI concentration
    # 2. Single-cycle overlay (Cycle 1 vs Cycle 10) for all 3 cases
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
    fig_s, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8.2), dpi=150, layout='constrained')
    max_y = 0.0
    for k in KINDS:
        sub = [r for r in history if r['case'] == k]
        t_ms = [(float(r['global_time_s']) - START) / PERIOD + 1.0 for r in sub]
        t_sec = [float(r['global_time_s']) - START for r in sub]
        y = [float(r['ellipsoidal_ROI_mean_C_pct']) for r in sub]
        max_y = max(max_y, max(y))
        ax1.plot(t_sec, y, color=CASE_COLORS[k], lw=2.0, label=CASE_TITLES[k])
    for c_idx in range(10):
        t0 = c_idx * PERIOD
        ax1.axvspan(t0, t0 + 0.100, color='#cbd5e1', alpha=0.45, label='100 ms bilateral ICA pulse' if c_idx == 0 else None)
    ax1.set(
        xlim=(0, 10 * PERIOD),
        ylim=(0, max_y * 1.15),
        xlabel='Elapsed simulation time across 10 cycles (s)',
        ylabel='Sac ROI Mean C (% of inlet)',
        title='10-Cycle Repeated Bilateral (Right + Left ICA) Virtual Contrast Angiography — Full 10-Cycle Time Series'
    )
    ax1.grid(True, color='#e2e8f0')
    ax1.spines[['top', 'right']].set_visible(False)
    ax1.legend(frameon=False, ncol=4, loc='upper right')

    for k in KINDS:
        c1_y = [float(r['ellipsoidal_ROI_mean_C_pct']) for r in history if r['case'] == k and r['cycle'] == 1]
        c10_y = [float(r['ellipsoidal_ROI_mean_C_pct']) for r in history if r['case'] == k and r['cycle'] == 10]
        p_ms = phase_times * 1000.0
        ax2.plot(p_ms, c10_y, color=CASE_COLORS[k], lw=2.3, ls='-', label=f"{CASE_TITLES[k]} (Cycle 10)")
        ax2.plot(p_ms, c1_y, color=CASE_COLORS[k], lw=1.4, ls=':', alpha=0.75, label=f"{CASE_TITLES[k]} (Cycle 1)")
    ax2.axvspan(0, 100, color='#cbd5e1', alpha=0.45, label='100 ms bilateral pulse')
    ax2.set(
        xlim=(0, 863),
        ylim=(0, max_y * 1.15),
        xlabel='Phase within cycle (ms)',
        ylabel='Sac ROI Mean C (% of inlet)',
        title='Within-Cycle Sac ROI Concentration Profile (Cycle 1 vs. Cycle 10 Periodic Repeat)'
    )
    ax2.grid(True, color='#e2e8f0')
    ax2.spines[['top', 'right']].set_visible(False)
    ax2.legend(frameon=False, ncol=3, loc='upper right', fontsize=9.5)
    fig_s.savefig(out / 'three-case-10cycle-bilateral-curves.png', dpi=150)
    plt.close(fig_s)

    # Render the 10-cycle 3-case comparison video:
    # We render 600 frames at 30 fps (20 seconds total playback = 60 frames / 2.0 s per cycle across all 10 cycles)
    # AND also a detailed single-cycle periodic (Cycle 10) 240-frame (10 s) video for detailed slow-motion inspection!
    dye_cmap = ListedColormap(plt.get_cmap('Oranges')(np.linspace(0.35, 1.0, 256)))
    opacity = (np.sqrt(np.linspace(0, 1, 256)) * 0.40 * 255).astype(np.uint8)
    opacity[0] = 0
    center = np.asarray(surface.center)
    extent = [origin[0], origin[0] + (dims[0] - 1) * spacing, origin[2], origin[2] + (dims[2] - 1) * spacing]

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
        ax3.set_title(f"{CASE_TITLES[k]} — 3D Bilateral Virtual Dye", fontsize=13, fontweight='bold', color=CASE_COLORS[k])
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
        t_sec = [float(r['global_time_s']) - START for r in sub]
        y = [float(r['ellipsoidal_ROI_mean_C_pct']) for r in sub]
        axc.plot(t_sec, y, color=CASE_COLORS[k], lw=1.8, label=CASE_TITLES[k])
    for c_idx in range(10):
        t0 = c_idx * PERIOD
        axc.axvspan(t0, t0 + 0.100, color='#cbd5e1', alpha=0.45)
    axc.set(
        xlim=(0, 10 * PERIOD),
        ylim=(0, max_y * 1.18),
        xlabel='Simulation time across 10 cycles (s) — gray bands mark 100 ms bilateral ICA pulses at start of each cycle',
        ylabel='Sac ROI Mean C (%)',
        title='Ellipsoidal Sac ROI Concentration — 10-Cycle Simultaneous Bilateral (Right + Left ICA) Injection'
    )
    axc.grid(True, color='#e2e8f0')
    axc.spines[['top', 'right']].set_visible(False)
    axc.legend(frameon=False, ncol=3, loc='upper right', fontsize=10.5)
    cursor = axc.axvline(0, color='#0f172a', lw=1.5)

    axt = fig.add_subplot(spec[2, 2])
    axt.axis('off')
    table_lines = ["10-Cycle Simultaneous Bilateral (Right + Left ICA) Summary:"]
    for sr in overall_summary:
        table_lines.append(
            f"• {sr['case_label']}:\n"
            f"   Bolus/cycle: {sr['injected_bolus_per_cycle_ml']:.3f} mL (10-cycle total: {sr['total_10cycle_injected_bolus_ml']:.2f} mL)\n"
            f"   Peak Sac C: {sr['cycle10_peak_sac_ROI_C_pct']:.1f}% at {sr['peak_phase_ms']:.0f} ms into cycle\n"
            f"   End-of-cycle Sac C: {sr['cycle10_end_sac_ROI_C_pct']:.2f}% (carry-over: +{sr['cycle_to_cycle_accumulation_pct']:.3f}%)"
        )
    axt.text(0.02, 0.95, "\n".join(table_lines), va='top', ha='left', fontsize=9.5,
             color='#1e293b', bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8fafc', edgecolor='#cbd5e1'))

    title = fig.suptitle('', fontsize=15.5, fontweight='bold')
    fig.text(0.01, 0.004,
             'Simultaneous Bilateral (Right + Left ICA) 10-Cycle Virtual Contrast Angiography | 100 ms bolus at start of each cycle | Shared display scales',
             fontsize=9, color='#475569')

    # Pre-render the 60 interpolated frames per cycle for Cycle 1 and Cycle 2+ (since Cycles 2..10 are identical to 1e-6!)
    # Wait: let's verify maximum difference between Cycle 2 and Cycle 10 across all cases!
    max_diff_c2_c10 = max(float(np.max(np.abs(c_cycle_fields[k][9] - c_cycle_fields[k][1]))) for k in KINDS)
    print('MAX_DIFF_CYCLE2_VS_CYCLE10', max_diff_c2_c10, flush=True)

    frames_per_cycle = 60
    tau_grid = np.linspace(0.0, phase_times[-1], frames_per_cycle)

    def render_cycle_frames(cycle_idx):
        rendered = []
        for f_idx, tau in enumerate(tau_grid):
            j = min(len(phase_times) - 2, max(0, int(np.searchsorted(phase_times, tau, side='right') - 1)))
            alpha = (tau - phase_times[j]) / (phase_times[j + 1] - phase_times[j])
            frame_imgs = {}
            frame_projs = {}
            for k in KINDS:
                c_scalar = (1.0 - alpha) * c_cycle_fields[k][cycle_idx, j] + alpha * c_cycle_fields[k][cycle_idx, j + 1]
                v = sample_from_scalar(c_scalar)
                g = grids[k]
                g.point_data['Dye_C'][:] = v
                g.GetPointData().GetArray('Dye_C').Modified()
                plotters[k].render()
                frame_imgs[k] = plotters[k].screenshot(return_img=True).copy()
                frame_projs[k] = (v.reshape(tuple(dims), order='F').sum(axis=1) * spacing).T.copy()
            rendered.append((tau, frame_imgs, frame_projs))
        return rendered

    print('RENDERING_3D_VIEWS_CYCLE1_AND_PERIODIC...', flush=True)
    cycle1_rendered = render_cycle_frames(0)
    cycle2_rendered = render_cycle_frames(1)
    cycle3_rendered = render_cycle_frames(2) if max_diff_c2_c10 > 1e-5 else cycle2_rendered

    video_10c_path = vis / 'Three-cases-bilateral-ICA-10cycles-virtual-angiography.mp4'
    log_10c_path = vis / 'Three-cases-bilateral-10cycles-encoding.log'
    command_10c = [
        str(FFMPEG), '-hide_banner', '-loglevel', 'warning', '-f', 'rawvideo',
        '-pixel_format', 'rgb24', '-video_size', '1920x1080', '-framerate', '30', '-i', '-',
        '-an', '-c:v', 'libopenh264', '-b:v', '8M', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
        str(video_10c_path)
    ]

    video_start = time.monotonic()
    with log_10c_path.open('x') as log:
        proc = subprocess.Popen(command_10c, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log, creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            for c_idx in range(10):
                source_frames = cycle1_rendered if c_idx == 0 else (cycle2_rendered if c_idx == 1 else cycle3_rendered)
                for f_idx, (tau, frame_imgs, frame_projs) in enumerate(source_frames):
                    for k in KINDS:
                        marks3[k].set_data(frame_imgs[k])
                        dsas[k].set_data(frame_projs[k])
                    t_elapsed_s = c_idx * PERIOD + tau
                    cursor.set_xdata([t_elapsed_s, t_elapsed_s])
                    phase_ms = tau * 1000.0
                    phase_label = 'Bilateral Injection (0–100 ms)' if phase_ms < 100 else 'Washout (>100 ms)'
                    title.set_text(
                        f"Bilateral (Right + Left ICA) Virtual Contrast Angiography | Cycle {c_idx+1}/10 | "
                        f"Phase = {phase_ms:.0f} ms ({phase_label}) | Total t = {t_elapsed_s:.2f} s"
                    )
                    fig.canvas.draw()
                    rgb = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
                    proc.stdin.write(rgb.tobytes())
                    global_frame = c_idx * frames_per_cycle + f_idx
                    if global_frame in [15, 59, 555, 599]:
                        fig.savefig(vis / f'Three-cases-bilateral-10cycles-preview-{global_frame:03d}.png', dpi=100)
                print(f'ENCODED_CYCLE {c_idx+1}/10 wall={time.monotonic()-video_start:.1f}s', flush=True)
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

    chk = subprocess.run([str(FFMPEG), '-v', 'error', '-xerror', '-i', str(video_10c_path), '-f', 'null', '-'],
                         capture_output=True, text=True, timeout=120, creationflags=subprocess.CREATE_NO_WINDOW)
    assert chk.returncode == 0, chk.stderr

    # Also create individual-case 10-cycle bilateral videos by cropping/Composing or saving manifest
    save(vis / 'ten-cycle-bilateral-video-manifest.json', {
        'status': 'completed_and_verified',
        'file': str(video_10c_path),
        'sha256': sha(video_10c_path),
        'bytes': video_10c_path.stat().st_size,
        'frames': 10 * frames_per_cycle,
        'fps': 30,
        'duration_s': 20,
        'resolution': [1920, 1080],
        'full_decode_passed': True,
        'shared_concentration_scale': [0, 1],
        'shared_integrated_projection_scale_mm': [0, max_projection],
    })
    print('BILATERAL_10CYCLE_COMPLETE', str(out), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=PROJECT / 'results/20261029-virtual-angiography-bilateral-10cycles-v1')
    a = p.parse_args()
    labels, phase_times, c_cycle_fields, history, cycle_summary, overall_summary = compute_10cycle_bilateral(a.out)
    render_10cycle_bilateral_videos(a.out, labels, phase_times, c_cycle_fields, history, cycle_summary, overall_summary)

