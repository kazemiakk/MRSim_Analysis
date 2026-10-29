# Concise-label copy of plot_scan_waveforms_noise.py (user request 2026-10-27).
"""Presentation: plot_projection_scan_waveforms.py layout with MRI-like background.

Image panels show the central plane through the vessel axis (P vs S, flow left to right,
+P down), NOT clipped to the wall, so background is visible: (1) 4D-flow MRI speed in its
measured plane (column through the mean vessel centroid; real background), (2) CFD speed
(lumen only; CFD has no background), (3) Koma speed with complex Gaussian noise added to
all four encodings at the scan's lumen SNR (noise from background air, Rayleigh; same rule
as plot_contours_with_noise.py v2/v3, seed fixed), (4) Koma (noiseless) minus CFD in the
lumen. Dashed line: Gaussian fit of the CFD wall (reference only). Bilinear 0.25 mm display
interpolation. Waveform row unchanged (v3 limits). Display only; registered numbers unchanged.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402

E060 = Path("D:/MRSim/outputs/scanner_geometry_e060")
sys.path.insert(0, str(E060/"code"))
import plot_projection_scan_waveforms as wf  # noqa: E402
import plot_contours_with_noise as nz  # noqa: E402
ws, sm, tri = wf.ws, wf.sm, wf.tri
SEED = nz.SEED


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def resample(img, z_rows, p_cols):
    """(rows=S, cols=P) native -> fine (SF, PF) bilinear, NaN kept as NaN."""
    rows = np.interp(sm.SF, z_rows, np.arange(z_rows.size), left=np.nan, right=np.nan)
    cols = np.interp(sm.PF, p_cols, np.arange(p_cols.size))
    R, C = np.meshgrid(rows, cols, indexing="ij")
    valid = np.isfinite(img).astype(float)
    num = ndimage.map_coordinates(np.nan_to_num(img), [np.nan_to_num(R), C], order=1, mode="nearest")
    den = ndimage.map_coordinates(valid, [np.nan_to_num(R), C], order=1, mode="nearest")
    out = np.where(den > .5, num/np.maximum(den, 1e-9), np.nan)
    return np.where(np.isfinite(R), out, np.nan)


def fill_nearest(img):
    """v3: fill NaN voxels from the nearest valid voxel so the wall-clipped display is smooth."""
    bad = ~np.isfinite(img)
    if not bad.any() or bad.all():
        return img
    idx = ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True)
    return img[tuple(idx)]


def main(target):
    target = Path(target)
    if target.exists():
        raise FileExistsError(target)
    rep = json.loads((ws.E059/"measured_v2/report.json").read_text())
    z = np.load(ws.PRIVATE)
    v, mag, pos = z["velocity_cm_s"]/100.0, z["magnitude"], z["slice_positions_mm"]
    throat = rep["throat_slice"]
    phase = int(np.argmax(rep["segmentation"]["0.25"]["slices"][throat]["flow_ml_s"]))
    _, _, _, snr_scan, _ = nz.scan_noise_and_panels()
    proj, zs, sphase, strig, shift, cy = ws.scan_projection()
    m = mag.mean(axis=1); thr = .25*np.percentile(m, 99.5)
    sign = rep["segmentation"]["0.25"]["vendor_to_forward_sign"]
    cxs = []
    for s in range(v.shape[0]):
        lab, n = ndimage.label(m[s] >= thr)
        flows = [abs((sign*v[s, :, :, :, 2][:, lab == c]).sum(axis=1).mean()) for c in range(1, n+1)]
        cxs.append(ndimage.center_of_mass(lab == (1+int(np.argmax(flows))))[1])
    cx = int(round(np.mean(cxs)))
    scan_plane = np.linalg.norm(v[:, phase, :, cx, :], axis=-1)               # (slice, y)
    p_scan = (np.arange(64)-cy)*1.25
    zrel = pos-pos[throat]
    report = json.loads((E060/"analysis_v1/report.json").read_text()); b = report["peak_bin"]-1
    images, vencs = tri.arms()["E060_VENC3"]
    with h5py.File(E060/"references_v2/references.h5") as f:
        ref, frac, zz, pfrac = f["fft_adc"][b], f["fraction_fft"][:], f["partition_z_m"][:]*1e3, f["plane_fraction"][:]
    lum = frac > 0
    rc = int(np.argmax(pfrac[tri.an.THROAT_K].sum(axis=0)))
    pc = ndimage.center_of_mass(pfrac[tri.an.THROAT_K])[0]
    p_sim = (np.arange(64)-pc)*1.25
    slab = (np.abs(zz) <= 30)[:, None, None]
    M = float(np.median(np.abs(images["REF"][b])[lum & slab]))
    sigma = M/snr_scan
    rng = np.random.default_rng(SEED)
    noisy = {e: images[e][b]+sigma*(rng.standard_normal(images[e][b].shape)+1j*rng.standard_normal(images[e][b].shape)) for e in tri.an.ENC}
    def speed(ims):
        return np.linalg.norm(np.stack([np.angle(ims[e]*ims["REF"].conj())*(vencs[i]/np.pi) for i, e in enumerate(tri.an.ENC[1:])], axis=-1), axis=-1)
    clean = {e: images[e][b] for e in tri.an.ENC}
    k_clean, k_noisy = speed(clean)[:, :, rc], speed(noisy)[:, :, rc]
    cfd = np.where(lum[:, :, rc], np.linalg.norm(np.nan_to_num(ref), axis=-1)[:, :, rc], np.nan)
    diff = np.where(lum[:, :, rc], k_clean-cfd, np.nan)
    rS = sm.wall(zz)
    inside = np.abs(sm.PF)[None, :] <= rS[:, None]   # v2: CFD and difference clipped to the Gaussian wall
    panels = [(resample(scan_plane, zrel, p_scan), "4D-flow MRI", "viridis"),
              (np.where(inside, resample(fill_nearest(cfd), zz, p_sim), np.nan), "CFD", "viridis"),
              (resample(k_noisy, zz, p_sim), "Koma", "viridis"),
              (np.where(inside, resample(fill_nearest(diff), zz, p_sim), np.nan), "Koma − CFD", "RdBu_r")]
    vmax = 3.0
    extent = [sm.PF[0]-sm.STEP/2, sm.PF[-1]+sm.STEP/2, sm.SF[0]-sm.STEP/2, sm.SF[-1]+sm.STEP/2]
    series = dict(scan=wf.scan_series()); series["cfd"], series["koma"] = wf.sim_series()
    shifts = {k: wf.PEAK_AT-int(np.argmax(f)) for k, (f, _) in series.items()}
    t = np.arange(30)*wf.DT
    stage = target.with_name(target.name+".building"); stage.mkdir()
    fig = plt.figure(figsize=(17, 16), layout="constrained")
    gs = fig.add_gridspec(2, 1, height_ratios=[3.3, 1])
    top = gs[0].subgridspec(2, 2); bottom = gs[1].subgridspec(1, 4)
    dmax = float(np.nanmax(np.abs(panels[3][0])))
    for i, (data, lab, cm) in enumerate(panels):
        ax = fig.add_subplot(top[i//2, i % 2])
        cmap = plt.get_cmap(cm).copy(); cmap.set_bad("#eeeeee")
        lim = (-dmax, dmax) if i == 3 else (0, vmax)
        im = sm.show(ax, data, extent, cmap=cmap, vmin=lim[0], vmax=lim[1])
        if i in (1, 3):   # v4: solid dark wall over the clipped edge (v3: white dashes looked like specks)
            ax.plot(sm.SF, rS, color="#3a3936", lw=2.0); ax.plot(sm.SF, -rS, color="#3a3936", lw=2.0)
        else:
            ax.plot(sm.SF, rS, color="white", lw=.8, ls=(0, (4, 3))); ax.plot(sm.SF, -rS, color="white", lw=.8, ls=(0, (4, 3)))
        ax.axvline(0, color="#999999", lw=.6, ls=(0, (3, 3))); ax.set_facecolor("#eeeeee")
        ax.set_title(lab, fontsize=15); ax.set_xlabel("z (mm)", fontsize=13); ax.set_ylabel("y (mm)", fontsize=13); ax.tick_params(labelsize=11)
        fig.colorbar(im, ax=ax, label="speed (m/s)" if i < 3 else "Δ speed (m/s)", shrink=.8)
    labels = dict(scan="MRI", cfd="CFD", koma="Koma")
    titles = ("Throat flow", "u, throat", "v, throat", "w, throat")
    for j in range(4):
        ax = fig.add_subplot(bottom[0, j])
        for key, (flow, uvw) in series.items():
            y = flow if j == 0 else uvw[:, j-1]
            ax.plot(t, wf.shifted(y, shifts[key]), color=wf.COLORS[key], lw=2, marker="o", ms=3, label=labels[key])
        ax.axhline(0, color="#888888", lw=.6)
        if j in (1, 2):
            ax.set_ylim(-0.1, 0.1)
        ax.set_title(titles[j], fontsize=15); ax.set_xlabel("time (ms)", fontsize=12)
        ax.set_ylabel("mL/s" if j == 0 else "m/s", fontsize=13); ax.grid(alpha=.25); ax.tick_params(labelsize=11)
        if j == 0:
            ax.legend(fontsize=12, frameon=False)
    fig.suptitle("Phantom: 4D-flow MRI vs CFD vs Koma", fontsize=19)
    fig.supxlabel("Central plane, peak systole; dashed = vessel wall. Curves rigidly shifted to align flow peaks. Display noise matched to the scan.", fontsize=12)
    fig.savefig(stage/"phantom_scan_cfd_koma_concise.png", dpi=150); plt.close(fig)
    (stage/"MANIFEST.json").write_text(json.dumps(dict(created_utc=datetime.now(timezone.utc).isoformat(), status="PRESENTATION_DERIVATIVE_ONLY",
        script_sha256=sha(__file__), scan_lumen_snr=snr_scan, koma_noise_sigma=sigma, seed=SEED, scan_plane_column=cx, sim_plane_R_index=int(rc),
        private_input_used=str(ws.PRIVATE), outputs={p.name: sha(p) for p in sorted(stage.glob("*.png"))}), indent=1)+"\n", encoding="utf-8")
    stage.rename(target)
    print("DONE", snr_scan, sigma)


if __name__ == "__main__":
    main(sys.argv[1])
