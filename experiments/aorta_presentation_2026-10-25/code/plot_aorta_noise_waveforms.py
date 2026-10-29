"""Aorta presentation figure in the style of the phantom figure (no 4D-flow scan exists).

Source: E-027 results_v2 (original 1M-spin / 8-bin Koma vs original aortic CFD; hash-checked
via its MANIFEST) and the Koma reconstruction pc_reconstruction.h5 (velocity + REF magnitude).
v2 top 2x2: E-028-style XY support-weighted projections through MRI Z (core15 x fraction weights;
v1 used one axial plane, which showed only part of the arch): CFD (lumen), Koma noiseless (lumen), Koma + synthetic noise (unclipped, background
visible), Koma minus CFD (lumen). Display: bilinear 0.5 mm interpolation; lumen panels clipped
where the interpolated CFD voxel fraction >= 0.5 (smooth boundary, nearest-filled inside).
Noise (display only): no aorta scan, so the phantom 4D-flow lumen SNR (84, E-060 noise figure)
is borrowed; complex images rebuilt as REF = |M|, encoded = |M| exp(i v pi/VENC) (VENC 2.5 m/s),
complex Gaussian noise sigma = median lumen |M| / 84 added to all four, velocity recomputed.
Bottom row: section X+000_C1 (plane normal to MRI X): net flow (E-027 section_series) and
section-mean Vx (through-plane), Vy, Vz for CFD and Koma over the 8 bins.
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

E027 = Path("D:/MRSim/MRSIM/MRSimTracks/outputs/aorta_hemodynamic_comparison/e027_v1/results_v2")
PC = Path("D:/MRSim/MRSIM/MRSimTracks/outputs/aorta_dp2_mri_pipeline/koma/multibin_8x4mm/simulation_motion_1m_8bin_4mm/pc_reconstruction.h5")
SNR, SEED, VENC, STEP = 84.4, 20261025, 2.5, 0.5
SECTION = "X+000_C1"
COLORS = dict(cfd="#0b0b0b", koma="#2a78d6")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fill_nearest(img):
    bad = ~np.isfinite(img)
    if not bad.any() or bad.all():
        return img
    return img[tuple(ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True))]


def fine(img, xs, ys, xf, yf):
    R, C = np.meshgrid(np.interp(yf, ys, np.arange(ys.size)), np.interp(xf, xs, np.arange(xs.size)), indexing="ij")
    return ndimage.map_coordinates(img, [R, C], order=1, mode="nearest")


def main(target):
    target = Path(target)
    if target.exists():
        raise FileExistsError(target)
    man = json.loads((E027/"MANIFEST.json").read_text())
    for name in ("report.json", "matched_reference.h5"):
        if sha(E027/name) != man[name]["sha256"]:
            raise ValueError("E-027 source changed: "+name)
    rep = json.loads((E027/"report.json").read_text())
    with h5py.File(E027/"matched_reference.h5") as f:
        ref, koma, frac, core = f["adc_time_average"][:], f["Koma"][:], f["fraction"][:], f["masks/core15"][:].astype(bool)
        sec = f["sections"][SECTION][:].astype(bool)
    with h5py.File(PC) as f:
        vel = np.moveaxis(f["velocity"][:], 0, -1)            # (bin, Z, Y, X, comp)
        mag = f["reference_magnitude"][:]
    if np.nanmax(np.abs(vel[:, core]-koma[:, core])) > 1e-4:
        raise ValueError("pc_reconstruction velocity is not the E-027 Koma field (axis mismatch)")
    nb, nz, ny, nx = mag.shape
    fov = rep["grid"]["fov"]
    xs = (np.arange(nx)-nx/2)*fov[0]/nx*1000; ys = (np.arange(ny)-ny/2)*fov[1]/ny*1000
    zsl = int(np.argmax(frac.sum(axis=(1, 2))))
    xf = np.arange(xs[0], xs[-1]+STEP/2, STEP); yf = np.arange(ys[0], ys[-1]+STEP/2, STEP)
    lumen_fine = fine(frac[zsl], xs, ys, xf, yf) >= .5
    peak = int(np.argmax([np.linalg.norm(ref[b][core], axis=-1).mean() for b in range(nb)]))
    b = peak
    M = float(np.median(mag[b][core]))
    sigma = M/SNR
    rng = np.random.default_rng(SEED)
    def noisy(z):
        return z+sigma*(rng.standard_normal(z.shape)+1j*rng.standard_normal(z.shape))
    REF = noisy(mag[b].astype(complex))
    vn = np.stack([np.angle(noisy(mag[b]*np.exp(1j*vel[b][..., c]*np.pi/VENC))*REF.conj())*VENC/np.pi for c in range(3)], axis=-1)
    wts = np.where(core, frac, 0)
    def proj(v):
        den = wts.sum(axis=0)
        return np.divide((np.nan_to_num(v)*wts).sum(axis=0), den, out=np.full(den.shape, np.nan), where=den > 0)
    sp_cfd = proj(np.linalg.norm(np.nan_to_num(ref[b]), axis=-1))
    sp_k = proj(np.linalg.norm(koma[b], axis=-1))
    # v6: noisy Koma uses the SAME support-weighted projection and lumen clip as the other panels
    # (v3-v5 showed a strongest-signal-voxel background that looked unrealistic).
    sp_kn = proj(np.linalg.norm(vn, axis=-1))
    lumen_fine = fine((wts.sum(axis=0) > 0).astype(float), xs, ys, xf, yf) >= .5
    panels = [(np.where(lumen_fine, fine(fill_nearest(sp_cfd), xs, ys, xf, yf), np.nan), f"CFD | bin {b+1}/{nb}", "viridis"),
              (np.where(lumen_fine, fine(fill_nearest(sp_k), xs, ys, xf, yf), np.nan), f"Koma, noiseless | bin {b+1}/{nb}", "viridis"),
              (np.where(lumen_fine, fine(fill_nearest(sp_kn), xs, ys, xf, yf), np.nan), f"Koma + synthetic noise (SNR {SNR:.0f}) | bin {b+1}/{nb}", "viridis"),
              (np.where(lumen_fine, fine(fill_nearest(sp_k-sp_cfd), xs, ys, xf, yf), np.nan), "Koma (noiseless) minus CFD", "RdBu_r")]
    vmax = float(np.nanmax(panels[0][0]))
    dmax = float(np.nanmax(np.abs(panels[3][0])))
    # v6 orientation: rotate 90 deg counter-clockwise then mirror left-right = transpose:
    # horizontal = MRI Y, vertical = MRI X (arch on top, branches pointing up).
    extent = [yf[0]-STEP/2, yf[-1]+STEP/2, xf[0]-STEP/2, xf[-1]+STEP/2]
    # Section waveforms.
    ix = int(np.argmin(abs(xs-[r for r in rep["section_series"] if r["section"] == SECTION][0]["x_m"]*1000)))
    flows = {m: [r["net_ml_s"] for r in sorted([r for r in rep["section_series"] if r["section"] == SECTION and r["method"] == ("CFD" if m == "cfd" else "Koma")], key=lambda r: r["bin"])] for m in ("cfd", "koma")}
    w = np.where(sec, frac[:, :, ix], 0)
    comps = {m: np.array([(np.nan_to_num(field[bb][:, :, ix])*w[..., None]).sum(axis=(0, 1))/w.sum() for bb in range(nb)])
             for m, field in (("cfd", ref), ("koma", koma))}
    # v6: forward flow positive (flow runs along -X here) and rigid circular shift so the cycle
    # starts at the minimum-flow bin (systole first; no time warping, no invented phases).
    sgn = -1.0 if np.mean(flows["cfd"]) < 0 else 1.0
    shift = -int(np.argmin(sgn*np.asarray(flows["cfd"])))
    flows = {m: np.roll(sgn*np.asarray(v), shift) for m, v in flows.items()}
    comps = {m: np.roll(v*np.array([sgn, 1, 1]), shift, axis=0) for m, v in comps.items()}
    bins_order = np.roll(np.arange(1, nb+1), shift)
    t = (np.arange(nb)+.5)*float(rep["period_s"])/nb*1000
    out = target.with_name(target.name+".building"); out.mkdir(parents=True)
    fig = plt.figure(figsize=(18, 12), layout="constrained")
    gs = fig.add_gridspec(2, 1, height_ratios=[2.0, 1])
    top = gs[0].subgridspec(1, 4); bottom = gs[1].subgridspec(1, 4)   # v7: one row of tall panels (v6 2x2 left gaps)
    for i, (data, lab, cm) in enumerate(panels):
        ax = fig.add_subplot(top[0, i])
        cmap = plt.get_cmap(cm).copy(); cmap.set_bad("#eeeeee")
        lim = (-dmax, dmax) if i == 3 else (0, vmax)
        im = ax.imshow(data.T, origin="lower", extent=extent, cmap=cmap, vmin=lim[0], vmax=lim[1], interpolation="nearest", aspect="equal")
        ax.contour(yf, xf, lumen_fine.T.astype(float), levels=[.5], colors="#3a3936", linewidths=1.6)
        ax.axhline(xs[ix], color="#eb6834", lw=1, ls=(0, (3, 3)))
        ax.set_facecolor("#eeeeee"); ax.set_title(lab, fontsize=10); ax.set(xlabel="MRI Y (mm)", ylabel="MRI X (mm)")
        fig.colorbar(im, ax=ax, label="speed (m/s)" if i < 3 else "m/s", shrink=.85)
    titles = (f"Forward flow, section {SECTION}", "Through-plane velocity (forward +), section mean", "Vy (in-plane), section mean", "Vz (in-plane), section mean")
    for j in range(4):
        ax = fig.add_subplot(bottom[0, j])
        for m in ("cfd", "koma"):
            y = flows[m] if j == 0 else comps[m][:, j-1]
            ax.plot(t, y, color=COLORS[m], lw=2, marker="o", ms=4, label="CFD" if m == "cfd" else "Koma")
        ax.axhline(0, color="#888888", lw=.6)
        if j in (2, 3):
            ax.set_ylim(-0.1, 0.1)
        for tt, bn, yy in zip(t, bins_order, flows["cfd"] if j == 0 else comps["cfd"][:, j-1]):
            if j == 0:
                ax.annotate(f"b{bn}", (tt, yy), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=6, color="#52514e")
        ax.set_title(titles[j], fontsize=10); ax.set_xlabel("time from cycle start, min-flow bin first (ms)", fontsize=8)
        ax.set_ylabel("mL/s" if j == 0 else "m/s", fontsize=9); ax.grid(alpha=.25); ax.tick_params(labelsize=8)
        if j == 0:
            ax.legend(fontsize=8, frameon=False)
    fig.suptitle(f"Aorta (E-027, 1M spins, 8 bins): CFD vs Koma at the peak bin and over the cycle; XY support-weighted projections", fontsize=13)
    fig.supxlabel("Images: projections through MRI Z (fixed 15 % core, fraction weights, as E-028), rotated 90° CCW then mirrored L-R (MRI Y horizontal, MRI X vertical); "
                  "all panels clipped to the smoothed projected support. Orange: section X+000_C1.\nNoise is synthetic, display only: no aorta 4D-flow scan exists, so the "
                  "phantom scan lumen SNR is borrowed. Curves: forward-positive section flow (E-027) and section-mean components; rigid circular shift only (8 bins; lines are guides).", fontsize=8)
    fig.savefig(out/"aorta_cfd_koma_with_waveforms_noise.png", dpi=150); plt.close(fig)
    (out/"MANIFEST.json").write_text(json.dumps(dict(created_utc=datetime.now(timezone.utc).isoformat(), status="PRESENTATION_DERIVATIVE_ONLY",
        script_sha256=sha(__file__), e027_report_sha256=sha(E027/"report.json"), pc_reconstruction_sha256=sha(PC), z_index=zsl, peak_bin=b+1,
        section=SECTION, section_x_index=ix, noise_sigma=sigma, snr=SNR, seed=SEED,
        outputs={p.name: sha(p) for p in sorted(out.glob("*.png"))}), indent=1)+"\n", encoding="utf-8")
    out.rename(target)
    print("DONE", zsl, b+1, ix)


if __name__ == "__main__":
    main(sys.argv[1])
