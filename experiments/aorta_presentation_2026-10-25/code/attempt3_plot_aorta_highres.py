"""v8 aorta presentation figure: native-resolution CFD layer and 20-bin CFD waveforms.

Why v8: v7 drew every panel from the E-027 Koma grid (72x32x32, 3.9 x 4 x 4 mm, 8 cardiac bins),
so the vessel outline was a 4 mm staircase and each curve had 8 points. The CFD source is 2.34 M
nodes x 72 frames (10 ms), so the CFD layer can be shown at full resolution. Koma cannot: its
k-space was acquired at 4 mm and 8 bins, so its panels stay 4 mm data (bilinear display only).

CFD layer (E-027 source point cloud, read-only, hashes checked against the E-027 report):
- outline: MRI-XY silhouette of all CFD nodes rasterised at 0.25 mm (0.5 mm closing, holes
  < 2 mm^2 filled, 0.25 mm Gaussian smoothing, 0.5 level)
- image: speed of the CFD velocity averaged over the Koma peak bin's ADC-time window (E-027
  frame_weights), IDW k=8 on a 0.5 x 0.5 x 1 mm grid, points within 0.75 mm of a node, mean
  along MRI Z (volume weighting)
- section X+000_C1: 0.25 mm plane, IDW k=8, points within 0.75 mm of a node (prototype: flux
  converged, 0.75 vs 1.5 mm within 0.3 %); 20 equal bins (box windows, 12 samples each) and the
  8 E-027 ADC windows
Koma panels: E-027 / pc_reconstruction 4 mm data as in v7, clipped to the CFD outline.
Noise (display only): as v7 (borrowed phantom-scan SNR 84; seed 20261025).
Waveforms: rigid circular shift only, cycle starts at the start of Koma bin 6 (minimum flow).
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from scripts.aorta_hemodynamic_reference import AortaGrid, temporal_weights  # noqa: E402  (frozen, read-only)

E027 = Path("D:/MRSim/MRSIM/MRSimTracks/outputs/aorta_hemodynamic_comparison/e027_v1/results_v2")
PC = Path("D:/MRSim/MRSIM/MRSimTracks/outputs/aorta_dp2_mri_pipeline/koma/multibin_8x4mm/simulation_motion_1m_8bin_4mm/pc_reconstruction.h5")
CLOUD = Path("D:/MRSim/MRSIM/external_inputs/aorta_pointcloud/processed/dp2")
SNR, SEED, VENC = 84.4, 20261025, 2.5
DISP, IMG, IMGZ, PLANE, R_IN = 0.25e-3, 0.5e-3, 1.0e-3, 0.25e-3, 0.75e-3
NB20, SUB20 = 20, 12
SECTION = "X+000_C1"
COLORS = dict(cfd20="#0b0b0b", match="#8a8984", koma="#2a78d6")   # attempt 1 keyed "cfd" (KeyError at plotting)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def fill_nearest(img):
    bad = ~np.isfinite(img)
    if not bad.any() or bad.all():
        return img
    return img[tuple(ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True))]


def fill_smooth(img, sigma=1.0):
    """v9: fill NaNs outward by normalised Gaussian convolution (v8 nearest fill left straight streaks)."""
    out = img.copy(); ok = np.isfinite(out)
    if not ok.any():
        return out
    while not ok.all():
        num = ndimage.gaussian_filter(np.where(ok, out, 0), sigma); den = ndimage.gaussian_filter(ok.astype(float), sigma)
        new = ~ok & (den > 1e-3)
        if not new.any():
            sigma *= 2
            continue
        out[new] = num[new]/den[new]; ok |= new
    return out


def resample(img, xs, ys, xf, yf):
    """Bilinear resample of a (Y, X) image on axes (ys, xs) to (yf, xf)."""
    R, C = np.meshgrid(np.interp(yf, ys, np.arange(ys.size)), np.interp(xf, xs, np.arange(xs.size)), indexing="ij")
    return ndimage.map_coordinates(img, [R, C], order=1, mode="nearest")


def idw(tree, pts):
    d, n = tree.query(pts, k=8, workers=-1)
    w = 1/np.maximum(d, 1e-9)**2
    return d[:, 0], n, (w/w.sum(axis=1, keepdims=True)).astype(np.float32)


def main(target):
    t_start = time.time()
    target = Path(target)
    if target.exists():
        raise FileExistsError(target)
    man = json.loads((E027/"MANIFEST.json").read_text())
    for name in ("report.json", "matched_reference.h5"):
        if sha(E027/name) != man[name]["sha256"]:
            raise ValueError("E-027 source changed: "+name)
    rep = json.loads((E027/"report.json").read_text())
    want = {Path(r["path"]).name: r["sha256"] for r in rep["inputs"]}
    for name in ("point_coordinates.npy", "velocities.npy"):
        if sha(CLOUD/name) != want[name]:
            raise ValueError("CFD point cloud changed: "+name)
    with h5py.File(E027/"matched_reference.h5") as f:
        ref, koma, frac, core = f["adc_time_average"][:], f["Koma"][:], f["fraction"][:], f["masks/core15"][:].astype(bool)
        sec = f["sections"][SECTION][:].astype(bool)
        fw = f["frame_weights"][:8]                                   # ADC-time frame weights, bins 1-8
    with h5py.File(PC) as f:
        vel = np.moveaxis(f["velocity"][:], 0, -1)                   # (bin, Z, Y, X, comp)
        mag = f["reference_magnitude"][:]
    if np.nanmax(np.abs(vel[:, core]-koma[:, core])) > 1e-4:
        raise ValueError("pc_reconstruction velocity is not the E-027 Koma field (axis mismatch)")
    if not np.allclose(fw.sum(axis=1), 1):
        raise ValueError("frame weights do not sum to one")
    nb, nz, ny, nx = mag.shape
    rr = float(rep["period_s"])
    grid = AortaGrid()
    order, cen = np.array(grid.order), np.array(grid.center)
    gx, gy, gz = grid.axes()                                          # metres, MRI X/Y/Z voxel centres
    xs, ys = gx*1000, gy*1000

    # ---- Koma (4 mm) panels, as v7 ----
    b = int(np.argmax([np.linalg.norm(ref[k][core], axis=-1).mean() for k in range(nb)]))
    M = float(np.median(mag[b][core])); sigma = M/SNR
    rng = np.random.default_rng(SEED)
    def noisy(z):
        return z+sigma*(rng.standard_normal(z.shape)+1j*rng.standard_normal(z.shape))
    REF = noisy(mag[b].astype(complex))
    vn = np.stack([np.angle(noisy(mag[b]*np.exp(1j*vel[b][..., c]*np.pi/VENC))*REF.conj())*VENC/np.pi for c in range(3)], axis=-1)
    wts = np.where(core, frac, 0)
    def proj(v):
        den = wts.sum(axis=0)
        return np.divide((np.nan_to_num(v)*wts).sum(axis=0), den, out=np.full(den.shape, np.nan), where=den > 0)
    sp_k, sp_kn = proj(np.linalg.norm(koma[b], axis=-1)), proj(np.linalg.norm(vn, axis=-1))
    sp_match = proj(np.linalg.norm(np.nan_to_num(ref[b]), axis=-1))

    # ---- CFD point cloud ----
    src = np.load(CLOUD/"point_coordinates.npy", allow_pickle=False).astype(np.float64)
    vel_src = np.load(CLOUD/"velocities.npy", mmap_mode="r", allow_pickle=False)   # (N, 3, 72)
    nodes = src[:, order]-cen[order]                                  # MRI coordinates (m)
    tree = cKDTree(nodes)
    print("cloud", nodes.shape, round(time.time()-t_start, 1), flush=True)

    # Outline: silhouette of all nodes in MRI XY at DISP.
    xf = np.arange(xs[0], xs[-1]+1e-9, DISP*1000); yf = np.arange(ys[0], ys[-1]+1e-9, DISP*1000)
    occ = np.zeros((yf.size, xf.size), bool)
    iy = np.round((nodes[:, 1]*1000-yf[0])/(DISP*1000)).astype(int); ix_ = np.round((nodes[:, 0]*1000-xf[0])/(DISP*1000)).astype(int)
    ok = (iy >= 0) & (iy < yf.size) & (ix_ >= 0) & (ix_ < xf.size)
    occ[iy[ok], ix_[ok]] = True
    disk = np.hypot(*np.mgrid[-2:3, -2:3]) <= 2
    occ = ndimage.binary_closing(occ, structure=disk, iterations=1)
    lab, n = ndimage.label(~occ)
    sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n+1))*(DISP*1000)**2
    occ |= np.isin(lab, 1+np.flatnonzero(sizes < 2.0))
    smooth = ndimage.gaussian_filter(occ.astype(float), 1.0)
    outline = smooth >= .5

    # CFD speed image at the Koma peak bin's ADC window, projected along MRI Z.
    V = np.empty((len(nodes), 3), np.float32)
    for s in range(0, len(nodes), 200000):
        V[s:s+200000] = np.einsum("ncf,f->nc", np.asarray(vel_src[s:s+200000], dtype=np.float32), fw[b].astype(np.float32))[:, order]
    lo, hi = nodes.min(axis=0)-1e-3, nodes.max(axis=0)+1e-3
    ax3 = [np.arange(lo[0], hi[0], IMG), np.arange(lo[1], hi[1], IMG), np.arange(lo[2], hi[2], IMGZ)]
    P = np.stack(np.meshgrid(*ax3, indexing="ij"), -1).reshape(-1, 3)
    d1, _ = tree.query(P, k=1, workers=-1, distance_upper_bound=R_IN*1.01)
    inside = d1 <= R_IN
    _, nq, wq = idw(tree, P[inside])
    spd = np.zeros(len(P), np.float32)
    spd[inside] = np.linalg.norm(np.einsum("qk,qkc->qc", wq, V[nq]), axis=1)
    # v9: same projection support as the Koma panels: only points whose 4 mm voxel is in core15
    # (v8 averaged every lumen point, incl. the boundary layer, so CFD read ~20 % lower than Koma).
    kv = [np.clip(np.round((P[:, a]-g[0])/(g[1]-g[0])).astype(int), 0, g.size-1) for a, g in enumerate((gx, gy, gz))]
    use = inside & core[kv[2], kv[1], kv[0]]
    shp = tuple(a.size for a in ax3)
    cnt = use.reshape(shp).sum(axis=2); tot = np.where(use, spd, 0).reshape(shp).sum(axis=2)
    cfd_img = np.divide(tot, cnt, out=np.full(cnt.shape, np.nan), where=cnt > 0).T     # (Y, X)
    cfd_disp = resample(fill_smooth(cfd_img, 2.0), ax3[0]*1000, ax3[1]*1000, xf, yf)
    print("image", int(inside.sum()), "points", round(time.time()-t_start, 1), flush=True)

    # Section X+000_C1 on a fine plane, all 72 frames.
    srow = [r for r in rep["section_series"] if r["section"] == SECTION]
    xsec = float(srow[0]["x_m"]); ix = int(np.argmin(abs(gx-xsec)))
    zz, yy = np.where(sec)
    py = np.arange(gy[yy].min()-.006, gy[yy].max()+.006, PLANE); pz = np.arange(gz[zz].min()-.006, gz[zz].max()+.006, PLANE)
    YY, ZZ = np.meshgrid(py, pz, indexing="ij")
    Q = np.stack([np.full(YY.size, xsec), YY.ravel(), ZZ.ravel()], -1)
    dq, nq2, wq2 = idw(tree, Q)
    inplane = (dq <= R_IN).reshape(YY.shape)
    lab2, n2 = ndimage.label(inplane)
    jy = np.clip(np.searchsorted(gy, YY+np.diff(gy)[0]/2)-1, 0, ny-1); jz = np.clip(np.searchsorted(gz, ZZ+np.diff(gz)[0]/2)-1, 0, nz-1)
    hit = np.bincount(lab2[sec[jz, jy] & inplane], minlength=n2+1); hit[0] = 0
    comp = (lab2 == int(np.argmax(hit))).ravel()
    used = np.unique(nq2[comp])
    vplane = np.asarray(vel_src[used], dtype=np.float32)[:, order, :]                 # (n, 3, 72) MRI comps
    vq = np.einsum("qk,qkcf->qcf", wq2[comp], vplane[np.searchsorted(used, nq2[comp])])
    area = comp.sum()*PLANE**2
    flux_f = vq[:, 0, :].sum(axis=0)*PLANE**2*1e6                                      # mL/s per source frame
    mean_f = vq.mean(axis=0).T                                                         # (72, 3) m/s
    W20 = np.stack([temporal_weights((j+(np.arange(SUB20)+.5)/SUB20)*rr/NB20, period_s=rr) for j in range(NB20)])
    cfd20 = dict(flow=W20 @ flux_f, comps=W20 @ mean_f)
    cfd8 = dict(flow=fw @ flux_f, comps=fw @ mean_f)
    e027 = {m: np.array([r["net_ml_s"] for r in sorted([r for r in srow if r["method"] == m], key=lambda r: r["bin"])]) for m in ("CFD", "Koma")}
    w4 = np.where(sec, frac[:, :, ix], 0)
    comps4 = {m: np.array([(np.nan_to_num(field[k][:, :, ix])*w4[..., None]).sum(axis=(0, 1))/w4.sum() for k in range(nb)])
              for m, field in (("match", ref), ("koma", koma))}
    print("section area mm2", round(area*1e6, 1), "fine CFD / E-027 CFD (ADC windows)", np.round(cfd8["flow"]/e027["CFD"], 3), flush=True)

    # Forward-positive, rigid circular shift: cycle starts at the start of the min-flow Koma bin.
    sgn = -1.0 if e027["CFD"].mean() < 0 else 1.0
    kmin = int(np.argmin(sgn*e027["CFD"])); t0 = kmin*rr/nb
    tau8 = ((np.arange(nb)+.5)*rr/nb-t0) % rr*1000
    tau20 = ((np.arange(NB20)+.5)*rr/NB20-t0) % rr*1000
    o8, o20 = np.argsort(tau8), np.argsort(tau20)
    sgnc = np.array([sgn, 1, 1])
    series = {"cfd20": (tau20[o20], sgn*cfd20["flow"][o20], cfd20["comps"][o20]*sgnc),
              "match": (tau8[o8], sgn*e027["CFD"][o8], comps4["match"][o8]*sgnc),
              "koma": (tau8[o8], sgn*e027["Koma"][o8], comps4["koma"][o8]*sgnc)}

    # ---- figure ----
    clip = lambda img: np.where(outline, img, np.nan)  # noqa: E731
    km = lambda img: clip(resample(fill_smooth(img), xs, ys, xf, yf))  # noqa: E731
    panels = [(clip(cfd_disp), f"CFD, native (0.5 mm) | bin {b+1}/{nb}", "viridis"),
              (km(sp_k), f"Koma, 4 mm | bin {b+1}/{nb}", "viridis"),
              (km(sp_kn), f"Koma + noise (SNR {SNR:.0f})", "viridis"),
              (km(sp_k-sp_match), "Koma − CFD (both 4 mm)", "RdBu_r")]
    vmax = float(max(np.nanpercentile(panels[0][0], 99.8), np.nanpercentile(panels[1][0], 99.8)))
    dmax = float(np.nanpercentile(np.abs(panels[3][0]), 99.5))
    extent = [yf[0]-DISP*500, yf[-1]+DISP*500, xf[0]-DISP*500, xf[-1]+DISP*500]
    out = target.with_name(target.name+".building"); out.mkdir(parents=True)
    fig = plt.figure(figsize=(18, 12.5), layout="constrained")
    gs = fig.add_gridspec(2, 1, height_ratios=[2.0, 1])
    top = gs[0].subgridspec(1, 4); bottom = gs[1].subgridspec(1, 4)
    for i, (data, lab, cm) in enumerate(panels):
        ax = fig.add_subplot(top[0, i])
        cmap = plt.get_cmap(cm).copy(); cmap.set_bad("#eeeeee")
        lim = (-dmax, dmax) if i == 3 else (0, vmax)
        # Orientation as v7: transpose = rotate 90 deg CCW then mirror L-R (MRI Y horizontal, MRI X vertical).
        im = ax.imshow(data.T, origin="lower", extent=extent, cmap=cmap, vmin=lim[0], vmax=lim[1], interpolation="nearest", aspect="equal")
        ax.contour(yf, xf, smooth.T, levels=[.5], colors="#3a3936", linewidths=1.2)
        ax.axhline(xsec*1000, color="#eb6834", lw=1, ls=(0, (3, 3)))
        ax.set_facecolor("#eeeeee"); ax.set_title(lab, fontsize=10); ax.set(xlabel="MRI Y (mm)", ylabel="MRI X (mm)")
        fig.colorbar(im, ax=ax, label="speed (m/s)" if i < 3 else "m/s", shrink=.85, extend="max" if i < 3 else "both")
    titles = (f"Forward flow, section {SECTION}", "Through-plane velocity (forward +), section mean", "Vy (in-plane), section mean", "Vz (in-plane), section mean")
    labels = {"cfd20": f"CFD, native, {NB20} bins ({rr/NB20*1000:.0f} ms)", "match": "CFD sampled like Koma (4 mm, 8 windows)",
              "koma": f"Koma, 8 bins ({rr/nb*1000:.0f} ms windows)"}
    for j in range(4):
        ax = fig.add_subplot(bottom[0, j])
        for m in ("cfd20", "match", "koma"):
            t, f_, c_ = series[m]
            y = f_ if j == 0 else c_[:, j-1]
            if m == "cfd20":   # close the periodic cycle with the first bin repeated at t = RR (line only)
                ax.plot(np.r_[t, rr*1000], np.r_[y, y[0]], color=COLORS[m], lw=2, label=labels[m])
                ax.plot(t, y, "o", color=COLORS[m], ms=3.5)
            elif m == "match":
                ax.plot(t, y, color=COLORS[m], lw=1.2, ls=(0, (4, 2)), marker="s", ms=4, mfc="white", label=labels[m])
            else:
                ax.errorbar(t, y, xerr=rr/nb*500, color=COLORS[m], lw=1.6, marker="o", ms=5, capsize=0, elinewidth=1, label=labels[m])
        ax.axhline(0, color="#888888", lw=.6)
        if j in (2, 3):
            ax.set_ylim(-0.1, 0.1)
        ax.set_xlim(0, rr*1000)
        ax.set_title(titles[j], fontsize=10); ax.set_xlabel("time from cycle start (ms); Koma bars = acquisition window", fontsize=8)
        ax.set_ylabel("mL/s" if j == 0 else "m/s", fontsize=9); ax.grid(alpha=.25); ax.tick_params(labelsize=8)
        if j == 3:   # v9: legend in the empty upper half of the Vz panel (v8 legend covered the flow peak)
            ax.legend(fontsize=8, frameon=False, loc="upper center")
    fig.suptitle("Aorta (E-027, 1M spins): CFD at native resolution vs Koma (4 mm, 8 bins); XY projections at the peak bin and section waveforms", fontsize=13)
    fig.supxlabel("Images: mean through MRI Z over the Koma core-15 % voxels, rotated 90° CCW then mirrored L-R (MRI Y horizontal, MRI X vertical). "
                  "Outline: silhouette of the 2.34 M CFD nodes (0.25 mm).\n"
                  "CFD panel: CFD velocity averaged over the Koma bin's acquisition window on a 0.5 mm grid. Koma panels: 4 mm reconstruction, bilinear display interpolation, "
                  "clipped to the CFD outline. Orange: section X+000_C1.\n"
                  "Noise is synthetic, display only (no aorta 4D-flow scan; phantom-scan SNR borrowed). "
                  f"Curves: forward-positive; rigid circular shift starting at Koma bin {kmin+1} (minimum flow); no time warping.", fontsize=8.5)
    fig.savefig(out/"aorta_cfd_koma_highres.png", dpi=150); plt.close(fig)
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), status="PRESENTATION_DERIVATIVE_ONLY", script_sha256=sha(__file__),
                  e027_report_sha256=sha(E027/"report.json"), pc_reconstruction_sha256=sha(PC), point_cloud_sha256={k: want[k] for k in ("point_coordinates.npy", "velocities.npy")},
                  peak_bin=b+1, section=SECTION, section_x_m=xsec, fine_section_area_mm2=area*1e6, r_in_m=R_IN, image_points_inside=int(inside.sum()),
                  noise_sigma=sigma, snr=SNR, seed=SEED, cycle_start_bin=kmin+1, rr_s=rr,
                  flow_ml_s=dict(cfd_native_20bins=(sgn*cfd20["flow"]).tolist(), cfd_native_8_adc_windows=(sgn*cfd8["flow"]).tolist(),
                                 cfd_e027_4mm=(sgn*e027["CFD"]).tolist(), koma_e027=(sgn*e027["Koma"]).tolist()),
                  fine_over_e027_cfd_flow=(cfd8["flow"]/e027["CFD"]).tolist(), runtime_s=time.time()-t_start)
    report["outputs"] = {p.name: sha(p) for p in sorted(out.glob("*.png"))}
    (out/"MANIFEST.json").write_text(json.dumps(report, indent=1)+"\n", encoding="utf-8")
    out.rename(target)
    print("DONE", b+1, ix, round(time.time()-t_start, 1))


if __name__ == "__main__":
    main(sys.argv[1])
