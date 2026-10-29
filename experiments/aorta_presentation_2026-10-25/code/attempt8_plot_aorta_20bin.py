"""v11 aorta presentation figure: Koma at 20 cardiac bins (E-062) next to the native CFD.

Changes from v10 (plot_aorta_highres.py):
- Koma panels and curves from E-062 (20 bins, 36 ms windows, 4 mm); the 8-bin Koma curve is kept
  for comparison.
- Timing fix: every series is placed at the circular mean phase of its actual ADC window
  (E-027 / E-062 sequence_timing). v8-v10 placed bins at nominal (k+0.5)RR/n, but the
  sequences start with 10 ramp shots, so the true 8-bin windows sit ~32 ms later.
- The native CFD is scaled by 0.72/RR, as in E-027 (v8-v10 omitted this 0.85 % factor).
- Thin grey line: the native CFD at its own 72 frames (10 ms), the waveform the bins average.
Images: E-062 peak bin; CFD native at that bin's ADC window; outline and projections as v10.
v12 (user request): display noise at an assumed clinical-4D-flow SNR of 12 instead of the
phantom-scan 84. One noise level for the whole acquisition (sigma = median core-15 |REF|,
averaged over bins, / 12), complex Gaussian noise added independently to the REF/X/Y/Z complex
images of every bin; the noisy Koma section waveforms are shown too. Display only.
"""
from datetime import datetime, timezone
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

sys.path.insert(0, str(Path(__file__).parent))
from plot_aorta_highres import CLOUD, DISP, E027, IMG, IMGZ, PLANE, R_IN, SEED, VENC, fill_smooth, idw, resample, sha  # noqa: E402
from mrsimtracks.hemodynamics import section_flow  # noqa: E402  (frozen, read-only)
from scripts.aorta_hemodynamic_reference import AortaGrid  # noqa: E402  (frozen, read-only)

E062 = Path("D:/MRSim/outputs/aorta_20bin_e062")
REC20 = E062/"combined/simulation_motion_1m_20bin_4mm/pc_reconstruction.h5"
SECTION = "X+000_C1"
SECTIONS = dict(asc="X+058_C1", desc="X+000_C1")   # v18: ascending and descending aorta side by side
SEC_STYLE = dict(asc=("#eb6834", "A"), desc=("#8e44ad", "D"))
SNR = 12.0   # v12: assumed clinical 4D-flow SNR (v11 borrowed the phantom-scan 84.4)
# v13: noise applied as for a VENC matched to this flow (Koma speeds <= 0.26 m/s). v12 used the
# simulated 2.5 m/s, ~10x the flow, so velocity noise (proportional to VENC) swamped it. Approximation:
# Koma's velocities are re-encoded at VENC_NOISE for the noise display only (Koma ran at 2.5 m/s).
VENC_NOISE = 0.3
COLORS = dict(frames="#c3c2bd", cfd="#0b0b0b", match="#8a8984", koma20="#2a78d6", koma8="#eb6834", noisy="#7fa8de")


def main(target):
    t_start = time.time()
    target = Path(target)
    if target.exists():
        raise FileExistsError(target)
    ana = json.loads((E062/"analysis_v1/report.json").read_text())
    if ana["status"] != "COMPLETE_E062_ANALYSIS" or not ana["regression_8bin_vs_e027"]["pass"]:
        raise ValueError("E-062 analysis not complete")
    man = json.loads((E062/"analysis_v1/MANIFEST.json").read_text())
    if sha(E062/"analysis_v1/matched_reference_20bin.h5") != man["matched_reference_20bin.h5"]:
        raise ValueError("E-062 matched reference changed")
    audit = json.loads((E062/"sequence_audit_v1/report.json").read_text())
    rep027 = json.loads((E027/"report.json").read_text())
    want = {Path(r["path"]).name: r["sha256"] for r in rep027["inputs"]}
    for name in ("point_coordinates.npy", "velocities.npy"):
        if sha(CLOUD/name) != want[name]:
            raise ValueError("CFD point cloud changed: "+name)
    with h5py.File(E062/"analysis_v1/matched_reference_20bin.h5") as f:
        ref, koma, frac, core = f["adc_time_average"][:], f["Koma"][:], f["fraction"][:], f["masks/core15"][:].astype(bool)
        secs = {k: f["sections"][k][:].astype(bool) for k in f["sections"]}; fw = f["frame_weights"][:]; pfrac = f["plane_fraction"][:]
    with h5py.File(E027/"matched_reference.h5") as f:
        koma8, frac8 = f["Koma"][:], f["fraction"][:]
    with h5py.File(REC20) as f:
        mag = f["reference_magnitude"][:]
        vel = np.moveaxis(f["velocity"][:], 0, -1)
    if np.nanmax(np.abs(vel[:, core]-koma[:, core])) > 1e-4:
        raise ValueError("reconstruction does not match the E-062 analysis field")
    nb, nz, ny, nx = mag.shape
    rr = float(ana["period_s"]); vscale = float(ana["source_velocity_time_scale"])
    grid = AortaGrid()
    order, cen = np.array(grid.order), np.array(grid.center)
    gx, gy, gz = grid.axes(); xs, ys = gx*1000, gy*1000
    tc20 = np.array([b["phase_circular_mean_s"] for b in audit["encodings"]["REF"]["bin_timing"]])
    tc8 = np.array([b["phase_circular_mean_s"] for b in rep027["timing_audit"]["REF"]["bin_timing"]])

    # ---- Koma 20-bin panels at the 20-bin peak ----
    b = int(np.argmax([np.linalg.norm(ref[k][core], axis=-1).mean() for k in range(nb)]))
    sigma = float(np.median(mag.mean(axis=0)[core]))/SNR      # v12: one noise level for the whole acquisition
    rng = np.random.default_rng(SEED)
    def noisy(z):
        return z+sigma*(rng.standard_normal(z.shape)+1j*rng.standard_normal(z.shape))
    REFall = noisy(mag.astype(complex))
    vnall = np.stack([np.angle(noisy(mag*np.exp(1j*vel[..., c]*np.pi/VENC_NOISE))*REFall.conj())*VENC_NOISE/np.pi for c in range(3)], axis=-1)
    vn = vnall[b]
    aliased = int((np.abs(vel[:, core]) > VENC_NOISE).sum())   # noiseless core samples beyond the display VENC
    # v15: scan-like background. The phantom holds only blood spins, so outside the vessel Koma's noisy
    # images are pure noise (as air in a scan). Shown there as the |REF|^2-weighted mean noisy velocity
    # vector through Z (speed of the mean vector, so the noise averages into speckle, not a flat bias).
    wbg = np.abs(REFall[b])**2
    bg = np.linalg.norm((vn*wbg[..., None]).sum(axis=0), axis=-1)/wbg.sum(axis=0)
    wts = np.where(core, frac, 0)
    def proj(v):
        den = wts.sum(axis=0)
        return np.divide((np.nan_to_num(v)*wts).sum(axis=0), den, out=np.full(den.shape, np.nan), where=den > 0)
    sp_k, sp_kn, sp_match = proj(np.linalg.norm(koma[b], axis=-1)), proj(np.linalg.norm(vn, axis=-1)), proj(np.linalg.norm(np.nan_to_num(ref[b]), axis=-1))

    # ---- native CFD (as v10, plus the 0.72/RR scale) ----
    src = np.load(CLOUD/"point_coordinates.npy", allow_pickle=False).astype(np.float64)
    vel_src = np.load(CLOUD/"velocities.npy", mmap_mode="r", allow_pickle=False)
    nodes = src[:, order]-cen[order]
    tree = cKDTree(nodes)
    xf = np.arange(xs[0], xs[-1]+1e-9, DISP*1000); yf = np.arange(ys[0], ys[-1]+1e-9, DISP*1000)
    occ = np.zeros((yf.size, xf.size), bool)
    iy = np.round((nodes[:, 1]*1000-yf[0])/(DISP*1000)).astype(int); ix_ = np.round((nodes[:, 0]*1000-xf[0])/(DISP*1000)).astype(int)
    ok = (iy >= 0) & (iy < yf.size) & (ix_ >= 0) & (ix_ < xf.size)
    occ[iy[ok], ix_[ok]] = True
    occ = ndimage.binary_closing(occ, structure=np.hypot(*np.mgrid[-2:3, -2:3]) <= 2)
    lab, n = ndimage.label(~occ)
    sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n+1))*(DISP*1000)**2
    occ |= np.isin(lab, 1+np.flatnonzero(sizes < 2.0))
    smooth = ndimage.gaussian_filter(occ.astype(float), 1.0); outline = smooth >= .5
    V = np.empty((len(nodes), 3), np.float32)
    for s in range(0, len(nodes), 200000):
        V[s:s+200000] = np.einsum("ncf,f->nc", np.asarray(vel_src[s:s+200000], dtype=np.float32), fw[b].astype(np.float32))[:, order]*vscale
    lo, hi = nodes.min(axis=0)-1e-3, nodes.max(axis=0)+1e-3
    ax3 = [np.arange(lo[0], hi[0], IMG), np.arange(lo[1], hi[1], IMG), np.arange(lo[2], hi[2], IMGZ)]
    P = np.stack(np.meshgrid(*ax3, indexing="ij"), -1).reshape(-1, 3)
    d1, _ = tree.query(P, k=1, workers=-1, distance_upper_bound=R_IN*1.01)
    inside = d1 <= R_IN
    _, nq, wq = idw(tree, P[inside])
    spd = np.zeros(len(P), np.float32)
    spd[inside] = np.linalg.norm(np.einsum("qk,qkc->qc", wq, V[nq]), axis=1)
    kv = [(P[:, a]-g[0])/(g[1]-g[0]) for a, g in enumerate((gx, gy, gz))]
    use = np.where(inside, ndimage.map_coordinates(wts.astype(np.float32), [kv[2], kv[1], kv[0]], order=1, mode="constant", cval=0.0), 0)
    shp = tuple(a.size for a in ax3)
    cnt = use.reshape(shp).sum(axis=2); tot = (use*spd).reshape(shp).sum(axis=2)
    cfd_img = np.divide(tot, cnt, out=np.full(cnt.shape, np.nan), where=cnt > 0).T
    cfd_disp = resample(fill_smooth(cfd_img, 2.0), ax3[0]*1000, ax3[1]*1000, xf, yf)

    # Sections on a fine plane: all 72 frames, then E-062 ADC windows. v18: one call per section.
    tf = ((np.arange(72)-18) % 72)/72*rr                     # sequence phase of each source frame (E-027 mapping)
    gsp = grid.spacing

    def section_set(name):
        sec = secs[name]
        srow = [r for r in ana["primary_20bin"]["section_series"] if r["section"] == name]
        xsec = float(srow[0]["x_m"]); ix = int(np.argmin(abs(gx-xsec)))
        zz, yy = np.where(sec)
        py = np.arange(gy[yy].min()-.006, gy[yy].max()+.006, PLANE); pz = np.arange(gz[zz].min()-.006, gz[zz].max()+.006, PLANE)
        YY, ZZ = np.meshgrid(py, pz, indexing="ij")
        dq, nq2, wq2 = idw(tree, np.stack([np.full(YY.size, xsec), YY.ravel(), ZZ.ravel()], -1))
        inplane = (dq <= R_IN).reshape(YY.shape)
        lab2, n2 = ndimage.label(inplane)
        jy = np.clip(np.round((YY-gy[0])/(gy[1]-gy[0])).astype(int), 0, ny-1); jz = np.clip(np.round((ZZ-gz[0])/(gz[1]-gz[0])).astype(int), 0, nz-1)
        hit = np.bincount(lab2[sec[jz, jy] & inplane], minlength=n2+1); hit[0] = 0
        comp = (lab2 == int(np.argmax(hit))).ravel()
        used = np.unique(nq2[comp])
        vplane = np.asarray(vel_src[used], dtype=np.float32)[:, order, :]*vscale
        vq = np.einsum("qk,qkcf->qcf", wq2[comp], vplane[np.searchsorted(used, nq2[comp])])
        flux_f = vq[:, 0, :].sum(axis=0)*PLANE**2*1e6; mean_f = vq.mean(axis=0).T
        cfd20 = dict(flow=fw @ flux_f, comps=fw @ mean_f)
        def section_series(rows, method, n):
            return np.array([r["net_ml_s"] for r in sorted([r for r in rows if r["section"] == name and r["method"] == method], key=lambda r: r["bin"])])[:n]
        q = dict(match=section_series(srow, "CFD", nb), koma20=section_series(srow, "Koma", nb),
                 koma8=section_series(rep027["section_series"], "Koma", 8))
        w4, w8 = np.where(sec, frac[:, :, ix], 0), np.where(sec, frac8[:, :, ix], 0)
        comp4 = lambda field, w, n: np.array([(np.nan_to_num(field[k][:, :, ix])*w[..., None]).sum(axis=(0, 1))/w.sum() for k in range(n)])  # noqa: E731
        qc = dict(match=comp4(ref, w4, nb), koma20=comp4(koma, w4, nb), koma8=comp4(koma8, w8, 8), noisy=comp4(vnall, w4, nb))
        # v12: noisy section flow with the E-062 definition (core15 section pixels, plane-fraction area);
        # the same code on the noiseless field must reproduce the analysis values.
        valid = sec & core[:, :, ix] & (pfrac[:, :, ix] > 0); area = pfrac[:, :, ix]*gsp[1]*gsp[2]
        flow4 = lambda field: np.array([section_flow(field[k, :, :, ix], area, valid=valid)["net_m3_s"]*1e6 for k in range(nb)])  # noqa: E731
        if np.max(np.abs(flow4(koma)-q["koma20"])) > 1e-6:
            raise ValueError("section-flow recomputation does not reproduce E-062: "+name)
        q["noisy"] = flow4(vnall)
        sgn = -1.0 if q["match"].mean() < 0 else 1.0
        yv = YY.ravel()[comp]
        return dict(name=name, xsec=xsec, yext=(float(yv.min()), float(yv.max())), sgn=sgn, flux_f=flux_f, mean_f=mean_f, cfd20=cfd20, q=q, qc=qc)

    S = {k: section_set(v) for k, v in SECTIONS.items()}
    # Forward-positive; rigid circular shift (both sections): cycle starts at the descending native-CFD minimum (as v11-v17).
    t0 = tf[int(np.argmin(S["desc"]["sgn"]*S["desc"]["flux_f"]))]
    tau = lambda t: ((np.asarray(t)-t0) % rr)*1000  # noqa: E731

    def build_series(s):
        sgn = s["sgn"]; sgnc = np.array([sgn, 1, 1]); q, qc = s["q"], s["qc"]
        out = {"frames": (tau(tf), sgn*s["flux_f"], s["mean_f"]*sgnc), "cfd": (tau(tc20), sgn*s["cfd20"]["flow"], s["cfd20"]["comps"]*sgnc),
               "match": (tau(tc20), sgn*q["match"], qc["match"]*sgnc), "koma20": (tau(tc20), sgn*q["koma20"], qc["koma20"]*sgnc),
               "koma8": (tau(tc8), sgn*q["koma8"], qc["koma8"]*sgnc), "noisy": (tau(tc20), sgn*q["noisy"], qc["noisy"]*sgnc)}
        return {k: (t[np.argsort(t)], f[np.argsort(t)], c[np.argsort(t)]) for k, (t, f, c) in out.items()}
    series = {k: build_series(v) for k, v in S.items()}

    # ---- figure ----
    clip = lambda img: np.where(outline, img, np.nan)  # noqa: E731
    km = lambda img: clip(resample(fill_smooth(img), xs, ys, xf, yf))  # noqa: E731
    # v17: concise text (user request) and every font 1.5x the v16 size.
    FS = 1.5
    panels = [(clip(cfd_disp), "CFD (0.5 mm)", "viridis"),
              (km(sp_k), "Koma (4 mm)", "viridis"),
              (np.where(outline, resample(fill_smooth(sp_kn), xs, ys, xf, yf), resample(bg, xs, ys, xf, yf)),
               f"Koma + noise (SNR {SNR:.0f})", "viridis"),
              (km(sp_k-sp_match), "Koma − CFD", "RdBu_r")]
    vmax = float(max(np.nanpercentile(panels[0][0], 99.8), np.nanpercentile(panels[1][0], 99.8)))
    dmax = float(np.nanpercentile(np.abs(panels[3][0]), 99.5))
    extent = [yf[0]-DISP*500, yf[-1]+DISP*500, xf[0]-DISP*500, xf[-1]+DISP*500]
    out = target.with_name(target.name+".building"); out.mkdir(parents=True)
    plt.rcParams["font.size"] = 10*FS
    fig = plt.figure(figsize=(20, 19), layout="constrained")
    gs = fig.add_gridspec(3, 1, height_ratios=[2.0, 1, 1])
    top = gs[0].subgridspec(1, 4); rows = [gs[1].subgridspec(1, 4), gs[2].subgridspec(1, 4)]
    for i, (data, lab_, cm) in enumerate(panels):
        ax = fig.add_subplot(top[0, i])
        cmap = plt.get_cmap(cm).copy(); cmap.set_bad("#eeeeee")
        lim = (-dmax, dmax) if i == 3 else (0, vmax)
        im = ax.imshow(data.T, origin="lower", extent=extent, cmap=cmap, vmin=lim[0], vmax=lim[1], interpolation="nearest", aspect="equal")
        ax.contour(yf, xf, smooth.T, levels=[.5], colors="#3a3936", linewidths=1.2)
        for k, s in S.items():   # v18: both sections, as segments across their own lumen
            col, tag = SEC_STYLE[k]; y0, y1 = s["yext"][0]*1000-4, s["yext"][1]*1000+4
            ax.plot([y0, y1], [s["xsec"]*1000]*2, color=col, lw=2.5, solid_capstyle="butt")
            ax.text(y1+2, s["xsec"]*1000, tag, color=col, fontsize=9*FS, fontweight="bold", va="center", ha="left")
        ax.set_facecolor("#eeeeee"); ax.set_title(lab_, fontsize=10*FS); ax.set(xlabel="Y (mm)", ylabel="X (mm)")
        fig.colorbar(im, ax=ax, label="Speed (m/s)" if i < 3 else "m/s", shrink=.85, extend="max" if i < 3 else "both")
    qname = ("Forward flow", "Through-plane velocity", "Vy (in-plane)", "Vz (in-plane)")
    side = dict(asc="ascending", desc="descending")
    labels = {"frames": "CFD (10 ms frames)", "cfd": f"CFD, {nb} windows", "match": f"CFD at 4 mm, {nb} windows",
              "koma20": f"Koma, {nb} bins", "koma8": "Koma, 8 bins", "noisy": f"Koma, {nb} bins + noise"}
    # v18 layout: row 1 = flow (A | D), through-plane (A | D); row 2 = Vy (A | D), Vz (A | D).
    cells = [(0, 0, 0, "asc"), (0, 1, 0, "desc"), (0, 2, 1, "asc"), (0, 3, 1, "desc"),
             (1, 0, 2, "asc"), (1, 1, 2, "desc"), (1, 2, 3, "asc"), (1, 3, 3, "desc")]
    first = {}
    for r_, c, j, k in cells:
        ax = fig.add_subplot(rows[r_][0, c], sharey=first.get(j))
        first.setdefault(j, ax)
        for m in ("frames", "cfd", "match", "koma8", "noisy", "koma20"):
            t, f_, c_ = series[k][m]
            y = f_ if j == 0 else c_[:, j-1]
            if m == "noisy":
                ax.plot(t, y, color=COLORS[m], lw=.9, ls=(0, (2, 2)), marker="x", ms=5, mew=1.3, label=labels[m], zorder=3)
            elif m == "frames":
                ax.plot(np.r_[t, t[0]+rr*1000], np.r_[y, y[0]], color=COLORS[m], lw=3, label=labels[m], zorder=1)
            elif m == "cfd":
                ax.plot(t, y, color=COLORS[m], lw=1.8, marker="o", ms=3.5, label=labels[m], zorder=3)
            elif m == "match":
                ax.plot(t, y, color=COLORS[m], lw=1.1, ls=(0, (4, 2)), marker="s", ms=3.5, mfc="white", label=labels[m], zorder=2)
            else:
                n_ = nb if m == "koma20" else 8
                ax.errorbar(t, y, xerr=rr/n_*500, color=COLORS[m], lw=1.5 if m == "koma20" else 1.0, marker="o", ms=4.5 if m == "koma20" else 5,
                            mfc=COLORS[m] if m == "koma20" else "white", capsize=0, elinewidth=.9, label=labels[m], zorder=4 if m == "koma20" else 2)
        ax.axhline(0, color="#888888", lw=.6)
        if j in (2, 3):
            ax.set_ylim(-0.1, 0.1)
        ax.set_xlim(0, rr*1000)
        ax.set_title(f"{qname[j]}, {side[k]} ({SEC_STYLE[k][1]})", fontsize=10*FS, color="black")
        ax.set_xlabel("Time (ms)", fontsize=8*FS)
        ax.set_ylabel("mL/s" if j == 0 else "m/s", fontsize=9*FS); ax.grid(alpha=.25); ax.tick_params(labelsize=8*FS)
        if (r_, c) == (1, 3):
            ax.legend(fontsize=7.5*FS, frameon=False, loc="upper center")
    fig.suptitle(f"Aorta: CFD vs Koma ({nb} cardiac bins, 4 mm)", fontsize=13*FS)
    fig.supxlabel(f"Images: peak bin ({b+1}/{nb}), mean speed through Z; outline from the CFD mesh; lines: ascending (A) and descending (D) flow sections.\n"
                  f"Noise (display only): SNR {SNR:.0f}, as for VENC {VENC_NOISE} m/s; outside the vessel, Koma's own noise.\n"
                  "Curves: forward flow positive; points at acquisition-window centres, bars show the window.", fontsize=8.5*FS)
    fig.savefig(out/"aorta_cfd_koma_20bin_asc_desc.png", dpi=150); plt.close(fig)
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), status="PRESENTATION_DERIVATIVE_ONLY", script_sha256=sha(__file__),
                  helper_script_sha256=sha(Path(__file__).parent/"plot_aorta_highres.py"), e062_analysis_report_sha256=sha(E062/"analysis_v1/report.json"),
                  e027_report_sha256=sha(E027/"report.json"), peak_bin=b+1, bins=nb, sections=SECTIONS, cycle_start_phase_s=float(t0),
                  source_velocity_time_scale=vscale, noise_sigma=sigma, snr=SNR, seed=SEED, venc_noise_m_s=VENC_NOISE,
                  aliased_core_samples_at_venc_noise=aliased,
                  flow_ml_s={s: {k: dict(time_ms=v[0].tolist(), flow=v[1].tolist()) for k, v in ser.items()} for s, ser in series.items()},
                  runtime_s=time.time()-t_start)
    report["outputs"] = {p.name: sha(p) for p in sorted(out.glob("*.png"))}
    (out/"MANIFEST.json").write_text(json.dumps(report, indent=1)+"\n", encoding="utf-8")
    out.rename(target)
    print("DONE", b+1, round(time.time()-t_start, 1))


if __name__ == "__main__":
    main(sys.argv[1])
