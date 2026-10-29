"""All-bin overview (user request): the v22 image row for every one of the 20 E-062 bins.

Rows: CFD (0.5 mm, Koma bin window), Koma (4 mm), Koma + noise (SNR 12 as for VENC 0.3 m/s, with
Koma's own background noise), Koma - CFD (both 4 mm). Columns: bins in time order from the v22
cycle start (rigid circular shift). Same definitions, outline, weights, noise seed and noise
realisation as figure_v22 (plot_aorta_20bin.py); one speed scale and one difference scale for all.
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
from plot_aorta_highres import CLOUD, DISP, E027, IMG, IMGZ, R_IN, SEED, fill_smooth, idw, resample, sha  # noqa: E402
from scripts.aorta_hemodynamic_reference import AortaGrid  # noqa: E402  (frozen, read-only)

ROOT = Path("D:/MRSim/outputs/aorta_presentation_2026-10-25")
E062 = Path("D:/MRSim/outputs/aorta_20bin_e062")
REC20 = E062/"combined/simulation_motion_1m_20bin_4mm/pc_reconstruction.h5"
SNR, VENC_NOISE, FS = 12.0, 0.3, 1.5


def main(target):
    t_start = time.time()
    target = Path(target)
    if target.exists():
        raise FileExistsError(target)
    v22 = json.loads((ROOT/"figure_v22/MANIFEST.json").read_text())
    ana = json.loads((E062/"analysis_v1/report.json").read_text())
    audit = json.loads((E062/"sequence_audit_v1/report.json").read_text())
    rep027 = json.loads((E027/"report.json").read_text())
    want = {Path(r["path"]).name: r["sha256"] for r in rep027["inputs"]}
    for name in ("point_coordinates.npy", "velocities.npy"):
        if sha(CLOUD/name) != want[name]:
            raise ValueError("CFD point cloud changed: "+name)
    with h5py.File(E062/"analysis_v1/matched_reference_20bin.h5") as f:
        ref, koma, frac, core = f["adc_time_average"][:], f["Koma"][:], f["fraction"][:], f["masks/core15"][:].astype(bool)
        fw = f["frame_weights"][:]
    with h5py.File(REC20) as f:
        mag = f["reference_magnitude"][:]
        vel = np.moveaxis(f["velocity"][:], 0, -1)
    nb = mag.shape[0]
    rr, vscale = float(ana["period_s"]), float(ana["source_velocity_time_scale"])
    grid = AortaGrid(); order, cen = np.array(grid.order), np.array(grid.center)
    gx, gy, gz = grid.axes(); xs, ys = gx*1000, gy*1000

    # Noise: identical generator, call order and shapes as plot_aorta_20bin.py (v22), so the realisation matches.
    sigma = float(np.median(mag.mean(axis=0)[core]))/SNR
    if abs(sigma-v22["noise_sigma"]) > 1e-12*max(1.0, sigma):
        raise ValueError("noise sigma differs from v22")
    rng = np.random.default_rng(SEED)
    def noisy(z):
        return z+sigma*(rng.standard_normal(z.shape)+1j*rng.standard_normal(z.shape))
    REFall = noisy(mag.astype(complex))
    vnall = np.stack([np.angle(noisy(mag*np.exp(1j*vel[..., c]*np.pi/VENC_NOISE))*REFall.conj())*VENC_NOISE/np.pi for c in range(3)], axis=-1)

    wts = np.where(core, frac, 0)
    def proj(v):
        den = wts.sum(axis=0)
        return np.divide((np.nan_to_num(v)*wts).sum(axis=0), den, out=np.full(den.shape, np.nan), where=den > 0)

    # CFD outline and per-bin native projection (as v22, all bins).
    src = np.load(CLOUD/"point_coordinates.npy", allow_pickle=False).astype(np.float64)
    vel_src = np.load(CLOUD/"velocities.npy", mmap_mode="r", allow_pickle=False)
    nodes = src[:, order]-cen[order]; tree = cKDTree(nodes)
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
    Vb = np.empty((len(nodes), 3, nb), np.float32)
    for s in range(0, len(nodes), 200000):
        Vb[s:s+200000] = np.einsum("ncf,bf->ncb", np.asarray(vel_src[s:s+200000], dtype=np.float32), fw.astype(np.float32))[:, order, :]*vscale
    lo, hi = nodes.min(axis=0)-1e-3, nodes.max(axis=0)+1e-3
    ax3 = [np.arange(lo[0], hi[0], IMG), np.arange(lo[1], hi[1], IMG), np.arange(lo[2], hi[2], IMGZ)]
    P = np.stack(np.meshgrid(*ax3, indexing="ij"), -1).reshape(-1, 3)
    d1, _ = tree.query(P, k=1, workers=-1, distance_upper_bound=R_IN*1.01)
    inside = d1 <= R_IN
    _, nq, wq = idw(tree, P[inside])
    kv = [(P[:, a]-g[0])/(g[1]-g[0]) for a, g in enumerate((gx, gy, gz))]
    use = np.where(inside, ndimage.map_coordinates(wts.astype(np.float32), [kv[2], kv[1], kv[0]], order=1, mode="constant", cval=0.0), 0)
    shp = tuple(a.size for a in ax3)
    cnt = use.reshape(shp).sum(axis=2)
    clip = lambda img: np.where(outline, img, np.nan)  # noqa: E731
    km = lambda img: clip(resample(fill_smooth(img), xs, ys, xf, yf))  # noqa: E731
    rows = {"cfd": [], "koma": [], "noisy": [], "diff": []}
    for b in range(nb):
        spd = np.zeros(len(P), np.float32)
        spd[inside] = np.linalg.norm(np.einsum("qk,qkc->qc", wq, Vb[nq, :, b]), axis=1)
        tot = (use*spd).reshape(shp).sum(axis=2)
        cfd_img = np.divide(tot, cnt, out=np.full(cnt.shape, np.nan), where=cnt > 0).T
        rows["cfd"].append(clip(resample(fill_smooth(cfd_img, 2.0), ax3[0]*1000, ax3[1]*1000, xf, yf)))
        sp_k = proj(np.linalg.norm(koma[b], axis=-1))
        rows["koma"].append(km(sp_k))
        wbg = np.abs(REFall[b])**2
        bg = np.linalg.norm((vnall[b]*wbg[..., None]).sum(axis=0), axis=-1)/wbg.sum(axis=0)
        rows["noisy"].append(np.where(outline, resample(fill_smooth(proj(np.linalg.norm(vnall[b], axis=-1))), xs, ys, xf, yf), resample(bg, xs, ys, xf, yf)))
        rows["diff"].append(km(sp_k-proj(np.linalg.norm(np.nan_to_num(ref[b]), axis=-1))))
        print("bin", b+1, round(time.time()-t_start, 1), flush=True)

    # Time order: v22 cycle start, bins at their ADC-window circular means.
    t0 = float(v22["cycle_start_phase_s"])
    tc = np.array([x["phase_circular_mean_s"] for x in audit["encodings"]["REF"]["bin_timing"]])
    tau = ((tc-t0) % rr)*1000; order_b = np.argsort(tau)
    vmax = float(np.nanmax([np.nanpercentile(np.stack(rows[k]), 99.8) for k in ("cfd", "koma")]))
    dmax = float(np.nanpercentile(np.abs(np.stack(rows["diff"])), 99.5))
    extent = [yf[0]-DISP*500, yf[-1]+DISP*500, xf[0]-DISP*500, xf[-1]+DISP*500]
    cm_s = plt.get_cmap("viridis").copy(); cm_s.set_bad("#eeeeee")
    cm_d = plt.get_cmap("RdBu_r").copy(); cm_d.set_bad("#eeeeee")
    out = target.with_name(target.name+".building"); out.mkdir(parents=True)
    plt.rcParams["font.size"] = 10*FS
    fig, axes = plt.subplots(4, nb, figsize=(34, 14.5), layout="constrained", sharex=True, sharey=True)
    names = (("cfd", "CFD (0.5 mm)"), ("koma", "Koma (4 mm)"), ("noisy", f"Koma + noise\n(SNR {SNR:.0f})"), ("diff", "Koma − CFD"))
    for r, (key, label) in enumerate(names):
        for c, b in enumerate(order_b):
            ax = axes[r, c]
            diff = key == "diff"
            im = ax.imshow(rows[key][b].T, origin="lower", extent=extent, cmap=cm_d if diff else cm_s,
                           vmin=-dmax if diff else 0, vmax=dmax if diff else vmax, interpolation="nearest", aspect="equal")
            ax.contour(yf, xf, smooth.T, levels=[.5], colors="#3a3936", linewidths=.7)
            ax.set_facecolor("#eeeeee"); ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"{tau[b]:.0f} ms", fontsize=8.5*FS)
            if c == 0:
                ax.set_ylabel(label, fontsize=9*FS)
            if diff:
                im_d = im
            elif r == 0:
                im_s = im
    fig.colorbar(im_s, ax=axes[:3, :], label="Speed (m/s)", shrink=.8, extend="max", pad=.01)
    fig.colorbar(im_d, ax=axes[3, :], label="m/s", shrink=.9, extend="both", pad=.01)
    fig.suptitle(f"Aorta: CFD vs Koma, all {nb} cardiac bins (time from cycle start)", fontsize=13*FS)
    fig.supxlabel("Mean speed through Z; outline from the CFD mesh. Noise (display only): SNR 12, as for VENC 0.3 m/s; outside the vessel, Koma's own noise.",
                  fontsize=8.5*FS)
    fig.savefig(out/"aorta_cfd_koma_allbins.png", dpi=110); plt.close(fig)
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), status="PRESENTATION_DERIVATIVE_ONLY", script_sha256=sha(__file__),
                  v22_manifest_sha256=sha(ROOT/"figure_v22/MANIFEST.json"), e062_analysis_report_sha256=sha(E062/"analysis_v1/report.json"),
                  bins_in_time_order=[int(b)+1 for b in order_b], time_ms=[float(tau[b]) for b in order_b], speed_vmax=vmax, diff_max=dmax,
                  noise_sigma=sigma, snr=SNR, venc_noise_m_s=VENC_NOISE, seed=SEED, runtime_s=time.time()-t_start)
    report["outputs"] = {p.name: sha(p) for p in sorted(out.glob("*.png"))}
    (out/"MANIFEST.json").write_text(json.dumps(report, indent=1)+"\n", encoding="utf-8")
    out.rename(target)
    print("DONE", round(time.time()-t_start, 1))


if __name__ == "__main__":
    main(sys.argv[1])
