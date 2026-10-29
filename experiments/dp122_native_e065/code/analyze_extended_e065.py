"""E-065 S4c (user request): extended dp122 evaluation. Four parts, all CPU, no new signal.

1. Standard metrics as E-060/E-061: frozen evaluate_arm (T1 throat flow, T2 peak throat velocity, T3 coverage,
   regional axial RMSE), vector RMSE/NRMSE on per-bin support within the CFD lumen (NRMSE = RMSE / reference vector
   RMS), and Bland-Altman of section flow and section peak velocity (24 sections x 30 bins, as bland_altman.py).
2. Peak-velocity overshoot mechanism at the peak bin: per-voxel axial error versus (a) REF magnitude / lumen median,
   (b) spins per voxel at the bin-centre time (phantom positions, E-060 rotation), (c) CFD axial-velocity spread
   across the voxel estimated as |grad v_z| * voxel size (the reference stores voxel means only, so this is a proxy).
   Maximum recomputed excluding voxels below 50 % (and 25 %, 75 %) of the lumen-median magnitude.
3. Throat-to-upstream flow ratio: E-059 scan method at fractions 0.20/0.25/0.30 for Koma and scan; per-slice mean
   flow along the vessel normalized by the upstream mean for Koma, CFD (plane reference) and scan.
4. Transverse jet profiles of forward axial velocity at the throat and 5, 10, 20 mm downstream (peak systole),
   through the section centroid, for scan, CFD and Koma.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tomllib

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402

ROOT = Path("D:/MRSim/outputs/dp122_native_e065")
E059 = Path("D:/MRSim/outputs/measured_scan_e059")
E060 = Path("D:/MRSim/outputs/scanner_geometry_e060")
E061 = Path("D:/MRSim/outputs/venc_unwrap_e061")
PRIVATE = Path("D:/MRSim/private_inputs/phantom_scan_2025-08-28/measured_v1/measured_30phase.npz")
sys.path.insert(0, str(E061/"code")); sys.path.insert(1, str(E060/"code"))
import analyze_e061 as a61  # noqa: E402
an, et = a61.an, a61.an.et
FS = 1.5
VENCS = (1.5, 1.5, 3.0)
C = dict(scan="#d9822b", cfd="#1f1f1f", koma="#1f5fa8")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def spins_per_voxel(b):
    """Spin counts on the (S, P, R) grid at the centre of bin b (phantom phase (b+0.5)/30 after the seam)."""
    info = tomllib.loads((ROOT/"inputs_v1/case_info.toml").read_text())
    shift = json.loads((E060/"phantom_v1/report.json").read_text())
    tz = float(np.float32(shift["transform"]["translation_new_z_m"])) if "transform" in shift else float(np.float32(shift["translation_new_z_m"]))
    knot = int(round((b+.5)/30*1440))
    with h5py.File(ROOT/"inputs_v1/case_inputs.h5") as f:
        dx, dy, dz = (f[k][knot, :300000] for k in ("dx", "dy", "dz"))
    x, y, z = dy, dz, dx+tz                                              # E-060 rotation (x,y,z)_new = (y, z, x + shift)
    ir = np.round(x/1.25e-3).astype(int)+32; ip = np.round(y/1.25e-3).astype(int)+32; iz = np.round(z/2.5e-3).astype(int)+15
    ok = (ir >= 0) & (ir < 64) & (ip >= 0) & (ip < 64) & (iz >= 0) & (iz < 30)
    counts = np.zeros((30, 64, 64), int); np.add.at(counts, (iz[ok], ip[ok], ir[ok]), 1)
    return counts, knot, info["period_s"]


def main(target):
    target = Path(target); stage = target.with_name(target.name+".building")
    if target.exists() or stage.exists():
        raise FileExistsError(target)
    plt.rcParams["font.size"] = 10*FS
    control = tomllib.loads((ROOT/"acquisition_v1/control_report.toml").read_text())
    images, audits = an.load_images(ROOT/"acquisition_v1", control, ROOT/"sequences_v1/timing_e060.h5")
    field = a61.velocity(images, VENCS, False)                                   # (bin, S, P, R, 3)
    refh5 = ROOT/"references_v1/references.h5"
    with h5py.File(refh5) as f:
        fft_adc, plane, frac, pfrac, zpart = (f[k][:] for k in ("fft_adc", "plane_reference", "fraction_fft", "plane_fraction", "partition_z_m"))
    zmm = zpart*1e3
    rep = dict(experiment="E-065", stage="S4c_extended_analysis", utc=datetime.now(timezone.utc).isoformat(), script_sha256=sha(__file__))
    # ---------------- 1. standard metrics ----------------
    std = a61.evaluate_arm(images, field, refh5)
    sup = np.ones(field.shape[:4], bool)
    for im in images.values():
        m = np.abs(im); sup &= m >= .05*m.max(axis=(1, 2, 3), keepdims=True)
    lum = frac > 0
    keep = sup & lum[None] & (np.abs(zmm) <= 30)[None, :, None, None]
    dv = field[keep]-fft_adc[keep]
    vec_rmse = float(np.sqrt(np.mean(np.sum(dv**2, axis=-1)))); ref_rms = float(np.sqrt(np.mean(np.sum(fft_adc[keep]**2, axis=-1))))
    rep["standard"] = dict(T1=std["targets_per_bin_support"]["T1_throat_flow"]["cycle_same_area_error"],
                           T1_worst_bin=std["targets_per_bin_support"]["T1_throat_flow"]["worst_bin_same_area_error"],
                           T2_max_error=std["targets_per_bin_support"]["T2_peak_throat_velocity"]["max_error"],
                           T2_p99_error=std["targets_per_bin_support"]["T2_peak_throat_velocity"]["p99_error"],
                           T2_koma_max=std["targets_per_bin_support"]["T2_peak_throat_velocity"]["Koma_max_m_s"],
                           T2_cfd_max=std["targets_per_bin_support"]["T2_peak_throat_velocity"]["CFD_region_max_m_s"],
                           T3_min_section_coverage=std["targets_per_bin_support"]["T3_coverage"]["min_section_area_coverage"],
                           T3_mean_whole_lumen_coverage=std["targets_per_bin_support"]["T3_coverage"]["mean_whole_lumen_coverage"],
                           passes=std["targets_per_bin_support"]["passes"], throat_axial_rmse_m_s=std["regional_axial"]["throat"]["axial_rmse_m_s"],
                           regional_axial=std["regional_axial"], vector_rmse_m_s=vec_rmse, vector_nrmse=vec_rmse/ref_rms,
                           vector_samples=int(keep.sum()), peak_bin=std["peak_bin"], wrapped_samples=std["wrapped_samples_total"])
    # Bland-Altman rows (as bland_altman.py phantom()).
    ft, vt, pt = an.to_e058(field, True), an.to_e058(fft_adc, True), an.to_e058(plane, True)
    fr, pf = frac.transpose(2, 1, 0), pfrac.transpose(2, 1, 0)
    it = {e: an.to_e058(v) for e, v in images.items()}
    support = et.per_bin_support(it)
    region = (fr > 0) & ((zpart >= an.LOW) & (zpart < an.HIGH))[None, None, :]
    peak = std["peak_bin"]-1
    rows = []
    for k in an.CENTRAL_K:
        a_k = pf[:, :, k]*an.AREA_MM2; full = a_k > 0
        t1 = et.evaluate(ft, support, vt, pt[:, :, :, k, 0], fr, a_k, region, k, peak)["T1_throat_flow"]
        for r in t1["bins"]:
            bb = r["bin"]-1; kk = support[bb][:, :, k] & full
            rows.append(dict(section=k, bin=bb+1, koma_flow=r["Koma_ml_s"], cfd_flow=r["CFD_same_area_ml_s"],
                             koma_peak=float(ft[bb, :, :, k, 0][kk].max()) if kk.any() else np.nan,
                             cfd_peak=float(vt[bb, :, :, k, 0][kk].max()) if kk.any() else np.nan))
    ba = {}
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), layout="constrained")
    for ax, q, unit in ((axes[0], "flow", "mL/s"), (axes[1], "peak", "m/s")):
        a = np.array([r["cfd_"+q] for r in rows], float); k = np.array([r["koma_"+q] for r in rows], float)
        ok = np.isfinite(a) & np.isfinite(k); a, k = a[ok], k[ok]; d = k-a; bias, sd = float(d.mean()), float(d.std(ddof=1))
        ba[q] = dict(n=int(d.size), bias=bias, sd=sd, loa_low=bias-1.96*sd, loa_high=bias+1.96*sd, relative_bias_pct=float(100*d.sum()/a.sum()),
                     r=float(np.corrcoef(a, k)[0, 1]))
        ax.scatter((a+k)/2, d, s=14, color=C["koma"], alpha=.55, edgecolor="none")
        for y, ls in ((bias, "-"), (bias+1.96*sd, (0, (4, 3))), (bias-1.96*sd, (0, (4, 3)))):
            ax.axhline(y, color=C["koma"], lw=1.4, ls=ls)
        ax.axhline(0, color="#888", lw=.7); ax.grid(alpha=.25)
        ax.set_xlabel(f"mean of Koma and CFD ({unit})"); ax.set_ylabel(f"Koma − CFD ({unit})")
        ax.set_title(f"Section {q}: bias {bias:+.3g}, LoA {bias-1.96*sd:+.3g} to {bias+1.96*sd:+.3g} {unit} (n = {d.size})", fontsize=10*FS)
    fig.suptitle("dp122 Koma vs CFD, Bland–Altman (24 sections × 30 phases; correlated samples, descriptive)", fontsize=12*FS)
    stage.mkdir(); fig.savefig(stage/"bland_altman_dp122.png", dpi=110); plt.close(fig)
    rep["bland_altman"] = ba
    # ---------------- 2. overshoot mechanism ----------------
    b = peak
    mag = np.abs(images["REF"][b]); counts, knot, period = spins_per_voxel(b)
    medlum = float(np.median(mag[lum & (np.abs(zmm) <= 30)[:, None, None]]))
    vz_c = fft_adc[b][..., 2]; vz_k = field[b][..., 2]
    g = np.gradient(np.nan_to_num(vz_c), 2.5e-3, 1.25e-3, 1.25e-3)
    spread = np.sqrt((g[0]*2.5e-3)**2+(g[1]*1.25e-3)**2+(g[2]*1.25e-3)**2)    # m/s across one voxel
    reg = lum & (np.abs(zmm) <= 30)[:, None, None]
    err = (vz_k-vz_c)[reg]; rel = (mag/medlum)[reg]; cnt = counts[reg]; spr = spread[reg]
    mech = dict(peak_bin=b+1, knot=knot, lumen_median_magnitude=medlum, voxels=int(reg.sum()))
    for name, x in (("relative_magnitude", rel), ("spins_per_voxel", cnt), ("cfd_spread_m_s", spr)):
        mech[f"spearman_abs_error_vs_{name}"] = float(np.corrcoef(np.argsort(np.argsort(np.abs(err))), np.argsort(np.argsort(x)))[0, 1])
    throat = lum & (np.abs(zmm) <= 10)[:, None, None]
    mx = {}
    for thr in (0, .25, .5, .75):
        sel = throat & (mag >= thr*medlum)
        mx[f"{thr:.2f}"] = dict(voxels=int(sel.sum()), koma_max=float(vz_k[sel].max()), cfd_max=float(vz_c[sel].max()),
                                koma_p99=float(np.percentile(vz_k[sel], 99)), cfd_p99=float(np.percentile(vz_c[sel], 99)))
    mech["throat_max_by_magnitude_threshold"] = mx
    big = np.abs(err) > .3
    mech["voxels_abs_error_gt_0.3"] = dict(count=int(big.sum()), median_relative_magnitude=float(np.median(rel[big])) if big.any() else None,
                                           median_spins=float(np.median(cnt[big])) if big.any() else None,
                                           median_cfd_spread=float(np.median(spr[big])) if big.any() else None)
    mech["all_voxels"] = dict(median_relative_magnitude=float(np.median(rel)), median_spins=float(np.median(cnt)), median_cfd_spread=float(np.median(spr)))
    fig, axes = plt.subplots(1, 4, figsize=(28, 7.5), layout="constrained")
    for ax, x, xl in ((axes[0], rel, "REF magnitude / lumen median"), (axes[1], cnt, "spins in voxel (bin centre)"), (axes[2], spr, "CFD |∇v_z| × voxel (m/s)")):
        ax.scatter(x, err, s=8, alpha=.4, color=C["koma"], edgecolor="none"); ax.axhline(0, color="#888", lw=.7); ax.grid(alpha=.25)
        ax.set_xlabel(xl); ax.set_ylabel("Koma − CFD axial velocity (m/s)")
    axes[0].axvline(.5, color="#c33", lw=1.2, ls=(0, (4, 3)))
    th = [0, .25, .5, .75]
    axes[3].plot(th, [mx[f"{t:.2f}"]["koma_max"] for t in th], "-s", color=C["koma"], lw=2, label="Koma max")
    axes[3].plot(th, [mx[f"{t:.2f}"]["cfd_max"] for t in th], "-o", color=C["cfd"], lw=2, mfc="white", label="CFD max")
    axes[3].plot(th, [mx[f"{t:.2f}"]["koma_p99"] for t in th], "--s", color=C["koma"], lw=1.4, label="Koma p99")
    axes[3].axhline(2.09, color=C["scan"], lw=1.5, label="scan peak (2.09)")
    axes[3].set_xlabel("excluded below fraction of lumen-median magnitude"); axes[3].set_ylabel("throat-region velocity (m/s)")
    axes[3].legend(frameon=False, fontsize=8*FS); axes[3].grid(alpha=.25)
    fig.suptitle(f"Peak-velocity overshoot: per-voxel error at peak systole (bin {b+1}/30, |S| ≤ 30 mm, CFD lumen)", fontsize=12*FS)
    fig.savefig(stage/"overshoot_mechanism.png", dpi=110); plt.close(fig)
    rep["overshoot"] = mech
    # ---------------- 3. throat/upstream ratio sensitivity and along-vessel flow ----------------
    ks = an.scan_style(field, images, zmm, .25)
    srep = json.loads((E059/"measured_v2/report.json").read_text())
    ratio = dict(koma={k: v["throat_vs_upstream_mean_flow"] for k, v in ks["sensitivity"].items()}, scan={})
    spos = np.array(srep["slice_positions_mm"]); st = srep["throat_slice"]
    for fk, seg in srep["segmentation"].items():
        mq = np.array([r["mean_flow_ml_s"] for r in seg["slices"]])
        ratio["scan"][fk] = float(mq[st]/mq[srep["upstream_slices"]].mean()-1)
    kq = np.array(ks["mean_flow_ml_s"]); zk = zmm[list(an.CENTRAL_K)]
    up_k = [i for i in range(len(zk)) if zk[i] <= zk[ks["throat_slice"]]-20]
    sec_c = np.einsum("bspr,spr->bs", np.nan_to_num(plane[..., 2].astype(float)), pfrac*1.5625).mean(axis=0)[list(an.CENTRAL_K)]
    sq = np.array([r["mean_flow_ml_s"] for r in srep["segmentation"]["0.25"]["slices"]])
    rep["flow_ratio"] = dict(throat_vs_upstream=ratio, koma_throat_slice_z_mm=float(zk[ks["throat_slice"]]),
                             along_vessel=dict(z_koma_mm=zk.tolist(), koma=(kq/kq[up_k].mean()).tolist(), cfd=(sec_c/sec_c[up_k].mean()).tolist(),
                                               z_scan_mm=(spos-spos[st]).tolist(), scan=(sq/sq[srep["upstream_slices"]].mean()).tolist()))
    fig, axes = plt.subplots(1, 2, figsize=(22, 7.5), layout="constrained")
    ax = axes[0]
    ax.plot(spos-spos[st], sq/sq[srep["upstream_slices"]].mean(), "-^", color=C["scan"], lw=2, label="4D-flow scan")
    ax.plot(zk, sec_c/sec_c[up_k].mean(), "-o", color=C["cfd"], lw=2, mfc="white", label="CFD (plane reference)")
    ax.plot(zk, kq/kq[up_k].mean(), "-s", color=C["koma"], lw=2, label="Koma")
    ax.axhline(1, color="#888", lw=.7); ax.grid(alpha=.25); ax.legend(frameon=False, fontsize=9*FS)
    ax.set_xlabel("distance from throat (mm)"); ax.set_ylabel("cycle-mean flow / upstream mean"); ax.set_title("Flow conservation along the vessel")
    ax = axes[1]; fx = ["0.20", "0.25", "0.30"]; w = .35; xs = np.arange(3)
    ax.bar(xs-w/2, [100*ratio["koma"][f] for f in fx], w, color=C["koma"], label="Koma")
    ax.bar(xs+w/2, [100*ratio["scan"][f] for f in fx], w, color=C["scan"], label="4D-flow scan")
    ax.set_xticks(xs, [f"{float(f):.0%} threshold" for f in fx]); ax.set_ylabel("throat vs upstream mean flow (%)")
    ax.axhline(0, color="#888", lw=.7); ax.legend(frameon=False, fontsize=9*FS); ax.grid(alpha=.25, axis="y")
    ax.set_title("Segmentation-threshold sensitivity (E-059 method)")
    fig.suptitle("Throat-to-upstream flow ratio: Koma vs scan", fontsize=12*FS)
    fig.savefig(stage/"flow_ratio.png", dpi=110); plt.close(fig)
    # ---------------- 4. transverse jet profiles ----------------
    sc = np.load(PRIVATE); sv, smag = sc["velocity_cm_s"]/100, sc["magnitude"]
    seg = srep["segmentation"]["0.25"]; sgn = seg["vendor_to_forward_sign"]
    s_up = np.array([r["flow_ml_s"] for r in seg["slices"]])[srep["upstream_slices"]].mean(axis=0); ps = int(np.argmax(s_up))
    m_s = smag.mean(axis=1); thr = .25*np.percentile(m_s, 99.5)
    prof = {}
    fig, axes = plt.subplots(2, 4, figsize=(28, 12), layout="constrained")
    for j, d in enumerate((0, 5, 10, 20)):
        kk = int(np.argmin(np.abs(zmm-d))); ss = int(np.argmin(np.abs((spos-spos[st])-d)))
        lab, n = ndimage.label(m_s[ss] >= thr)
        fl = [abs((sgn*sv[ss, :, :, :, 2][:, lab == c]).sum(axis=1).mean()) for c in range(1, n+1)]
        cy, cx = (int(round(v)) for v in ndimage.center_of_mass(lab == 1+int(np.argmax(fl))))
        x = (np.arange(64)-32)*1.25
        cfd_r = np.where(frac[kk, 32, :] > 0, fft_adc[b][kk, 32, :, 2], np.nan); cfd_p = np.where(frac[kk, :, 32] > 0, fft_adc[b][kk, :, 32, 2], np.nan)
        kom_r = np.where(frac[kk, 32, :] > 0, field[b][kk, 32, :, 2], np.nan); kom_p = np.where(frac[kk, :, 32] > 0, field[b][kk, :, 32, 2], np.nan)
        scn_r = sgn*sv[ss, ps, cy, :, 2]; scn_p = sgn*sv[ss, ps, :, cx, 2]
        xs_r = (np.arange(64)-cx)*1.25; xs_p = (np.arange(64)-cy)*1.25
        for row, (cf, km, sn, xsn, lab2) in enumerate(((cfd_r, kom_r, scn_r, xs_r, "R / scan column"), (cfd_p, kom_p, scn_p, xs_p, "P / scan row"))):
            ax = axes[row, j]
            ax.plot(xsn, sn, "-^", color=C["scan"], lw=2, ms=5, label="4D-flow scan")
            ax.plot(x, cf, "-o", color=C["cfd"], lw=2, ms=5, mfc="white", label="CFD")
            ax.plot(x, km, "-s", color=C["koma"], lw=2, ms=5, label="Koma")
            ax.set_xlim(-16, 16); ax.set_ylim(-0.6, 2.6); ax.axhline(0, color="#888", lw=.7); ax.grid(alpha=.25)
            ax.set_xlabel(f"across vessel, {lab2} (mm)"); ax.set_ylabel("forward axial velocity (m/s)")
            ax.set_title(f"{'throat' if d == 0 else f'{d} mm downstream'} (Koma z = {zmm[kk]:+.1f}, scan {spos[ss]-spos[st]:+.1f} mm)", fontsize=9*FS)
            if j == 0 and row == 0:
                ax.legend(frameon=False, fontsize=8*FS)
        prof[str(d)] = dict(koma_partition=kk, scan_slice=ss, koma_peak_center=float(np.nanmax(kom_r[28:37])), cfd_peak_center=float(np.nanmax(cfd_r[28:37])),
                            scan_peak_center=float(np.nanmax(scn_r[cx-4:cx+5])),
                            fwhm_mm=dict(koma=float(np.sum(kom_r >= .5*np.nanmax(kom_r))*1.25), cfd=float(np.sum(cfd_r >= .5*np.nanmax(cfd_r))*1.25),
                                         scan=float(np.sum(scn_r[cx-12:cx+13] >= .5*np.nanmax(scn_r[cx-12:cx+13]))*1.25)))
    fig.suptitle(f"Transverse jet profiles at peak systole (Koma/CFD bin {b+1}/30, scan phase {ps+1}/30; scan axes centred on its section centroid, not registered)", fontsize=12*FS)
    fig.savefig(stage/"jet_profiles.png", dpi=110); plt.close(fig)
    rep["jet_profiles"] = prof
    rep["evidence"] = "Koma vs CFD: same-source internal consistency (one 300k sample, noiseless). Scan comparisons: peak-inflow matched only; descriptive."
    (stage/"report.json").write_text(json.dumps(rep, indent=1, default=float)+chr(10), encoding="utf-8")
    (stage/"MANIFEST.json").write_text(json.dumps({p.name: sha(p) for p in sorted(stage.iterdir())}, indent=1)+chr(10), encoding="utf-8")
    stage.rename(target)
    print(json.dumps(dict(standard={k: rep["standard"][k] for k in ("T1", "T2_max_error", "T2_p99_error", "T3_min_section_coverage", "passes", "throat_axial_rmse_m_s", "vector_rmse_m_s", "vector_nrmse")},
                          bland_altman=ba, overshoot={k: v for k, v in mech.items() if k != "knot"}, flow_ratio=ratio, jet_profiles=prof), indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1])
