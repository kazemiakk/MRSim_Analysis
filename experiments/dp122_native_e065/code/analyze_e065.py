"""E-065 S4b: dp122 Koma (native 0.72-s cycle, scan VENCs) versus its CFD reference and the measured 4D-flow scan.

Koma and scan are measured with the identical frozen E-059 scan method (magnitude-threshold vessel segmentation per
slice, fraction 0.25, central 24 partitions = scan slice count): peak upstream flow, throat flow, peak forward
velocity. Koma versus CFD uses the actual-ADC references (references_v1) on the same grid. Waveforms are shown on
each dataset's own time axis with one rigid circular shift (peak at 200 ms); no time warping. Figures:
  comparison.png   flow waveforms, jet peak velocity along the vessel, centreline velocity, summary table
  longaxis.png     long-axis axial velocity at peak systole: scan, CFD, Koma, Koma + scan-matched noise
  peak_systole.png Koma axial throat slice + long-axis slice (magnitude and 3 velocity encodings), noise SNR 84
Presentation and comparison; the scan comparison is peak-inflow matched only (cycle length, waveform shape, fluid
and in-plane registration not matched).
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
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402

ROOT = Path("D:/MRSim/outputs/dp122_native_e065")
E059 = Path("D:/MRSim/outputs/measured_scan_e059")
E060 = Path("D:/MRSim/outputs/scanner_geometry_e060")
PRIVATE = Path("D:/MRSim/private_inputs/phantom_scan_2025-08-28/measured_v1/measured_30phase.npz")
sys.path.insert(0, str(E060/"code"))
import analyze_e060 as an  # noqa: E402

FS, SEED, SNR = 1.5, 20261025, 84.4
VENC = dict(X=1.5, Y=1.5, Z=3.0)
RR_KOMA, RR_SCAN, ALIGN_MS = 720.0, 1012.0, 200.0
VOX = 1.25
C = dict(scan="#d9822b", cfd="#1f1f1f", koma="#1f5fa8")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def tukey3(img, alpha=.5):
    w = np.ones(img.shape[-3:])
    for ax, n in enumerate(img.shape[-3:]):
        u = np.abs(np.fft.fftfreq(n))/.5
        wa = np.where(u <= 1-alpha, 1., .5*(1+np.cos(np.pi*(u-(1-alpha))/alpha)))
        sh = [1, 1, 1]; sh[ax] = n; w = w*wa.reshape(sh)
    return np.fft.ifftn(np.fft.fftn(img, axes=(-3, -2, -1))*w, axes=(-3, -2, -1))


def shift_to(t_ms, y, period_ms):
    """Rigid circular shift so the maximum sits at ALIGN_MS (no warping)."""
    d = ALIGN_MS - t_ms[int(np.argmax(y))]
    tt = np.mod(t_ms + d, period_ms); o = np.argsort(tt)
    return tt[o], np.asarray(y)[o], d


def per_slice_profiles(vz, mag_mean, frac=.25):
    """Per slice: lumen mask (scan method), centroid-centred 3x3 mean axial velocity and lumen maximum."""
    thr = frac*np.percentile(mag_mean, 99.5)
    cen, mx, masks = [], [], []
    for s in range(vz.shape[0]):
        lab, n = ndimage.label(mag_mean[s] >= thr)
        if n == 0:
            cen.append(np.nan); mx.append(np.nan); masks.append(None); continue
        sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n+1))
        best = 1+int(np.argmax([abs(vz[s][lab == c].sum()) if sizes[c-1] >= 5 else 0 for c in range(1, n+1)]))
        m = lab == best; cy, cx = ndimage.center_of_mass(m); cy, cx = int(round(cy)), int(round(cx))
        cen.append(float(vz[s, cy-1:cy+2, cx-1:cx+2].mean())); mx.append(float(vz[s][m].max())); masks.append((m, cy, cx))
    return np.array(cen), np.array(mx), masks


def main(target):
    target = Path(target); stage = target.with_name(target.name+".building")
    if target.exists() or stage.exists():
        raise FileExistsError(target)
    plt.rcParams["font.size"] = 10*FS
    control = tomllib.loads((ROOT/"acquisition_v1/control_report.toml").read_text())
    if control["status"] != "COMPLETE_E065_F64_300K_DP122_NATIVE_ACQUISITIONS":
        raise ValueError("acquisition not complete")
    images, audits = an.load_images(ROOT/"acquisition_v1", control, ROOT/"sequences_v1/timing_e060.h5")
    field = np.stack([np.angle(images[e]*images["REF"].conj())*VENC[e]/np.pi for e in ("X", "Y", "Z")], -1)   # (bin,S,P,R,3) m/s
    with h5py.File(ROOT/"references_v1/references.h5") as f:
        ref, frac, pref, pfr, zz = f["fft_adc"][:], f["fraction_fft"][:], f["plane_reference"][:], f["plane_fraction"][:], f["partition_z_m"][:]*1e3
    rrep = json.loads((ROOT/"references_v1/report.json").read_text())
    # --- Koma with the frozen scan method (central partitions as the scan's 24 slices) ---
    ks = an.scan_style(field, images, zz, .25)
    kq = np.array(ks["flow_ml_s"])                                         # (24 slices, 30 bins)
    k_up = kq[ks["upstream_slices"]].mean(axis=0)
    b = int(np.argmax(k_up))
    # --- CFD reference ---
    q_cfd = np.array(rrep["upstream_flow_ml_s"]); b_cfd = int(np.argmax(q_cfd))
    lum = frac > 0; throat_reg = lum & (np.abs(zz) <= 10)[:, None, None]
    cfd_vmax = float(ref[b][..., 2][throat_reg].max()); koma_vmax_lumen = float(field[b][..., 2][throat_reg].max())
    koma_p99 = float(np.percentile(field[b][..., 2][throat_reg], 99)); cfd_p99 = float(np.percentile(ref[b][..., 2][throat_reg], 99))
    # --- scan (E-059 frozen analysis, own peak) ---
    srep = json.loads((E059/"measured_v2/report.json").read_text()); seg = srep["segmentation"]["0.25"]
    sq = np.array([r["flow_ml_s"] for r in seg["slices"]])
    s_up = sq[srep["upstream_slices"]].mean(axis=0); ps = int(np.argmax(s_up))
    sc = np.load(PRIVATE); sv, smag, spos, trig = sc["velocity_cm_s"]/100, sc["magnitude"], sc["slice_positions_mm"], sc["trigger_ms"]
    sgn = seg["vendor_to_forward_sign"]
    # --- per-slice jet profiles at each dataset's peak (centreline and lumen maximum) ---
    kc = list(an.CENTRAL_K)
    k_cen, k_max, _ = per_slice_profiles(field[b][kc][..., 2], np.abs(images["REF"][:, kc]).mean(axis=0))
    c_cen = np.array([ref[b][k, 31:34, 31:34, 2].mean() for k in kc]); c_max = np.array([ref[b][k][..., 2][frac[k] > 0].max() for k in kc])
    s_cen, s_max, s_masks = per_slice_profiles(sgn*sv[:, ps][..., 2], smag.mean(axis=1))
    z_k = zz[kc]; z_s = spos - spos[srep["throat_slice"]]
    rep = dict(experiment="E-065", stage="S4b_analysis", utc=datetime.now(timezone.utc).isoformat(), script_sha256=sha(__file__),
               reconstruction_audits=audits, koma_peak_bin=b+1, cfd_peak_bin=b_cfd+1, scan_peak_phase=ps+1,
               upstream_peak_flow_ml_s=dict(cfd=float(q_cfd.max()), koma=float(k_up.max()), scan=float(s_up.max())),
               upstream_mean_flow_ml_s=dict(cfd=float(q_cfd.mean()), koma=float(k_up.mean()), scan=float(s_up.mean())),
               peak_forward_velocity_m_s=dict(koma_scan_method=ks["peak_forward_velocity_cm_s"]/100, scan=srep["peak_forward_velocity_cm_s"]/100,
                                              cfd_throat_region=cfd_vmax, koma_throat_region=koma_vmax_lumen, cfd_p99=cfd_p99, koma_p99=koma_p99),
               koma_vs_cfd=dict(peak_throat_velocity_error=koma_vmax_lumen/cfd_vmax-1, p99_error=koma_p99/cfd_p99-1,
                                peak_upstream_flow_error=float(k_up.max()/q_cfd.max()-1), mean_upstream_flow_error=float(k_up.mean()/q_cfd.mean()-1)),
               koma_vs_scan=dict(peak_upstream_flow_ratio=float(k_up.max()/s_up.max()), peak_velocity_ratio=ks["peak_forward_velocity_cm_s"]/srep["peak_forward_velocity_cm_s"],
                                 throat_vs_upstream_mean_flow=dict(koma=ks["throat_vs_upstream_mean_flow"], scan=srep["throat_vs_upstream_mean_flow"]),
                                 throat_diameter_mm=dict(koma=ks["throat_diameter_mm"], scan=srep["throat_diameter_mm"])),
               profiles=dict(z_koma_mm=z_k.tolist(), z_scan_mm=z_s.tolist(), koma_centre=k_cen.tolist(), koma_max=k_max.tolist(),
                             cfd_centre=c_cen.tolist(), cfd_max=c_max.tolist(), scan_centre=s_cen.tolist(), scan_max=s_max.tolist()),
               evidence=("Koma vs CFD: same-source internal consistency. Koma vs scan: peak-inflow-matched comparison with an independent "
                         "measurement; cycle length (0.72 vs 1.012 s), waveform shape, fluid properties and in-plane registration not matched."))
    stage.mkdir()
    # ---------------- comparison figure ----------------
    fig, ax = plt.subplots(2, 2, figsize=(22, 14), layout="constrained")
    a = ax[0, 0]
    tk = (np.arange(30)+.5)/30*RR_KOMA; tsc = np.asarray(trig, float)
    for name, t, y, per, col, mk in (("CFD reference (dp122)", tk, q_cfd, RR_KOMA, C["cfd"], "o"), ("Koma (dp122)", tk, k_up, RR_KOMA, C["koma"], "s"),
                                     ("4D-flow scan", tsc, s_up, RR_SCAN, C["scan"], "^")):
        tt, yy, d = shift_to(t, y, per)
        a.plot(tt, yy, "-"+mk, color=col, lw=2, ms=6, mfc="white" if name.startswith("CFD") else col, label=f"{name} (cycle {per:.0f} ms)")
    a.set_xlabel("time in cycle (ms); each curve circularly shifted so its peak is at 200 ms"); a.set_ylabel("upstream flow (mL/s)")
    a.set_title("Upstream flow waveform"); a.legend(fontsize=8*FS, frameon=False); a.grid(color="#ddd")
    a = ax[0, 1]
    a.plot(z_s, s_max, "-^", color=C["scan"], lw=2, label=f"4D-flow scan (phase {ps+1}/30)")
    a.plot(z_k, c_max, "-o", color=C["cfd"], lw=2, mfc="white", label=f"CFD reference (bin {b+1}/30)")
    a.plot(z_k, k_max, "-s", color=C["koma"], lw=2, label=f"Koma (bin {b+1}/30)")
    a.set_xlabel("distance from throat along vessel (mm)"); a.set_ylabel("maximum forward velocity in section (m/s)")
    a.set_title("Jet peak velocity along the vessel, peak systole"); a.legend(fontsize=8*FS, frameon=False); a.grid(color="#ddd")
    a = ax[1, 0]
    a.plot(z_s, s_cen, "-^", color=C["scan"], lw=2, label="4D-flow scan"); a.plot(z_k, c_cen, "-o", color=C["cfd"], lw=2, mfc="white", label="CFD reference")
    a.plot(z_k, k_cen, "-s", color=C["koma"], lw=2, label="Koma")
    a.set_xlabel("distance from throat along vessel (mm)"); a.set_ylabel("centreline forward velocity, 3x3 mean (m/s)")
    a.set_title("Centreline velocity, peak systole"); a.legend(fontsize=8*FS, frameon=False); a.grid(color="#ddd")
    a = ax[1, 1]; a.axis("off")
    u, pv = rep["upstream_peak_flow_ml_s"], rep["peak_forward_velocity_m_s"]
    rows = [["", "CFD (dp122)", "Koma", "4D-flow scan"],
            ["Peak upstream flow (mL/s)", f"{u['cfd']:.1f}", f"{u['koma']:.1f}", f"{u['scan']:.1f}"],
            ["Mean upstream flow (mL/s)", f"{rep['upstream_mean_flow_ml_s']['cfd']:.1f}", f"{rep['upstream_mean_flow_ml_s']['koma']:.1f}", f"{rep['upstream_mean_flow_ml_s']['scan']:.1f}"],
            ["Peak forward velocity, scan method (m/s)", "–", f"{pv['koma_scan_method']:.2f}", f"{pv['scan']:.2f}"],
            ["Peak throat-region velocity (m/s)", f"{cfd_vmax:.2f}", f"{koma_vmax_lumen:.2f}", "–"],
            ["Throat diameter, scan method (mm)", "9.16 (geometry)", f"{ks['throat_diameter_mm']:.2f}", f"{srep['throat_diameter_mm']:.2f}"],
            ["Cardiac cycle (ms)", "720", "720", "1012"]]
    t = a.table(cellText=rows, loc="center", cellLoc="center"); t.auto_set_font_size(False); t.set_fontsize(9*FS); t.scale(1, 2.6)
    a.set_title("Summary (scan: measured; Koma/CFD: native dp122 cycle)")
    fig.suptitle("Stenotic phantom: inflow-matched CFD (dp122), Koma and the measured 4D-flow scan", fontsize=12*FS)
    fig.supxlabel("Koma and scan analysed with the identical E-059 scan method. Peak inflow matched (CFD 120 vs scan ~118 mL/s); cycle length, waveform "
                  "shape, fluid and in-plane registration not matched, so timing is not compared.", fontsize=8*FS)
    fig.savefig(stage/"comparison.png", dpi=110); plt.close(fig)
    # ---------------- long-axis axial velocity, peak systole ----------------
    sl = list(kc)
    s_cy, s_cx = s_masks[srep["throat_slice"]][1], s_masks[srep["throat_slice"]][2]
    scan_la = (sgn*sv[:, ps, :, s_cx, 2]).T                               # rows = scan rows, cols = slices
    cfd_la = ref[b][sl, :, 32, 2].T; cfd_la = np.where(frac[sl][:, :, 32].T > 0, cfd_la, np.nan)
    koma_la = field[b][sl, :, 32, 2].T
    rng = np.random.default_rng(SEED)
    sigma = float(np.median(np.abs(images["REF"][b])[(frac > 0) & (np.abs(zz) <= 30)[:, None, None]]))/SNR
    noisy = {e: tukey3(images[e][b]+sigma*(rng.standard_normal(images[e][b].shape)+1j*rng.standard_normal(images[e][b].shape))) for e in ("REF", "Z")}
    koma_noisy_la = (np.angle(noisy["Z"]*noisy["REF"].conj())*VENC["Z"]/np.pi)[sl, :, 32].T
    fig, axs = plt.subplots(1, 4, figsize=(26, 7.5), layout="constrained")
    ext_k = [z_k[0]-1.25, z_k[-1]+1.25, -40, 40]; ext_s = [z_s[0]-1.25, z_s[-1]+1.25, -(s_cy-32)*VOX-40, -(s_cy-32)*VOX+40]
    for a, img, ext, title in ((axs[0], scan_la, ext_s, f"4D-flow scan (phase {ps+1}/30)"), (axs[1], cfd_la, ext_k, f"CFD reference (bin {b+1}/30)"),
                               (axs[2], koma_la, ext_k, "Koma, noiseless"), (axs[3], koma_noisy_la, ext_k, "Koma + scan-matched noise (SNR 84)")):
        im = a.imshow(img[::-1], cmap="viridis", vmin=0, vmax=2.5, extent=ext, aspect="equal", interpolation="nearest")
        a.set_title(title, fontsize=10*FS); a.set_xlabel("along vessel from throat (mm)"); a.set_ylim(-20, 20); a.set_facecolor("#e8e8e8")
    axs[0].set_ylabel("in-plane (mm)")
    fig.colorbar(im, ax=axs.tolist(), shrink=.8, label="forward axial velocity (m/s)")
    fig.suptitle("Long-axis slice through the vessel axis at peak systole (scan VENC 3 m/s; Koma VENC 3 m/s)", fontsize=12*FS)
    fig.supxlabel("Scan: own peak phase, plane through the throat centroid, in-plane axes not registered to the CFD. Display windows identical.", fontsize=8*FS)
    fig.savefig(stage/"longaxis.png", dpi=110); plt.close(fig)
    rep["display_noise_sigma"] = sigma
    (stage/"report.json").write_text(json.dumps(rep, indent=1, default=float)+chr(10), encoding="utf-8")
    (stage/"MANIFEST.json").write_text(json.dumps({p.name: sha(p) for p in sorted(stage.iterdir())}, indent=1)+chr(10), encoding="utf-8")
    stage.rename(target)
    print(json.dumps({k: rep[k] for k in ("upstream_peak_flow_ml_s", "peak_forward_velocity_m_s", "koma_vs_cfd", "koma_vs_scan")}, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1])
