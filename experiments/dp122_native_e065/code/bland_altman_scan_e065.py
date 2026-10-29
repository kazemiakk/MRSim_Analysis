"""E-065 (user request): Bland-Altman of the measured scan against dp122 CFD and Koma, scan cropped to the 720-ms cycle.

Sections: the scan's 24 slices and Koma's central 24 partitions share the 2.5-mm grid relative to their throats
(-30 ... +27.5 mm), paired one to one. Time: each dataset is circularly shifted (rigid, no warping) so its upstream
flow peak sits at 0 ms; scan phases with shifted time in [-PRE, 720-PRE) are kept, and CFD/Koma are sampled at those
times by linear interpolation along their own periodic 0.72-s cycle. Flow: E-059 scan-method section flow (Koma and
scan identical method; CFD = plane-reference section flow). Peak: maximum forward axial velocity in the scan-method
vessel mask (CFD: in its lumen). Descriptive only: pairs waveforms of different shape and cycle length, so diastolic
shape differences count as disagreement; samples are correlated within sections and phases.
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
PRIVATE = Path("D:/MRSim/private_inputs/phantom_scan_2025-08-28/measured_v1/measured_30phase.npz")
sys.path.insert(0, "D:/MRSim/outputs/scanner_geometry_e060/code")
import analyze_e060 as an  # noqa: E402
FS, PRE = 1.5, 200.0          # peak placed 200 ms into the kept 720-ms window
COL = dict(cfd="#1f1f1f", koma="#1f5fa8")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def masks(vz, mag_mean, frac=.25):
    """E-059 vessel mask per slice: largest |mean flow| component of time-mean magnitude >= frac * p99.5."""
    thr = frac*np.percentile(mag_mean, 99.5); out = []
    for s in range(vz.shape[0]):
        lab, n = ndimage.label(mag_mean[s] >= thr)
        q = [abs(vz[s][:, lab == c].sum(axis=1).mean()) for c in range(1, n+1)]
        out.append(lab == 1+int(np.argmax(q)))
    return out


def periodic_interp(t, y, period, tq):
    t = np.r_[t, t[0]+period]; y = np.r_[y, y[0]]
    return np.interp(np.mod(tq-t[0], period)+t[0], t, y)


def main(target):
    target = Path(target); stage = target.with_name(target.name+".building")
    if target.exists() or stage.exists():
        raise FileExistsError(target)
    plt.rcParams["font.size"] = 10*FS
    control = tomllib.loads((ROOT/"acquisition_v1/control_report.toml").read_text())
    images, _ = an.load_images(ROOT/"acquisition_v1", control, ROOT/"sequences_v1/timing_e060.h5")
    kc = list(an.CENTRAL_K)
    kvz = (np.angle(images["Z"]*images["REF"].conj())*3.0/np.pi)[:, kc].transpose(1, 0, 2, 3)       # (24, 30, P, R) m/s
    kmag = np.abs(images["REF"][:, kc]).transpose(1, 0, 2, 3)
    km = masks(kvz, kmag.mean(axis=1))
    k_flow = np.array([kvz[s][:, km[s]].sum(axis=1)*an.AREA_MM2 for s in range(24)])                # mL/s (m/s x mm^2)
    k_peak = np.array([kvz[s][:, km[s]].max(axis=1) for s in range(24)])
    with h5py.File(ROOT/"references_v1/references.h5") as f:
        ref, frac, plane, pfrac = f["fft_adc"][:, kc][..., 2], f["fraction_fft"][kc], f["plane_reference"][:, kc][..., 2], f["plane_fraction"][kc]
    c_flow = np.einsum("bspr,spr->sb", np.nan_to_num(plane.astype(float)), pfrac*an.AREA_MM2)
    c_peak = np.array([[np.nanmax(np.where(frac[s] > 0, ref[b, s], np.nan)) for b in range(30)] for s in range(24)])
    srep = json.loads((E059/"measured_v2/report.json").read_text()); seg = srep["segmentation"]["0.25"]; sgn = seg["vendor_to_forward_sign"]
    sc = np.load(PRIVATE); svz = sgn*sc["velocity_cm_s"][..., 2]/100; smag = sc["magnitude"]; trig = np.asarray(sc["trigger_ms"], float)
    sm = masks(svz, smag.mean(axis=1))
    s_flow = np.array([r["flow_ml_s"] for r in seg["slices"]])
    s_peak = np.array([svz[s][:, sm[s]].max(axis=1) for s in range(24)])
    # time alignment: upstream-flow peak -> PRE ms in the kept window
    tk = (np.arange(30)+.5)/30*720.
    k_up = k_flow[:6].mean(axis=0); c_up = c_flow[:6].mean(axis=0); s_up = s_flow[srep["upstream_slices"]].mean(axis=0)
    tk_k = tk-tk[int(np.argmax(k_up))]+PRE; tk_c = tk-tk[int(np.argmax(c_up))]+PRE
    ts = np.mod(trig-trig[int(np.argmax(s_up))]+PRE, 1012.)
    keep = ts < 720.
    rows = []
    for s in range(24):
        for i in np.nonzero(keep)[0]:
            rows.append(dict(section=s, z_mm=-30+2.5*s, phase=int(i)+1, t_ms=float(ts[i]),
                             scan_flow=float(s_flow[s, i]), scan_peak=float(s_peak[s, i]),
                             cfd_flow=float(periodic_interp(np.sort(np.mod(tk_c, 720)), c_flow[s][np.argsort(np.mod(tk_c, 720))], 720., ts[i])),
                             cfd_peak=float(periodic_interp(np.sort(np.mod(tk_c, 720)), c_peak[s][np.argsort(np.mod(tk_c, 720))], 720., ts[i])),
                             koma_flow=float(periodic_interp(np.sort(np.mod(tk_k, 720)), k_flow[s][np.argsort(np.mod(tk_k, 720))], 720., ts[i])),
                             koma_peak=float(periodic_interp(np.sort(np.mod(tk_k, 720)), k_peak[s][np.argsort(np.mod(tk_k, 720))], 720., ts[i]))))
    stats = {}
    fig, axes = plt.subplots(1, 2, figsize=(22, 8.5), layout="constrained")
    for ax, q, unit in ((axes[0], "flow", "mL/s"), (axes[1], "peak", "m/s")):
        for who in ("cfd", "koma"):
            a = np.array([r["scan_"+q] for r in rows]); b = np.array([r[who+"_"+q] for r in rows])
            ok = np.isfinite(a) & np.isfinite(b); a, b = a[ok], b[ok]; d = b-a
            bias, sd = float(d.mean()), float(d.std(ddof=1))
            stats[f"{who}_vs_scan_{q}"] = dict(n=int(d.size), bias=bias, sd=sd, loa_low=bias-1.96*sd, loa_high=bias+1.96*sd,
                                               relative_bias_pct=float(100*d.sum()/a.sum()), r=float(np.corrcoef(a, b)[0, 1]))
            lab = "CFD" if who == "cfd" else "Koma"
            ax.scatter((a+b)/2, d, s=12, alpha=.45, color=COL[who], edgecolor="none",
                       label=f"{lab} − MRI: bias {bias:+.3g}, LoA {bias-1.96*sd:+.3g} to {bias+1.96*sd:+.3g} {unit}")
            for y, ls in ((bias, "-"), (bias+1.96*sd, (0, (4, 3))), (bias-1.96*sd, (0, (4, 3)))):
                ax.axhline(y, color=COL[who], lw=1.3, ls=ls)
        ax.axhline(0, color="#888", lw=.7); ax.grid(alpha=.25); ax.legend(frameon=False, fontsize=8*FS, loc="upper left")
        ax.set_xlabel(f"mean of simulation and MRI ({unit})"); ax.set_ylabel(f"simulation − MRI ({unit})")
        ax.set_title(f"Section {'flow' if q == 'flow' else 'peak velocity'} ({int(keep.sum())} MRI phases × 24 sections)", fontsize=10*FS)
    fig.suptitle("dp122 CFD and Koma vs 4D-flow MRI, MRI cropped to the 720-ms CFD/Koma cycle", fontsize=12*FS)
    fig.supxlabel("Peaks aligned by rigid circular shift; CFD/Koma sampled at the MRI phase times (linear, periodic). Waveform shape and cycle length "
                  "differ, so differences include inflow-shape mismatch. Correlated samples; descriptive.", fontsize=8*FS)
    stage.mkdir(); fig.savefig(stage/"bland_altman_scan_dp122.png", dpi=110); plt.close(fig)
    rep = dict(experiment="E-065", stage="scan_bland_altman", utc=datetime.now(timezone.utc).isoformat(), script_sha256=sha(__file__),
               kept_scan_phases=[int(i)+1 for i in np.nonzero(keep)[0]], peak_alignment_ms=PRE, stats=stats, rows=rows,
               evidence="Peak-inflow-matched comparison with an independent measurement; descriptive; waveform shape not matched.")
    (stage/"report.json").write_text(json.dumps(rep, indent=1)+chr(10), encoding="utf-8")
    (stage/"MANIFEST.json").write_text(json.dumps({p.name: sha(p) for p in sorted(stage.iterdir())}, indent=1)+chr(10), encoding="utf-8")
    stage.rename(target); print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
