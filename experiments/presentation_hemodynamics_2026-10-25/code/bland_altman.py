"""Bland-Altman agreement, Koma vs CFD (user request): section flow and section peak velocity.

Phantom: E-060 sections k = 3..26 (60 mm slab) x 30 bins, through the registered E-058/E-060 evaluator
(et.evaluate, per-bin support, same-area CFD flow); peak = maximum axial velocity over the same supported
section voxels (Koma field vs CFD volume reference). Arms: E-060 VENC 3 and E-061 VENC 4 (Z re-acquired),
reference REF2. Aorta: E-062 sections (4) x 20 bins, registered section_series (core15 section pixels,
plane-fraction area, forward-positive); peak = maximum forward normal velocity over the same section
pixels; E-027 8-bin points for comparison. Difference = Koma - CFD; bias and 95 % limits (+-1.96 SD).
Same-source internal consistency, noiseless. Presentation only.
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

E060 = Path("D:/MRSim/outputs/scanner_geometry_e060")
E061 = Path("D:/MRSim/outputs/venc_unwrap_e061")
sys.path.insert(0, str(E061/"code")); sys.path.insert(1, str(E060/"code"))
sys.path.insert(2, "D:/MRSim/outputs/aorta_presentation_2026-10-25/code")
import analyze_e061 as a61  # noqa: E402
from aorta_common import Aorta, E027  # noqa: E402

an = a61.an; et = an.et
FS = 1.5
COL = dict(v3="#2a78d6", v4="#1baf7a", k20="#2a78d6", k8="#eb6834")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def phantom():
    control = tomllib.loads((E060/"acquisition_v1/control_report.toml").read_text())
    images, _ = an.load_images(E060/"acquisition_v1", control, E060/"sequences_v2/timing_e060.h5")
    z4, _ = a61.load_z4()
    images4 = dict(images); images4["Z"] = z4
    with h5py.File(E060/"references_v2/references.h5") as f:
        fft_adc, plane, frac, pfrac, zpart = (f[k][:] for k in ("fft_adc", "plane_reference", "fraction_fft", "plane_fraction", "partition_z_m"))
    vt, pt = an.to_e058(fft_adc, True), an.to_e058(plane, True)
    fr, pf = frac.transpose(2, 1, 0), pfrac.transpose(2, 1, 0)
    region = (fr > 0) & ((zpart >= an.LOW) & (zpart < an.HIGH))[None, None, :]
    out = {}
    for key, ims, vencs in (("v3", images, (1.5, 1.5, 3.0)), ("v4", images4, (1.5, 1.5, 4.0))):
        ft = an.to_e058(a61.velocity(ims, vencs, False), True)
        it = {e: an.to_e058(v) for e, v in ims.items()}
        support = et.per_bin_support(it)
        area_t = pf[:, :, an.THROAT_K]*an.AREA_MM2; full_t = area_t > 0
        peak = int(np.argmax([pt[b, :, :, an.THROAT_K, 0][full_t]@area_t[full_t] for b in range(an.SHAPE[0])]))
        rows = []
        for k in an.CENTRAL_K:
            a_k = pf[:, :, k]*an.AREA_MM2; full = a_k > 0
            t1 = et.evaluate(ft, support, vt, pt[:, :, :, k, 0], fr, a_k, region, k, peak)["T1_throat_flow"]
            for r in t1["bins"]:
                b = r["bin"]-1; keep = support[b][:, :, k] & full
                rows.append(dict(section=k, bin=b+1, koma_flow=r["Koma_ml_s"], cfd_flow=r["CFD_same_area_ml_s"],
                                 koma_peak=float(ft[b, :, :, k, 0][keep].max()) if keep.any() else np.nan,
                                 cfd_peak=float(vt[b, :, :, k, 0][keep].max()) if keep.any() else np.nan))
        out[key] = rows
    return out


def aorta():
    A = Aorta(); g = A.grid.spacing
    out = {}
    for key, rows_src, ref, koma, core, frac in (("k20", A.ana["primary_20bin"]["section_series"], A.ref, A.koma, A.core, A.frac),
                                                 ("k8", A.rep027["section_series"], A.ref8, A.koma8, A.core8, A.frac8)):
        rows = []
        for name, sec in A.secs.items():
            sr = [r for r in rows_src if r["section"] == name]
            x = sr[0]["x_m"]; ix = int(np.argmin(abs(A.gx-x)))
            cf = {r["bin"]: r["net_ml_s"] for r in sr if r["method"] == "CFD"}; km = {r["bin"]: r["net_ml_s"] for r in sr if r["method"] == "Koma"}
            sgn = -1.0 if np.mean(list(cf.values())) < 0 else 1.0
            valid = sec & core[:, :, ix] & (A.pfrac[:, :, ix] > 0)
            for b in sorted(cf):
                rows.append(dict(section=name, bin=b, koma_flow=sgn*km[b], cfd_flow=sgn*cf[b],
                                 koma_peak=float((sgn*koma[b-1][:, :, ix, 0])[valid].max()), cfd_peak=float((sgn*np.nan_to_num(ref[b-1][:, :, ix, 0]))[valid].max())))
        out[key] = rows
    return out


def ba(ax, sets, quantity, unit, title):
    stats = {}
    for key, rows, lab in sets:
        a = np.array([r["cfd_"+quantity] for r in rows], float); k = np.array([r["koma_"+quantity] for r in rows], float)
        ok = np.isfinite(a) & np.isfinite(k); a, k = a[ok], k[ok]
        mean, diff = (a+k)/2, k-a
        bias, sd = float(diff.mean()), float(diff.std(ddof=1))
        ax.scatter(mean, diff, s=14, color=COL[key], alpha=.55, edgecolor="none", label=f"{lab}: bias {bias:+.3g}, LoA ±{1.96*sd:.3g} {unit}")
        for y, ls in ((bias, "-"), (bias+1.96*sd, (0, (4, 3))), (bias-1.96*sd, (0, (4, 3)))):
            ax.axhline(y, color=COL[key], lw=1.4, ls=ls)
        stats[key] = dict(n=int(len(diff)), bias=bias, sd=sd, loa_low=bias-1.96*sd, loa_high=bias+1.96*sd,
                          relative_bias_pct=float(100*diff.sum()/a.sum()), r=float(np.corrcoef(a, k)[0, 1]))
    ax.axhline(0, color="#888888", lw=.7)
    ax.set_title(title, fontsize=10*FS); ax.set_xlabel(f"Mean of Koma and CFD ({unit})", fontsize=8*FS)
    ax.set_ylabel(f"Koma − CFD ({unit})", fontsize=9*FS); ax.grid(alpha=.25); ax.tick_params(labelsize=8*FS)
    ax.legend(fontsize=7*FS, frameon=False, loc="upper left")
    return stats


def main(target):
    target = Path(target)
    if target.exists():
        raise FileExistsError(target)
    P = phantom(); Q = aorta()
    plt.rcParams["font.size"] = 10*FS
    out = target.with_name(target.name+".building"); out.mkdir(parents=True)
    fig, axes = plt.subplots(2, 2, figsize=(22, 15), layout="constrained")
    stats = dict(
        phantom_flow=ba(axes[0, 0], (("v4", P["v4"], "VENC 4"), ("v3", P["v3"], "VENC 3")), "flow", "mL/s", "Phantom: section flow (24 sections × 30 phases)"),
        phantom_peak=ba(axes[0, 1], (("v4", P["v4"], "VENC 4"), ("v3", P["v3"], "VENC 3")), "peak", "m/s", "Phantom: section peak velocity"),
        aorta_flow=ba(axes[1, 0], (("k8", Q["k8"], "8 bins"), ("k20", Q["k20"], "20 bins")), "flow", "mL/s", "Aorta: section flow (4 sections × 20 / 8 phases)"),
        aorta_peak=ba(axes[1, 1], (("k8", Q["k8"], "8 bins"), ("k20", Q["k20"], "20 bins")), "peak", "m/s", "Aorta: section peak velocity"))
    fig.suptitle("Bland–Altman agreement: Koma vs CFD (same voxels, same acquisition windows)", fontsize=13*FS)
    fig.supxlabel("Solid: bias; dashed: 95 % limits of agreement (±1.96 SD). Phantom: E-060/E-061 registered section definitions (per-bin support, same area). "
                  "Aorta: E-062/E-027 section definitions (core-15 % pixels).\nPeak = maximum forward through-plane velocity over the same section voxels. "
                  "Same-source simulation, no noise; the 4D-flow scan is not included because its flow differs from the CFD by about 60 %.", fontsize=8.5*FS)
    fig.savefig(out/"bland_altman_koma_cfd.png", dpi=110); plt.close(fig)
    rep = dict(created_utc=datetime.now(timezone.utc).isoformat(), status="PRESENTATION_DERIVATIVE_ONLY", script_sha256=sha(__file__), stats=stats,
               points=dict(phantom=P, aorta=Q))
    rep["outputs"] = {p.name: sha(p) for p in sorted(out.glob("*.png"))}
    (out/"MANIFEST.json").write_text(json.dumps(rep, indent=1, default=float)+"\n", encoding="utf-8")
    out.rename(target)
    for k, s in stats.items():
        print(k, {a: {m: round(v, 4) if isinstance(v, float) else v for m, v in b.items()} for a, b in s.items()})


if __name__ == "__main__":
    main(sys.argv[1])
