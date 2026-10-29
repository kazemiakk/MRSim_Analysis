"""Presentation: plot_projection_with_scan.py figure plus a second row of time plots:
throat flow waveform, and throat-section mean u, v, w (4D-flow MRI / CFD / Koma VENC 3).

Throat flow: scan = E-059 throat-slice flow (25 % mask); CFD = full-section throat flow and
Koma = same-area throat flow, both from E-060 analysis_v1 (registered numbers).
u, v, w: area-mean over the throat section (scan: E-059 vessel mask; CFD: lumen, fraction-
weighted; Koma: lumen and per-bin 5 % support, fraction-weighted). u, v are in-plane
(scan x, y vs CFD/Koma R, P; scan orientation not registered, so in-plane signs/axes are
not comparable one-to-one); w is through-plane (forward positive). Each dataset rigidly
circularly shifted so its flow peak sits at phase 3 (no time warping); 33.73 ms per phase.
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
import plot_projection_with_scan as ws  # noqa: E402
sm, tri = ws.sm, ws.tri

PEAK_AT, DT = 3, 1012/30
COLORS = dict(scan="#eb6834", cfd="#0b0b0b", koma="#2a78d6")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def shifted(x, shift):
    return np.roll(np.asarray(x, float), shift, axis=0)


def scan_series():
    rep = json.loads((ws.E059/"measured_v2/report.json").read_text())
    z = np.load(ws.PRIVATE)
    v, mag = z["velocity_cm_s"]/100.0, z["magnitude"]
    sign = rep["segmentation"]["0.25"]["vendor_to_forward_sign"]
    t = rep["throat_slice"]
    m = mag.mean(axis=1); thr = .25*np.percentile(m, 99.5)
    lab, n = ndimage.label(m[t] >= thr)
    flows = [abs((sign*v[t, :, :, :, 2][:, lab == c]).sum(axis=1).mean()) for c in range(1, n+1)]
    mask = lab == (1+int(np.argmax(flows)))
    uvw = np.stack([v[t, :, :, :, 0][:, mask].mean(axis=1), v[t, :, :, :, 1][:, mask].mean(axis=1),
                    sign*v[t, :, :, :, 2][:, mask].mean(axis=1)], axis=-1)
    flow = np.array(rep["segmentation"]["0.25"]["slices"][t]["flow_ml_s"])
    return flow, uvw


def sim_series():
    report = json.loads((E060/"analysis_v1/report.json").read_text())
    bins = report["targets_per_bin_support"]["T1_throat_flow"]["bins"]
    images, vencs = tri.arms()["E060_VENC3"]
    k = tri.an.THROAT_K
    with h5py.File(E060/"references_v2/references.h5") as f:
        ref, frac = f["fft_adc"][:, k], f["fraction_fft"][k]
    lum = frac > 0
    cfd_uvw, koma_uvw = [], []
    for b in range(ref.shape[0]):
        sup = np.ones(lum.shape, bool)
        for im in images.values():
            mm = np.abs(im[b]); sup &= mm[k] >= .05*mm.max()
        kv = np.stack([np.angle(images[e][b, k]*images["REF"][b, k].conj())*(vencs[i]/np.pi) for i, e in enumerate(tri.an.ENC[1:])], axis=-1)
        w_c = np.where(lum, frac, 0); w_k = np.where(lum & sup, frac, 0)
        # v2: CFD stores NaN outside its geometry (zero weight there); v1 let NaN void the whole mean.
        cfd_uvw.append((np.nan_to_num(ref[b])*w_c[..., None]).sum(axis=(0, 1))/w_c.sum())
        koma_uvw.append((kv*w_k[..., None]).sum(axis=(0, 1))/w_k.sum())
    cfd_flow = np.array([r["CFD_full_section_ml_s"] for r in bins])
    koma_flow = np.array([r["Koma_ml_s"] for r in bins])
    return (cfd_flow, np.array(cfd_uvw)), (koma_flow, np.array(koma_uvw))


def main(target):
    target = Path(target)
    if target.exists():
        raise FileExistsError(target)
    series = dict(scan=scan_series())
    series["cfd"], series["koma"] = sim_series()
    shifts = {k: PEAK_AT-int(np.argmax(f)) for k, (f, _) in series.items()}
    t = np.arange(30)*DT
    # Top block: reuse the image panels of plot_projection_with_scan by rendering them here.
    with h5py.File(E060/"references_v2/references.h5") as f:
        ref, frac, z = f["fft_adc"][:], f["fraction_fft"][:], f["partition_z_m"][:]*1e3
    rS = sm.wall(z)
    scan, zs, sphase, strig, shift, cy = ws.scan_projection()
    report = json.loads((E060/"analysis_v1/report.json").read_text()); b = report["peak_bin"]-1
    images, vencs = tri.arms()["E060_VENC3"]
    field = np.stack([np.angle(images[e][b]*images["REF"][b].conj())*(vencs[i]/np.pi) for i, e in enumerate(tri.an.ENC[1:])], axis=-1)
    sup = np.ones(images["REF"].shape[1:], bool)
    for im in images.values():
        mm = np.abs(im[b]); sup &= mm >= .05*mm.max()
    core = sup & (frac > 0)
    a = sm.smooth(tri.project(np.linalg.norm(ref[b].astype(float), axis=-1), core, frac, 2), z, rS)
    k = sm.smooth(tri.project(np.linalg.norm(field, axis=-1), core, frac, 2), z, rS)
    s = sm.smooth(scan, zs, rS)
    vmax = float(np.nanmax(a)); d = k-a; dmax = float(np.nanmax(np.abs(d)))
    extent = [sm.PF[0]-sm.STEP/2, sm.PF[-1]+sm.STEP/2, sm.SF[0]-sm.STEP/2, sm.SF[-1]+sm.STEP/2]
    cmap = plt.get_cmap("viridis").copy(); cmap.set_bad("#eeeeee")
    dmap = plt.get_cmap("RdBu_r").copy(); dmap.set_bad("#eeeeee")
    stage = target.with_name(target.name+".building"); stage.mkdir()
    fig = plt.figure(figsize=(17, 16), layout="constrained")
    gs = fig.add_gridspec(2, 1, height_ratios=[3.3, 1])
    top = gs[0].subgridspec(2, 2); bottom = gs[1].subgridspec(1, 4)
    panels = [(s, f"4D-flow MRI (phase {sphase}, own peak)", cmap, (0, vmax)), (a, f"CFD (bin {b+1}/30)", cmap, (0, vmax)),
              (k, f"Koma, VENC 3 m/s (bin {b+1}/30)", cmap, (0, vmax)), (d, "Koma minus CFD", dmap, (-dmax, dmax))]
    for i, (data, lab, cm, lim) in enumerate(panels):
        ax = fig.add_subplot(top[i//2, i % 2])
        im = sm.show(ax, data, extent, cmap=cm, vmin=lim[0], vmax=lim[1])
        sm.outline(ax, rS); ax.axvline(0, color="#555555", lw=.6, ls=(0, (3, 3))); ax.set_facecolor("#eeeeee")
        ax.set_title(f"{lab} | max {np.nanmax(data):.2f} m/s" if i < 3 else lab, fontsize=10)
        ax.set(xlabel="S, along vessel (mm; flow left to right)", ylabel="P (mm)")
        fig.colorbar(im, ax=ax, label="m/s", shrink=.8)
    labels = dict(scan="4D-flow MRI", cfd="CFD", koma="Koma VENC 3")
    titles = ("Throat flow", "u (in-plane 1), throat mean", "v (in-plane 2), throat mean", "w (through-plane), throat mean")
    for j in range(4):
        ax = fig.add_subplot(bottom[0, j])
        for key, (flow, uvw) in series.items():
            y = flow if j == 0 else uvw[:, j-1]
            ax.plot(t, shifted(y, shifts[key]), color=COLORS[key], lw=2, marker="o", ms=3, label=labels[key])
        ax.axhline(0, color="#888888", lw=.6)
        if j in (1, 2):
            ax.set_ylim(-0.1, 0.1)   # v3: common in-plane range (v2 used automatic limits)
        ax.set_title(titles[j], fontsize=10); ax.set_xlabel("time (ms); flow peaks aligned at 101 ms", fontsize=8)
        ax.set_ylabel("mL/s" if j == 0 else "m/s", fontsize=9); ax.grid(alpha=.25); ax.tick_params(labelsize=8)
        if j == 0:
            ax.legend(fontsize=8, frameon=False)
    fig.suptitle("Phantom at peak systole and over the cycle: 4D-flow MRI vs CFD vs Koma", fontsize=13)
    fig.supxlabel("Images: support-weighted speed projection, bilinear 0.25 mm display clipped to a Gaussian CFD wall; flow left to right. "
                  "Curves: each dataset rigidly circularly shifted so its flow peak is at phase 3 (no time warping).\n"
                  "Scan in-plane axes (u, v) are not registered to CFD R, P; CFD carries ~60 % more flow than the scan. "
                  "Koma flow is same-area (registered T1 numbers); CFD flow is full section.", fontsize=8)
    fig.savefig(stage/"phantom_scan_cfd_koma_with_waveforms.png", dpi=150); plt.close(fig)
    meta = dict(created_utc=datetime.now(timezone.utc).isoformat(), status="PRESENTATION_DERIVATIVE_ONLY", script_sha256=sha(__file__),
                circular_shift_phases=shifts, scan_phase=sphase, sim_bin=b+1, private_input_used=str(ws.PRIVATE),
                series={k: dict(flow_ml_s=f.tolist(), uvw_m_s=u.tolist()) for k, (f, u) in series.items()},
                outputs={p.name: sha(p) for p in sorted(stage.glob("*.png"))})
    (stage/"MANIFEST.json").write_text(json.dumps(meta, indent=1)+"\n", encoding="utf-8")
    stage.rename(target)
    print("DONE", json.dumps(shifts))


if __name__ == "__main__":
    main(sys.argv[1])
