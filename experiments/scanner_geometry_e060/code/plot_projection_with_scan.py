"""Presentation: peak-phase speed projection with the measured 4D-flow scan added
(rows: 4D-flow MRI, CFD, Koma, Koma - CFD), same display as plot_projection_smooth.py v4
(flow left to right, +P down, bilinear 0.25 mm, Gaussian CFD wall, 60 mm slab).

Scan: E-059 private arrays at its own peak throat-flow phase; speed = |v| of the
background-corrected velocity; per-slice E-059 vessel mask (25 % of the 99.5th percentile
time-mean magnitude, component with the largest |flow|); unweighted mean through the
in-plane x axis inside the mask; P centred on the mean vessel centroid. Slice positions
relative to the scan throat. The scan in-plane orientation is not registered to the CFD;
qualitative (CFD carries ~60 % more flow). No identifiers are read or written.
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
E059 = Path("D:/MRSim/outputs/measured_scan_e059")
PRIVATE = Path("D:/MRSim/private_inputs/phantom_scan_2025-08-28/measured_v1/measured_30phase.npz")
sys.path.insert(0, str(E060/"code"))
import plot_projection_smooth as sm  # noqa: E402
tri = sm.tri


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def scan_projection():
    rep = json.loads((E059/"measured_v2/report.json").read_text())
    z = np.load(PRIVATE)
    v, mag, pos = z["velocity_cm_s"]/100.0, z["magnitude"], z["slice_positions_mm"]
    sign = rep["segmentation"]["0.25"]["vendor_to_forward_sign"]
    throat = rep["throat_slice"]
    phase = int(np.argmax(rep["segmentation"]["0.25"]["slices"][throat]["flow_ml_s"]))
    m = mag.mean(axis=1); thr = .25*np.percentile(m, 99.5)
    speed = np.linalg.norm(v[:, phase], axis=-1)            # (slice, y, x)
    masks = []
    for s in range(v.shape[0]):
        lab, n = ndimage.label(m[s] >= thr)
        flows = [abs((sign*v[s, :, :, :, 2][:, lab == c]).sum(axis=1).mean()) for c in range(1, n+1)]
        masks.append(lab == (1+int(np.argmax(flows))))
    masks = np.array(masks)
    cy = float(np.mean([ndimage.center_of_mass(mk)[0] for mk in masks]))
    w = masks.astype(float)
    proj = np.divide((speed*w).sum(axis=2), w.sum(axis=2), out=np.full(w.shape[:2], np.nan), where=w.sum(axis=2) > 0)  # (slice, y)
    shift = int(round(32-cy))
    proj = np.roll(proj, shift, axis=1)
    if shift > 0:
        proj[:, :shift] = np.nan
    elif shift < 0:
        proj[:, shift:] = np.nan
    return proj, pos-pos[throat], phase+1, float(rep["trigger_ms"][phase]), shift, cy


def main(target):
    target = Path(target)
    if target.exists():
        raise FileExistsError(target)
    with h5py.File(E060/"references_v2/references.h5") as f:
        ref, frac, z = f["fft_adc"][:], f["fraction_fft"][:], f["partition_z_m"][:]*1e3
    rS = sm.wall(z)
    scan, zs, sphase, strig, shift, cy = scan_projection()
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
    for norm in (False, True):
        fig, axes = plt.subplots(4, 1, figsize=(9, 12.5), layout="constrained")
        panels = [(s, f"4D-flow MRI scan (phase {sphase}, {strig:.0f} ms; own peak)", cmap),
                  (a, f"CFD (bin {b+1}/30)", cmap), (k, f"Koma, VENC 3 m/s (bin {b+1}/30)", cmap), (d, "Koma minus CFD", dmap)]
        for i, (ax, (data, lab, cm)) in enumerate(zip(axes, panels)):
            if i < 3:
                sc = np.nanmax(data) if norm else 1.0
                im = sm.show(ax, data/sc, extent, cmap=cm, vmin=0, vmax=1.0 if norm else vmax)
            else:
                im = sm.show(ax, data, extent, cmap=cm, vmin=-dmax, vmax=dmax)
            sm.outline(ax, rS); ax.axvline(0, color="#555555", lw=.6, ls=(0, (3, 3)))
            ax.set_facecolor("#eeeeee"); ax.set_title(f"{lab} | max {np.nanmax(data):.2f} m/s" if i < 3 else lab, fontsize=10)
            ax.set(xlabel="S, along vessel (mm; flow left to right)", ylabel="P (mm)")
            fig.colorbar(im, ax=ax, label=("speed / own max" if (norm and i < 3) else "m/s"), shrink=.9)
        fig.suptitle("Phantom speed, support-weighted projection: 4D-flow MRI vs CFD vs Koma"+(" (each normalized to its own max)" if norm else ""), fontsize=12)
        fig.supxlabel("Display: bilinear 0.25 mm, clipped to a Gaussian fit of the CFD wall; values unchanged. Flow left to right; 60 mm slab.\n"
                      "Scan: E-059 vessel mask, unweighted mean through x, own peak phase; orientation not registered; CFD carries ~60 % more flow.", fontsize=8)
        fig.savefig(stage/f"phantom_speed_scan_cfd_koma_peak{'_normalized' if norm else ''}.png", dpi=150); plt.close(fig)
    (stage/"MANIFEST.json").write_text(json.dumps(dict(created_utc=datetime.now(timezone.utc).isoformat(), status="PRESENTATION_DERIVATIVE_ONLY",
        script_sha256=sha(__file__), scan_phase=sphase, scan_trigger_ms=strig, sim_bin=b+1, scan_centroid_row=cy, scan_shift_pixels=shift,
        private_input_used=str(PRIVATE), outputs={p.name: sha(p) for p in sorted(stage.glob("*.png"))}), indent=2)+"\n", encoding="utf-8")
    stage.rename(target)
    print("DONE", target)


if __name__ == "__main__":
    main(sys.argv[1])
