"""Presentation: through-plane (axial) velocity contours, 3x3 panel.

Rows: measured 4D-flow scan (E-059 private arrays; background-corrected, forward
sign per E-059), dp150 CFD (E-060 references_v2, actual-ADC average), Koma
(E-060 acquisition as registered, axial VENC 3 m/s). Columns: proximal (-25 mm),
throat (0), distal (+15 mm) relative to each dataset's throat. Each dataset at its
own peak-throat-flow phase. Scan vessel mask = E-059 rule (25 % of the 99.5th
percentile of the time-mean magnitude, component with the largest |flow|); CFD
mask = lumen; Koma mask = lumen and per-bin 5 % support. In-plane orientation of
the scan relative to the CFD is not registered; panels are centred on each
vessel. Qualitative: the CFD carries ~60 % more flow than the scan.
No identifiers are read or written.
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
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402

E060 = Path("D:/MRSim/outputs/scanner_geometry_e060")
E059 = Path("D:/MRSim/outputs/measured_scan_e059")
PRIVATE = Path("D:/MRSim/private_inputs/phantom_scan_2025-08-28/measured_v1/measured_30phase.npz")
sys.path.insert(0, str(E060/"code"))
import analyze_e060 as an  # noqa: E402

OUT = E060/"figures_v3"
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
SEQ = LinearSegmentedColormap.from_list("seq", ["#f0efec", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
COLUMNS = (("Proximal", -25.0), ("Throat", 0.0), ("Distal", 15.0))
HALF = 16.0


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def scan_data():
    rep = json.loads((E059/"measured_v2/report.json").read_text())
    z = np.load(PRIVATE)
    v, mag, pos = z["velocity_cm_s"][..., 2]/100.0, z["magnitude"], z["slice_positions_mm"]
    sign = rep["segmentation"]["0.25"]["vendor_to_forward_sign"]
    throat = rep["throat_slice"]
    m = mag.mean(axis=1); thr = .25*np.percentile(m, 99.5)
    q_throat = np.array(rep["segmentation"]["0.25"]["slices"][throat]["flow_ml_s"])
    phase = int(np.argmax(q_throat))
    rel = pos-pos[throat]
    panels = []
    for name, d in COLUMNS:
        s = int(np.argmin(abs(rel-d)))
        lab, n = ndimage.label(m[s] >= thr)
        best, mask = -1, None
        for c in range(1, n+1):
            mk = lab == c; f = abs((sign*v[s][:, mk]).sum(axis=1).mean())
            if f > best:
                best, mask = f, mk
        img = np.where(mask, sign*v[s, phase], np.nan)
        cy, cx = ndimage.center_of_mass(mask)
        panels.append(dict(img=img, pos=float(rel[s]), center=((cx-32)*1.25, (cy-32)*1.25)))
    return panels, phase+1, float(rep["trigger_ms"][phase])


def sim_data():
    report = json.loads((E060/"analysis_v1/report.json").read_text())
    b = report["peak_bin"]-1
    control = an.tomllib.loads((E060/"acquisition_v1/control_report.toml").read_text())
    images, _ = an.load_images(E060/"acquisition_v1", control, E060/"sequences_v2/timing_e060.h5")
    koma = np.angle(images["Z"][b]*images["REF"][b].conj())*(3.0/np.pi)
    sup = np.ones(koma.shape, bool)
    for im in images.values():
        mm = np.abs(im[b]); sup &= mm >= .05*mm.max()
    with h5py.File(E060/"references_v2/references.h5") as f:
        cfd, frac, zz = f["fft_adc"][b][..., 2], f["fraction_fft"][:], f["partition_z_m"][:]*1e3
    lum = frac > 0
    cp, kp = [], []
    for name, d in COLUMNS:
        k = int(np.argmin(abs(zz-d)))
        cp.append(dict(img=np.where(lum[k], cfd[k], np.nan), pos=float(zz[k]), center=(0.0, 0.0)))
        kp.append(dict(img=np.where(lum[k] & sup[k], koma[k], np.nan), pos=float(zz[k]), center=(0.0, 0.0)))
    return cp, kp, b+1


def draw(rows, labels, norm, path, title):
    fig, axes = plt.subplots(3, 3, figsize=(13, 13), facecolor=SURFACE)
    x = (np.arange(64)-32)*1.25
    scales = [np.nanmax(np.abs(r[1]["img"])) if norm else 1.0 for r in rows]
    vmax = 1.0 if norm else max(np.nanmax(p["img"]) for r in rows for p in r)
    levels = np.linspace(0, vmax, 11)
    for i, (row, lab, sc) in enumerate(zip(rows, labels, scales)):
        for j, p in enumerate(row):
            ax = axes[i, j]; ax.set_facecolor(SURFACE)
            img = np.clip(p["img"]/sc, 0, None)
            cf = ax.contourf(x, x, np.ma.masked_invalid(img), levels=levels, cmap=SEQ, extend="max")
            ax.contour(x, x, np.ma.masked_invalid(img), levels=levels[1::2], colors=INK, linewidths=.5)
            ax.contour(x, x, np.isfinite(p["img"]).astype(float), levels=[.5], colors=INK2, linewidths=1.0)
            cx, cy = p["center"]
            ax.set_xlim(cx-HALF, cx+HALF); ax.set_ylim(cy-HALF, cy+HALF); ax.set_aspect("equal")
            ax.tick_params(labelsize=7, colors=INK2)
            for s in ax.spines.values():
                s.set_color("#c3c2b7")
            if i == 0:
                ax.set_title(f"{COLUMNS[j][0]} ({COLUMNS[j][1]:+.0f} mm)", fontsize=12, color=INK)
            if j == 0:
                ax.set_ylabel(lab, fontsize=11, color=INK)
            pk = np.nanmax(p["img"])
            ax.text(.03, .03, f"z = {p['pos']:+.1f} mm · max {pk:.2f} m/s", transform=ax.transAxes, fontsize=8, color=INK2)
    cb = fig.colorbar(cf, ax=axes, shrink=.6, pad=.02)
    cb.set_label("through-plane velocity / own throat peak" if norm else "through-plane velocity (m/s)", fontsize=10, color=INK2)
    cb.ax.tick_params(labelsize=8, colors=INK2)
    fig.suptitle(title, fontsize=12, color=INK, x=.02, ha="left", y=.97)
    fig.savefig(path, dpi=130, facecolor=SURFACE, bbox_inches="tight"); plt.close(fig)


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    scan, sphase, strig = scan_data()
    cfd, koma, kbin = sim_data()
    stage = OUT.with_name(OUT.name+".building"); stage.mkdir()
    labels = (f"4D-flow scan\n(phase {sphase}, {strig:.0f} ms)", f"CFD\n(bin {kbin} of 30)", f"Koma (E-060)\n(bin {kbin} of 30)")
    base = ("Through-plane velocity contours: measured 4D-flow scan vs CFD vs Koma, each at its own peak throat-flow phase.\n"
            "Mask outline in grey. Qualitative: CFD carries ~60 % more flow than the scan; scan in-plane orientation not registered to CFD.")
    draw((scan, cfd, koma), labels, False, stage/"contours_3x3_absolute.png", base)
    draw((scan, cfd, koma), labels, True, stage/"contours_3x3_normalized.png",
         base+"\nEach row normalized by its own maximum at the throat.")
    written = sorted(stage.glob("*.png"))
    (stage/"MANIFEST.json").write_text(json.dumps(dict(created_utc=datetime.now(timezone.utc).isoformat(), status="PRESENTATION_DERIVATIVE_ONLY",
        inputs={str(p): sha(p) for p in (Path(__file__), E060/"analysis_v1/report.json", E060/"references_v2/references.h5", E059/"measured_v2/report.json")},
        private_input_used=str(PRIVATE), outputs={p.name: sha(p) for p in written}), indent=2)+"\n", encoding="utf-8")
    stage.rename(OUT)
    print("DONE", OUT)


if __name__ == "__main__":
    main()
