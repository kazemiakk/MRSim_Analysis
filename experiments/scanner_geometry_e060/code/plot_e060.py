"""E-060 presentation figures (no new metrics; reads the registered S4 outputs).

Fig 1: flow waveforms (each normalized by its own cycle mean, rigid circular
shift only so the systolic peak sits at phase 3; no time warping) and mean flow
along the slab normalized to upstream: measured scan vs Koma (same E-059 method)
vs CFD. Fig 2: per-bin same-area throat flow error with the +/-8 % band. Fig 3:
peak-bin axial velocity, CFD vs Koma vs difference, throat cross-section and the
along-vessel plane. Qualitative versus the scan (CFD carries ~60 % more flow).
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

E060 = Path("D:/MRSim/outputs/scanner_geometry_e060")
sys.path.insert(0, str(E060/"code"))
import analyze_e060 as an  # noqa: E402

OUT = E060/"figures_v1"
SURFACE, INK, INK2, GRID, SPINE = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#c3c2b7"
SCAN, KOMA, CFD = "#eb6834", "#2a78d6", INK
SEQ = LinearSegmentedColormap.from_list("seq", ["#f0efec", "#9ec5f4", "#3987e5", "#184f95", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#184f95", "#6da7ec", "#f0efec", "#f0a07c", "#b4441b"])
PEAK_AT = 3


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def style(ax):
    ax.set_facecolor(SURFACE); ax.tick_params(labelsize=8, colors=INK2); ax.grid(color=GRID, lw=.5)
    for s in ax.spines.values():
        s.set_color(SPINE)


def aligned(q):
    q = np.asarray(q, float); q = q/q.mean()
    return np.roll(q, PEAK_AT-int(np.argmax(q)))


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    report = json.loads((E060/"analysis_v1/report.json").read_text())
    scan = json.loads((an.E059/"measured_v2/report.json").read_text())
    rows = scan["segmentation"]["0.25"]["slices"]
    koma = report["scan_method"]["koma"]
    t1 = report["targets_per_bin_support"]["T1_throat_flow"]["bins"]
    stage = OUT.with_name(OUT.name+".building"); stage.mkdir()
    phase = np.arange(1, 31)

    # Fig 1
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.6), facecolor=SURFACE)
    s_th, s_up = scan["throat_slice"], scan["upstream_slices"]
    k_th, k_up = koma["throat_slice"], koma["upstream_slices"]
    a.plot(phase, aligned(rows[s_th]["flow_ml_s"]), color=SCAN, lw=2, label="scan, throat")
    a.plot(phase, aligned(np.mean([rows[s]["flow_ml_s"] for s in s_up], axis=0)), color=SCAN, lw=1.5, ls=(0, (4, 2)), label="scan, upstream")
    a.plot(phase, aligned(koma["flow_ml_s"][k_th]), color=KOMA, lw=2, label="Koma, throat")
    a.plot(phase, aligned(np.mean([koma["flow_ml_s"][s] for s in k_up], axis=0)), color=KOMA, lw=1.5, ls=(0, (4, 2)), label="Koma, upstream")
    a.plot(phase, aligned([r["CFD_full_section_ml_s"] for r in t1]), color=CFD, lw=1.2, ls=":", label="CFD, throat (full section)")
    a.set_xlabel("cardiac phase (rigid circular shift: systolic peak at phase 3)", fontsize=9, color=INK2)
    a.set_ylabel("flow / own cycle mean", fontsize=9, color=INK2)
    a.set_title("A  Flow waveform shape (each curve normalized by its own mean)", fontsize=10, color=INK, loc="left")
    a.legend(fontsize=8, frameon=False)
    pos_s = np.array(scan["slice_positions_mm"])-scan["throat_position_mm"]
    q_s = np.array([r["mean_flow_ml_s"] for r in rows]); q_k = np.array(koma["mean_flow_ml_s"])
    pos_k = (np.arange(len(q_k))-12)*2.5
    b.plot(pos_s, q_s/q_s[s_up].mean(), color=SCAN, lw=2, marker="o", ms=3, label=f"scan ({100*scan['throat_vs_upstream_mean_flow']:+.1f} % at throat)")
    b.plot(pos_k, q_k/q_k[k_up].mean(), color=KOMA, lw=2, marker="o", ms=3, label=f"Koma ({100*koma['throat_vs_upstream_mean_flow']:+.1f} % at throat)")
    b.axhspan(.92, 1.08, color=GRID, zorder=0); b.axhline(1, color=INK2, lw=.6); b.axvline(0, color=INK2, lw=.6, ls=(0, (3, 3)))
    b.set_xlabel("position along vessel from scan throat slice (mm)", fontsize=9, color=INK2)
    b.set_ylabel("cycle-mean flow / upstream mean", fontsize=9, color=INK2)
    b.set_title("B  Flow conservation along the slab (E-059 method, band ±8 %)", fontsize=10, color=INK, loc="left")
    b.legend(fontsize=8, frameon=False, loc="lower left")
    for ax in (a, b):
        style(ax)
    fig.suptitle("E-060 vs measured scan, qualitative (CFD carries ~60 % more flow than the scan; phase origins differ)", fontsize=11, color=INK, x=.02, ha="left")
    fig.savefig(stage/"fig1_flow_vs_scan.png", dpi=130, facecolor=SURFACE, bbox_inches="tight"); plt.close(fig)

    # Fig 2
    fig, ax = plt.subplots(figsize=(8, 4), facecolor=SURFACE)
    x = [r["bin"] for r in t1 if r["same_area_relative_error"] is not None]
    y = [100*r["same_area_relative_error"] for r in t1 if r["same_area_relative_error"] is not None]
    ax.axhspan(-8, 8, color=GRID, zorder=0); ax.axhline(0, color=INK2, lw=.6)
    ax.plot(x, y, color=KOMA, lw=2, marker="o", ms=4)
    ax.axvline(report["peak_bin"], color=INK2, lw=.6, ls=(0, (3, 3)))
    ax.text(report["peak_bin"]+.3, min(y), "peak bin", fontsize=8, color=INK2)
    T = report["targets_per_bin_support"]
    ax.set_title(f"Throat same-area flow error per bin: cycle {100*T['T1_throat_flow']['cycle_same_area_error']:+.1f} %, "
                 f"min section coverage {100*T['T1_throat_flow']['min_section_area_coverage']:.1f} %", fontsize=10, color=INK, loc="left")
    ax.set_xlabel("cardiac bin", fontsize=9, color=INK2); ax.set_ylabel("Koma / CFD − 1 (%)", fontsize=9, color=INK2)
    style(ax)
    fig.savefig(stage/"fig2_throat_flow_per_bin.png", dpi=130, facecolor=SURFACE, bbox_inches="tight"); plt.close(fig)

    # Fig 3
    control = an.tomllib.loads((E060/"acquisition_v1/control_report.toml").read_text())
    images, _ = an.load_images(E060/"acquisition_v1", control, E060/"sequences_v2/timing_e060.h5")
    field = np.stack([np.angle(images[e]*images["REF"].conj())*(an.VENC[i]/np.pi) for i, e in enumerate(an.ENC[1:])], axis=-1)
    with h5py.File(E060/"references_v2/references.h5") as f:
        ref, frac, z = f["fft_adc"][:], f["fraction_fft"][:], f["partition_z_m"][:]
    b = report["peak_bin"]-1
    support = np.ones(images["REF"].shape[1:], bool)
    for im in images.values():
        m = np.abs(im[b]); support &= m >= .05*m.max()
    lumen = frac > 0
    kz = an.THROAT_K
    r_c = int(np.argmax(frac[kz].sum(axis=0)))
    vmax = float(np.nanmax(ref[b][..., 2][lumen]))
    panels = [("CFD", ref[b][..., 2], lumen), ("Koma", field[b][..., 2], lumen & support), ("Koma − CFD", field[b][..., 2]-ref[b][..., 2], lumen & support)]
    fig, axes = plt.subplots(2, 3, figsize=(14, 9.5), facecolor=SURFACE, gridspec_kw=dict(height_ratios=[1, 1.15]))
    mm = (np.arange(64)-32)*1.25
    for j, (name, data, mask) in enumerate(panels):
        diff = j == 2
        kw = dict(cmap=DIV, vmin=-.5, vmax=.5) if diff else dict(cmap=SEQ, vmin=0, vmax=vmax)
        top = np.where(mask[kz], data[kz], np.nan)
        im = axes[0, j].pcolormesh(mm, mm, top, shading="nearest", **kw)
        axes[0, j].set_xlim(-15, 15); axes[0, j].set_ylim(-15, 15); axes[0, j].set_aspect("equal")
        axes[0, j].set_title(f"{name}: throat cross-section (z = 0)", fontsize=10, color=INK, loc="left")
        axes[0, j].set_xlabel("R (mm)", fontsize=8, color=INK2); axes[0, j].set_ylabel("P (mm)", fontsize=8, color=INK2)
        side = np.where(mask[:, :, r_c], data[:, :, r_c], np.nan).T
        im2 = axes[1, j].pcolormesh(z*1e3, mm, side, shading="nearest", **kw)
        axes[1, j].set_ylim(-15, 15); axes[1, j].axvline(0, color=INK2, lw=.6, ls=(0, (3, 3)))
        axes[1, j].axvspan(-37.5, -30, color=GRID, alpha=.6); axes[1, j].axvspan(30, 37.5, color=GRID, alpha=.6)
        axes[1, j].set_title(f"{name}: along-vessel plane (R = {mm[r_c]:.1f} mm)", fontsize=10, color=INK, loc="left")
        axes[1, j].set_xlabel("S, along vessel (mm; grey = outside 60 mm slab)", fontsize=8, color=INK2); axes[1, j].set_ylabel("P (mm)", fontsize=8, color=INK2)
        for ax, h in ((axes[0, j], im), (axes[1, j], im2)):
            style(ax); ax.grid(False)
            cb = fig.colorbar(h, ax=ax, shrink=.85, pad=.02); cb.ax.tick_params(labelsize=7, colors=INK2)
            cb.set_label("m/s" if not diff else "Δ m/s", fontsize=8, color=INK2)
    t2 = T["T2_peak_throat_velocity"]
    fig.suptitle(f"E-060 peak bin {b+1}: axial velocity. Koma throat max {t2['Koma_max_m_s']:.2f} vs CFD {t2['CFD_region_max_m_s']:.2f} m/s "
                 f"({100*t2['max_error']:+.1f} %). Same-source internal consistency, noiseless.", fontsize=11, color=INK, x=.02, ha="left")
    fig.savefig(stage/"fig3_peak_bin_maps.png", dpi=120, facecolor=SURFACE, bbox_inches="tight"); plt.close(fig)

    manifest = dict(utc=datetime.now(timezone.utc).isoformat(), script_sha256=sha(__file__),
                    inputs={str(p): sha(p) for p in (E060/"analysis_v1/report.json", an.E059/"measured_v2/report.json", E060/"references_v2/references.h5")},
                    outputs={p.name: sha(p) for p in sorted(stage.iterdir())})
    (stage/"MANIFEST.json").write_text(json.dumps(manifest, indent=1)+"\n", encoding="utf-8")
    stage.rename(OUT)
    print("E060_FIGURES_DONE", OUT)


if __name__ == "__main__":
    main()
