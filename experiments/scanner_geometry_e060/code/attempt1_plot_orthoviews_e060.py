"""Presentation derivative: E-060 orthogonal projections and center slices, CFD vs Koma.

Same layout as slice_comparison_e055_v1/code/plot_orthoviews.py (max projections
over center slices; CFD speed / Koma speed / Koma REF magnitude), on the E-060
scanner grid: S (slab, along the vessel) horizontal, R/P across the vessel.
Koma speed only inside its own per-bin 5% all-encoding support. Nothing fitted,
filled or smoothed; projections combine depth and are not planes.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import warnings

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
import numpy as np  # noqa: E402

E060 = Path("D:/MRSim/outputs/scanner_geometry_e060")
sys.path.insert(0, str(E060/"code"))
import analyze_e060 as an  # noqa: E402

OUT = E060/"orthoviews_v1"
SURFACE, INK, INK2, OUTLINE = "#fcfcfb", "#0b0b0b", "#52514e", "#3a3936"
SPEED = LinearSegmentedColormap.from_list("blue_sequential",
    ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
SPEED.set_bad(SURFACE)
GRAY = plt.get_cmap("gray").copy(); GRAY.set_bad(SURFACE)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def edges(n, d):
    c = (np.arange(n)-n//2)*d
    return np.r_[c-d/2, c[-1]+d/2]


S_E, P_E, R_E = edges(30, 2.5), edges(64, 1.25), edges(64, 1.25)


def mip(volume, axis):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmax(volume, axis=axis)


def outline(ax, x_edges, y_edges, mask):
    padded = np.pad(mask, 1); segments = []
    for r, c in zip(*np.nonzero(padded[1:, 1:-1] != padded[:-1, 1:-1])):
        segments.append([(x_edges[c], y_edges[r]), (x_edges[c+1], y_edges[r])])
    for r, c in zip(*np.nonzero(padded[1:-1, 1:] != padded[1:-1, :-1])):
        segments.append([(x_edges[c], y_edges[r]), (x_edges[c], y_edges[r+1])])
    ax.add_collection(LineCollection(segments, colors=OUTLINE, linewidths=.5))


def views(v, rc, pc):
    """v is (S, P, R). Along-vessel panels are transposed so S is horizontal."""
    return dict(proj_ps=mip(v, 2).T, proj_rs=mip(v, 1).T, proj_rp=mip(v, 0),
                center_ps=v[:, :, rc].T, center_rs=v[:, pc, :].T, center_rp=v[an.THROAT_K])


def draw_block(fig, grid, row0, label, data, cmap, vmin, vmax, lumen, rc, pc, unit):
    layout = (("proj_ps", "Projection (max) - P vs S"), ("proj_rs", "Projection (max) - R vs S"), ("proj_rp", "Projection (max) - R vs P"),
              ("center_ps", f"Center slice - P vs S (R = {(rc-32)*1.25:.1f} mm)"), ("center_rs", f"Center slice - R vs S (P = {(pc-32)*1.25:.1f} mm)"),
              ("center_rp", "Throat slice - R vs P (S = 0)"))
    masks = dict(center_ps=lumen[:, :, rc].T, center_rs=lumen[:, pc, :].T, center_rp=lumen[an.THROAT_K])
    mesh = None
    for i, (key, title) in enumerate(layout):
        ax = fig.add_subplot(grid[row0+i//3, i % 3]); ax.set_facecolor(SURFACE)
        if key.endswith("rp"):
            xe, ye, xl, yl = R_E, P_E, "R / source x (mm)", "P / source y (mm)"
            ax.set_xlim(-20, 20); ax.set_ylim(-20, 20)
        else:
            xe, xl = S_E, "S, slab along vessel (mm)"
            ye, yl = (P_E, "P (mm)") if key.endswith("ps") else (R_E, "R (mm)")
            ax.set_ylim(-20, 20)
        mesh = ax.pcolormesh(xe, ye, data[key], cmap=cmap, vmin=vmin, vmax=vmax, shading="flat")
        if key in masks:
            outline(ax, xe, ye, masks[key])
        if not key.endswith("rp"):
            ax.axvline(0, color=INK2, lw=.6, ls=(0, (3, 3)))
            for x in (-30, 30):
                ax.axvline(x, color=INK2, lw=.6, ls=(0, (1, 2)))
        ax.set_aspect("equal"); ax.set_title(title, fontsize=9, color=INK)
        if i == 0:
            ax.text(-.07, 1.25, label, transform=ax.transAxes, fontsize=12, fontweight="bold", color=INK, ha="left")
        ax.set_xlabel(xl, fontsize=8, color=INK2); ax.set_ylabel(yl, fontsize=8, color=INK2)
        ax.tick_params(labelsize=7, colors=INK2)
        for side in ax.spines.values():
            side.set_color("#c3c2b7")
    bar = fig.colorbar(mesh, ax=[fig.axes[-1], fig.axes[-4]], shrink=.9, pad=.02)
    bar.set_label(unit, fontsize=8, color=INK2); bar.ax.tick_params(labelsize=7, colors=INK2)


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    report = json.loads((E060/"analysis_v1/report.json").read_text())
    peak = report["peak_bin"]
    bins = ((1, "early cycle"), (peak, "peak systole"), (20, "diastole"))
    control = an.tomllib.loads((E060/"acquisition_v1/control_report.toml").read_text())
    images, _ = an.load_images(E060/"acquisition_v1", control, E060/"sequences_v2/timing_e060.h5")
    field = np.stack([np.angle(images[e]*images["REF"].conj())*(an.VENC[i]/np.pi) for i, e in enumerate(an.ENC[1:])], axis=-1)
    refs = E060/"references_v2/references.h5"
    with h5py.File(refs) as f:
        ref, fraction = f["fft_adc"][:], f["fraction_fft"][:]
    lumen = fraction > 0
    rc = int(np.argmax(fraction[an.THROAT_K].sum(axis=0))); pc = int(np.argmax(fraction[an.THROAT_K].sum(axis=1)))
    stage = OUT.with_name(OUT.name+".building"); stage.mkdir()
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    written = []
    for b, phase in bins:
        cfd = np.where(lumen, np.linalg.norm(ref[b-1].astype(float), axis=-1), np.nan)
        support = np.ones(lumen.shape, bool)
        for im in images.values():
            m = np.abs(im[b-1]); support &= m >= .05*m.max()
        koma = np.where(support, np.linalg.norm(field[b-1].astype(float), axis=-1), np.nan)
        mag = np.abs(images["REF"][b-1]).astype(float); mag /= mag.max()
        vmax = float(np.ceil(np.nanpercentile(cfd, 99.9)*10)/10)
        fig = plt.figure(figsize=(22, 17), facecolor=SURFACE)
        grid = fig.add_gridspec(6, 3, width_ratios=[4, 4, 1.6], hspace=.6, wspace=.15, top=.9)
        draw_block(fig, grid, 0, "CFD speed", views(cfd, rc, pc), SPEED, 0, vmax, lumen, rc, pc, "speed |v| (m/s), CFD and Koma")
        draw_block(fig, grid, 2, "Koma speed", views(koma, rc, pc), SPEED, 0, vmax, lumen, rc, pc, "speed |v| (m/s), CFD and Koma")
        draw_block(fig, grid, 4, "Koma magnitude", views(mag, rc, pc), GRAY, 0, 1, lumen, rc, pc, "REF magnitude (normalized)")
        fig.suptitle(f"dp150 CFD vs KomaMRI in the measured scanner geometry (E-060, 300k, Float64) | cardiac bin {b} of 30, {phase}\n"
                     "Rows 1-2 CFD speed (averaged at actual ADC times, E-060 grid) · rows 3-4 Koma speed on its own per-bin 5% support · "
                     "rows 5-6 Koma REF magnitude.\nDashed line: throat (S = 0). Dotted: 60 mm slab edges. Outline: CFD lumen. "
                     "Same-source internal consistency, noiseless.", fontsize=11, color=INK, y=.975)
        path = stage/f"orthoview_bin{b:02d}.png"
        fig.savefig(path, dpi=110, facecolor=SURFACE, bbox_inches="tight"); plt.close(fig); written.append(path)
        print(path, flush=True)
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), status="PRESENTATION_DERIVATIVE_ONLY",
                    inputs={str(p): sha(p) for p in (Path(__file__), refs, E060/"analysis_v1/report.json")},
                    outputs={p.name: sha(p) for p in written}, bins=[b for b, _ in bins])
    (stage/"MANIFEST.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    stage.rename(OUT)


if __name__ == "__main__":
    main()
