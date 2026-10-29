"""Presentation derivative: orthogonal projections and center slices, CFD vs Koma.

Same layout as the project's static reconstruction view (XY/XZ/YZ maximum
projections over center slices), for the completed E055 300k acquisition and
the frozen E048 actual-ADC averaged CFD. Nothing is fitted, filled or smoothed.
Koma speed is shown only inside its own fixed 5% support; projections are
maximum-intensity projections (they combine depth and are not planes).
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tomllib
import warnings

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

E048 = Path("D:/MRSim/MRSIM/MRSimTracks/outputs/phantom_diagnosis_e048")
E055 = Path("D:/MRSim/outputs/phantom_density_e055")
OUT = Path("D:/MRSim/outputs/slice_comparison_e055_v1/orthoviews_v2")  # v1: clipped YZ titles
BINS = ((1, "early cycle"), (4, "peak systole"), (12, "diastole"))
SURFACE, INK, INK2, OUTLINE = "#fcfcfb", "#0b0b0b", "#52514e", "#3a3936"
SPEED = LinearSegmentedColormap.from_list("blue_sequential",
    ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
SPEED.set_bad(SURFACE)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def edges(n, d):
    c = (np.arange(n)-n//2)*d
    return np.r_[c-d/2, c[-1]+d/2]


R_E, P_E, S_E = edges(400, 1.25), edges(64, 1.25), edges(32, 2.5)


def complex_array(dataset):
    a = dataset[()]
    return a if np.iscomplexobj(a) else a["r"]+1j*a["i"]


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


def views(volume):
    """volume is (slice, phase, readout). Returns projections and center slices."""
    return dict(proj_xy=mip(volume, 0), proj_xz=mip(volume, 1), proj_yz=mip(volume, 2),
                center_xy=volume[16], center_xz=volume[:, 32], center_yz=volume[:, :, 200])


def draw_block(fig, grid, row0, label, data, cmap, vmin, vmax, lumen, stenosis_mm, unit):
    layout = (("proj_xy", "Projection (max) - XY"), ("proj_xz", "Projection (max) - XZ"), ("proj_yz", "Projection - YZ"),
              ("center_xy", "Center slice - XY"), ("center_xz", "Center slice - XZ"), ("center_yz", "Center slice - YZ"))
    masks = dict(center_xy=lumen[16], center_xz=lumen[:, 32], center_yz=lumen[:, :, 200])
    mesh = None
    for i, (key, title) in enumerate(layout):
        ax = fig.add_subplot(grid[row0+i//3, i % 3]); ax.set_facecolor(SURFACE)
        if key.endswith("yz"):
            xe, ye, xl, yl = P_E, S_E, "Phase / source x (mm)", "Slice / source y (mm)"
        else:
            xe = R_E; xl = "Readout / source z (mm)"
            ye, yl = (P_E, "Phase / source x (mm)") if key.endswith("xy") else (S_E, "Slice / source y (mm)")
        mesh = ax.pcolormesh(xe, ye, data[key], cmap=cmap, vmin=vmin, vmax=vmax, shading="flat")
        if key in masks:
            outline(ax, xe, ye, masks[key])
        if not key.endswith("yz"):
            ax.axvline(stenosis_mm, color=INK2, lw=.6, ls=(0, (3, 3)))
        ax.set_aspect("equal"); ax.set_title(title, fontsize=9, color=INK)
        if i == 0:
            ax.text(-.07, 1.32, label, transform=ax.transAxes, fontsize=12, fontweight="bold", color=INK, ha="left")
        ax.set_xlabel(xl, fontsize=8, color=INK2); ax.set_ylabel(yl, fontsize=8, color=INK2)
        ax.tick_params(labelsize=7, colors=INK2)
        for side in ax.spines.values():
            side.set_color("#c3c2b7")
    bar = fig.colorbar(mesh, ax=[fig.axes[-1], fig.axes[-4]], shrink=.9, pad=.02)
    bar.set_label(unit, fontsize=8, color=INK2); bar.ax.tick_params(labelsize=7, colors=INK2)


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    reference = E048/"reference_v1/reference_fields.h5"
    local = E048/"local_v1/local_fields.h5"
    fields = E055/"analysis_v2/paired_fields.h5"
    ref_image = E055/"mapped_f64_v1/ref_result.h5"
    if sha(reference) != "f4c4ec3153d30a2a9dfd360d8ff1c21811e14e6f942697cdd990b669fec75694":
        raise ValueError("E048 reference changed")
    if sha(fields) != json.loads((E055/"analysis_v2/MANIFEST.json").read_text())["outputs"]["paired_fields.h5"]:
        raise ValueError("E055 fields changed")
    report = tomllib.loads((E055/"mapped_f64_v1/control_report.toml").read_text())
    if sha(ref_image) != report["result_sha256"]["REF"]:
        raise ValueError("E055 REF result changed")
    with h5py.File(local) as f:
        z = f["source_z_fft_m"][:]
    stenosis_mm = (int(np.argmin(abs(z)))-200)*1.25   # readout coordinate of source z = 0
    stage = OUT.with_name(OUT.name+".building"); stage.mkdir(parents=True)
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    written = []
    for b, phase in BINS:
        with h5py.File(reference) as f:
            fraction = f["fraction_fft"][:]
            cfd = np.linalg.norm(f["fft_adc"][b-1].astype(float), axis=-1)
        lumen = fraction > 0
        cfd = np.where(lumen, cfd, np.nan)
        with h5py.File(fields) as f:
            koma = np.linalg.norm(f["candidate300k"][b-1].astype(float), axis=-1)
            support = f["candidate300k_fixed5_mask"][:]
        koma = np.where(support, koma, np.nan)
        with h5py.File(ref_image) as f:
            magnitude = np.abs(complex_array(f["image"])[b-1]).astype(float)
        magnitude /= magnitude.max()
        vmax = float(np.ceil(np.nanpercentile(cfd, 99.9)*10)/10)
        fig = plt.figure(figsize=(24, 17.5), facecolor=SURFACE)
        grid = fig.add_gridspec(6, 3, width_ratios=[5, 5, 1.25], hspace=.62, wspace=.12, top=.9)
        draw_block(fig, grid, 0, "CFD speed", views(cfd), SPEED, 0, vmax, lumen, stenosis_mm, "speed |v| (m/s), CFD and Koma")
        draw_block(fig, grid, 2, "Koma speed", views(koma), SPEED, 0, vmax, lumen, stenosis_mm, "speed |v| (m/s), CFD and Koma")
        draw_block(fig, grid, 4, "Koma magnitude", views(magnitude), "gray", 0, 1, lumen, stenosis_mm, "REF magnitude (normalized)")
        fig.suptitle(f"dp150 CFD vs KomaMRI (E055 300k, Float64) | cardiac bin {b}, {phase}\n"
                     "Rows 1-2 CFD speed (E048, averaged at actual ADC times) · rows 3-4 Koma speed on its own 5% support · "
                     "rows 5-6 Koma REF magnitude.\nDashed line: stenosis throat (readout "
                     f"{stenosis_mm:.1f} mm). Outline: CFD lumen. YZ center slice is at readout 0 mm, downstream of the throat. "
                     "Same-source internal consistency, noiseless.", fontsize=11, color=INK, y=.97)
        path = stage/f"orthoview_bin{b:02d}.png"
        fig.savefig(path, dpi=110, facecolor=SURFACE, bbox_inches="tight"); plt.close(fig); written.append(path)
        print(path, flush=True)
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), status="PRESENTATION_DERIVATIVE_ONLY",
        inputs={str(p): sha(p) for p in (Path(__file__), reference, local, fields, ref_image)},
        outputs={p.name: sha(p) for p in written}, bins=[b for b, _ in BINS], stenosis_readout_mm=stenosis_mm,
        note="Projections are maximum-intensity projections; Koma speed only inside its own fixed 5% support.")
    (stage/"MANIFEST.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    stage.rename(OUT)


if __name__ == "__main__":
    main()
