"""Presentation derivative: CFD versus Koma axial velocity on true MRI-grid planes.

Reads the completed E055 300k Koma field and the frozen E048 actual-ADC averaged
CFD reference. Nothing is fitted, filled, smoothed or interpolated; voxels are
drawn at their true 1.25 x 1.25 x 2.5 mm size. Lumen voxels without Koma 5%
signal support are hatched, never drawn as zero velocity.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import textwrap

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np

E048 = Path("D:/MRSim/MRSIM/MRSimTracks/outputs/phantom_diagnosis_e048")
E055 = Path("D:/MRSim/outputs/phantom_density_e055")
OUT = Path("D:/MRSim/outputs/slice_comparison_e055_v1")
FIGURES = OUT/"figures_v3"    # v1: contour hatching missed isolated voxels; v2: one clipped title
PLANES = (("Proximal", -.035), ("Stenosis", 0.), ("Distal", .030))  # source z, as compare_dp150_hemodynamics.PLANES
BINS = ((4, "peak systole"), (12, "diastole"))
SAGITTAL_SLICE = 16           # MRI slice index through the vessel axis
SAGITTAL_Z_MM = (-100, 150)
DIFF_LIMIT = 1.0              # common difference scale, m/s, for every panel and figure

SURFACE, INK, INK2, OUTLINE, HATCH = "#fcfcfb", "#0b0b0b", "#52514e", "#3a3936", "#8f8e89"
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red",
    ["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec", "#f4b6ac", "#e34948", "#8a1f1f"])
DIVERGING.set_bad(SURFACE)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def edges(n, spacing_mm, offset_mm=0.):
    centers = (np.arange(n)-n//2)*spacing_mm+offset_mm
    return np.r_[centers-spacing_mm/2, centers[-1]+spacing_mm/2], centers


def nice_limit(value):
    return max(.1, np.ceil(value*10)/10)


def outline(ax, x_edges, y_edges, mask):
    """Lumen boundary along voxel edges (mask rows follow y, columns follow x)."""
    padded = np.pad(mask, 1)
    segments = []
    for r, c in zip(*np.nonzero(padded[1:, 1:-1] != padded[:-1, 1:-1])):
        segments.append([(x_edges[c], y_edges[r]), (x_edges[c+1], y_edges[r])])
    for r, c in zip(*np.nonzero(padded[1:-1, 1:] != padded[1:-1, :-1])):
        segments.append([(x_edges[c], y_edges[r]), (x_edges[c], y_edges[r+1])])
    ax.add_collection(LineCollection(segments, colors=OUTLINE, linewidths=.8))


def hatch(ax, x_edges, y_edges, mask):
    """One hatched rectangle per unsupported lumen voxel."""
    boxes = [Rectangle((x_edges[c], y_edges[r]), x_edges[c+1]-x_edges[c], y_edges[r+1]-y_edges[r])
             for r, c in zip(*np.nonzero(mask))]
    if boxes:
        ax.add_collection(PatchCollection(boxes, facecolor="none", edgecolor=HATCH, hatch="////", linewidth=0))


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ax.spines.values():
        side.set_color("#c3c2b7")
    ax.tick_params(colors=INK2, labelsize=8)


def stats(koma, cfd, keep):
    if not keep.any():
        return "no Koma-supported lumen voxels"
    d = koma[keep]-cfd[keep]
    return f"RMSE {np.sqrt(np.mean(d**2)):.2f} · mean {d.mean():+.2f} m/s (n={keep.sum()})"


def wrap(text, width):
    return "\n".join(textwrap.wrap(text, width))


def main():
    if FIGURES.exists():
        raise FileExistsError(FIGURES)
    inputs = {E048/"reference_v1/reference_fields.h5": "f4c4ec3153d30a2a9dfd360d8ff1c21811e14e6f942697cdd990b669fec75694",
              E048/"local_v1/local_fields.h5": "6c57a8708ea4a5aff09f61db232b0ba1528ae4d000bf60d9ca00524516a27885"}
    for path, digest in inputs.items():
        if sha(path) != digest:
            raise ValueError("Frozen E048 reference changed: "+str(path))
    fields_path = E055/"analysis_v2/paired_fields.h5"
    manifest = json.loads((E055/"analysis_v2/MANIFEST.json").read_text())["outputs"]
    if sha(fields_path) != manifest["paired_fields.h5"]:
        raise ValueError("E055 field changed")
    with h5py.File(E048/"reference_v1/reference_fields.h5") as f:
        fraction = f["fraction_fft"][:]
        cfd_all = {b: f["fft_adc"][b-1][..., 0] for b, _ in BINS}
    with h5py.File(E048/"local_v1/local_fields.h5") as f:
        z_m = f["source_z_fft_m"][:]
    with h5py.File(fields_path) as f:
        koma_all = {b: f["candidate300k"][b-1][..., 0] for b, _ in BINS}
        support = f["candidate300k_fixed5_mask"][:]
    lumen = fraction > 0
    z_mm = z_m*1000
    z_edges = np.r_[z_mm-.625, z_mm[-1]+.625]
    p_edges, p_mm = edges(64, 1.25)
    s_edges, s_mm = edges(32, 2.5)
    figures = FIGURES.with_name(FIGURES.name+".building")
    figures.mkdir(parents=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": INK2})
    note = ("E055 Koma 300k (Inf precession, 50 µs RF) vs E048 CFD averaged at the actual ADC times. "
            "Same-source internal consistency, noiseless; not independent validation. "
            "Hatched: lumen voxels without Koma 5% signal support.")
    written = []
    for b, phase in BINS:
        cfd, koma = cfd_all[b], koma_all[b]
        # Axial (cross-sectional) planes.
        fig, axes = plt.subplots(3, 3, figsize=(11.5, 10.5), facecolor=SURFACE, constrained_layout=True)
        for row, (name, target) in enumerate(PLANES):
            i = int(np.argmin(abs(z_m-target)))
            lum = lumen[:, :, i]; keep = lum & support[:, :, i]
            c = np.where(lum, cfd[:, :, i], np.nan); k = np.where(keep, koma[:, :, i], np.nan)
            d = np.where(keep, koma[:, :, i]-cfd[:, :, i], np.nan)
            limit = nice_limit(np.nanmax(abs(c)))
            s_idx, p_idx = np.nonzero(lum)
            xlim = (p_edges[p_idx.min()]-4, p_edges[p_idx.max()+1]+4)
            ylim = (s_edges[s_idx.min()]-4, s_edges[s_idx.max()+1]+4)
            panels = ((c, f"CFD · {name}\nz = {z_mm[i]:+.1f} mm", limit),
                      (k, f"Koma\nsupport {keep.sum()}/{lum.sum()} lumen voxels", limit),
                      (d, "Koma − CFD\n" + stats(koma[:, :, i], cfd[:, :, i], keep), DIFF_LIMIT))
            for col, (data, title, lim) in enumerate(panels):
                ax = axes[row, col]; style(ax)
                mesh = ax.pcolormesh(p_edges, s_edges, data, cmap=DIVERGING, norm=TwoSlopeNorm(0, -lim, lim), shading="flat")
                if col > 0:
                    hatch(ax, p_edges, s_edges, lum & ~support[:, :, i])
                outline(ax, p_edges, s_edges, lum)
                ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal")
                ax.set_title(title, fontsize=9, color=INK, loc="left")
                ax.set_xlabel("phase-encode axis (mm)", fontsize=8)
                if col == 0:
                    ax.set_ylabel("slice axis (mm)", fontsize=8)
                if col != 0:
                    bar = fig.colorbar(mesh, ax=ax, shrink=.85, extend="both")
                    bar.set_label("axial velocity (m/s), CFD and Koma" if col == 1 else "Koma − CFD (m/s)", fontsize=8, color=INK2)
                    bar.ax.tick_params(labelsize=7, colors=INK2)
        fig.suptitle(f"Axial velocity on cross-sectional planes · bin {b}/15, {phase}\n{wrap(note, 150)}",
                     fontsize=10, color=INK, x=.01, ha="left")
        path = figures/f"axial_planes_bin{b:02d}.png"
        fig.savefig(path, dpi=170, facecolor=SURFACE); plt.close(fig); written.append(path)
        # Sagittal (central longitudinal) plane through the vessel axis.
        zsel = (z_mm >= SAGITTAL_Z_MM[0]) & (z_mm <= SAGITTAL_Z_MM[1])
        lum = lumen[SAGITTAL_SLICE]; keep = lum & support[SAGITTAL_SLICE]
        c = np.where(lum, cfd[SAGITTAL_SLICE], np.nan); k = np.where(keep, koma[SAGITTAL_SLICE], np.nan)
        d = np.where(keep, koma[SAGITTAL_SLICE]-cfd[SAGITTAL_SLICE], np.nan)
        limit = nice_limit(np.nanmax(abs(c[:, zsel])))
        p_idx = np.nonzero(lum[:, zsel].any(axis=1))[0]
        ylim = (p_edges[p_idx.min()]-3, p_edges[p_idx.max()+1]+3)
        fig, axes = plt.subplots(3, 1, figsize=(13, 7.6), facecolor=SURFACE, constrained_layout=True)
        win = keep[:, zsel]
        panels = ((c, "CFD", limit), (k, f"Koma · support {win.sum()}/{lum[:, zsel].sum()} lumen voxels in view", limit),
                  (d, "Koma − CFD · " + stats(koma[SAGITTAL_SLICE][:, zsel], cfd[SAGITTAL_SLICE][:, zsel], win), DIFF_LIMIT))
        for ax, (data, title, lim) in zip(axes, panels):
            style(ax)
            mesh = ax.pcolormesh(z_edges, p_edges, data, cmap=DIVERGING, norm=TwoSlopeNorm(0, -lim, lim), shading="flat")
            if title != "CFD":
                hatch(ax, z_edges, p_edges, lum & ~support[SAGITTAL_SLICE])
            outline(ax, z_edges, p_edges, lum)
            for name, target in PLANES:
                ax.axvline(target*1000, color=INK2, lw=.8, ls=(0, (4, 3)))
                if ax is axes[0]:
                    ax.text(target*1000, ylim[1]+.6, name, ha="center", va="bottom", fontsize=8, color=INK2)
            ax.set_xlim(*SAGITTAL_Z_MM); ax.set_ylim(*ylim); ax.set_aspect("equal")
            ax.set_title(title, fontsize=9, color=INK, loc="left", pad=14 if ax is axes[0] else 4)
            ax.set_ylabel("phase-encode axis (mm)", fontsize=8)
            bar = fig.colorbar(mesh, ax=ax, shrink=.9, extend="both", pad=.01)
            bar.set_label("difference (m/s)" if lim == DIFF_LIMIT else "axial velocity (m/s)", fontsize=8, color=INK2)
            bar.ax.tick_params(labelsize=7, colors=INK2)
        axes[-1].set_xlabel("vessel axis z (mm, stenosis at 0)", fontsize=8)
        fig.suptitle(f"Axial velocity on the central longitudinal (sagittal-like) plane, MRI slice {SAGITTAL_SLICE} · "
                     f"bin {b}/15, {phase}\n{wrap(note+' Anatomical orientation is not certified.', 170)}",
                     fontsize=10, color=INK, x=.01, ha="left")
        path = figures/f"sagittal_plane_bin{b:02d}.png"
        fig.savefig(path, dpi=170, facecolor=SURFACE); plt.close(fig); written.append(path)
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), status="PRESENTATION_DERIVATIVE_ONLY",
        inputs={str(p): sha(p) for p in [Path(__file__), fields_path, *inputs]},
        outputs={p.name: sha(p) for p in written},
        planes_source_z_m=dict(PLANES), bins=[b for b, _ in BINS], difference_scale_m_s=DIFF_LIMIT,
        note="Koma voxels shown only inside its own fixed 5% all-bin/all-encoding support; no filling or fitting.")
    (figures/"MANIFEST.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    figures.rename(FIGURES)
    print("\n".join(str(FIGURES/p.name) for p in written))


if __name__ == "__main__":
    main()
