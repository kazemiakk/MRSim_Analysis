"""E-060 comparison figures (presentation only; no new registered metrics).

Reads the registered acquisition and S4a references. Koma values are shown only
inside the per-bin all-encoding 5% support; CFD inside its lumen. Nothing fitted,
filled or smoothed. Same-source internal consistency, noiseless.
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
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402
import numpy as np  # noqa: E402

E060 = Path("D:/MRSim/outputs/scanner_geometry_e060")
sys.path.insert(0, str(E060/"code"))
import analyze_e060 as an  # noqa: E402

OUT = E060/"figures_v2"
SURFACE, INK, INK2, GRID, SPINE = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#c3c2b7"
KOMA, CFD = "#2a78d6", INK
SEQ = LinearSegmentedColormap.from_list("seq", ["#f0efec", "#9ec5f4", "#3987e5", "#184f95", "#0d366b"]); SEQ.set_bad(SURFACE)
DIV = LinearSegmentedColormap.from_list("div", ["#184f95", "#6da7ec", "#f0efec", "#f0a07c", "#b4441b"]); DIV.set_bad(SURFACE)
GRAY = plt.get_cmap("gray").copy(); GRAY.set_bad(SURFACE)
MM = (np.arange(64)-32)*1.25
COMP = ("R (source x)", "P (source y)", "S (axial, along vessel)")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def style(ax, grid=True):
    ax.set_facecolor(SURFACE); ax.tick_params(labelsize=7, colors=INK2)
    if grid:
        ax.grid(color=GRID, lw=.5)
    for s in ax.spines.values():
        s.set_color(SPINE)


def cbar(fig, h, ax, label):
    c = fig.colorbar(h, ax=ax, shrink=.85, pad=.02); c.ax.tick_params(labelsize=7, colors=INK2); c.set_label(label, fontsize=8, color=INK2)


def save(fig, stage, name, written):
    p = stage/name; fig.savefig(p, dpi=120, facecolor=SURFACE, bbox_inches="tight"); plt.close(fig); written.append(p); print(p, flush=True)


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    report = json.loads((E060/"analysis_v1/report.json").read_text())
    control = an.tomllib.loads((E060/"acquisition_v1/control_report.toml").read_text())
    images, _ = an.load_images(E060/"acquisition_v1", control, E060/"sequences_v2/timing_e060.h5")
    field = np.stack([np.angle(images[e]*images["REF"].conj())*(an.VENC[i]/np.pi) for i, e in enumerate(an.ENC[1:])], axis=-1)
    refs = E060/"references_v2/references.h5"
    with h5py.File(refs) as f:
        ref, frac, z = f["fft_adc"][:], f["fraction_fft"][:], f["partition_z_m"][:]
    with h5py.File(E060/"acquisition_v1/ref_result.h5") as f:
        kspace = f["rawsort"][:]
    nb = field.shape[0]
    support = np.ones(field.shape[:4], bool)
    for im in images.values():
        m = np.abs(im); support &= m >= .05*m.max(axis=(1, 2, 3), keepdims=True)
    lumen = frac > 0
    pb = report["peak_bin"]-1
    kz = an.THROAT_K
    rc = int(np.argmax(frac[kz].sum(axis=0))); pc = int(np.argmax(frac[kz].sum(axis=1)))
    zmm = z*1e3
    koma_ax = np.where(support[pb] & lumen, field[pb][..., 2], np.nan)
    cfd_ax = np.where(lumen, ref[pb][..., 2], np.nan)
    vmax = float(np.nanmax(cfd_ax))
    stage = OUT.with_name(OUT.name+".building"); stage.mkdir()
    written = []
    title = f"E-060 peak bin {pb+1} of 30 · same-source internal consistency, noiseless"

    # 1. Contours: Koma axial velocity with CFD iso-velocity contours.
    sections = [(5, "upstream"), (kz, "throat"), (21, "downstream")]
    levels = np.arange(.5, vmax, .5)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5), facecolor=SURFACE, gridspec_kw=dict(height_ratios=[1, .9]))
    for j, (k, name) in enumerate(sections):
        ax = axes[0, j]
        h = ax.pcolormesh(MM, MM, koma_ax[k], cmap=SEQ, vmin=0, vmax=vmax, shading="nearest")
        cs = ax.contour(MM, MM, np.nan_to_num(cfd_ax[k], nan=0), levels=levels, colors=INK, linewidths=.8)
        ax.clabel(cs, fontsize=6, fmt="%.1f")
        ax.contour(MM, MM, lumen[k].astype(float), levels=[.5], colors=INK2, linewidths=.8, linestyles="dashed")
        ax.set_xlim(-16, 16); ax.set_ylim(-16, 16); ax.set_aspect("equal")
        ax.set_title(f"{name} section, S = {zmm[k]:+.1f} mm", fontsize=10, color=INK, loc="left")
        ax.set_xlabel("R (mm)", fontsize=8, color=INK2); ax.set_ylabel("P (mm)", fontsize=8, color=INK2)
        style(ax, False); cbar(fig, h, ax, "Koma axial (m/s)")
    for j, (plane, lab) in enumerate((("P", f"R = {MM[rc]:.1f} mm"), ("R", f"P = {MM[pc]:.1f} mm"))):
        ax = axes[1, j]
        kv = koma_ax[:, :, rc].T if plane == "P" else koma_ax[:, pc, :].T
        cv = cfd_ax[:, :, rc].T if plane == "P" else cfd_ax[:, pc, :].T
        lm = lumen[:, :, rc].T if plane == "P" else lumen[:, pc, :].T
        h = ax.pcolormesh(zmm, MM, kv, cmap=SEQ, vmin=0, vmax=vmax, shading="nearest")
        cs = ax.contour(zmm, MM, np.nan_to_num(cv, nan=0), levels=levels, colors=INK, linewidths=.8)
        ax.clabel(cs, fontsize=6, fmt="%.1f")
        ax.contour(zmm, MM, lm.astype(float), levels=[.5], colors=INK2, linewidths=.8, linestyles="dashed")
        ax.set_ylim(-16, 16); ax.axvline(0, color=INK2, lw=.6, ls=(0, (3, 3)))
        ax.set_title(f"along vessel, {plane} vs S ({lab})", fontsize=10, color=INK, loc="left")
        ax.set_xlabel("S (mm)", fontsize=8, color=INK2); ax.set_ylabel(f"{plane} (mm)", fontsize=8, color=INK2)
        style(ax, False); cbar(fig, h, ax, "Koma axial (m/s)")
    ax = axes[1, 2]
    for k, name in sections:
        r = ~np.isnan(koma_ax[k]) & ~np.isnan(cfd_ax[k])
        ax.scatter(cfd_ax[k][r], koma_ax[k][r], s=10, alpha=.7, label=name)
    ax.plot([0, vmax], [0, vmax], color=INK2, lw=.8)
    ax.set_xlabel("CFD axial (m/s)", fontsize=8, color=INK2); ax.set_ylabel("Koma axial (m/s)", fontsize=8, color=INK2)
    ax.set_title("section voxels, Koma vs CFD", fontsize=10, color=INK, loc="left"); ax.legend(fontsize=7, frameon=False); style(ax)
    fig.suptitle(f"Contours: Koma axial velocity (colour) with CFD iso-velocity lines (black, 0.5 m/s steps) and CFD lumen (dashed) · {title}",
                 fontsize=11, color=INK, x=.02, ha="left")
    save(fig, stage, "c1_contours_koma_vs_cfd.png", written)

    # 2. Images Koma generated: magnitude, phase difference, velocity per encoding + k-space.
    fig, axes = plt.subplots(3, 4, figsize=(17, 12), facecolor=SURFACE)
    mag = np.abs(images["REF"][pb]); mag = mag/mag.max()
    h = axes[0, 0].pcolormesh(MM, MM, mag[kz], cmap=GRAY, vmin=0, vmax=1, shading="nearest")
    axes[0, 0].set_title("REF magnitude, throat slice", fontsize=9, color=INK, loc="left"); cbar(fig, h, axes[0, 0], "normalized")
    h = axes[1, 0].pcolormesh(zmm, MM, mag[:, :, rc].T, cmap=GRAY, vmin=0, vmax=1, shading="nearest")
    axes[1, 0].set_title("REF magnitude, along vessel", fontsize=9, color=INK, loc="left"); cbar(fig, h, axes[1, 0], "normalized")
    ks = np.abs(kspace[pb][kz//1])
    h = axes[2, 0].imshow(np.abs(np.fft.fftshift(np.fft.ifftshift(kspace[pb]), axes=(0,)))[15] if False else np.abs(kspace[pb][15]),
                          cmap="magma", norm=LogNorm(vmin=max(float(ks.max())*1e-5, 1e-12), vmax=float(ks.max())), origin="lower",
                          extent=(-32, 32, -32, 32))
    axes[2, 0].set_title("REF k-space |k|, central partition (log)", fontsize=9, color=INK, loc="left"); cbar(fig, h, axes[2, 0], "|signal|")
    axes[2, 0].set_xlabel("kx index", fontsize=8, color=INK2); axes[2, 0].set_ylabel("ky index", fontsize=8, color=INK2)
    for j, e in enumerate(("X", "Y", "Z")):
        dphi = np.angle(images[e][pb]*images["REF"][pb].conj())
        dphi = np.where(mag >= .05, dphi, np.nan)
        h = axes[0, j+1].pcolormesh(MM, MM, dphi[kz], cmap=DIV, vmin=-np.pi, vmax=np.pi, shading="nearest")
        axes[0, j+1].set_title(f"{e} − REF phase difference, throat slice", fontsize=9, color=INK, loc="left"); cbar(fig, h, axes[0, j+1], "rad")
        h = axes[1, j+1].pcolormesh(zmm, MM, dphi[:, :, rc].T, cmap=DIV, vmin=-np.pi, vmax=np.pi, shading="nearest")
        axes[1, j+1].set_title(f"{e} − REF phase difference, along vessel", fontsize=9, color=INK, loc="left"); cbar(fig, h, axes[1, j+1], "rad")
        v = np.where(support[pb], field[pb][..., j], np.nan)
        lim = an.VENC[j]
        h = axes[2, j+1].pcolormesh(zmm, MM, v[:, :, rc].T, cmap=DIV, vmin=-lim, vmax=lim, shading="nearest")
        axes[2, j+1].set_title(f"velocity {COMP[j]}, VENC {lim:.1f} m/s", fontsize=9, color=INK, loc="left"); cbar(fig, h, axes[2, j+1], "m/s")
    for ax in axes.flat:
        style(ax, False)
        if ax is not axes[2, 0]:
            if ax.get_xlim()[1] > 30 or ax in axes[1:, :].flat:
                ax.set_ylim(-16, 16)
            else:
                ax.set_xlim(-16, 16); ax.set_ylim(-16, 16)
    fig.suptitle(f"Images generated by Koma (REF/X/Y/Z encodings, no masking except |REF| ≥ 5 % for phase) · {title}",
                 fontsize=11, color=INK, x=.02, ha="left")
    save(fig, stage, "c2_koma_generated_images.png", written)

    # 3. Velocity profiles.
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), facecolor=SURFACE)
    for ax, (vk, vc, x, lab) in zip(axes[:2], ((koma_ax[kz, pc, :], cfd_ax[kz, pc, :], MM, "R (mm), throat, P = %.1f mm" % MM[pc]),
                                                (koma_ax[kz, :, rc], cfd_ax[kz, :, rc], MM, "P (mm), throat, R = %.1f mm" % MM[rc]))):
        ax.plot(x, vc, color=CFD, lw=2, label="CFD"); ax.plot(x, vk, color=KOMA, lw=2, marker="o", ms=3, label="Koma")
        ax.set_xlim(-10, 10); ax.set_xlabel(lab, fontsize=8, color=INK2); ax.set_ylabel("axial velocity (m/s)", fontsize=8, color=INK2)
        ax.set_title("profile across the throat", fontsize=10, color=INK, loc="left"); ax.legend(fontsize=8, frameon=False); style(ax)
    ax = axes[2]
    ax.plot(zmm, cfd_ax[:, pc, rc], color=CFD, lw=2, label="CFD, centreline")
    ax.plot(zmm, koma_ax[:, pc, rc], color=KOMA, lw=2, marker="o", ms=3, label="Koma, centreline")
    ax.axvspan(-37.5, -30, color=GRID); ax.axvspan(30, 37.5, color=GRID); ax.axvline(0, color=INK2, lw=.6, ls=(0, (3, 3)))
    ax.set_xlabel("S along vessel (mm; grey = outside slab)", fontsize=8, color=INK2); ax.set_ylabel("axial velocity (m/s)", fontsize=8, color=INK2)
    ax.set_title("centreline axial velocity", fontsize=10, color=INK, loc="left"); ax.legend(fontsize=8, frameon=False); style(ax)
    fig.suptitle(f"Velocity profiles · {title}", fontsize=11, color=INK, x=.02, ha="left")
    save(fig, stage, "c3_velocity_profiles.png", written)

    # 4. Voxel agreement: scatter and Bland-Altman (all three components, peak bin; axial all bins).
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.5), facecolor=SURFACE)
    keep = support[pb] & lumen
    for j in range(3):
        a, b = ref[pb][..., j][keep], field[pb][..., j][keep]
        ax = axes[0, j]; ax.hexbin(a, b, gridsize=60, bins="log", cmap=SEQ, mincnt=1)
        lo, hi = float(min(a.min(), b.min())), float(max(a.max(), b.max())); ax.plot([lo, hi], [lo, hi], color=INK2, lw=.8)
        rmse = float(np.sqrt(np.mean((b-a)**2)))
        ax.set_title(f"{COMP[j]}: RMSE {rmse:.3f} m/s, n = {a.size}", fontsize=9, color=INK, loc="left")
        ax.set_xlabel("CFD (m/s)", fontsize=8, color=INK2); ax.set_ylabel("Koma (m/s)", fontsize=8, color=INK2); style(ax)
        ax = axes[1, j]; mean, diff = (a+b)/2, b-a
        ax.hexbin(mean, diff, gridsize=60, bins="log", cmap=SEQ, mincnt=1)
        bias, sd = float(diff.mean()), float(diff.std())
        for y, ls in ((bias, "-"), (bias+1.96*sd, "--"), (bias-1.96*sd, "--")):
            ax.axhline(y, color=INK2, lw=.8, ls=ls)
        ax.set_title(f"Bland–Altman: bias {bias:+.3f}, 95 % limits ±{1.96*sd:.3f} m/s", fontsize=9, color=INK, loc="left")
        ax.set_xlabel("mean of Koma and CFD (m/s)", fontsize=8, color=INK2); ax.set_ylabel("Koma − CFD (m/s)", fontsize=8, color=INK2); style(ax)
    ka = np.where(support & lumen[None], field[..., 2], np.nan); ca = np.where(lumen[None], ref[..., 2], np.nan)
    r = ~np.isnan(ka) & ~np.isnan(ca)
    ax = axes[0, 3]; ax.hexbin(ca[r], ka[r], gridsize=60, bins="log", cmap=SEQ, mincnt=1)
    ax.plot([ca[r].min(), ca[r].max()], [ca[r].min(), ca[r].max()], color=INK2, lw=.8)
    ax.set_title(f"axial, all 30 bins: RMSE {float(np.sqrt(np.mean((ka[r]-ca[r])**2))):.3f} m/s", fontsize=9, color=INK, loc="left")
    ax.set_xlabel("CFD (m/s)", fontsize=8, color=INK2); ax.set_ylabel("Koma (m/s)", fontsize=8, color=INK2); style(ax)
    ax = axes[1, 3]
    rel = np.array([np.nanmean(np.where(support[b] & lumen, np.abs(field[b][..., 2]-ref[b][..., 2]), np.nan)) for b in range(nb)])
    ax.plot(np.arange(1, nb+1), rel, color=KOMA, lw=2, marker="o", ms=3)
    ax.set_title("mean |Koma − CFD| axial per bin (supported lumen)", fontsize=9, color=INK, loc="left")
    ax.set_xlabel("cardiac bin", fontsize=8, color=INK2); ax.set_ylabel("m/s", fontsize=8, color=INK2); style(ax)
    fig.suptitle(f"Voxel agreement on Koma's own per-bin support (log-count hexbins) · {title}", fontsize=11, color=INK, x=.02, ha="left")
    save(fig, stage, "c4_voxel_agreement.png", written)

    # 5. All three components at the throat cross-section.
    fig, axes = plt.subplots(3, 3, figsize=(13, 12), facecolor=SURFACE)
    for j in range(3):
        c = np.where(lumen[kz], ref[pb][kz][..., j], np.nan)
        k = np.where(support[pb][kz] & lumen[kz], field[pb][kz][..., j], np.nan)
        lim = max(float(np.nanmax(np.abs(c))), .05) if j < 2 else vmax
        cm, lo = (DIV, -lim) if j < 2 else (SEQ, 0)
        for i, (data, lab) in enumerate(((c, "CFD"), (k, "Koma"), (k-c, "Koma − CFD"))):
            ax = axes[j, i]
            if i == 2:
                dl = max(float(np.nanmax(np.abs(k-c))) if np.isfinite(k-c).any() else .1, .05)
                h = ax.pcolormesh(MM, MM, data, cmap=DIV, vmin=-dl, vmax=dl, shading="nearest")
            else:
                h = ax.pcolormesh(MM, MM, data, cmap=cm, vmin=lo, vmax=lim, shading="nearest")
            ax.set_xlim(-8, 8); ax.set_ylim(-8, 8); ax.set_aspect("equal")
            ax.set_title(f"{lab}: {COMP[j]}", fontsize=9, color=INK, loc="left"); style(ax, False); cbar(fig, h, ax, "m/s")
    fig.suptitle(f"Throat cross-section (S = 0), all velocity components · {title}", fontsize=11, color=INK, x=.02, ha="left")
    save(fig, stage, "c5_throat_components.png", written)

    # 6. Time course.
    region = lumen & ((zmm >= -10) & (zmm < 10))[:, None, None]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), facecolor=SURFACE)
    km = [np.nanmean(np.where(support[b] & region, field[b][..., 2], np.nan)) for b in range(nb)]
    cmn = [np.nanmean(np.where(support[b] & region, ref[b][..., 2], np.nan)) for b in range(nb)]
    kx = [np.nanmax(np.where(support[b] & region, field[b][..., 2], np.nan)) for b in range(nb)]
    cx = [np.nanmax(np.where(region, ref[b][..., 2], np.nan)) for b in range(nb)]
    bb = np.arange(1, nb+1)
    axes[0].plot(bb, cmn, color=CFD, lw=2, label="CFD"); axes[0].plot(bb, km, color=KOMA, lw=2, marker="o", ms=3, label="Koma")
    axes[0].set_title("mean axial velocity, throat region (same support)", fontsize=10, color=INK, loc="left")
    axes[1].plot(bb, cx, color=CFD, lw=2, label="CFD"); axes[1].plot(bb, kx, color=KOMA, lw=2, marker="o", ms=3, label="Koma")
    axes[1].set_title("maximum axial velocity, throat region", fontsize=10, color=INK, loc="left")
    cov = [r["whole_lumen_coverage"] for r in report["targets_per_bin_support"]["T1_throat_flow"]["bins"]]
    sec = [r["section_area_coverage"] for r in report["targets_per_bin_support"]["T1_throat_flow"]["bins"]]
    axes[2].plot(bb, 100*np.array(sec), color=KOMA, lw=2, marker="o", ms=3, label="throat section")
    axes[2].plot(bb, 100*np.array(cov), color=INK2, lw=2, ls=(0, (4, 2)), label="whole lumen")
    axes[2].axhline(90, color=INK2, lw=.6, ls=(0, (3, 3)))
    axes[2].set_title("coverage per bin (per-bin 5 % support)", fontsize=10, color=INK, loc="left"); axes[2].set_ylabel("%", fontsize=8, color=INK2)
    for ax in axes:
        ax.axvline(pb+1, color=INK2, lw=.6, ls=(0, (3, 3))); ax.set_xlabel("cardiac bin", fontsize=8, color=INK2)
        ax.legend(fontsize=8, frameon=False); style(ax)
    axes[0].set_ylabel("m/s", fontsize=8, color=INK2); axes[1].set_ylabel("m/s", fontsize=8, color=INK2)
    fig.suptitle("Over the cardiac cycle (dashed: peak bin) · same-source internal consistency, noiseless", fontsize=11, color=INK, x=.02, ha="left")
    save(fig, stage, "c6_time_course.png", written)

    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), status="PRESENTATION_DERIVATIVE_ONLY",
                    inputs={str(p): sha(p) for p in (Path(__file__), refs, E060/"analysis_v1/report.json", E060/"acquisition_v1/control_report.toml")},
                    outputs={p.name: sha(p) for p in written})
    (stage/"MANIFEST.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    stage.rename(OUT)


if __name__ == "__main__":
    main()
