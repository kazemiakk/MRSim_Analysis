"""E-058 post-hoc diagnostic (not preregistered): why does T2 fail in the fine arms?

Reads the completed arms and references only. Reports where the hottest throat
voxels sit, a partial-volume sensitivity (CFD fraction == 1) and the axial shift
of the Koma section-mean profile against CFD. Changes no registered verdict.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import tomllib

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import evaluate_targets as et  # noqa: E402

OUT = et.E058/"t2_diagnostic_v1"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
# Reference categorical slots 1-3 (validated default palette); line style is secondary encoding.
STYLE = dict(baseline_coarse=dict(color="#2a78d6", ls=(0, (4, 2)), label="baseline, coarse step (E-055)"),
             baseline_fine=dict(color="#eb6834", ls="-", label="baseline, fine step"),
             short_echo_fine=dict(color="#1baf7a", ls="-", label="short echo, fine step"))
SHIFT_RANGE = 8


def profile_shift(koma, cfd, keep, z_lo, z_hi):
    """Integer z shift (voxels, + = Koma downstream) minimising RMS of same-support section means."""
    def means(v):
        out = np.full(v.shape[-1], np.nan)
        for k in range(z_lo, z_hi):
            m = keep[..., k]
            if m.sum() >= 4:
                out[k] = v[..., k][m].mean()
        return out
    mk, mc = means(koma), means(cfd)
    best = None
    for s in range(-SHIFT_RANGE, SHIFT_RANGE+1):
        a, b = mk[z_lo+SHIFT_RANGE:z_hi-SHIFT_RANGE], mc[z_lo+SHIFT_RANGE-s:z_hi-SHIFT_RANGE-s]
        ok = np.isfinite(a) & np.isfinite(b)
        rms = float(np.sqrt(np.mean((a[ok]-b[ok])**2)))
        if best is None or rms < best[1]:
            best = (s, rms)
    return dict(best_shift_voxels=best[0], rms_at_best_m_s=best[1],
                rms_at_zero_m_s=float(np.sqrt(np.nanmean((mk[z_lo:z_hi]-mc[z_lo:z_hi])**2))))


def main():
    if OUT.exists() or OUT.with_name(OUT.name+".building").exists():
        raise FileExistsError(OUT)
    targets = json.loads((et.E058/"targets_v1/report.json").read_text())
    info = tomllib.loads((et.E055/"inputs_v1/case_info.toml").read_text())
    ref48 = et.bind_reference(et.E048/"reference_v1", "reference_fields.h5", "f4c4ec3153d30a2a9dfd360d8ff1c21811e14e6f942697cdd990b669fec75694")
    local48 = et.bind_reference(et.E048/"local_v1", "local_fields.h5", "6c57a8708ea4a5aff09f61db232b0ba1528ae4d000bf60d9ca00524516a27885")
    with h5py.File(ref48) as f:
        fraction = f["fraction_fft"][:]
    with h5py.File(local48) as f:
        source_z = f["source_z_fft_m"][:]
    dz = float(np.median(np.diff(source_z)))
    low, high = et.REGIONS["throat"]
    region = (fraction > 0) & ((source_z >= low) & (source_z < high))[None, None, :]
    b = targets["peak_bin"]-1
    z_mm = source_z*1e3
    window = np.flatnonzero((z_mm >= -40) & (z_mm <= 60))
    z_lo, z_hi = int(window[0]), int(window[-1])+1
    results, curves = {}, {}
    for arm, spec in et.ARMS.items():
        report = tomllib.loads((spec["folder"]/"control_report.toml").read_text())
        images, _ = et.load_arm(spec["folder"], report, candidate=True)
        field = et.velocities(images, info["Venc_m_s"])
        support = et.per_bin_support(images)[b]
        if spec["timing"] == "baseline":
            with h5py.File(ref48) as f:
                cfd = f["fft_adc"][b][..., 0]
        else:
            with h5py.File(et.E058/"references_v1/references.h5") as f:
                cfd = f["fft_adc_short_echo"][b][..., 0]
        koma = field[b][..., 0]
        rel = np.min([np.abs(images[k][b])/np.abs(images[k][b]).max() for k in images], axis=0)
        reg = region & support
        cfd_max = float(np.nanmax(cfd[region]))
        hot = np.unravel_index(np.argmax(np.where(reg, koma, -np.inf)), koma.shape)
        interior = reg & (fraction >= 1)
        lumen = (fraction > 0) & support
        results[arm] = dict(
            registered_T2_max_error=targets["arms"][arm]["per_bin_support"]["T2_peak_throat_velocity"]["max_error"],
            hottest_voxel=dict(index=[int(i) for i in hot], z_mm=float(z_mm[hot[2]]), Koma_m_s=float(koma[hot]),
                               CFD_m_s=float(cfd[hot]), cfd_fraction=float(fraction[hot]), min_relative_magnitude=float(rel[hot])),
            voxels_above_CFD_region_max=int(np.sum(koma[reg] > cfd_max)),
            above_max_with_partial_fraction=int(np.sum((koma > cfd_max) & reg & (fraction < 1))),
            interior_only=dict(voxels=int(interior.sum()), Koma_max_m_s=float(koma[interior].max()),
                               CFD_interior_max_m_s=float(np.nanmax(cfd[region & (fraction >= 1)])),
                               max_error=float(koma[interior].max()/np.nanmax(cfd[region & (fraction >= 1)])-1),
                               median_voxel_error_m_s=float(np.median(koma[interior]-cfd[interior]))),
            axial_shift=profile_shift(koma, cfd, lumen, z_lo, z_hi))
        sec_k = np.array([koma[..., k][lumen[..., k]].max() if lumen[..., k].any() else np.nan for k in range(z_lo, z_hi)])
        sec_c = np.array([np.nanmax(cfd[..., k][fraction[..., k] > 0]) if (fraction[..., k] > 0).any() else np.nan for k in range(z_lo, z_hi)])
        curves[arm] = dict(section_max=sec_k, cfd_section_max=sec_c, rel=rel[reg], err=koma[reg]-cfd[reg], frac=fraction[reg])
        print("E058_T2_DIAG", arm, json.dumps(results[arm]), flush=True)
        del images, field
    stage = OUT.with_name(OUT.name+".building")
    stage.mkdir()
    fig = plt.figure(figsize=(15, 9.5), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=.42, wspace=.28)
    ax = fig.add_subplot(gs[0, :2])
    zz = z_mm[z_lo:z_hi]
    ax.plot(zz, curves["baseline_fine"]["cfd_section_max"], color=INK, lw=2, label="CFD reference (baseline ADC timing)")
    for arm, st in STYLE.items():
        ax.plot(zz, curves[arm]["section_max"], color=st["color"], ls=st["ls"], lw=2, label=st["label"])
    ax.axvspan(low*1e3, high*1e3, color=GRID, zorder=0)
    ax.text(0, .02, "throat region", transform=ax.get_xaxis_transform(), ha="center", fontsize=8, color=INK2)
    ax.set_xlabel("axial position from throat (mm)", fontsize=9, color=INK2)
    ax.set_ylabel("section max axial velocity (m/s)", fontsize=9, color=INK2)
    ax.set_title(f"A  Peak bin {b+1}: per-slice maximum axial velocity on own per-bin support", fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax2 = fig.add_subplot(gs[0, 2])
    for arm, st in STYLE.items():
        rows = targets["arms"][arm]["per_bin_support"]["T1_throat_flow"]["bins"]
        x = [r["bin"] for r in rows if r["same_area_relative_error"] is not None]
        y = [100*r["same_area_relative_error"] for r in rows if r["same_area_relative_error"] is not None]
        ax2.plot(x, y, color=st["color"], ls=st["ls"], lw=2, marker="o", ms=4, label=st["label"])
    ax2.axhspan(-8, 8, color=GRID, zorder=0)
    ax2.axhline(0, color=INK2, lw=.6)
    ax2.set_xlabel("cardiac bin", fontsize=9, color=INK2)
    ax2.set_ylabel("throat same-area flow error (%)", fontsize=9, color=INK2)
    ax2.set_title("B  Throat flow error per bin (band = ±8 % target)", fontsize=10, color=INK, loc="left")
    for i, (arm, st) in enumerate(STYLE.items()):
        a = fig.add_subplot(gs[1, i])
        c = curves[arm]
        part = c["frac"] < 1
        a.scatter(100*c["rel"][~part], c["err"][~part], s=10, color=st["color"], alpha=.7, lw=0, label="fully inside lumen")
        a.scatter(100*c["rel"][part], c["err"][part], s=14, facecolor="none", edgecolor=st["color"], lw=.8, label="partial-volume voxel")
        a.axhline(0, color=INK2, lw=.6)
        a.axvline(5, color=INK2, lw=.6, ls=(0, (3, 3)))
        a.set_xscale("log"); a.set_ylim(-2.5, 4.5)
        a.set_xlabel("min relative magnitude across encodings (%)", fontsize=9, color=INK2)
        if i == 0:
            a.set_ylabel("Koma − CFD axial velocity (m/s)", fontsize=9, color=INK2)
        r = results[arm]
        a.set_title(f"{'CDE'[i]}  {st['label']}\nT2 max error {100*r['registered_T2_max_error']:+.1f} %; interior-only {100*r['interior_only']['max_error']:+.1f} %",
                    fontsize=9, color=INK, loc="left")
        a.legend(fontsize=7, frameon=False, loc="upper right")
    for a in fig.axes:
        a.set_facecolor(SURFACE); a.tick_params(labelsize=8, colors=INK2); a.grid(color=GRID, lw=.5)
        for side in a.spines.values():
            side.set_color("#c3c2b7")
    fig.suptitle("E-058 post-hoc T2 diagnostic: same-source internal consistency, noiseless, single spin sample",
                 fontsize=11, color=INK, x=.06, ha="left")
    fig.savefig(stage/"t2_diagnostic.png", dpi=120, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    report = dict(experiment="E-058", stage="post-hoc T2 diagnostic (not preregistered; registered verdicts unchanged)",
                  utc=datetime.now(timezone.utc).isoformat(), peak_bin=b+1, z_spacing_mm=dz*1e3, arms=results,
                  evidence_class="Same-source internal consistency, noiseless; exploratory.")
    et.dump(stage/"report.json", report)
    et.dump(stage/"MANIFEST.json", dict(script_sha256=et.sha(Path(__file__)), targets_report_sha256=et.sha(et.E058/"targets_v1/report.json"),
                                        outputs={p.name: et.sha(p) for p in sorted(stage.iterdir()) if p.name != "MANIFEST.json"}))
    stage.rename(OUT)
    print("E058_T2_DIAG_DONE", OUT)


if __name__ == "__main__":
    main()
