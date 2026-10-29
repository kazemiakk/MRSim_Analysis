"""Presentation: through-plane velocity contours WITH background, incl. Koma + scan-matched noise.

Rows: 4D-flow MRI, CFD, Koma (noiseless), Koma + synthetic noise. Columns: proximal -25,
throat 0, distal +15 mm; each dataset at its own peak throat-flow phase; no circular mask
(square +/-16 mm around each vessel), so background is visible.

Noise model (display only; registered metrics unchanged): the scan velocity noise sigma_v
per axis is the median over static-object voxels (non-vessel magnitude components, E-059
25 % rule) of the temporal standard deviation over the 30 phases. Complex white Gaussian
noise (std sigma per real/imaginary part, fixed seed) is added to each Koma encoding image
with sigma = sigma_v,z * pi * |M| / (sqrt(2) * VENC_z), |M| = median REF magnitude in the
CFD lumen within the slab at the peak bin; velocity then recomputed from the phase difference.
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
import analyze_e060 as an  # noqa: E402
import plot_contours_circular as base  # noqa: E402

HALF, SEED = 16.0, 20261025


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def scan_noise_and_panels():
    rep = json.loads((E059/"measured_v2/report.json").read_text())
    z = np.load(PRIVATE)
    v, mag, pos = z["velocity_cm_s"]/100.0, z["magnitude"], z["slice_positions_mm"]
    sign = rep["segmentation"]["0.25"]["vendor_to_forward_sign"]
    throat = rep["throat_slice"]
    m = mag.mean(axis=1); thr = .25*np.percentile(m, 99.5)
    phase = int(np.argmax(rep["segmentation"]["0.25"]["slices"][throat]["flow_ml_s"]))
    stds, panels = [], []
    rel = pos-pos[throat]
    for s in range(v.shape[0]):
        lab, n = ndimage.label(m[s] >= thr)
        flows = [abs((sign*v[s, :, :, :, 2][:, lab == c]).sum(axis=1).mean()) for c in range(1, n+1)]
        vessel = 1+int(np.argmax(flows))
        for c in range(1, n+1):
            if c != vessel and (lab == c).sum() >= 20:
                stds.append(v[s][:, lab == c].std(axis=0))          # (voxels, 3)
    sigma_v = np.median(np.concatenate(stds), axis=0)
    for name, d in base.COLUMNS:
        s = int(np.argmin(abs(rel-d)))
        lab, n = ndimage.label(m[s] >= thr)
        flows = [abs((sign*v[s, :, :, :, 2][:, lab == c]).sum(axis=1).mean()) for c in range(1, n+1)]
        mask = lab == (1+int(np.argmax(flows)))
        cy, cx = ndimage.center_of_mass(mask)
        panels.append(dict(img=sign*v[s, phase, :, :, 2], pos=float(rel[s]), center=((cx-32)*1.25, (cy-32)*1.25), mask=mask))
    return sigma_v, panels, phase+1


def main(target):
    target = Path(target)
    if target.exists():
        raise FileExistsError(target)
    sigma_v, scan, sphase = scan_noise_and_panels()
    report = json.loads((E060/"analysis_v1/report.json").read_text()); b = report["peak_bin"]-1
    control = an.tomllib.loads((E060/"acquisition_v1/control_report.toml").read_text())
    images, _ = an.load_images(E060/"acquisition_v1", control, E060/"sequences_v2/timing_e060.h5")
    with h5py.File(E060/"references_v2/references.h5") as f:
        cfd, frac, zz, pfrac = f["fft_adc"][b][..., 2], f["fraction_fft"][:], f["partition_z_m"][:]*1e3, f["plane_fraction"][:]
    lum = frac > 0
    slab = (np.abs(zz) <= 30)[:, None, None]
    M = float(np.median(np.abs(images["REF"][b])[lum & slab]))
    sigma = float(sigma_v[2]*np.pi*M/(np.sqrt(2)*3.0))
    rng = np.random.default_rng(SEED)
    noisy = {e: images[e][b]+sigma*(rng.standard_normal(images[e][b].shape)+1j*rng.standard_normal(images[e][b].shape)) for e in an.ENC}
    k_clean = np.angle(images["Z"][b]*images["REF"][b].conj())*(3.0/np.pi)
    k_noisy = np.angle(noisy["Z"]*noisy["REF"].conj())*(3.0/np.pi)
    bg = ~lum & slab
    achieved = dict(lumen_koma_noise_sd_m_s=float(np.std((k_noisy-k_clean)[lum & slab])),
                    background_velocity_sd_m_s=float(np.std(k_noisy[bg])))
    rows = [scan, [], [], []]
    for name, d in base.COLUMNS:
        k = int(np.argmin(abs(zz-d)))
        cy, cx = ndimage.center_of_mass(pfrac[k]); c = ((cx-32)*1.25, (cy-32)*1.25)
        rows[1].append(dict(img=np.where(lum[k], cfd[k], np.nan), pos=float(zz[k]), center=c, mask=lum[k]))
        rows[2].append(dict(img=k_clean[k], pos=float(zz[k]), center=c, mask=lum[k]))
        rows[3].append(dict(img=k_noisy[k], pos=float(zz[k]), center=c, mask=lum[k]))
    labels = (f"4D-flow MRI\n(phase {sphase})", f"CFD\n(bin {b+1})", f"Koma, noiseless\n(bin {b+1})", f"Koma + scan-matched noise\n(bin {b+1})")
    x = (np.arange(64)-32)*1.25
    stage = target.with_name(target.name+".building"); stage.mkdir()
    for norm in (False, True):
        fig, axes = plt.subplots(4, 3, figsize=(12, 16), facecolor=base.SURFACE, layout="constrained")
        scales = [np.nanmax(np.where(r[1]["mask"], r[1]["img"], np.nan)) if norm else 1.0 for r in rows]
        vmax = 1.0 if norm else 3.0
        for i, (row, lab, sc) in enumerate(zip(rows, labels, scales)):
            for j, p in enumerate(row):
                ax = axes[i, j]
                img = p["img"]/sc
                im = ax.pcolormesh(x, x, img, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="nearest")
                ax.contour(x, x, np.nan_to_num(img, nan=0), levels=np.linspace(.2*vmax, vmax, 5), colors="#0b0b0b", linewidths=.5)
                ax.contour(x, x, p["mask"].astype(float), levels=[.5], colors="#52514e", linewidths=.9, linestyles="dashed")
                cx, cy = p["center"]
                ax.set_xlim(cx-HALF, cx+HALF); ax.set_ylim(cy-HALF, cy+HALF); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
                if i == 0:
                    ax.set_title(f"{base.COLUMNS[j][0]} ({base.COLUMNS[j][1]:+.0f} mm)", fontsize=12)
                if j == 0:
                    ax.set_ylabel(lab, fontsize=11)
        axes[-1, 0].plot([-HALF+1+row[0]["center"][0] for row in [rows[3]]]*2, [0, 0], alpha=0)
        cb = fig.colorbar(im, ax=axes, shrink=.5)
        cb.set_label("through-plane velocity / own throat peak" if norm else "through-plane velocity (m/s)")
        fig.suptitle("Through-plane velocity with background: 4D-flow MRI vs CFD vs Koma (noiseless and with scan-matched noise)\n"
                     f"Scan velocity noise from static objects: sigma_v = {sigma_v[0]:.3f}/{sigma_v[1]:.3f}/{sigma_v[2]:.3f} m/s (x/y/z); "
                     f"Koma noise added in all 4 encodings (seed {SEED}); dashed = vessel mask. Display only.", fontsize=10)
        fig.savefig(stage/f"contours_with_noise{'_normalized' if norm else ''}.png", dpi=140, facecolor=base.SURFACE); plt.close(fig)
    meta = dict(created_utc=datetime.now(timezone.utc).isoformat(), status="PRESENTATION_DERIVATIVE_ONLY", script_sha256=sha(__file__),
                scan_sigma_v_m_s=sigma_v.tolist(), koma_image_noise_sigma=sigma, koma_median_ref_magnitude=M, seed=SEED, achieved=achieved,
                private_input_used=str(PRIVATE), outputs={p.name: sha(p) for p in sorted(stage.glob("*.png"))})
    (stage/"MANIFEST.json").write_text(json.dumps(meta, indent=1)+"\n", encoding="utf-8")
    stage.rename(target)
    print(json.dumps({k: meta[k] for k in ("scan_sigma_v_m_s", "koma_image_noise_sigma", "achieved")}))


if __name__ == "__main__":
    main(sys.argv[1])
