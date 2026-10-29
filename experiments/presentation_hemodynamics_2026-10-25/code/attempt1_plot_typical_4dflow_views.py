"""Scanner-style 4D-flow display (user example): one slice, 2x2 = magnitude + through-plane + two in-plane phase images.

Phase images: velocity = phase difference x VENC / pi, grey -VENC (black) .. +VENC (white), zero = mid-grey,
unmasked (background noise visible), native pixels (nearest). Magnitude: mean over all bins of the noisy |REF|.
Phantom: E-060 VENC 3 images + scan-matched noise (SNR 84), peak bin, slice along the vessel through its centre
(S-P plane; through-plane = R). Aorta: E-062 + v22 display noise (SNR 12, phase as for VENC 0.3 m/s; Koma ran at
2.5 m/s), peak bin, the Y-Z slice with the most vessel (through-plane = X). No static tissue exists in either
phantom, so the area outside the vessel is noise, as air in a scan. Presentation only.
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

E060 = Path("D:/MRSim/outputs/scanner_geometry_e060")
sys.path.insert(0, str(E060/"code")); sys.path.insert(1, "D:/MRSim/outputs/aorta_presentation_2026-10-25/code")
FS = 1.5


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def figure(path, mag, vels, vencs_cm, labels, extent, xl, yl, title):
    fig, axes = plt.subplots(2, 2, figsize=(16, 16*(extent[3]-extent[2])/(extent[1]-extent[0])*1.02+1.5), layout="constrained")
    ax = axes[0, 0]
    ax.imshow(mag, origin="lower", extent=extent, cmap="gray", vmin=0, vmax=np.percentile(mag, 99.5), aspect="equal", interpolation="nearest")
    ax.set_title("Magnitude", fontsize=10*FS)
    for k, ax in enumerate((axes[0, 1], axes[1, 0], axes[1, 1])):
        im = ax.imshow(vels[k], origin="lower", extent=extent, cmap="gray", vmin=-vencs_cm[k], vmax=vencs_cm[k], aspect="equal", interpolation="nearest")
        ax.set_title(f"{labels[k]}  |  VENC {vencs_cm[k]:.0f} cm/s", fontsize=10*FS)
        cb = fig.colorbar(im, ax=ax, shrink=.8, ticks=[-vencs_cm[k], 0, vencs_cm[k]]); cb.set_label("cm/s", fontsize=8*FS); cb.ax.tick_params(labelsize=8*FS)
    for ax in axes.ravel():
        ax.set_xlabel(xl, fontsize=8*FS); ax.set_ylabel(yl, fontsize=8*FS); ax.tick_params(labelsize=8*FS); ax.set_facecolor("black")
    fig.suptitle(title, fontsize=12*FS)
    fig.savefig(path, dpi=110); plt.close(fig)


def main(target):
    target = Path(target)
    if target.exists():
        raise FileExistsError(target)
    plt.rcParams["font.size"] = 10*FS
    stage = target.with_name(target.name+".building"); stage.mkdir()
    import plot_projection_triptych as tri
    import plot_contours_with_noise as nz
    images, vencs = tri.arms()["E060_VENC3"]
    b = json.loads((E060/"analysis_v1/report.json").read_text())["peak_bin"]-1
    with h5py.File(E060/"references_v2/references.h5") as f:
        zz, frac, pfr = f["partition_z_m"][:]*1e3, f["fraction_fft"][:], f["plane_fraction"][:]
    snr = nz.scan_noise_and_panels()[3]
    sig = float(np.median(np.abs(images["REF"]).mean(axis=0)[(frac > 0) & (np.abs(zz) <= 30)[:, None, None]]))/snr
    rng = np.random.default_rng(20261025)
    nzy = {e: images[e]+sig*(rng.standard_normal(images[e].shape)+1j*rng.standard_normal(images[e].shape)) for e in ("REF", "X", "Y", "Z")}
    ks = int(np.argmin(abs(zz))); rc = int(np.argmax(pfr[ks].sum(axis=0))); pr = (np.arange(64)-32)*1.25
    sl = lambda a: a[:, :, rc]  # noqa: E731   (S, P) at the vessel centre in R
    v = {e: sl(np.angle(nzy[e][b]*nzy["REF"][b].conj()))*vencs[i]*100/np.pi for i, e in enumerate(("X", "Y", "Z"))}
    figure(stage/"phantom_4dflow_style.png", sl(np.abs(nzy["REF"]).mean(axis=0)), [v["X"], v["Y"], v["Z"]],
           [vencs[0]*100, vencs[1]*100, vencs[2]*100],
           ["PC through-plane (R)", "PC in-plane horizontal (P)", "PC in-plane vertical (S, along vessel)"],
           [pr[0]-.625, pr[-1]+.625, zz[0]-1.25, zz[-1]+1.25], "P (mm)", "S (mm)",
           f"Phantom (E-060) Koma | slice along the vessel | peak bin {b+1}/30 | noise SNR {snr:.0f}")
    from aorta_common import Aorta, VENC_NOISE
    A = Aorta()
    with h5py.File(Path("D:/MRSim/outputs/aorta_20bin_e062/analysis_v1/matched_reference_20bin.h5")) as f:
        ref = f["adc_time_average"][:]
    ba = int(np.argmax([np.linalg.norm(ref[k][A.core], axis=-1).mean() for k in range(A.nb)]))
    kx = int(np.argmax(A.core.sum(axis=(0, 1))))
    nz_, ny, nx = A.core.shape
    ax_x = (np.arange(nx)-nx//2)*280/nx; ax_y = (np.arange(ny)-ny//2)*4.0; ax_z = (np.arange(nz_)-nz_//2)*4.0
    va = A.vnall[ba][:, :, kx, :]*100                                                  # (Z, Y, 3) cm/s
    figure(stage/"aorta_4dflow_style.png", np.abs(A.REFall).mean(axis=0)[:, :, kx], [va[..., 0], va[..., 1], va[..., 2]],
           [VENC_NOISE*100]*3, ["PC through-plane (X)", "PC in-plane horizontal (Y)", "PC in-plane vertical (Z)"],
           [ax_y[0]-2, ax_y[-1]+2, ax_z[0]-2, ax_z[-1]+2], "Y (mm)", "Z (mm)",
           f"Aorta (E-062) Koma | Y–Z slice at X = {ax_x[kx]:.0f} mm | peak bin {ba+1}/20 | noise SNR 12, as for VENC {VENC_NOISE} m/s")
    (stage/"MANIFEST.json").write_text(json.dumps(dict(created_utc=datetime.now(timezone.utc).isoformat(), status="PRESENTATION_DERIVATIVE_ONLY",
        script_sha256=sha(__file__), phantom_bin=b+1, aorta_bin=ba+1, aorta_x_mm=float(ax_x[kx]),
        outputs={p.name: sha(p) for p in stage.glob("*.png")}), indent=1)+"\n", encoding="utf-8")
    stage.rename(target)
    print("DONE")


if __name__ == "__main__":
    main(sys.argv[1])
