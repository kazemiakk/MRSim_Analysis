"""Koma phase-contrast phase images (user request): delta-phi = angle(encoded x conj(REF)) per encoding, peak bin.

Phantom: E-060 VENC 3 complex images (VENC 1.5/1.5/3.0 m/s), slices across the vessel at the throat (R-P) and
along it (P-S through the vessel centre). Aorta: E-062 velocity x pi / VENC (2.5 m/s), slices X-Y and Y-Z
through the planes with the most vessel. Unmasked (background phase is undefined where there is no signal,
as on a scanner). Display: 4x upsampling of exp(i phi) (wrap-safe), grey -pi..pi. Presentation only.
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
E062 = Path("D:/MRSim/outputs/aorta_20bin_e062")
sys.path.insert(0, str(E060/"code"))
FS, UP = 1.5, 4


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def up(phi):
    """Wrap-safe upsampling of a 2-D phase map."""
    re = ndimage.zoom(np.cos(phi), UP, order=1); im = ndimage.zoom(np.sin(phi), UP, order=1)
    return np.arctan2(im, re)


def figure(path, slices, names_venc, title):
    """slices: list of (label, phi (3, rows, cols) for X/Y/Z, extent, xlabel, ylabel)."""
    fig, axes = plt.subplots(len(slices), 3, figsize=(20, 7*len(slices)), layout="constrained", squeeze=False)
    for r, (lab, phis, extent, xl, yl) in enumerate(slices):
        for c in range(3):
            ax = axes[r, c]
            im = ax.imshow(up(phis[c]), origin="lower", extent=extent, cmap="gray", vmin=-np.pi, vmax=np.pi, aspect="equal", interpolation="bilinear")
            ax.set_title(f"{names_venc[c]} | {lab}", fontsize=10*FS); ax.set_xlabel(xl, fontsize=8*FS); ax.set_ylabel(yl, fontsize=8*FS)
            ax.tick_params(labelsize=8*FS)
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=.6, ticks=[-np.pi, 0, np.pi]); cb.ax.set_yticklabels(["−π", "0", "π"])
    cb.set_label("Phase difference (rad)")
    fig.suptitle(title, fontsize=13*FS)
    fig.savefig(path, dpi=110); plt.close(fig)


def main(target):
    target = Path(target)
    if target.exists():
        raise FileExistsError(target)
    plt.rcParams["font.size"] = 10*FS
    stage = target.with_name(target.name+".building"); stage.mkdir()
    import plot_projection_triptych as tri
    images, vencs = tri.arms()["E060_VENC3"]
    b = json.loads((E060/"analysis_v1/report.json").read_text())["peak_bin"]-1
    with h5py.File(E060/"references_v2/references.h5") as f:
        zz, pfrac = f["partition_z_m"][:]*1e3, f["plane_fraction"][:]
    # v2: scan-like noise (scan lumen SNR, one sigma, fixed seed) so the background shows random phase as on a scanner.
    import plot_contours_with_noise as nz
    snr = nz.scan_noise_and_panels()[3]
    with h5py.File(E060/"references_v2/references.h5") as f:
        frac = f["fraction_fft"][:]
    sig = float(np.median(np.abs(images["REF"]).mean(axis=0)[(frac > 0) & (np.abs(zz) <= 30)[:, None, None]]))/snr
    rng = np.random.default_rng(20261025)
    nzy = {e: images[e][b]+sig*(rng.standard_normal(images[e][b].shape)+1j*rng.standard_normal(images[e][b].shape)) for e in ("REF", "X", "Y", "Z")}
    phi = np.stack([np.angle(nzy[e]*nzy["REF"].conj()) for e in ("X", "Y", "Z")])   # (3, S, P, R)
    ks = int(np.argmin(abs(zz))); rc = int(np.argmax(pfrac[ks].sum(axis=0)))
    pr = (np.arange(64)-32)*1.25
    figure(stage/"phantom_e060_phase_views.png",
           [("across vessel at throat", phi[:, ks], [pr[0], pr[-1], pr[0], pr[-1]], "R, readout (mm)", "P, phase (mm)"),
            ("along vessel", np.transpose(phi[:, :, :, rc], (0, 1, 2)), [pr[0], pr[-1], zz[0], zz[-1]], "P, phase (mm)", "S, slab (mm)")],
           [f"X (R), VENC {vencs[0]}", f"Y (P), VENC {vencs[1]}", f"Z (S, along vessel), VENC {vencs[2]}"],
           f"Phantom (E-060) Koma phase images | bin {b+1}/30 (peak) | scan-matched noise, SNR {snr:.0f}")
    with h5py.File(E062/"combined/simulation_motion_1m_20bin_4mm/pc_reconstruction.h5") as f:
        vel, mag = f["velocity"][:], f["reference_magnitude"][:]                  # (3, bin, Z, Y, X), (bin, Z, Y, X)
    with h5py.File(E062/"analysis_v1/matched_reference_20bin.h5") as f:
        ref, core = f["adc_time_average"][:], f["masks/core15"][:].astype(bool)
    ba = int(np.argmax([np.linalg.norm(ref[k][core], axis=-1).mean() for k in range(ref.shape[0])]))
    # v2: aorta VENC 2.5 m/s uses ~6 % of the phase range; show the v22 display realisation instead
    # (SNR 12, phase as for VENC 0.3 m/s; Koma ran at 2.5 m/s).
    sys.path.insert(1, "D:/MRSim/outputs/aorta_presentation_2026-10-25/code")
    from aorta_common import Aorta, VENC_NOISE
    A = Aorta(); venc = VENC_NOISE
    ph = np.moveaxis(A.vnall[ba], -1, 0)*np.pi/venc
    kz = int(np.argmax(core.sum(axis=(1, 2)))); kx = int(np.argmax(core.sum(axis=(0, 1))))
    nz, ny, nx = core.shape
    ax_x = (np.arange(nx)-nx//2)*280/nx; ax_y = (np.arange(ny)-ny//2)*4.0; ax_z = (np.arange(nz)-nz//2)*4.0
    figure(stage/"aorta_e062_phase_views.png",
           [(f"X-Y slice, Z = {ax_z[kz]:.0f} mm", ph[:, kz], [ax_x[0], ax_x[-1], ax_y[0], ax_y[-1]], "X, readout (mm)", "Y, phase (mm)"),
            (f"Y-Z slice, X = {ax_x[kx]:.0f} mm", ph[:, :, :, kx], [ax_y[0], ax_y[-1], ax_z[0], ax_z[-1]], "Y, phase (mm)", "Z, partition (mm)")],
           [f"X (readout), VENC {venc}", f"Y (phase), VENC {venc}", f"Z (partition), VENC {venc}"],
           f"Aorta (E-062) Koma phase images | bin {ba+1}/20 (peak) | noise SNR 12, phase as for VENC {venc} m/s")
    (stage/"MANIFEST.json").write_text(json.dumps(dict(created_utc=datetime.now(timezone.utc).isoformat(), status="PRESENTATION_DERIVATIVE_ONLY",
        script_sha256=sha(__file__), phantom_bin=b+1, aorta_bin=ba+1, outputs={p.name: sha(p) for p in stage.glob("*.png")}), indent=1)+"\n", encoding="utf-8")
    stage.rename(target)
    print("DONE")


if __name__ == "__main__":
    main(sys.argv[1])
