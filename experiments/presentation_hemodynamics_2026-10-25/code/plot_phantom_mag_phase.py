"""Phantom magnitude + 3 phase images in one figure (user request).

Peak bin, E-060 VENC 3 images with scan-matched noise (as phase_views_v3, same seed). Columns: magnitude, X, Y, Z
phase differences. Rows: along the vessel (projection through R: |sum REF| for magnitude, phase of sum enc x conj(REF))
and across the vessel at the throat (slice). 4x upsampling (complex for phase), phase masked where the signal is
< 12 % of its 99.5th percentile. Presentation only.
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
sys.path.insert(0, str(E060/"code")); sys.path.insert(1, str(Path(__file__).parent))
from plot_phase_views_v3 import CMAP, UP, render  # noqa: E402
FS = 1.5


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(target):
    target = Path(target)
    if target.exists():
        raise FileExistsError(target)
    import plot_projection_triptych as tri
    import plot_contours_with_noise as nz
    images, vencs = tri.arms()["E060_VENC3"]
    b = json.loads((E060/"analysis_v1/report.json").read_text())["peak_bin"]-1
    with h5py.File(E060/"references_v2/references.h5") as f:
        zz, frac = f["partition_z_m"][:]*1e3, f["fraction_fft"][:]
    snr = nz.scan_noise_and_panels()[3]
    sig = float(np.median(np.abs(images["REF"]).mean(axis=0)[(frac > 0) & (np.abs(zz) <= 30)[:, None, None]]))/snr
    rng = np.random.default_rng(20261025)
    nzy = {e: images[e][b]+sig*(rng.standard_normal(images[e][b].shape)+1j*rng.standard_normal(images[e][b].shape)) for e in ("REF", "X", "Y", "Z")}
    C = {e: nzy[e]*nzy["REF"].conj() for e in ("X", "Y", "Z")}
    # v2: true centre slices (v1 projected through R, which cancels the symmetric in-plane flow and made X/Y flat),
    # one magnitude-based mask per row shared by all panels (v1 masked each phase panel separately -> holes).
    with h5py.File(E060/"references_v2/references.h5") as f:
        pfr = f["plane_fraction"][:]
    ks = int(np.argmin(abs(zz))); pr = (np.arange(64)-32)*1.25
    rc = int(np.argmax(pfr[ks].sum(axis=0))); pc = int(np.argmax(pfr[ks].sum(axis=1)))
    mref = float(np.percentile(np.abs(nzy["REF"]), 99.5))
    rows = [("Along vessel, S–P slice", lambda a: a[:, :, rc], [pr[0], pr[-1], zz[0], zz[-1]], "P (mm)", "S, along vessel (mm)"),
            ("Along vessel, S–R slice", lambda a: a[:, pc, :], [pr[0], pr[-1], zz[0], zz[-1]], "R (mm)", "S, along vessel (mm)"),
            ("Across vessel at throat", lambda a: a[ks], [pr[0], pr[-1], pr[0], pr[-1]], "R (mm)", "P (mm)")]
    plt.rcParams["font.size"] = 10*FS
    stage = target.with_name(target.name+".building"); stage.mkdir()
    fig, axes = plt.subplots(3, 4, figsize=(24, 18), layout="constrained")
    names = ("Magnitude", f"Phase X (R), VENC {vencs[0]}", f"Phase Y (P), VENC {vencs[1]}", f"Phase Z (S), VENC {vencs[2]}")
    for r, (lab, red, ext, xl, yl) in enumerate(rows):
        mag = np.clip(ndimage.zoom(np.abs(red(nzy["REF"])), UP, order=3), 0, None)
        keep = ndimage.gaussian_filter(mag, UP) >= .15*mref                       # shared mask from smoothed magnitude
        ax = axes[r, 0]
        im_m = ax.imshow(mag, origin="lower", extent=ext, cmap="gray", vmin=0, vmax=mref, aspect="equal", interpolation="bilinear")
        for c, e in enumerate(("X", "Y", "Z")):
            ax = axes[r, c+1]; cc = red(C[e])
            ph = np.arctan2(ndimage.zoom(cc.imag, UP, order=3), ndimage.zoom(cc.real, UP, order=3))
            im_p = ax.imshow(np.where(keep, ph, np.nan), origin="lower", extent=ext, cmap=CMAP, vmin=-np.pi, vmax=np.pi, aspect="equal", interpolation="bilinear")
            ax.set_facecolor("black")
        for c in range(4):
            ax = axes[r, c]
            ax.set_title(f"{names[c]}" if r == 0 else "", fontsize=10*FS); ax.set_xlabel(xl, fontsize=8*FS); ax.tick_params(labelsize=8*FS)
            if c == 0:
                ax.set_ylabel(f"{lab}\n{yl}", fontsize=8.5*FS)
    fig.colorbar(im_m, ax=axes[:, 0].tolist(), shrink=.6, label="Magnitude (a.u.)", location="bottom")
    cb = fig.colorbar(im_p, ax=axes[:, 1:].ravel().tolist(), shrink=.6, ticks=[-np.pi, 0, np.pi], location="right")
    cb.ax.set_yticklabels(["−π", "0", "π"]); cb.set_label("Phase difference (rad); black = no signal")
    fig.suptitle(f"Phantom (E-060) Koma: magnitude and phase images | peak bin {b+1}/30 | scan-matched noise, SNR {snr:.0f}", fontsize=13*FS)
    fig.supxlabel("Centre slices; phase masked where the smoothed magnitude is < 15 % (black). In-plane phases: flow converges toward the axis before the throat "
                  "and diverges after it (opposite signs on the two sides); X is ~0 in the S–P plane and Y in the S–R plane by symmetry.", fontsize=8.5*FS)
    fig.savefig(stage/"phantom_magnitude_phase.png", dpi=110); plt.close(fig)
    (stage/"MANIFEST.json").write_text(json.dumps(dict(created_utc=datetime.now(timezone.utc).isoformat(), status="PRESENTATION_DERIVATIVE_ONLY",
        script_sha256=sha(__file__), peak_bin=b+1, snr=snr, outputs={p.name: sha(p) for p in stage.glob("*.png")}), indent=1)+"\n", encoding="utf-8")
    stage.rename(target)
    print("DONE")


if __name__ == "__main__":
    main(sys.argv[1])
