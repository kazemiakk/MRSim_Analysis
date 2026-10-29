"""Koma phase-contrast phase images, v3 (user: v2 quality poor).

Phase of the magnitude-weighted complex sum C = sum(encoded x conj(REF)) through the third axis (whole vessel
visible; a slice only cut part of it), 4x cubic upsampling of Re/Im before the angle (no noise blobs from phase
interpolation), background masked where |C| < 12 % of its 99.5th percentile (black). Grey -pi (dark grey) .. +pi
(white). Noise as v2: phantom scan-matched SNR 84 (VENC 1.5/1.5/3.0 m/s); aorta v22 display realisation (SNR 12,
phase as for VENC 0.3 m/s; Koma ran at 2.5 m/s). Phantom: across the vessel at the throat (slice) and along
the vessel (projection through R). Aorta: projections through Z and through X. Peak bin. Presentation only.
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
from scipy import ndimage  # noqa: E402

E060 = Path("D:/MRSim/outputs/scanner_geometry_e060")
sys.path.insert(0, str(E060/"code")); sys.path.insert(1, "D:/MRSim/outputs/aorta_presentation_2026-10-25/code")
FS, UP = 1.5, 4
CMAP = LinearSegmentedColormap.from_list("phase", ["#3a3a3a", "#ffffff"]); CMAP.set_bad("black")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def render(c2d):
    re, im = (ndimage.zoom(x, UP, order=3) for x in (c2d.real, c2d.imag))
    amp = np.hypot(re, im)
    return np.where(amp >= .12*np.percentile(amp, 99.5), np.arctan2(im, re), np.nan)


def figure(path, rows, enc_names, title):
    fig, axes = plt.subplots(len(rows), 3, figsize=(21, 6.5*len(rows)), layout="constrained", squeeze=False)
    for r, (lab, cs, extent, xl, yl) in enumerate(rows):
        for c in range(3):
            ax = axes[r, c]
            im = ax.imshow(render(cs[c]), origin="lower", extent=extent, cmap=CMAP, vmin=-np.pi, vmax=np.pi, aspect="equal", interpolation="bilinear")
            ax.set_facecolor("black"); ax.set_title(f"{enc_names[c]} | {lab}", fontsize=10*FS)
            ax.set_xlabel(xl, fontsize=8*FS); ax.set_ylabel(yl, fontsize=8*FS); ax.tick_params(labelsize=8*FS)
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=.6, ticks=[-np.pi, 0, np.pi]); cb.ax.set_yticklabels(["−π", "0", "π"])
    cb.set_label("Phase difference (rad); black = no signal")
    fig.suptitle(title, fontsize=13*FS)
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
        zz, frac = f["partition_z_m"][:]*1e3, f["fraction_fft"][:]
    snr = nz.scan_noise_and_panels()[3]
    sig = float(np.median(np.abs(images["REF"]).mean(axis=0)[(frac > 0) & (np.abs(zz) <= 30)[:, None, None]]))/snr
    rng = np.random.default_rng(20261025)
    nzy = {e: images[e][b]+sig*(rng.standard_normal(images[e][b].shape)+1j*rng.standard_normal(images[e][b].shape)) for e in ("REF", "X", "Y", "Z")}
    C = [nzy[e]*nzy["REF"].conj() for e in ("X", "Y", "Z")]                       # (S, P, R)
    ks = int(np.argmin(abs(zz))); pr = (np.arange(64)-32)*1.25
    figure(stage/"phantom_e060_phase_views.png",
           [("across vessel at throat (slice)", [c[ks] for c in C], [pr[0], pr[-1], pr[0], pr[-1]], "R, readout (mm)", "P, phase (mm)"),
            ("along vessel (projection through R)", [c.sum(axis=2) for c in C], [pr[0], pr[-1], zz[0], zz[-1]], "P, phase (mm)", "S, slab (mm)")],
           [f"X (R), VENC {vencs[0]}", f"Y (P), VENC {vencs[1]}", f"Z (S, along vessel), VENC {vencs[2]}"],
           f"Phantom (E-060) Koma phase images | bin {b+1}/30 (peak) | scan-matched noise, SNR {snr:.0f}")
    from aorta_common import Aorta, VENC_NOISE
    A = Aorta()
    with h5py.File(Path("D:/MRSim/outputs/aorta_20bin_e062/analysis_v1/matched_reference_20bin.h5")) as f:
        ref = f["adc_time_average"][:]
    ba = int(np.argmax([np.linalg.norm(ref[k][A.core], axis=-1).mean() for k in range(A.nb)]))
    w = np.abs(A.REFall[ba])**2
    Ca = [w*np.exp(1j*A.vnall[ba][..., c]*np.pi/VENC_NOISE) for c in range(3)]       # (Z, Y, X)
    nz_, ny, nx = w.shape
    ax_x = (np.arange(nx)-nx//2)*280/nx; ax_y = (np.arange(ny)-ny//2)*4.0; ax_z = (np.arange(nz_)-nz_//2)*4.0
    figure(stage/"aorta_e062_phase_views.png",
           [("projection through Z", [c.sum(axis=0) for c in Ca], [ax_x[0], ax_x[-1], ax_y[0], ax_y[-1]], "X, readout (mm)", "Y, phase (mm)"),
            ("projection through X", [c.sum(axis=2) for c in Ca], [ax_y[0], ax_y[-1], ax_z[0], ax_z[-1]], "Y, phase (mm)", "Z, partition (mm)")],
           [f"X (readout), VENC {VENC_NOISE}", f"Y (phase), VENC {VENC_NOISE}", f"Z (partition), VENC {VENC_NOISE}"],
           f"Aorta (E-062) Koma phase images | bin {ba+1}/20 (peak) | noise SNR 12, phase as for VENC {VENC_NOISE} m/s")
    (stage/"MANIFEST.json").write_text(json.dumps(dict(created_utc=datetime.now(timezone.utc).isoformat(), status="PRESENTATION_DERIVATIVE_ONLY",
        script_sha256=sha(__file__), phantom_bin=b+1, aorta_bin=ba+1, outputs={p.name: sha(p) for p in stage.glob("*.png")}), indent=1)+"\n", encoding="utf-8")
    stage.rename(target)
    print("DONE")


if __name__ == "__main__":
    main(sys.argv[1])
