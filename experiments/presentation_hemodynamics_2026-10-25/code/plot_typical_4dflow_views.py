"""Scanner-style 4D-flow display (user example): magnitude + through-plane + two in-plane phase images, 2x2.

v2 (user: phantom phase full of artefacts; aorta should show inlet, branches, ascending and descending aorta):
- Reconstruction as on a scanner: Tukey (alpha 0.5) k-space filter on every complex image (3-D), then 2x zero-filled
  interpolation of the displayed plane. v1 showed raw truncation ringing (sidelobes ~1 % of the vessel signal, the same
  size as the noise), which printed coherent stripes into the background.
- Phantom: E-061 VENC 4 m/s along the vessel (v1: VENC 3, aliased jet), slice along the vessel through its centre,
  flow left to right. Scan-matched noise (SNR 84). Peak bin.
- Aorta: thick-slab phase projection through Z (phase of the magnitude-weighted complex sum, as PC-MRA-style viewers do),
  so inlet, ascending aorta, arch, branches and descending aorta are all visible; v1 was one Y-Z slice. Candy-cane
  orientation (horizontal Y, vertical X, arch on top). v22 display noise (SNR 12, phase as for VENC 0.3 m/s; Koma ran at
  2.5 m/s). Peak bin.
Phase images: velocity, -VENC (black) .. +VENC (white), unmasked. Magnitude: mean over bins. Presentation only.
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
FS, ALPHA = 1.5, 0.5


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def tukey_k(shape):
    """Separable Tukey window on unshifted FFT frequencies (real, symmetric: no image shift)."""
    w = np.ones(shape)
    for ax, n in enumerate(shape):
        u = np.abs(np.fft.fftfreq(n))/0.5
        wa = np.where(u <= 1-ALPHA, 1.0, 0.5*(1+np.cos(np.pi*(u-(1-ALPHA))/ALPHA)))
        sh = [1]*len(shape); sh[ax] = n
        w = w*wa.reshape(sh)
    return w


def filt(img):
    """Apply the k-space filter over the last three axes."""
    w = tukey_k(img.shape[-3:])
    return np.fft.ifftn(np.fft.fftn(img, axes=(-3, -2, -1))*w, axes=(-3, -2, -1))


def zip2(img):
    """2x zero-filled interpolation of a 2-D complex image (sample 2i = original i)."""
    n0, n1 = img.shape
    k = np.fft.fftshift(np.fft.fft2(img))
    kp = np.zeros((2*n0, 2*n1), complex); kp[n0//2:n0//2+n0, n1//2:n1//2+n1] = k
    return np.fft.ifft2(np.fft.ifftshift(kp))*4


def figure(path, mag, vels, vencs_cm, labels, extent, xl, yl, title, note):
    asp = (extent[3]-extent[2])/(extent[1]-extent[0])
    fig, axes = plt.subplots(2, 2, figsize=(17, 17*asp+3), layout="constrained")
    ax = axes[0, 0]
    ax.imshow(mag, origin="lower", extent=extent, cmap="gray", vmin=0, vmax=np.percentile(mag, 99.7), aspect="equal", interpolation="nearest")
    ax.set_title("Magnitude", fontsize=10*FS)
    for k, ax in enumerate((axes[0, 1], axes[1, 0], axes[1, 1])):
        im = ax.imshow(vels[k], origin="lower", extent=extent, cmap="gray", vmin=-vencs_cm[k], vmax=vencs_cm[k], aspect="equal", interpolation="nearest")
        ax.set_title(f"{labels[k]}  |  VENC {vencs_cm[k]:.0f} cm/s", fontsize=10*FS)
        cb = fig.colorbar(im, ax=ax, shrink=.8, ticks=[-vencs_cm[k], 0, vencs_cm[k]]); cb.set_label("cm/s", fontsize=8*FS); cb.ax.tick_params(labelsize=8*FS)
    for ax in axes.ravel():
        ax.set_xlabel(xl, fontsize=8*FS); ax.set_ylabel(yl, fontsize=8*FS); ax.tick_params(labelsize=8*FS); ax.set_facecolor("black")
    fig.suptitle(title, fontsize=12*FS); fig.supxlabel(note, fontsize=8*FS)
    fig.savefig(path, dpi=110); plt.close(fig)


def main(target):
    target = Path(target)
    if target.exists():
        raise FileExistsError(target)
    plt.rcParams["font.size"] = 10*FS
    stage = target.with_name(target.name+".building"); stage.mkdir()
    import plot_projection_triptych as tri
    import plot_contours_with_noise as nz
    images, vencs = tri.arms()["E061_VENC4"]
    b = json.loads((E060/"analysis_v1/report.json").read_text())["peak_bin"]-1
    with h5py.File(E060/"references_v2/references.h5") as f:
        zz, frac, pfr = f["partition_z_m"][:]*1e3, f["fraction_fft"][:], f["plane_fraction"][:]
    snr = nz.scan_noise_and_panels()[3]
    sig = float(np.median(np.abs(images["REF"]).mean(axis=0)[(frac > 0) & (np.abs(zz) <= 30)[:, None, None]]))/snr
    rng = np.random.default_rng(20261025)
    nzy = {e: filt(images[e]+sig*(rng.standard_normal(images[e].shape)+1j*rng.standard_normal(images[e].shape))) for e in ("REF", "X", "Y", "Z")}
    ks = int(np.argmin(abs(zz))); rc = int(np.argmax(pfr[ks].sum(axis=0))); pr = (np.arange(64)-32)*1.25
    sl = lambda a: zip2(a[:, :, rc]).T  # noqa: E731   (S, P) at the vessel centre -> (P, S): flow left to right
    ref_b = sl(nzy["REF"][b])
    v = {e: np.angle(sl(nzy[e][b])*ref_b.conj())*vencs[i]*100/np.pi for i, e in enumerate(("X", "Y", "Z"))}
    mag = np.mean([np.abs(sl(nzy["REF"][k])) for k in range(nzy["REF"].shape[0])], axis=0)
    figure(stage/"phantom_4dflow_style.png", mag, [v["X"], v["Z"], v["Y"]], [vencs[0]*100, vencs[2]*100, vencs[1]*100],
           ["PC through-plane (R)", "PC in-plane horizontal (S, along vessel)", "PC in-plane vertical (P)"],
           [zz[0]-1.25, zz[-1]+1.25, pr[0]-.625, pr[-1]+.625], "S, along vessel (mm; flow left to right)", "P (mm)",
           f"Phantom (E-060/E-061) Koma | slice along the vessel | peak bin {b+1}/30 | noise SNR {snr:.0f}",
           "Scanner-style reconstruction: Tukey k-space filter + 2× zero-filled interpolation. VENC 400 cm/s along the vessel (no aliasing), 150 cm/s in-plane.")
    from aorta_common import Aorta, VENC_NOISE
    A = Aorta()
    with h5py.File(Path("D:/MRSim/outputs/aorta_20bin_e062/analysis_v1/matched_reference_20bin.h5")) as f:
        ref = f["adc_time_average"][:]
    ba = int(np.argmax([np.linalg.norm(ref[k][A.core], axis=-1).mean() for k in range(A.nb)]))
    R = filt(A.REFall[ba])                                                          # (Z, Y, X)
    E = [filt(A.REFall[ba]*np.exp(1j*A.vnall[ba][..., c]*np.pi/VENC_NOISE)) for c in range(3)]
    proj = lambda c: zip2((c*R.conj()).sum(axis=0)).T  # noqa: E731   (Y, X) -> (X, Y): candy-cane orientation
    va = [np.angle(proj(E[c]))*VENC_NOISE*100/np.pi for c in range(3)]
    mag = np.abs(zip2(np.abs(filt(A.REFall)).mean(axis=0).mean(axis=0).astype(complex))).T
    nz_, ny, nx = A.core.shape
    ax_x = (np.arange(nx)-nx//2)*280/nx; ax_y = (np.arange(ny)-ny//2)*4.0
    figure(stage/"aorta_4dflow_style.png", mag, [va[2], va[1], va[0]], [VENC_NOISE*100]*3,
           ["PC through-plane (Z)", "PC in-plane horizontal (Y)", "PC in-plane vertical (X)"],
           [ax_y[0]-2, ax_y[-1]+2, ax_x[0]-1.9, ax_x[-1]+1.9], "Y (mm)", "X (mm)",
           f"Aorta (E-062) Koma | thick-slab projection through Z | peak bin {ba+1}/20 | noise SNR 12, as for VENC {VENC_NOISE} m/s",
           "Whole aorta: phase of the magnitude-weighted complex sum through Z; Tukey k-space filter + 2× zero-filled interpolation.")
    (stage/"MANIFEST.json").write_text(json.dumps(dict(created_utc=datetime.now(timezone.utc).isoformat(), status="PRESENTATION_DERIVATIVE_ONLY",
        script_sha256=sha(__file__), phantom_bin=b+1, aorta_bin=ba+1, tukey_alpha=ALPHA, outputs={p.name: sha(p) for p in stage.glob("*.png")}), indent=1)+"\n", encoding="utf-8")
    stage.rename(target)
    print("DONE")


if __name__ == "__main__":
    main(sys.argv[1])
