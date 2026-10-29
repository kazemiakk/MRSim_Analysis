"""E-063 presentation v2: aorta figure with the standard candy-cane plane (generated from combine_e063.py;
same data, noise realisation and display; only the oblique plane changes)."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tomllib

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402

ROOT = Path("D:/MRSim/outputs/static_background_e063")
E060 = Path("D:/MRSim/outputs/scanner_geometry_e060")
E061 = Path("D:/MRSim/outputs/venc_unwrap_e061")
E062 = Path("D:/MRSim/outputs/aorta_20bin_e062")
sys.path.insert(0, str(E061/"code")); sys.path.insert(1, str(E060/"code"))
sys.path.insert(2, "D:/MRSim/outputs/presentation_hemodynamics_2026-10-25/code")
sys.path.insert(3, "D:/MRSim/outputs/aorta_presentation_2026-10-25/code")
FS, SEED = 1.5, 20261025


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def rel(a, b):
    return float(np.sqrt(np.sum(np.abs(a-b)**2)/np.sum(np.abs(b)**2)))


def tukey3(img, alpha=.5):
    w = np.ones(img.shape[-3:])
    for ax, n in enumerate(img.shape[-3:]):
        u = np.abs(np.fft.fftfreq(n))/.5
        wa = np.where(u <= 1-alpha, 1., .5*(1+np.cos(np.pi*(u-(1-alpha))/alpha)))
        sh = [1, 1, 1]; sh[ax] = n; w = w*wa.reshape(sh)
    return np.fft.ifftn(np.fft.fftn(img, axes=(-3, -2, -1))*w, axes=(-3, -2, -1))


def jld_image(path):
    with h5py.File(path) as f:
        d = f["image"][()]
    return (d["re"]+1j*d["im"]).astype(np.complex64)


def panel_row(axes, mag, vel, venc, labels, extent, ylab, title, origin="upper"):
    axes[0].imshow(mag, cmap="gray", vmin=0, vmax=np.percentile(mag, 99.7), extent=extent, interpolation="nearest", origin=origin)
    axes[0].set_title(f"{title}\nMagnitude", fontsize=9.5*FS); axes[0].set_ylabel(ylab, fontsize=8*FS)
    ims = []
    for k in range(3):
        ims.append(axes[k+1].imshow(vel[k], cmap="gray", vmin=-venc[k], vmax=venc[k], extent=extent, interpolation="nearest", origin=origin))
        axes[k+1].set_title(f"{labels[k]}\n±{venc[k]:.0f} cm/s", fontsize=9.5*FS)
    for ax in axes:
        ax.tick_params(labelsize=7*FS); ax.set_facecolor("black")
    return ims


def aorta(stage, rep):
    from aorta_common import Aorta, VENC_NOISE
    A = Aorta()
    fdir = E062/"combined/simulation_motion_1m_20bin_4mm"; sdir = ROOT/"aorta_combined/simulation_static_tissue_20bin_4mm"
    s1 = tomllib.loads((sdir/"simulation_report.toml").read_text())
    if s1["status"] != "PASS":
        raise ValueError("aorta static run not passed")
    F = {e: jld_image(fdir/f"{e.lower()}_result.jld2") for e in ("REF", "X", "Y", "Z")}
    S = {e: jld_image(sdir/f"{e.lower()}_result.jld2") for e in ("REF", "X", "Y", "Z")}
    with h5py.File(fdir/"pc_reconstruction.h5") as f:
        vpc = f["velocity"][:]
    venc = 2.5
    for c, e in enumerate(("X", "Y", "Z")):
        vv = np.angle(F[e]*F["REF"].conj())*venc/np.pi
        if np.max(np.abs(vv-vpc[c])[np.abs(F["REF"]) > .05*np.abs(F["REF"]).max()]) > 1e-3:
            raise ValueError("flow image read does not reproduce pc_reconstruction "+e)
    rep["aorta_static_encoding_vs_REF_relative_L2"] = {e: rel(S[e], S["REF"]) for e in ("X", "Y", "Z")}
    mref = np.abs(S["REF"]); big = mref > .2*mref.max()
    rep["aorta_static_phase_p99_rad"] = {e: float(np.percentile(np.abs(np.angle(S[e]*S["REF"].conj())[big]), 99)) for e in ("X", "Y", "Z")}
    # Amendment A1: f32 simulator; attempt 1 used relative L2 <= 1e-4 and stopped at ~2e-3 (rounding only).
    if max(rep["aorta_static_encoding_vs_REF_relative_L2"].values()) > 1e-2 or max(rep["aorta_static_phase_p99_rad"].values()) > 1e-2:
        raise ValueError("aorta static encodings differ from REF")
    geo = json.loads((ROOT/"geometry_v1/report.json").read_text())["aorta"]; a = geo["scale_factor"]
    tot = {"REF": F["REF"]+a*S["REF"]}
    for e in ("X", "Y", "Z"):   # flow phase re-encoded for display at VENC 0.3 (Koma ran at 2.5); stationary phase unchanged
        phi = np.angle(F[e]*F["REF"].conj())
        tot[e] = F[e]*np.exp(1j*(phi*venc/VENC_NOISE-phi))+a*S[e]
    sigma = float(np.median(np.abs(tot["REF"]).mean(axis=0)[A.core]))/12.0
    rng = np.random.default_rng(SEED)
    noisy = {e: tukey3(v+sigma*(rng.standard_normal(v.shape)+1j*rng.standard_normal(v.shape))) for e, v in tot.items()}
    with h5py.File(E062/"analysis_v1/matched_reference_20bin.h5") as f:
        ref = f["adc_time_average"][:]
    ba = int(np.argmax([np.linalg.norm(ref[k][A.core], axis=-1).mean() for k in range(A.nb)]))
    vel = np.stack([np.angle(noisy[e][ba]*noisy["REF"][ba].conj())*VENC_NOISE*100/np.pi for e in ("X", "Y", "Z")], -1)   # (Z, Y, X, 3) cm/s
    mag = np.abs(noisy["REF"]).mean(axis=0)
    rep["aorta"] = dict(static_scale=a, noise_sigma=sigma, peak_bin=ba+1,
                        tissue_to_lumen_magnitude=float(np.median(np.abs(a*S["REF"]).mean(axis=0)[~(A.frac > 0)])/np.median(np.abs(F["REF"]).mean(axis=0)[A.core])))
    V = VENC_NOISE*100
    fig, axes = plt.subplots(2, 4, figsize=(26, 15), layout="constrained", gridspec_kw=dict(height_ratios=[1, 1.9]))
    ix = int(np.argmin(abs(A.gx-.058)))
    ext = [A.gy[0]*1e3-2, A.gy[-1]*1e3+2, A.gz[0]*1e3-2, A.gz[-1]*1e3+2]
    sl = lambda img: img[:, :, ix]  # noqa: E731   (Z, Y)
    ims = panel_row(axes[0], sl(mag), [sl(vel[..., 0]), sl(vel[..., 1]), sl(vel[..., 2])], (V, V, V),
                    ("Through-plane (X)", "In-plane horizontal (Y)", "In-plane vertical (Z)"), ext, "Z (mm)",
                    f"Axial slice X = {A.gx[ix]*1e3:.0f} mm (ascending + descending)", origin="lower")
    for ax in axes[0]:
        ax.set_xlabel("Y (mm)", fontsize=8*FS)
    # Oblique candy-cane slab (12 mm) in the plane fitted to the vessel, as aorta_candycane_v4 (velocity rotated).
    zi, yi, xi = np.nonzero(A.core)
    P = np.stack([A.gx[xi], A.gy[yi], A.gz[zi]], -1)
    # v2 (user: v1 candy-cane tilted): standard view. Vertical = descending-aorta axis (principal direction of the
    # vessel below X = 0, oriented toward the arch); the plane contains that axis and the ascending-aorta centroid,
    # ascending on the left. v1 used the whole-vessel PCA plane, which left the descending aorta diagonal.
    lower = P[P[:, 0] < 0]
    dc = lower.mean(0); dv = np.linalg.svd(lower-dc, full_matrices=False)[2][0]
    e2 = dv if dv[0] > 0 else -dv
    asc = np.array([.058, -.033, -.019])
    w_ = (asc-dc)-((asc-dc)@e2)*e2
    e1 = -w_/np.linalg.norm(w_)                                                # ascending on the left (negative horizontal)
    n = np.cross(e1, e2)
    c0 = dc+((P-dc)@e2).mean()*e2
    q = (P-c0)@np.stack([e1, e2, n]).T
    lo, hi = q.min(0)-.015, q.max(0)+.015
    s1v = np.arange(lo[0], hi[0], 1.5e-3); s2v = np.arange(lo[1], hi[1], 1.5e-3); s3v = np.arange(-.010, .0101, 2e-3)
    S1, S2, S3 = np.meshgrid(s1v, s2v, s3v, indexing="ij")
    pts = c0+S1.ravel()[:, None]*e1+S2.ravel()[:, None]*e2+S3.ravel()[:, None]*n
    kk = [(pts[:, i]-g[0])/(g[1]-g[0]) for i, g in enumerate((A.gx, A.gy, A.gz))]
    def smp(vol):
        f = lambda u: ndimage.map_coordinates(u, [kk[2], kk[1], kk[0]], order=1, mode="constant", cval=0.0)  # noqa: E731
        return (f(vol.real)+1j*f(vol.imag)).reshape(S1.shape) if np.iscomplexobj(vol) else f(vol).reshape(S1.shape)
    R = noisy["REF"][ba]
    vm = np.stack([np.angle(smp(noisy[e][ba]*R.conj()).sum(axis=2))*VENC_NOISE*100/np.pi for e in ("X", "Y", "Z")], -1)
    inside = smp(np.ones(R.shape)).mean(axis=2) > .3
    vv = [np.where(inside, vm@u, np.nan).T for u in (n, e1, e2)]
    mg = smp(mag).mean(axis=2).T
    ext2 = [s1v[0]*1e3, s1v[-1]*1e3, s2v[0]*1e3, s2v[-1]*1e3]
    ims2 = panel_row(axes[1], mg, vv, (V, V, V), ("Through-plane", "In-plane horizontal", "In-plane vertical"), ext2, "vertical (mm)",
                     "Candy-cane slab, 20 mm (descending axis vertical)", origin="lower")
    for ax in axes[1]:
        ax.set_xlabel("horizontal (mm)", fontsize=8*FS)
    fig.colorbar(ims2[0], ax=axes[:, 1:].ravel().tolist(), shrink=.5, label=f"cm/s (display VENC {V:.0f} cm/s)")
    fig.suptitle(f"Aorta (E-062 + E-063 stationary tissue) Koma | peak bin {ba+1}/20 | noise SNR 12", fontsize=12*FS)
    fig.supxlabel("Stationary tissue block (synthetic; T1 0.9 s, T2 45 ms, PD 0.8 x blood) added by exact superposition. Flow phase shown as for VENC 0.3 m/s "
                  "(Koma ran at 2.5 m/s; display re-encoding). Tukey k-space filter.", fontsize=8*FS)
    p = stage/"aorta_with_tissue_candycane_v2.png"; fig.savefig(p, dpi=110); plt.close(fig)
    rep["candycane_axes"] = dict(vertical=e2.tolist(), horizontal=e1.tolist(), normal=n.tolist())
    return {"aorta": p.name}


def main(target):
    target = Path(target)
    if target.exists():
        raise FileExistsError(target)
    stage = target.with_name(target.name+".building"); stage.mkdir()
    plt.rcParams["font.size"] = 10*FS
    rep = dict(experiment="E-063", stage="presentation_candycane_v2", utc=datetime.now(timezone.utc).isoformat(), script_sha256=sha(__file__))
    rep["figures"] = aorta(stage, rep); rep["status"] = "PRESENTATION_DERIVATIVE_ONLY"
    (stage/"report.json").write_text(json.dumps(rep, indent=1, default=float)+chr(10), encoding="utf-8")
    (stage/"MANIFEST.json").write_text(json.dumps({p.name: sha(p) for p in sorted(stage.iterdir())}, indent=1)+chr(10), encoding="utf-8")
    stage.rename(target)
    print("DONE", rep["candycane_axes"])


if __name__ == "__main__":
    main(sys.argv[1])
