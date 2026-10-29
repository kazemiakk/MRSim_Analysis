"""E-063 S2: add the stationary-spin images to the flow images (exact superposition) and render scanner-style figures.

Phantom: E-060 REF/X/Y + E-061 Z (VENC 4) flow images + a x stationary image (REF static for Y/Z per G1, full X static
for X), a calibrated so that median bottle |M| / median time-mean lumen |M| equals the real scan's ratio. Display noise
SNR 84 (scan-matched, one sigma), Tukey k-space filter. Figure: Koma vs real scan at the throat and upstream slices,
each at its own systolic peak; magnitude, through-plane (along the vessel) and the two in-plane phase images, all
velocity windows +-150 cm/s in-plane, +-300 cm/s through-plane (the scan's VENCs).
Aorta: E-062 flow images + 22.75 x tissue images (PD 0.8 x blood per volume), flow phase shown as for VENC 0.3 m/s
(display re-encoding, labelled), display noise SNR 12, Tukey filter. Figure: axial slice through the ascending and
descending aorta, and a 12 mm oblique candy-cane slab. Presentation; no registered metric changes.
"""
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


def phantom(stage, rep):
    import analyze_e061 as a61
    an = a61.an
    control = tomllib.loads((E060/"acquisition_v1/control_report.toml").read_text())
    flow, _ = an.load_images(E060/"acquisition_v1", control, E060/"sequences_v2/timing_e060.h5")
    z4, _ = a61.load_z4(); flow = dict(flow); flow["Z"] = z4                     # VENC 4 along the vessel
    g1 = tomllib.loads((ROOT/"identity_v1/report.toml").read_text())
    if g1["status"] != "PASS_E063_G1_PREFIX_IDENTITY":
        raise ValueError("G1 not passed")
    st = {}
    for e, d in (("REF", "phantom_ref_v1"), ("X", "phantom_x_v1")):
        c = tomllib.loads((ROOT/d/"control_report.toml").read_text())
        if c["status"] != "COMPLETE_E063_STATIC_SUBSET" or sha(ROOT/d/f"{e.lower()}_static_result.h5") != c["result_sha256"][e]:
            raise ValueError("static result missing/changed "+e)
        with h5py.File(ROOT/d/f"{e.lower()}_static_result.h5") as f:
            st[e] = f["image"][:]
    rep["phantom_full_X_vs_REF_static_relative_L2"] = rel(st["X"], st["REF"])
    if rep["phantom_full_X_vs_REF_static_relative_L2"] > 1e-6:
        raise ValueError("full static X differs from REF")
    geo = json.loads((ROOT/"geometry_v1/report.json").read_text())["phantom"]
    with h5py.File(E060/"references_v2/references.h5") as f:
        frac, zz = f["fraction_fft"][:], f["partition_z_m"][:]*1e3
    lum = (frac > 0) & (np.abs(zz) <= 30)[:, None, None]
    sref = np.abs(st["REF"]).mean(axis=0)
    bottle = sref >= .5*np.percentile(sref, 99.5)
    a = geo["scan_bottle_to_vessel_magnitude_ratio"]*float(np.median(np.abs(flow["REF"]).mean(axis=0)[lum]))/float(np.median(sref[bottle]))
    tot = {e: flow[e]+a*(st["X"] if e == "X" else st["REF"]) for e in ("REF", "X", "Y", "Z")}
    sigma = float(np.median(np.abs(tot["REF"]).mean(axis=0)[lum]))/84.4
    rng = np.random.default_rng(SEED)
    noisy = {e: tukey3(v+sigma*(rng.standard_normal(v.shape)+1j*rng.standard_normal(v.shape))) for e, v in tot.items()}
    rep["phantom"] = dict(static_scale=a, noise_sigma=sigma, bottle_voxels=int(bottle.sum()))
    # Orientation: show the bottles at the bottom, as in the scan (rows = P).
    bot_p = np.nonzero(bottle.any(axis=(0, 2)))[0].mean()
    flipP = bot_p < 32
    b = json.loads((E060/"analysis_v1/report.json").read_text())["peak_bin"]-1
    vencs = (1.5, 1.5, 4.0)
    v = {e: np.angle(noisy[e][b]*noisy["REF"][b].conj())*vencs[i]*100/np.pi for i, e in enumerate(("X", "Y", "Z"))}
    mag = np.abs(noisy["REF"]).mean(axis=0)
    # Real scan.
    sc = np.load("D:/MRSim/private_inputs/phantom_scan_2025-08-28/measured_v1/measured_30phase.npz")
    srep = json.loads((Path("D:/MRSim/outputs/measured_scan_e059")/"measured_v2/report.json").read_text())
    sv, smag, pos = sc["velocity_cm_s"], sc["magnitude"], sc["slice_positions_mm"]
    t_s = srep["throat_slice"]; sgn = srep["segmentation"]["0.25"]["vendor_to_forward_sign"]
    ps = int(np.argmax(srep["segmentation"]["0.25"]["slices"][t_s]["flow_ml_s"]))
    zrel = pos-pos[t_s]
    ext = [-40, 40, 40, -40]
    out = {}
    for name, zmm in (("throat", 0.0), ("upstream", -20.0)):
        k = int(np.argmin(abs(zz-zmm))); s = int(np.argmin(abs(zrel-zmm)))
        fl = (lambda img: img[::-1]) if flipP else (lambda img: img)  # noqa: E731
        fig, axes = plt.subplots(2, 4, figsize=(24, 13.5), layout="constrained")
        koma = [fl(v["Z"][k]), fl(v["X"][k]), fl(v["Y"][k])]
        ims = panel_row(axes[0], fl(mag[k]), koma, (300, 150, 150), ("Through-plane (along vessel)", "In-plane horizontal (R)", "In-plane vertical (P)"),
                        ext, "Koma (P, mm)", f"Koma, bin {b+1}/30 (peak)")
        scan = [sgn*sv[s, ps, :, :, 2], sv[s, ps, :, :, 0], sv[s, ps, :, :, 1]]
        panel_row(axes[1], smag[s].mean(axis=0), scan, (300, 150, 150), ("Through-plane (along vessel)", "In-plane horizontal", "In-plane vertical"),
                  ext, "4D-flow MRI (mm)", f"4D-flow MRI scan, phase {ps+1}/30 (peak)")
        for j in (1, 2, 3):
            fig.colorbar(ims[j-1], ax=axes[:, j].tolist(), shrink=.6, location="bottom", label="cm/s")
        fig.suptitle(f"Phantom, slice at {name} (S = {zz[k]:+.1f} mm Koma, {zrel[s]:+.1f} mm scan): Koma with stationary bottles vs real scan",
                     fontsize=12*FS)
        fig.supxlabel("Koma: E-060 flow + stationary bottles (E-063, exact superposition; bottle intensity calibrated to the scan), VENC 4 m/s along the vessel, "
                      "noise SNR 84, Tukey filter.\nScan: own VENC (300 / 150 / 150 cm/s), CFD carries ~60 % more flow than the scan. "
                      "In-plane orientation of the scan not registered to the CFD.", fontsize=8*FS)
        p = stage/f"phantom_vs_scan_{name}.png"; fig.savefig(p, dpi=110); plt.close(fig); out[name] = p.name
    return out


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
    if max(rep["aorta_static_encoding_vs_REF_relative_L2"].values()) > 1e-4:
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
    P = np.stack([A.gx[xi], A.gy[yi], A.gz[zi]], -1); w = A.mag.mean(axis=0)[A.core]
    c0 = (P*w[:, None]).sum(0)/w.sum(); ev, vec = np.linalg.eigh(np.cov((P-c0).T, aweights=w))
    n, e2 = vec[:, 0], (vec[:, 2] if vec[0, 2] > 0 else -vec[:, 2]); e1 = np.cross(e2, n)
    if (np.array([.058, -.033, -.019])-np.array([.0, .027, .025]))@e1 > 0:
        e1, n = -e1, -n
    q = (P-c0)@np.stack([e1, e2, n]).T
    lo, hi = q.min(0)-.015, q.max(0)+.015
    s1v = np.arange(lo[0], hi[0], 1.5e-3); s2v = np.arange(lo[1], hi[1], 1.5e-3); s3v = np.arange(-.006, .0061, 2e-3)
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
                     "Oblique candy-cane slab, 12 mm", origin="lower")
    for ax in axes[1]:
        ax.set_xlabel("horizontal (mm)", fontsize=8*FS)
    fig.colorbar(ims2[0], ax=axes[:, 1:].ravel().tolist(), shrink=.5, label=f"cm/s (display VENC {V:.0f} cm/s)")
    fig.suptitle(f"Aorta (E-062 + E-063 stationary tissue) Koma | peak bin {ba+1}/20 | noise SNR 12", fontsize=12*FS)
    fig.supxlabel("Stationary tissue block (synthetic; T1 0.9 s, T2 45 ms, PD 0.8 x blood) added by exact superposition. Flow phase shown as for VENC 0.3 m/s "
                  "(Koma ran at 2.5 m/s; display re-encoding). Tukey k-space filter.", fontsize=8*FS)
    p = stage/"aorta_with_tissue.png"; fig.savefig(p, dpi=110); plt.close(fig)
    return {"aorta": p.name}


def main(target):
    target = Path(target)
    if target.exists():
        raise FileExistsError(target)
    stage = target.with_name(target.name+".building"); stage.mkdir()
    plt.rcParams["font.size"] = 10*FS
    rep = dict(experiment="E-063", stage="S2_combine", utc=datetime.now(timezone.utc).isoformat(), script_sha256=sha(__file__))
    figs = {}
    figs.update(phantom(stage, rep)); print("phantom done", rep["phantom"], flush=True)
    figs.update(aorta(stage, rep)); print("aorta done", rep["aorta"], flush=True)
    rep["figures"] = figs; rep["status"] = "COMPLETE_E063_S2"
    (stage/"report.json").write_text(json.dumps(rep, indent=1, default=float)+"\n", encoding="utf-8")
    (stage/"MANIFEST.json").write_text(json.dumps({p.name: sha(p) for p in sorted(stage.iterdir())}, indent=1)+"\n", encoding="utf-8")
    stage.rename(target)
    print("DONE")


if __name__ == "__main__":
    main(sys.argv[1])
