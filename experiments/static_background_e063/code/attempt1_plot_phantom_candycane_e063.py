"""Stenosis phantom, aorta-candy-cane-style figure (user request after E-064 figure v5): axial slice at the throat and a
long-axis slice through the vessel axis, magnitude + three velocity encodings, at peak systole.

Data and combination exactly as combine_e063.phantom(): E-060 REF/X/Y + E-061 Z (VENC 4) flow images + a x stationary
bottle images (exact superposition, bottle intensity calibrated to the scan), display noise SNR 84.4 (scan-matched,
same seed), Tukey k-space filter. The vessel is straight along S and centred at P = R = 0 (references_v2 centroids), so
the long-axis plane is the R = 0 partition-P plane, which also contains the bottles. Presentation only.
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

ROOT = Path("D:/MRSim/outputs/static_background_e063")
E060 = Path("D:/MRSim/outputs/scanner_geometry_e060")
E061 = Path("D:/MRSim/outputs/venc_unwrap_e061")
sys.path.insert(0, str(E061/"code")); sys.path.insert(1, str(E060/"code"))
FS, SEED = 1.5, 20261025
VOX, NPR = 1.25, 64
CROP = 2                      # in-plane edge voxels removed (bottle edge at the FOV boundary)


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


def noisy_phantom(rep):
    """combine_e063.phantom() up to the noisy images (same calibration, sigma and seed)."""
    import analyze_e061 as a61
    an = a61.an
    control = tomllib.loads((E060/"acquisition_v1/control_report.toml").read_text())
    flow, _ = an.load_images(E060/"acquisition_v1", control, E060/"sequences_v2/timing_e060.h5")
    z4, _ = a61.load_z4(); flow = dict(flow); flow["Z"] = z4
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
    if rel(st["X"], st["REF"]) > 1e-6:
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
    rep.update(static_scale=a, noise_sigma=sigma, snr=84.4, seed=SEED)
    return noisy, bottle, frac, zz


def main(target):
    target = Path(target)
    if target.exists():
        raise FileExistsError(target)
    stage = target.with_name(target.name+".building"); stage.mkdir()
    plt.rcParams["font.size"] = 10*FS
    rep = dict(experiment="E-063 presentation", figure="phantom candy-cane style", utc=datetime.now(timezone.utc).isoformat(), script_sha256=sha(__file__))
    noisy, bottle, frac, zz = noisy_phantom(rep)
    # Peak systole = peak upstream flow (E-060 scan-method Koma flows; upstream velocities are far below VENC 3).
    a60 = json.loads((E060/"analysis_v1/report.json").read_text())
    km = a60["scan_method"]["koma"]
    q_up = np.array(km["flow_ml_s"])[km["upstream_slices"]].mean(axis=0)
    b = int(np.argmax(q_up))
    if b+1 != a60["peak_bin"]:
        raise ValueError("peak-flow bin differs from the E-060 peak bin")
    rr_s = 1.012                                             # scan RR (E-060 RESULTS); bins are RR/30 wide, nominal midpoints
    t_ms = (b+.5)/30*rr_s*1e3
    vencs = dict(X=1.5, Y=1.5, Z=4.0)                        # R, P, S (S = along the vessel, E-061 VENC 4)
    v = {e: np.angle(noisy[e][b]*noisy["REF"][b].conj())*vencs[e]*100/np.pi for e in ("X", "Y", "Z")}   # (S, P, R) cm/s
    mag = np.abs(noisy["REF"]).mean(axis=0)
    # Display orientation: bottles at the bottom, vertical positive up (sP = +1 if bottles lie at low P).
    sP = 1 if np.nonzero(bottle.any(axis=(0, 2)))[0].mean() < NPR/2 else -1
    pr = (np.arange(NPR)-NPR//2)*VOX                          # centred FFT grid, mm
    vert = sP*pr; order = np.argsort(vert)
    cp = slice(CROP, NPR-CROP)
    def axial(img):                                           # (P, R) -> rows = vertical (low->high), cols = R
        return img[order][cp, cp]
    def longax(vol):                                          # (S, P, R) at R = 0 -> rows = vertical, cols = S
        return vol[:, order, NPR//2].T[cp]
    vv, rr = vert[order][cp], pr[cp]
    ext1 = [rr[0]-VOX/2, rr[-1]+VOX/2, vv[0]-VOX/2, vv[-1]+VOX/2]
    ext2 = [zz[0]-1.25, zz[-1]+1.25, vv[0]-VOX/2, vv[-1]+VOX/2]
    k0 = int(np.argmin(np.abs(zz)))                           # throat partition (S = 0)
    W = dict(Z=300, X=150, Y=150)                              # display windows = the scan's VENCs
    top = dict(Z=axial(v["Z"][k0]), X=axial(v["X"][k0]), Y=sP*axial(v["Y"][k0]))
    bot = dict(Z=longax(v["Z"]), X=longax(v["X"]), Y=sP*longax(v["Y"]))
    t_top = dict(Z="Through-plane (along vessel)", X="In-plane horizontal", Y="In-plane vertical")
    t_bot = dict(Z="In-plane horizontal (along vessel)", X="Through-plane", Y="In-plane vertical")
    fig, axes = plt.subplots(2, 4, figsize=(24, 13.5), layout="constrained")
    for row, (img_m, comps, titles, ext, xl, title) in enumerate((
            (axial(mag[k0]), top, t_top, ext1, "horizontal (mm)", f"Axial slice at the throat (S = {zz[k0]:+.0f} mm)"),
            (longax(mag), bot, t_bot, ext2, "along vessel (mm)", "Long-axis slice through the vessel axis"))):
        ax = axes[row]
        ax[0].imshow(img_m, cmap="gray", vmin=0, vmax=np.percentile(img_m, 99.7), extent=ext, origin="lower", interpolation="nearest")
        ax[0].set_title(f"{title}\nMagnitude", fontsize=9.5*FS)
        ims = []
        for j, e in enumerate(("Z", "X", "Y")):
            ims.append(ax[j+1].imshow(comps[e], cmap="gray", vmin=-W[e], vmax=W[e], extent=ext, origin="lower", interpolation="nearest"))
            ax[j+1].set_title(f"{titles[e]}\n±{W[e]} cm/s", fontsize=9.5*FS)
        for a_ in ax:
            a_.set_xlabel(xl, fontsize=8*FS); a_.set_ylabel("vertical (mm)", fontsize=8*FS); a_.tick_params(labelsize=7*FS)
    for j, e in enumerate(("Z", "X", "Y")):
        fig.colorbar(ims[j], ax=axes[:, j+1].tolist(), shrink=.6, location="bottom",
                     label="cm/s, " + ("along vessel (S)" if e == "Z" else "horizontal (R)" if e == "X" else "vertical (P)"))
    # Anatomy labels on the long-axis magnitude (geometry from the references_v2 lumen fraction).
    r_s = np.sqrt((frac > 0).sum(axis=(1, 2))*VOX**2/np.pi)    # equivalent lumen radius per partition, mm
    kq = int(np.argmin(np.abs(zz-30)))
    marks = {"Upstream": ((zz[3], 0.0), (-6, 22)), f"Stenosis throat (d = {2*r_s[k0]:.0f} mm)": ((zz[k0], r_s[k0]), (2, 18)),
             "Jet": ((14.0, 0.0), (6, -12)), "Water bottles (static)": ((0.0, -28.0), (0, -6)),
             "Downstream": ((zz[kq], r_s[kq]), (-4, 12))}
    for k, (xy, d) in marks.items():
        axes[1][0].annotate(k, xy, xytext=(xy[0]+d[0], xy[1]+d[1]), color="#ffd24d", fontsize=8.5*FS, fontweight="bold", ha="center",
                            arrowprops=None if k.startswith("Water") else dict(arrowstyle="-|>", color="#ffd24d", lw=1.4))
    axes[1][0].annotate("", (zz[-2], -18), xytext=(zz[1], -18), arrowprops=dict(arrowstyle="-|>", color="#ffd24d", lw=2))
    axes[1][0].text(zz[1], -15.5, "flow", color="#ffd24d", fontsize=8.5*FS, fontweight="bold")
    fig.suptitle(f"Stenosis phantom Koma, scanner geometry | peak systole: bin {b+1}/30, t = {t_ms:.0f} ms "
                 f"(upstream flow {q_up[b]:.0f} mL/s) | noise SNR 84", fontsize=12*FS)
    fig.supxlabel("Stationary water bottles (E-063; intensity calibrated to the scan) added by exact superposition. Koma VENC 1.5 / 1.5 / 4 m/s (R, P, along vessel); "
                  "display windows = scan VENCs.\nNoise SNR 84 (scan-matched). Tukey k-space filter. Air has no signal, so its phase is random, as in the scan.", fontsize=8*FS)
    p = stage/"phantom_candycane_peak_systole.png"; fig.savefig(p, dpi=110); plt.close(fig)
    rep.update(peak_bin=b+1, t_ms=t_ms, rr_s=rr_s, upstream_flow_ml_s=float(q_up[b]), throat_partition=k0, long_axis_R_index=NPR//2,
               vertical_sign_P=sP, crop_voxels=CROP, throat_diameter_mm=float(2*r_s[k0]), status="PRESENTATION_DERIVATIVE_ONLY")
    (stage/"report.json").write_text(json.dumps(rep, indent=1, default=float)+chr(10), encoding="utf-8")
    (stage/"MANIFEST.json").write_text(json.dumps({p.name: sha(p) for p in sorted(stage.iterdir())}, indent=1)+chr(10), encoding="utf-8")
    stage.rename(target)
    print("DONE", b+1, round(t_ms), round(float(q_up[b]), 1), "sP", sP)


if __name__ == "__main__":
    main(sys.argv[1])
