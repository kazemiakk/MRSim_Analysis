"""Stenosis phantom, aorta-candy-cane-style figure WITHOUT the scan's water bottles (user: the bottles are in the 4D-flow
scan, not in the CFD; show only the circular phantom and background noise).

Koma flow images only: E-060 REF/X/Y + E-061 Z (VENC 4 along the vessel). The E-063 stationary bottle images are not
added. Display noise matched to the 4D-flow scan: sigma = median lumen |REF| at the peak bin / scan lumen SNR, where the
scan SNR (84.4) is the scan's median vessel magnitude at its peak phase over the air-noise sigma (E-060
contours_noise_v3). Cross-check from the scan's static-bottle phase noise is reported. Same seed and Tukey filter as
before. Axial slice at the throat + long-axis slice through the vessel axis, at peak systole. Presentation only.
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
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path("D:/MRSim/outputs/static_background_e063")
E060 = Path("D:/MRSim/outputs/scanner_geometry_e060")
E061 = Path("D:/MRSim/outputs/venc_unwrap_e061")
sys.path.insert(0, str(E061/"code")); sys.path.insert(1, str(E060/"code"))
FS, SEED = 1.5, 20261025
VOX, NPR = 1.25, 64
SP = -1                      # display orientation as v1/v2 and the scan display (vertical = -P)
SCAN_VENC = (1.5, 1.5, 3.0)  # R, P, S


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def tukey3(img, alpha=.5):
    w = np.ones(img.shape[-3:])
    for ax, n in enumerate(img.shape[-3:]):
        u = np.abs(np.fft.fftfreq(n))/.5
        wa = np.where(u <= 1-alpha, 1., .5*(1+np.cos(np.pi*(u-(1-alpha))/alpha)))
        sh = [1, 1, 1]; sh[ax] = n; w = w*wa.reshape(sh)
    return np.fft.ifftn(np.fft.fftn(img, axes=(-3, -2, -1))*w, axes=(-3, -2, -1))


def scan_noise(rep):
    """Scan lumen SNR (air sigma) and the independent cross-check from the static-bottle phase noise."""
    m = json.loads((E060/"contours_noise_v3/MANIFEST.json").read_text())
    def find(d, key):
        if isinstance(d, dict):
            if key in d:
                return d[key]
            for v in d.values():
                r = find(v, key)
                if r is not None:
                    return r
        return None
    snr = float(find(m, "scan_lumen_snr")); sv = np.array(find(m, "scan_sigma_v_static_m_s"))
    ratio = json.loads((ROOT/"geometry_v1/report.json").read_text())["phantom"]["scan_bottle_to_vessel_magnitude_ratio"]
    snr_bottle = np.sqrt(2)*np.array(SCAN_VENC)/(np.pi*sv)             # phase-difference noise sqrt(2)/SNR
    rep.update(scan_lumen_snr_air=snr, scan_static_velocity_sd_m_s=sv.tolist(), scan_bottle_snr_from_phase=snr_bottle.tolist(),
               scan_lumen_snr_from_bottle_phase=float(snr_bottle.mean()/ratio))
    return snr


def main(target):
    target = Path(target)
    if target.exists():
        raise FileExistsError(target)
    stage = target.with_name(target.name+".building"); stage.mkdir()
    plt.rcParams["font.size"] = 10*FS
    rep = dict(experiment="E-060/E-061 presentation", figure="phantom candy-cane style, flow only (no bottles)",
               utc=datetime.now(timezone.utc).isoformat(), script_sha256=sha(__file__))
    import analyze_e061 as a61
    an = a61.an
    control = tomllib.loads((E060/"acquisition_v1/control_report.toml").read_text())
    flow, _ = an.load_images(E060/"acquisition_v1", control, E060/"sequences_v2/timing_e060.h5")
    z4, _ = a61.load_z4(); flow = dict(flow); flow["Z"] = z4                     # VENC 4 along the vessel
    with h5py.File(E060/"references_v2/references.h5") as f:
        frac, zz = f["fraction_fft"][:], f["partition_z_m"][:]*1e3
    a60 = json.loads((E060/"analysis_v1/report.json").read_text())
    km = a60["scan_method"]["koma"]
    q_up = np.array(km["flow_ml_s"])[km["upstream_slices"]].mean(axis=0)
    b = int(np.argmax(q_up))
    if b+1 != a60["peak_bin"]:
        raise ValueError("peak-flow bin differs from the E-060 peak bin")
    snr = scan_noise(rep)
    lum = (frac > 0) & (np.abs(zz) <= 30)[:, None, None]
    sigma = float(np.median(np.abs(flow["REF"][b])[lum]))/snr                 # same definition as the scan SNR (peak phase)
    rng = np.random.default_rng(SEED)
    noisy = {e: tukey3(v+sigma*(rng.standard_normal(v.shape)+1j*rng.standard_normal(v.shape))) for e, v in flow.items()}
    rr_s = 1.012
    t_ms = (b+.5)/30*rr_s*1e3
    vencs = dict(X=1.5, Y=1.5, Z=4.0)
    v = {e: np.angle(noisy[e][b]*noisy["REF"][b].conj())*vencs[e]*100/np.pi for e in ("X", "Y", "Z")}   # (S, P, R) cm/s
    mag = np.abs(noisy["REF"]).mean(axis=0)
    pr = (np.arange(NPR)-NPR//2)*VOX
    vert = SP*pr; order = np.argsort(vert)
    def axial(img):
        return img[order]
    def longax(vol):
        return vol[:, order, NPR//2].T
    vv = vert[order]
    ext1 = [pr[0]-VOX/2, pr[-1]+VOX/2, vv[0]-VOX/2, vv[-1]+VOX/2]
    ext2 = [zz[0]-1.25, zz[-1]+1.25, vv[0]-VOX/2, vv[-1]+VOX/2]
    k0 = int(np.argmin(np.abs(zz)))
    W = dict(Z=300, X=150, Y=150)
    top = dict(Z=axial(v["Z"][k0]), X=axial(v["X"][k0]), Y=SP*axial(v["Y"][k0]))
    bot = dict(Z=longax(v["Z"]), X=longax(v["X"]), Y=SP*longax(v["Y"]))
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
    r_s = np.sqrt((frac > 0).sum(axis=(1, 2))*VOX**2/np.pi)
    halo = [pe.withStroke(linewidth=3, foreground="black")]
    marks = {"Upstream": ((-30.0, -4.0), (-26.0, -22.0)), "Downstream": ((30.0, -4.0), (26.0, -22.0)),
             "Stenosis throat" + chr(10) + f"(d = {2*r_s[k0]:.0f} mm)": ((0.0, r_s[k0]), (0.0, 22.0)), "Jet": ((14.0, 1.0), (24.0, 20.0))}
    for k, (xy, xyt) in marks.items():
        axes[1][0].annotate(k, xy, xytext=xyt, color="#ffd24d", fontsize=8.5*FS, fontweight="bold", ha="center", va="center", path_effects=halo,
                            arrowprops=dict(arrowstyle="-|>", color="#ffd24d", lw=1.4))
    axes[1][0].annotate("", (30, 33), xytext=(-18, 33), arrowprops=dict(arrowstyle="-|>", color="#ffd24d", lw=2))
    axes[1][0].text(-27, 33, "flow", color="#ffd24d", fontsize=8.5*FS, fontweight="bold", ha="center", va="center", path_effects=halo)
    fig.suptitle(f"Stenosis phantom Koma | peak systole: bin {b+1}/30, t = {t_ms:.0f} ms (upstream flow {q_up[b]:.0f} mL/s) | "
                 f"noise matched to the 4D-flow scan", fontsize=12*FS)
    fig.supxlabel(f"CFD phantom only (no stationary objects). Noise matched to the scan: vessel SNR {snr:.0f} (scan air noise; "
                  f"cross-check from the scan's static phase noise {rep['scan_lumen_snr_from_bottle_phase']:.0f}).\n"
                  "Koma VENC 1.5 / 1.5 / 4 m/s (R, P, along vessel); display windows = scan VENCs. Tukey k-space filter. "
                  "Air has no signal, so its phase is random, as in the scan.", fontsize=8*FS)
    p = stage/"phantom_candycane_peak_systole_flow_only.png"; fig.savefig(p, dpi=110); plt.close(fig)
    rep.update(peak_bin=b+1, t_ms=t_ms, rr_s=rr_s, upstream_flow_ml_s=float(q_up[b]), noise_sigma=sigma, seed=SEED, throat_partition=k0,
               long_axis_R_index=NPR//2, vertical_sign_P=SP, throat_diameter_mm=float(2*r_s[k0]),
               expected_lumen_velocity_sd_cm_s={e: float(np.sqrt(2)*vencs[e]*100/(np.pi*snr)) for e in vencs},
               status="PRESENTATION_DERIVATIVE_ONLY")
    (stage/"report.json").write_text(json.dumps(rep, indent=1, default=float)+chr(10), encoding="utf-8")
    (stage/"MANIFEST.json").write_text(json.dumps({p.name: sha(p) for p in sorted(stage.iterdir())}, indent=1)+chr(10), encoding="utf-8")
    stage.rename(target)
    print("DONE", b+1, round(t_ms), "snr air", round(snr, 1), "snr bottle-phase", round(rep["scan_lumen_snr_from_bottle_phase"], 1), "sigma", sigma)


if __name__ == "__main__":
    main(sys.argv[1])
