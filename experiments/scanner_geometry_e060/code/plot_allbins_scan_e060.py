"""All-phase overview for the phantom (user request), like the aorta all-bins figure, plus the scan.

Rows: 4D-flow MRI (measured, own background), CFD, Koma (VENC 3, noiseless), Koma + scan-matched
noise (own background), Koma - CFD. Images as plot_scan_waveforms_noise.py v4: central plane through
the vessel axis, speed |v|, flow left to right, bilinear 0.25 mm, CFD/Koma/difference clipped to the
Gaussian wall fit. Columns: the 30 phases with each dataset rigidly circularly shifted so its flow
peak sits at phase index 3 (101 ms), as in the v4 waveforms (CFD and Koma share the CFD shift, since
they share acquisition bins). Noise: one sigma for the whole acquisition (median lumen |REF| over all
bins / scan lumen SNR), complex Gaussian on all four encodings, fixed seed. Display only.
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
sys.path.insert(0, str(E060/"code"))
import plot_projection_scan_waveforms as wf  # noqa: E402
import plot_contours_with_noise as nz  # noqa: E402
import plot_scan_waveforms_noise as v4  # noqa: E402
ws, sm, tri = wf.ws, wf.sm, wf.tri
FS, PER_ROW = 1.5, 15


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(target):
    target = Path(target)
    if target.exists():
        raise FileExistsError(target)
    rep = json.loads((ws.E059/"measured_v2/report.json").read_text())
    z = np.load(ws.PRIVATE)
    v, mag, pos = z["velocity_cm_s"]/100.0, z["magnitude"], z["slice_positions_mm"]
    throat = rep["throat_slice"]
    _, _, _, snr_scan, _ = nz.scan_noise_and_panels()
    _, _, _, _, _, cy = ws.scan_projection()
    m = mag.mean(axis=1); thr = .25*np.percentile(m, 99.5)
    sign = rep["segmentation"]["0.25"]["vendor_to_forward_sign"]
    cxs = []
    for s in range(v.shape[0]):
        lab, n = ndimage.label(m[s] >= thr)
        flows = [abs((sign*v[s, :, :, :, 2][:, lab == c]).sum(axis=1).mean()) for c in range(1, n+1)]
        cxs.append(ndimage.center_of_mass(lab == (1+int(np.argmax(flows))))[1])
    cx = int(round(np.mean(cxs)))                                          # as v4
    p_scan = (np.arange(64)-cy)*1.25; zrel = pos-pos[throat]
    images, vencs = tri.arms()["E060_VENC3"]
    with h5py.File(E060/"references_v2/references.h5") as f:
        ref, frac, zz, pfrac = f["fft_adc"][:], f["fraction_fft"][:], f["partition_z_m"][:]*1e3, f["plane_fraction"][:]
    nb = ref.shape[0]
    lum = frac > 0
    rc = int(np.argmax(pfrac[tri.an.THROAT_K].sum(axis=0)))
    pc = ndimage.center_of_mass(pfrac[tri.an.THROAT_K])[0]
    p_sim = (np.arange(64)-pc)*1.25
    slab = (np.abs(zz) <= 30)[:, None, None]
    M = float(np.median(np.abs(images["REF"]).mean(axis=0)[lum & slab]))
    sigma = M/snr_scan
    rng = np.random.default_rng(v4.SEED)
    noisy = {e: images[e]+sigma*(rng.standard_normal(images[e].shape)+1j*rng.standard_normal(images[e].shape)) for e in tri.an.ENC}
    def speed(ims):
        return np.linalg.norm(np.stack([np.angle(ims[e]*ims["REF"].conj())*(vencs[i]/np.pi) for i, e in enumerate(tri.an.ENC[1:])], axis=-1), axis=-1)
    k_clean, k_noisy = speed(images)[..., rc], speed(noisy)[..., rc]                # (bin, S, P)
    cfd = np.where(lum[None, :, :, rc], np.linalg.norm(np.nan_to_num(ref), axis=-1)[..., rc], np.nan)
    rS = sm.wall(zz)
    inside = np.abs(sm.PF)[None, :] <= rS[:, None]
    wall_clip = lambda img: np.where(inside, v4.resample(v4.fill_nearest(img), zz, p_sim), np.nan)  # noqa: E731
    # Peak alignment (v4 waveform rule): scan by its own flow peak, CFD and Koma by the CFD peak.
    scan_flow, _ = wf.scan_series(); (cfd_flow, _), _ = wf.sim_series()
    sh_scan, sh_sim = wf.PEAK_AT-int(np.argmax(scan_flow)), wf.PEAK_AT-int(np.argmax(cfd_flow))
    rows = {"scan": [], "cfd": [], "koma": [], "noisy": [], "diff": []}
    cols = []
    for k in range(nb):
        ps, bs = (k-sh_scan) % nb, (k-sh_sim) % nb
        cols.append(dict(time_ms=k*wf.DT, scan_phase=ps+1, sim_bin=bs+1))
        rows["scan"].append(v4.resample(np.linalg.norm(v[:, ps, :, cx, :], axis=-1), zrel, p_scan))
        rows["cfd"].append(wall_clip(cfd[bs]))
        rows["koma"].append(wall_clip(np.where(lum[:, :, rc], k_clean[bs], np.nan)))
        rows["noisy"].append(v4.resample(k_noisy[bs], zz, p_sim))
        rows["diff"].append(wall_clip(np.where(lum[:, :, rc], k_clean[bs]-cfd[bs], np.nan)))
    vmax = 3.0                                                               # v4 scale
    dmax = float(np.nanpercentile(np.abs(np.stack(rows["diff"])), 99.5))
    extent = [sm.PF[0]-sm.STEP/2, sm.PF[-1]+sm.STEP/2, sm.SF[0]-sm.STEP/2, sm.SF[-1]+sm.STEP/2]
    cm_s = plt.get_cmap("viridis").copy(); cm_s.set_bad("#eeeeee")
    cm_d = plt.get_cmap("RdBu_r").copy(); cm_d.set_bad("#eeeeee")
    names = (("scan", "4D-flow MRI"), ("cfd", "CFD"), ("koma", "Koma"), ("noisy", f"Koma + noise\n(SNR {snr_scan:.0f})"), ("diff", "Koma − CFD"))
    plt.rcParams["font.size"] = 10*FS
    stage = target.with_name(target.name+".building"); stage.mkdir()
    fig = plt.figure(figsize=(34, 17), layout="constrained")
    subs = fig.subfigures(2, 1, hspace=.02)
    for blk, sub in enumerate(subs):
        axes = sub.subplots(len(names), PER_ROW, sharex=True, sharey=True)
        for r, (key, label) in enumerate(names):
            for c in range(PER_ROW):
                k = blk*PER_ROW+c; ax = axes[r, c]
                diff = key == "diff"
                im = sm.show(ax, rows[key][k], extent, cmap=cm_d if diff else cm_s, vmin=-dmax if diff else 0, vmax=dmax if diff else vmax)
                clipped = key in ("cfd", "koma", "diff")
                ax.plot(sm.SF, rS, color="#3a3936" if clipped else "white", lw=1.0 if clipped else .6, ls="-" if clipped else (0, (4, 3)))
                ax.plot(sm.SF, -rS, color="#3a3936" if clipped else "white", lw=1.0 if clipped else .6, ls="-" if clipped else (0, (4, 3)))
                ax.set_facecolor("#eeeeee"); ax.set_xticks([]); ax.set_yticks([])
                if r == 0:
                    ax.set_title(f"{cols[k]['time_ms']:.0f} ms", fontsize=8.5*FS)
                if c == 0:
                    ax.set_ylabel(label, fontsize=8.5*FS)
                if diff:
                    im_d = im
                else:
                    im_s = im
        sub.colorbar(im_s, ax=axes[:4, :], label="Speed (m/s)", shrink=.85, pad=.005)
        sub.colorbar(im_d, ax=axes[4, :], label="m/s", shrink=.9, extend="both", pad=.005)
    fig.suptitle("Phantom: 4D-flow MRI vs CFD vs Koma, all 30 phases (flow peaks aligned at 101 ms)", fontsize=13*FS)
    fig.supxlabel("Central plane through the vessel axis, speed; flow left to right. Lines: Gaussian fit of the CFD wall. "
                  f"Noise (display only): scan-matched SNR {snr_scan:.0f}; scan and noisy Koma show their own background.\n"
                  "Scan not registered in-plane to the CFD; the CFD carries about 60 % more flow than the scan.", fontsize=8.5*FS)
    fig.savefig(stage/"phantom_scan_cfd_koma_allphases.png", dpi=110); plt.close(fig)
    (stage/"MANIFEST.json").write_text(json.dumps(dict(created_utc=datetime.now(timezone.utc).isoformat(), status="PRESENTATION_DERIVATIVE_ONLY",
        script_sha256=sha(__file__), arm="E060_VENC3", scan_lumen_snr=snr_scan, koma_noise_sigma=sigma, seed=v4.SEED, scan_plane_column=cx,
        sim_plane_R_index=rc, shift_scan=sh_scan, shift_sim=sh_sim, columns=cols, diff_max=dmax, private_input_used=str(ws.PRIVATE),
        outputs={p.name: sha(p) for p in sorted(stage.glob("*.png"))}), indent=1)+"\n", encoding="utf-8")
    stage.rename(target)
    print("DONE", snr_scan, sigma, sh_scan, sh_sim)


if __name__ == "__main__":
    main(sys.argv[1])
