"""E-060/E-061 version of the 'matched longitudinal profiles' figure (user request).

Per S partition (vessel axis, 60 mm slab): fraction-weighted mean axial velocity, fraction-weighted mean
speed and 99th-percentile speed over lumen voxels (CFD: fraction > 0; Koma: same voxels with 5 % all-encoding
magnitude support). Phases: CFD peak bin and CFD minimum-flow bin; the scan uses its own phases matched by the
rigid peak-aligned shift (as the phantom waveform figures), with its own vessel mask (E-059, 25 %).
Same-source internal consistency for CFD/Koma; the scan carries ~60 % less flow. Presentation only.
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
ws, tri = wf.ws, wf.tri
FS = 1.5
COL = dict(cfd="#0b0b0b", v3="#2a78d6", v4="#1baf7a", scan="#eb6834")
LAB = dict(cfd="CFD", v3="Koma, VENC 3", v4="Koma, VENC 4", scan="4D-flow MRI")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def wstats(vals_ax, speed, w):
    if w.sum() <= 0:
        return np.nan, np.nan, np.nan
    return float((vals_ax*w).sum()/w.sum()), float((speed*w).sum()/w.sum()), float(np.percentile(speed[w > 0], 99))


def main(target):
    target = Path(target)
    if target.exists():
        raise FileExistsError(target)
    with h5py.File(E060/"references_v2/references.h5") as f:
        ref, frac, zz = f["fft_adc"][:], f["fraction_fft"][:], f["partition_z_m"][:]*1e3
    arms = tri.arms()
    (cfd_flow, _), _ = wf.sim_series(); scan_flow, _ = wf.scan_series()
    b_pk, b_lo = int(np.argmax(cfd_flow)), int(np.argmin(cfd_flow))
    sh_sim, sh_scan = wf.PEAK_AT-b_pk, wf.PEAK_AT-int(np.argmax(scan_flow))
    slab = np.abs(zz) <= 30
    prof = {}
    for tag, b in (("peak", b_pk), ("low", b_lo)):
        rows = {"cfd": []}
        for s in np.flatnonzero(slab):
            w = np.where(frac[s] > 0, frac[s], 0); v = np.nan_to_num(ref[b, s])
            rows["cfd"].append((zz[s],)+wstats(v[..., 2], np.linalg.norm(v, axis=-1), w))
        for key, arm in (("v3", "E060_VENC3"), ("v4", "E061_VENC4")):
            images, vencs = arms[arm]
            field = np.stack([np.angle(images[e][b]*images["REF"][b].conj())*(vencs[i]/np.pi) for i, e in enumerate(tri.an.ENC[1:])], axis=-1)
            sup = np.ones(frac.shape, bool)
            for e in tri.an.ENC:
                m = np.abs(arms["E060_VENC3"][0][e][b]); sup &= m >= .05*m.max()
            rows[key] = []
            for s in np.flatnonzero(slab):
                w = np.where((frac[s] > 0) & sup[s], frac[s], 0)
                rows[key].append((zz[s],)+wstats(field[s][..., 2], np.linalg.norm(field[s], axis=-1), w))
        prof[tag] = dict(bin=b+1, rows={k: np.array(v) for k, v in rows.items()})
    rep = json.loads((ws.E059/"measured_v2/report.json").read_text())
    z = np.load(ws.PRIVATE)
    v, mag, pos = z["velocity_cm_s"]/100.0, z["magnitude"], z["slice_positions_mm"]
    zrel = pos-pos[rep["throat_slice"]]; sign = rep["segmentation"]["0.25"]["vendor_to_forward_sign"]
    mm = mag.mean(axis=1); thr = .25*np.percentile(mm, 99.5)
    for tag in ("peak", "low"):
        p_scan = ((prof[tag]["bin"]-1)+sh_sim-sh_scan) % 30                    # scan phase at the same aligned time
        rows = []
        for s in range(v.shape[0]):
            if abs(zrel[s]) > 30:
                continue
            lab, n = ndimage.label(mm[s] >= thr)
            flows = [abs((sign*v[s, :, :, :, 2][:, lab == c]).sum(axis=1).mean()) for c in range(1, n+1)]
            msk = (lab == (1+int(np.argmax(flows)))).astype(float)
            vv = v[s, p_scan]
            rows.append((zrel[s],)+wstats(sign*vv[..., 2], np.linalg.norm(vv, axis=-1), msk))
        prof[tag]["rows"]["scan"] = np.array(rows); prof[tag]["scan_phase"] = p_scan+1
    plt.rcParams["font.size"] = 10*FS
    stage = target.with_name(target.name+".building"); stage.mkdir()
    fig, axes = plt.subplots(3, 2, figsize=(22, 17), layout="constrained")
    qn = ("Mean axial velocity", "Mean speed", "Speed p99")
    for c, tag in enumerate(("peak", "low")):
        for r in range(3):
            ax = axes[r, c]
            for k in ("cfd", "v3", "v4", "scan"):
                a = prof[tag]["rows"][k]
                ax.plot(a[:, 0], a[:, r+1], color=COL[k], lw=2.2 if k != "v4" else 1.6, marker="s" if k == "scan" else "o", ms=4,
                        ls=(0, (4, 2)) if k == "v4" else "-", label=LAB[k])
            ax.axvline(0, color="#999999", lw=.7, ls=(0, (3, 3)))
            ax.set_title(f"{'Peak' if tag == 'peak' else 'Lowest flow'} | bin {prof[tag]['bin']} | {qn[r]}", fontsize=10*FS)
            ax.set_xlabel("S, along vessel (mm; throat at 0)", fontsize=8*FS); ax.set_ylabel("m/s", fontsize=9*FS)
            ax.grid(alpha=.25); ax.tick_params(labelsize=8*FS)
    axes[0, 0].legend(fontsize=8*FS, frameon=False)
    fig.suptitle("Phantom | E-060 scanner geometry | 30 bins\nLongitudinal profiles along the vessel axis", fontsize=13*FS)
    fig.supxlabel("Per 2.5 mm partition, lumen voxels (fraction-weighted means; p99 unweighted). Scan at the matching peak-aligned phase, own vessel mask; "
                  "it carries about 60 % less flow than the CFD.\nSame-source simulation for CFD/Koma; VENC 3 aliases in the systolic jet.", fontsize=8.5*FS)
    fig.savefig(stage/"phantom_longitudinal_profiles.png", dpi=110); plt.close(fig)
    (stage/"MANIFEST.json").write_text(json.dumps(dict(created_utc=datetime.now(timezone.utc).isoformat(), status="PRESENTATION_DERIVATIVE_ONLY",
        script_sha256=sha(__file__), bins=dict(peak=prof["peak"]["bin"], low=prof["low"]["bin"]),
        scan_phases=dict(peak=int(prof["peak"]["scan_phase"]), low=int(prof["low"]["scan_phase"])),
        outputs={p.name: sha(p) for p in stage.glob("*.png")}), indent=1)+"\n", encoding="utf-8")
    stage.rename(target)
    print("DONE", prof["peak"]["bin"], prof["low"]["bin"])


if __name__ == "__main__":
    main(sys.argv[1])
