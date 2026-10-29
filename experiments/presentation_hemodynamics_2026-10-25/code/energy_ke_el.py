"""Kinetic energy and viscous energy loss over the cycle (user request), phantom and aorta.

KE = rho/2 sum |v|^2 dV; viscous energy loss rate EL = mu sum 2 S:S dV (resolved, Newtonian; frozen
mrsimtracks.hemodynamics velocity_gradient / differential_metrics, central differences, no one-sided or
wall-imposed gradients). Assumed blood-like properties: rho = 1060 kg/m^3, mu = 3.5 mPa s (neither CFD's
fluid properties are stored with the data). Weights: voxel lumen fraction x voxel volume (simulation
grids); voxel volume (scan, own vessel mask). Gradient component mapping is checked on the data: the
mapping must give a small divergence (|div| / |grad u|) for the CFD; the scan mapping is chosen as the
in-plane order/sign with the smallest divergence.
Phantom (60 mm slab, |S| <= 30 mm): CFD (E-060 reference, 1.25 x 1.25 x 2.5 mm), Koma VENC 3 and VENC 4
(same grid; 5 % all-encoding magnitude support), Koma VENC 3 + scan-matched noise (SNR 84, one sigma),
4D-flow scan (E-059 vessel mask). Time: each dataset rigidly shifted so its flow peaks at 101 ms.
Aorta: CFD native on a 1 mm grid (lumen), CFD at 4 mm (E-062), Koma 20 bins, Koma + display noise (SNR 12
as for VENC 0.3 m/s). Time from the v22 cycle start. Presentation only; no registered metric.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402

E060 = Path("D:/MRSim/outputs/scanner_geometry_e060")
sys.path.insert(0, str(E060/"code"))
sys.path.insert(1, "D:/MRSim/outputs/aorta_presentation_2026-10-25/code")
import plot_projection_scan_waveforms as wf  # noqa: E402
import plot_contours_with_noise as nz  # noqa: E402
import plot_scan_waveforms_noise as v4  # noqa: E402
from aorta_common import Aorta  # noqa: E402
from mrsimtracks.hemodynamics import differential_metrics, velocity_gradient  # noqa: E402  (frozen, read-only)

ws, tri = wf.ws, wf.tri
RHO, MU, FS = 1060.0, 3.5e-3, 1.5
COL = dict(cfd="#0b0b0b", match="#8a8984", koma="#2a78d6", koma4="#1baf7a", noisy="#7fa8de", scan="#eb6834")
LAB = dict(cfd="CFD", match="CFD at 4 mm", koma="Koma", koma4="Koma, VENC 4", noisy="Koma + noise", scan="4D-flow MRI")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def ke_el(u, spacing, valid, weight):
    """u (z,y,x,3) m/s, components (x,y,z); returns KE (J), EL (W), divergence ratio, gradient-covered weight."""
    keep = valid & np.isfinite(u).all(axis=-1)
    ke = 0.5*RHO*float((np.sum(np.where(keep[..., None], u, 0)**2, axis=-1)*np.where(keep, weight, 0)).sum())
    grad, st = velocity_gradient(np.where(keep[..., None], u, np.nan), spacing, valid=keep)
    met = differential_metrics(grad)
    ok = st & np.isfinite(met["dissipation_per_viscosity_s_inv2"])
    el = MU*float((met["dissipation_per_viscosity_s_inv2"][ok]*weight[ok]).sum())
    gnorm = np.sqrt(np.sum(grad[ok]**2, axis=(-2, -1)))
    divr = float(np.median(np.abs(met["divergence_s_inv"][ok])/np.maximum(gnorm, 1e-9))) if ok.any() else np.nan
    return ke, el, divr, float(weight[ok].sum())


def phantom():
    rep = json.loads((ws.E059/"measured_v2/report.json").read_text())
    z = np.load(ws.PRIVATE)
    v, mag, pos = z["velocity_cm_s"]/100.0, z["magnitude"], z["slice_positions_mm"]
    throat = rep["throat_slice"]; zrel = pos-pos[throat]
    sign = rep["segmentation"]["0.25"]["vendor_to_forward_sign"]
    m = mag.mean(axis=1); thr = .25*np.percentile(m, 99.5)
    vmask = np.zeros(m.shape, bool)
    for s in range(v.shape[0]):
        lab, n = ndimage.label(m[s] >= thr)
        flows = [abs((sign*v[s, :, :, :, 2][:, lab == c]).sum(axis=1).mean()) for c in range(1, n+1)]
        vmask[s] = lab == (1+int(np.argmax(flows)))
    vmask &= (np.abs(zrel) <= 30)[:, None, None]
    sp_scan = np.array([1.25e-3, 1.25e-3, 2.5e-3])
    # Scan component mapping: choose the in-plane order/sign and slice sign with the smallest divergence (peak phase).
    pk = int(np.argmax(rep["segmentation"]["0.25"]["slices"][throat]["flow_ml_s"]))
    best = None
    for perm in ((0, 1), (1, 0)):
        for s0 in (1, -1):
            for s1 in (1, -1):
                for s2 in (1, -1):
                    u = np.stack([s0*v[:, pk, :, :, perm[0]], s1*v[:, pk, :, :, perm[1]], s2*v[:, pk, :, :, 2]], axis=-1)
                    d = ke_el(u, sp_scan, vmask, np.full(vmask.shape, np.prod(sp_scan)))[2]
                    if best is None or d < best[0]:
                        best = (d, perm, (s0, s1, s2))
    _, perm, sg = best
    scan = []
    for p in range(v.shape[1]):
        u = np.stack([sg[0]*v[:, p, :, :, perm[0]], sg[1]*v[:, p, :, :, perm[1]], sg[2]*v[:, p, :, :, 2]], axis=-1)
        scan.append(ke_el(u, sp_scan, vmask, np.full(vmask.shape, np.prod(sp_scan))))
    _, _, _, snr_scan, _ = nz.scan_noise_and_panels()
    arms = tri.arms()
    with h5py.File(E060/"references_v2/references.h5") as f:
        ref, frac, zz = f["fft_adc"][:], f["fraction_fft"][:], f["partition_z_m"][:]*1e3
    sp = np.array([1.25e-3, 1.25e-3, 2.5e-3])                                     # x=R, y=P, z=S
    region = (frac > 0) & (np.abs(zz) <= 30)[:, None, None]
    w = frac*np.prod(sp)
    out = dict(scan=np.array(scan), scan_mapping=dict(perm=perm, signs=sg, median_div_ratio=best[0]))
    out["cfd"] = np.array([ke_el(np.nan_to_num(ref[b]).astype(float), sp, region, w) for b in range(ref.shape[0])])
    images3 = arms["E060_VENC3"][0]
    M = float(np.median(np.abs(images3["REF"]).mean(axis=0)[(frac > 0) & (np.abs(zz) <= 30)[:, None, None]]))
    sigma = M/snr_scan
    rng = np.random.default_rng(v4.SEED)
    noisy = {e: images3[e]+sigma*(rng.standard_normal(images3[e].shape)+1j*rng.standard_normal(images3[e].shape)) for e in tri.an.ENC}
    for key, (images, vencs) in (("koma", arms["E060_VENC3"]), ("koma4", arms["E061_VENC4"]), ("noisy", (noisy, arms["E060_VENC3"][1]))):
        rows = []
        for b in range(ref.shape[0]):
            field = np.stack([np.angle(images[e][b]*images["REF"][b].conj())*(vencs[i]/np.pi) for i, e in enumerate(tri.an.ENC[1:])], axis=-1)
            sup = np.ones(region.shape, bool)
            for e in tri.an.ENC:
                mm = np.abs(arms["E060_VENC3"][0][e][b]); sup &= mm >= .05*mm.max()      # support from the noiseless images
            rows.append(ke_el(field, sp, region & sup, w))
        out[key] = np.array(rows)
    scan_flow, _ = wf.scan_series(); (cfd_flow, _), _ = wf.sim_series()
    out["shift"] = dict(scan=wf.PEAK_AT-int(np.argmax(scan_flow)), sim=wf.PEAK_AT-int(np.argmax(cfd_flow)))
    out["snr"], out["sigma"] = snr_scan, sigma
    return out


def aorta():
    A = Aorta(); nb = A.nb
    dV = float(np.prod(A.spacing)); w = A.frac*dV
    out = {}
    out["match"] = np.array([ke_el(np.nan_to_num(A.ref[b]), A.spacing, (A.frac > 0) & np.isfinite(A.ref[b]).all(axis=-1), w) for b in range(nb)])
    for key, field in (("koma", A.koma), ("noisy", A.vnall)):
        out[key] = np.array([ke_el(field[b], A.spacing, (A.frac > 0) & (A.mag[b] >= .05*A.mag[b].max()), w) for b in range(nb)])
    step = 1e-3
    lo, hi = A.nodes.min(axis=0)-2e-3, A.nodes.max(axis=0)+2e-3
    ax_ = [np.arange(lo[i], hi[i], step) for i in range(3)]
    Zg, Yg, Xg = np.meshgrid(ax_[2], ax_[1], ax_[0], indexing="ij")
    Pg = np.stack([Xg.ravel(), Yg.ravel(), Zg.ravel()], -1)
    d1, _ = A.tree.query(Pg, k=1, workers=-1, distance_upper_bound=1.01e-3)
    lumen = d1 <= 1e-3
    vnat = A.native_bins(Pg[lumen])
    rows = []
    for b in range(nb):
        u = np.full(Pg.shape, np.nan); u[lumen] = vnat[b]
        rows.append(ke_el(u.reshape(Zg.shape+(3,)), np.array([step]*3), lumen.reshape(Zg.shape), np.full(Zg.shape, step**3)))
    out["cfd"] = np.array(rows)
    out["tau"] = A.tau_ms(A.tc20); out["rr"] = A.rr
    out["volumes_ml"] = dict(native_lumen=float(lumen.sum()*1e-3), four_mm_support=float(w[A.frac > 0].sum()*1e6))
    return out


def main(target):
    t_start = time.time()
    target = Path(target)
    if target.exists():
        raise FileExistsError(target)
    P = phantom(); print("phantom done", round(time.time()-t_start, 1), P["scan_mapping"], flush=True)
    Q = aorta(); print("aorta done", round(time.time()-t_start, 1), Q["volumes_ml"], flush=True)
    for name, D in (("phantom", P), ("aorta", Q)):
        for k in ("cfd", "match", "koma", "koma4", "noisy", "scan"):
            if k in D:
                print(name, k, "median div ratio %.3f" % np.nanmedian(D[k][:, 2]), "KE max %.3f mJ" % (D[k][:, 0].max()*1e3), "EL max %.3f mW" % (D[k][:, 1].max()*1e3), flush=True)
    plt.rcParams["font.size"] = 10*FS
    out = target.with_name(target.name+".building"); out.mkdir(parents=True)
    fig, axes = plt.subplots(2, 3, figsize=(26, 14), layout="constrained", gridspec_kw=dict(width_ratios=[1, 1, .7]))
    dt = wf.DT
    for r, (name, D, keys) in enumerate((("Phantom", P, ("cfd", "koma", "koma4", "noisy", "scan")), ("Aorta", Q, ("cfd", "match", "koma", "noisy")))):
        totals = {}
        for k in keys:
            ke, el = D[k][:, 0]*1e3, D[k][:, 1]*1e3                              # mJ, mW
            if name == "Phantom":
                sh = D["shift"]["scan" if k == "scan" else "sim"]
                t = np.arange(len(ke))*dt; ke, el = np.roll(ke, sh), np.roll(el, sh)
                period = 30*dt
            else:
                o = np.argsort(D["tau"]); t = D["tau"][o]; ke, el = ke[o], el[o]; period = D["rr"]*1000
            sty = dict(color=COL[k], lw=2.2 if k in ("cfd", "koma", "scan") else 1.4, marker="o", ms=3.5, label=LAB[k],
                       ls=(0, (4, 2)) if k == "match" else ((0, (2, 2)) if k == "noisy" else "-"))
            axes[r, 0].plot(t, ke, **sty); axes[r, 1].plot(t, el, **sty)
            totals[k] = (ke.mean(), el.mean()*period/1000)                       # mean KE (mJ), EL per beat (mJ)
        for j, (title, unit) in enumerate((("Kinetic energy", "mJ"), ("Viscous energy loss rate", "mW"))):
            ax = axes[r, j]; ax.set_title(f"{name}: {title}", fontsize=10*FS); ax.set_xlabel("Time (ms)", fontsize=8*FS)
            ax.set_ylabel(unit, fontsize=9*FS); ax.grid(alpha=.25); ax.tick_params(labelsize=8*FS); ax.set_ylim(bottom=0)
        axes[r, 0].legend(fontsize=7.5*FS, frameon=False)
        ax = axes[r, 2]; xs = np.arange(len(keys)); wbar = .38
        ax.bar(xs-wbar/2, [totals[k][0] for k in keys], wbar, color=[COL[k] for k in keys], label="mean KE (mJ)")
        ax.bar(xs+wbar/2, [totals[k][1] for k in keys], wbar, color=[COL[k] for k in keys], alpha=.45, hatch="//", label="EL per beat (mJ)")
        ax.set_xticks(xs, [LAB[k] for k in keys], rotation=30, ha="right", fontsize=7.5*FS)
        ax.set_title(f"{name}: per beat", fontsize=10*FS); ax.set_ylabel("mJ", fontsize=9*FS); ax.grid(alpha=.25, axis="y")
        ax.legend(fontsize=7.5*FS, frameon=False); ax.tick_params(labelsize=8*FS)
    fig.suptitle("Kinetic energy and viscous energy loss: CFD vs Koma vs 4D-flow MRI", fontsize=13*FS)
    fig.supxlabel("Assumed ρ = 1060 kg/m³, μ = 3.5 mPa·s. Phantom: 60 mm slab, flow peaks aligned at 101 ms; the CFD carries about 60 % more flow than the scan. "
                  "Aorta: whole lumen; CFD on a 1 mm grid, others 4 mm.\nEnergy loss uses resolved velocity gradients only (central differences); coarse grids miss near-wall shear and read low. "
                  "Noise is display-level (phantom SNR 84; aorta SNR 12 as for VENC 0.3 m/s) and inflates both quantities.", fontsize=8.5*FS)
    fig.savefig(out/"energy_ke_el.png", dpi=110); plt.close(fig)
    ser = lambda D, ks: {k: dict(ke_J=D[k][:, 0].tolist(), el_W=D[k][:, 1].tolist(), median_div_ratio=float(np.nanmedian(D[k][:, 2])),  # noqa: E731
                                 gradient_covered_volume_ml=float(np.nanmedian(D[k][:, 3])*1e6)) for k in ks if k in D}
    rep = dict(created_utc=datetime.now(timezone.utc).isoformat(), status="PRESENTATION_DERIVATIVE_ONLY", script_sha256=sha(__file__),
               rho_kg_m3=RHO, mu_pa_s=MU, phantom=dict(series=ser(P, ("cfd", "koma", "koma4", "noisy", "scan")), shift=P["shift"], scan_mapping=P["scan_mapping"],
               noise_snr=P["snr"], noise_sigma=P["sigma"]), aorta=dict(series=ser(Q, ("cfd", "match", "koma", "noisy")), volumes_ml=Q["volumes_ml"]),
               private_input_used=str(ws.PRIVATE), runtime_s=time.time()-t_start)
    rep["outputs"] = {p.name: sha(p) for p in sorted(out.glob("*.png"))}
    (out/"MANIFEST.json").write_text(json.dumps(rep, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))+"\n", encoding="utf-8")
    out.rename(target)
    print("DONE", round(time.time()-t_start, 1))


if __name__ == "__main__":
    main(sys.argv[1])
