"""Aorta secondary flow (user request): in-plane vector maps on planes A/D and helicity over the cycle.

Maps: planes perpendicular to the cycle-mean flow (v21), background = through-plane velocity, arrows =
in-plane velocity, at peak systole, late systole (flow first <= 50 % of peak) and diastole (minimum flow
after late systole) chosen from the native CFD flow at A. Rows: CFD native (0.25 mm plane, Koma bin
window), Koma (4 mm, fraction-weighted trilinear onto the plane), Koma + display noise (v22 realisation).
Curves: mean in-plane speed on A and D; volume helicity H = sum(v . curl v) dV and sum|v . curl v| dV.
Helicity region: voxels whose full central-difference stencil lies in the E-062 20-bin core15 (same set
for all 4 mm fields); native CFD on a 1 mm grid (points within 1 mm of a CFD node) restricted to those
voxels. Display and presentation only; no registered metric.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from aorta_common import Aorta, PLANE, SNR, VENC_NOISE  # noqa: E402
from plot_aorta_highres import sha  # noqa: E402
from mrsimtracks.hemodynamics import differential_metrics, velocity_gradient  # noqa: E402  (frozen, read-only)

FS = 1.5
COL = dict(cfd="#0b0b0b", match="#8a8984", koma="#2a78d6", noisy="#7fa8de")
LAB = dict(cfd="CFD", match="CFD at 4 mm", koma="Koma", noisy="Koma + noise")


def volume_helicity(field, spacing, valid, weight):
    """Signed and absolute helicity integrals (m^4/s^2) over stencil-valid voxels."""
    grad, stencil = velocity_gradient(field, spacing, valid=valid)
    h = differential_metrics(grad, field)["helicity_m_s_inv2"]
    keep = stencil & np.isfinite(h)
    return float((h[keep]*weight[keep]).sum()), float((np.abs(h[keep])*weight[keep]).sum()), stencil


def main(target):
    t_start = time.time()
    target = Path(target)
    if target.exists():
        raise FileExistsError(target)
    A = Aorta(); nb = A.nb
    tau20 = A.tau_ms(A.tc20); order_t = np.argsort(tau20)
    planes = {k: A.plane(k) for k in ("asc", "desc")}
    data = {}
    for k, p in planes.items():
        B = p["basis"]
        data[k] = dict(cfd=np.einsum("bqc,dc->bqd", A.native_bins(p["pts"]), B),
                       match=np.einsum("bqc,dc->bqd", A.interp4(A.ref, A.frac, p["pts"]), B),
                       koma=np.einsum("bqc,dc->bqd", A.interp4(A.koma, A.frac, p["pts"]), B),
                       noisy=np.einsum("bqc,dc->bqd", A.interp4(A.vnall, A.frac, p["pts"]), B))
        print("plane", k, len(p["pts"]), round(time.time()-t_start, 1), flush=True)
    # Times from the native CFD flow at A (in time order).
    flowA = data["asc"]["cfd"][:, :, 0].sum(axis=1)*PLANE**2*1e6
    seq = [int(b) for b in order_t]; fl = flowA[seq]
    i_pk = int(np.argmax(fl))
    i_late = next(i for i in range(i_pk, nb) if fl[i] <= .5*fl[i_pk])
    i_dia = i_late+int(np.argmin(fl[i_late:]))
    picks = [("Peak systole", seq[i_pk]), ("Late systole", seq[i_late]), ("Diastole", seq[i_dia])]
    # Secondary-flow index: mean in-plane speed on each plane.
    sec_idx = {k: {m: np.linalg.norm(v[:, :, 1:], axis=-1).mean(axis=1) for m, v in d.items()} for k, d in data.items()}

    # Volume helicity: 4 mm fields on the common stencil region; native CFD on 1 mm.
    dV = float(np.prod(A.spacing)); wvox = A.frac*dV
    hel = {}
    for m, field in (("match", np.nan_to_num(A.ref)), ("koma", A.koma), ("noisy", A.vnall)):
        hel[m] = np.array([volume_helicity(field[b], A.spacing, A.core, wvox)[:2] for b in range(nb)])
    _, stencil = velocity_gradient(np.nan_to_num(A.ref[0]), A.spacing, valid=A.core)
    step = 1e-3
    lo, hi = A.nodes.min(axis=0)-2e-3, A.nodes.max(axis=0)+2e-3
    ax_ = [np.arange(lo[i], hi[i], step) for i in range(3)]
    Zg, Yg, Xg = np.meshgrid(ax_[2], ax_[1], ax_[0], indexing="ij")                 # (z, y, x)
    Pg = np.stack([Xg.ravel(), Yg.ravel(), Zg.ravel()], -1)
    d1, _ = A.tree.query(Pg, k=1, workers=-1, distance_upper_bound=1.51e-3)
    vox = [np.clip(np.round((Pg[:, a]-g[0])/(g[1]-g[0])).astype(int), 0, len(g)-1) for a, g in enumerate((A.gx, A.gy, A.gz))]
    shape = Zg.shape
    region = stencil[vox[2], vox[1], vox[0]] & (d1 <= 1.5e-3)
    # v2: gradients may use lumen neighbours outside the region; the integral is still over the region only
    # (v1 required all six neighbours inside the region and lost a 1 mm shell, ~30 % of its 81 mL).
    from scipy import ndimage as _nd
    near = _nd.binary_dilation(region.reshape(shape), iterations=2).ravel()
    lumen = near & (d1 <= 1.5e-3)
    pin = Pg[lumen]
    print("native 1 mm points", int(region.sum()), "region,", len(pin), "sampled", round(time.time()-t_start, 1), flush=True)
    vnat = A.native_bins(pin)                                                       # (bins, points, 3)
    hel["cfd"] = []
    wreg = np.where(region, step**3, 0.0).reshape(shape)
    for b in range(nb):
        u = np.full(Pg.shape, np.nan); u[lumen] = vnat[b]
        H, Habs, st1 = volume_helicity(u.reshape(shape+(3,)), np.array([step]*3), lumen.reshape(shape), wreg)
        hel["cfd"].append((H, Habs))
    cover = float((st1 & region.reshape(shape)).sum()*step**3*1e6)
    print("native region volume mL", region.sum()*1e-3, "gradient-covered", cover, "| 4 mm region mL", float((wvox[stencil]).sum()*1e6), flush=True)
    hel["cfd"] = np.array(hel["cfd"])
    print("helicity done", round(time.time()-t_start, 1), flush=True)

    # ---- figure ----
    plt.rcParams["font.size"] = 10*FS
    out = target.with_name(target.name+".building"); out.mkdir(parents=True)
    fig = plt.figure(figsize=(30, 21), layout="constrained")
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1.05])
    top = gs[0].subgridspec(3, 6); bot = gs[1].subgridspec(1, 4)
    vmax = float(max(np.percentile(data[k]["cfd"][:, :, 0], 99.5) for k in data))
    ip_ref = float(np.percentile(np.concatenate([np.linalg.norm(data[k]["cfd"][b, :, 1:], axis=-1) for k in data for _, b in picks]), 95))
    K = 4e-3/ip_ref                                                                  # arrow length 4 mm at the 95th percentile
    cm = plt.get_cmap("viridis").copy(); cm.set_bad("#eeeeee")
    tag = dict(asc="A", desc="D")
    for r, m in enumerate(("cfd", "koma", "noisy")):
        for c, (k, (lab, b)) in enumerate([(k, pk) for k in ("asc", "desc") for pk in picks]):
            ax = fig.add_subplot(top[r, c]); p = planes[k]; v = data[k][m][b]
            img = np.full(p["shape"], np.nan); img[p["ij"][:, 0], p["ij"][:, 1]] = v[:, 0]
            half = (p["shape"][0]-1)*PLANE/2*1000
            s_mm = (p["ij"][:, 0]*PLANE*1000)-half; t_mm = (p["ij"][:, 1]*PLANE*1000)-half
            im = ax.imshow(img.T, origin="lower", extent=[-half-PLANE*500, half+PLANE*500]*2, cmap=cm, vmin=0, vmax=vmax, interpolation="nearest")
            sel = (p["ij"][:, 0] % 8 == 0) & (p["ij"][:, 1] % 8 == 0)
            q = ax.quiver(s_mm[sel], t_mm[sel], v[sel, 1]*K*1000, v[sel, 2]*K*1000, angles="xy", scale_units="xy", scale=1,
                          color="white", width=.004, headwidth=4)
            if r == 0 and c == 0:
                q0, ax0 = q, ax
            ax.set_xlim(s_mm.min()-2, s_mm.max()+2); ax.set_ylim(t_mm.min()-2, t_mm.max()+2); ax.set_aspect("equal")
            ax.set_facecolor("#eeeeee"); ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"{tag[k]} | {lab}\n{tau20[b]:.0f} ms", fontsize=9*FS)
            if c == 0:
                ax.set_ylabel(LAB[m] + (f"\n(SNR {SNR:.0f})" if m == "noisy" else ""), fontsize=9*FS)
    fig.colorbar(im, ax=fig.axes[:18], label="Through-plane velocity (m/s)", shrink=.6, pad=.01)
    ax0.quiverkey(q0, .25, -.07, ip_ref*K*1000, f"{ip_ref*100:.1f} cm/s", labelpos="E", coordinates="axes", fontproperties={"size": 8*FS})
    curves = [("In-plane speed, A", lambda m: sec_idx["asc"][m], "m/s", ("cfd", "match", "koma", "noisy")),
              ("In-plane speed, D", lambda m: sec_idx["desc"][m], "m/s", ("cfd", "match", "koma", "noisy")),
              ("Helicity (signed)", lambda m: hel[m][:, 0]*1e4, "10⁻⁴ m⁴/s²", ("cfd", "match", "koma", "noisy")),
              ("Helicity (absolute)", lambda m: hel[m][:, 1]*1e4, "10⁻⁴ m⁴/s²", ("cfd", "match", "koma", "noisy"))]
    for j, (title, get, unit, ms) in enumerate(curves):
        ax = fig.add_subplot(bot[0, j])
        for m in ms:
            y = np.asarray(get(m))
            ax.plot(tau20[order_t], y[order_t], color=COL[m], lw=2 if m in ("cfd", "koma") else 1.3,
                    ls=(0, (4, 2)) if m == "match" else ((0, (2, 2)) if m == "noisy" else "-"), marker="o", ms=3.5, label=LAB[m])
        for lab, b in picks:
            ax.axvline(tau20[b], color="#bbbbbb", lw=.8, ls=(0, (3, 3)))
        ax.axhline(0, color="#888888", lw=.6)
        ax.set_title(title, fontsize=10*FS); ax.set_xlabel("Time (ms)", fontsize=8*FS); ax.set_ylabel(unit, fontsize=9*FS)
        ax.set_xlim(0, A.rr*1000); ax.grid(alpha=.25); ax.tick_params(labelsize=8*FS)
        if j == 0:
            ax.legend(fontsize=7.5*FS, frameon=False)
    fig.suptitle("Aorta secondary flow: in-plane velocity on planes A and D, and helicity over the cycle", fontsize=13*FS)
    fig.supxlabel("Planes perpendicular to the local mean flow; colour: through-plane velocity, arrows: in-plane velocity (same scale everywhere). "
                  f"Noise: SNR {SNR:.0f}, as for VENC {VENC_NOISE} m/s (display only).\n"
                  "Helicity = Σ v·(∇×v) dV over the same interior region for all datasets (CFD on a 1 mm grid; others 4 mm). "
                  "Dotted verticals: the three times shown.", fontsize=8.5*FS)
    fig.savefig(out/"aorta_secondary_flow_helicity.png", dpi=110); plt.close(fig)
    rep = dict(created_utc=datetime.now(timezone.utc).isoformat(), status="PRESENTATION_DERIVATIVE_ONLY", script_sha256=sha(__file__),
               common_sha256=sha(Path(__file__).parent/"aorta_common.py"), times={lab: dict(bin=b+1, time_ms=float(tau20[b])) for lab, b in picks},
               plane_normals={k: p["n"].tolist() for k, p in planes.items()}, arrow_ref_m_s=ip_ref,
               secondary_index_m_s={k: {m: v.tolist() for m, v in d.items()} for k, d in sec_idx.items()},
               helicity_m4_s2={m: v.tolist() for m, v in hel.items()}, native_points=int(len(pin)), time_ms=tau20.tolist(),
               region_ml=dict(native=float(region.sum()*1e-3), native_gradient_covered=cover, four_mm=float((wvox[stencil]).sum()*1e6)),
               runtime_s=time.time()-t_start)
    rep["outputs"] = {p.name: sha(p) for p in sorted(out.glob("*.png"))}
    (out/"MANIFEST.json").write_text(json.dumps(rep, indent=1)+"\n", encoding="utf-8")
    out.rename(target)
    print("DONE", round(time.time()-t_start, 1))


if __name__ == "__main__":
    main(sys.argv[1])
