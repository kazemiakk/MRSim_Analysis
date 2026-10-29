"""Aorta scanner-style 4D-flow display in the oblique 'candy-cane' view (user: show inlet, ascending aorta, arch, branches,
descending aorta). A projection through MRI Z foreshortens the ascending aorta (it runs mostly along Z).

View plane = best-fit plane of the vessel (PCA of the core-15 % voxel centres weighted by mean magnitude); normal n is the
smallest-variance axis, e1 the largest (horizontal), e2 = n x e1 (vertical, sign chosen so the arch is on top; e1 sign so the
ascending aorta is on the left). Thick slab through the vessel along n (2 mm steps), 1.5 mm in-plane resampling.
Magnitude: mean over bins of the Tukey-filtered |REF|, mean through the slab. Velocity: for each MRI axis, phase of the
slab-summed product E_c x conj(REF) (magnitude-weighted), converted to velocity, then rotated into the view frame
(through-plane n, in-plane e1, e2). Noise: v22 display realisation (SNR 12, phase as for VENC 0.3 m/s; Koma ran at 2.5 m/s).
Tukey (alpha 0.5) k-space filter as in typical_4dflow_views_v2. Peak bin. Unmasked. Presentation only.
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

sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(1, "D:/MRSim/outputs/aorta_presentation_2026-10-25/code")
from plot_typical_4dflow_views import filt  # noqa: E402
FS, STEP, DEPTH = 1.5, 1.5e-3, 2.0e-3


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sample(vol, pts, A):
    """Trilinear sampling of a (Z, Y, X) volume (real or complex) at MRI points (m, columns x, y, z)."""
    kk = [(pts[:, a]-g[0])/(g[1]-g[0]) for a, g in enumerate((A.gx, A.gy, A.gz))]
    coords = [kk[2], kk[1], kk[0]]
    f = lambda v: ndimage.map_coordinates(v, coords, order=1, mode="constant", cval=0.0)  # noqa: E731
    return f(vol.real)+1j*f(vol.imag) if np.iscomplexobj(vol) else f(vol)


def main(target):
    target = Path(target)
    if target.exists():
        raise FileExistsError(target)
    from aorta_common import Aorta, VENC_NOISE
    A = Aorta()
    with h5py.File(Path("D:/MRSim/outputs/aorta_20bin_e062/analysis_v1/matched_reference_20bin.h5")) as f:
        ref = f["adc_time_average"][:]
    ba = int(np.argmax([np.linalg.norm(ref[k][A.core], axis=-1).mean() for k in range(A.nb)]))
    zi, yi, xi = np.nonzero(A.core)
    P = np.stack([A.gx[xi], A.gy[yi], A.gz[zi]], -1); w = A.mag.mean(axis=0)[A.core]
    c0 = (P*w[:, None]).sum(0)/w.sum()
    ev, vec = np.linalg.eigh(np.cov((P-c0).T, aweights=w))
    # v2: classic candy-cane orientation: vertical = largest-variance axis (descending aorta), arch (large MRI X) on top;
    # horizontal chosen so the ascending aorta is left of the descending (v1 laid the cane on its side).
    n, ev_ = vec[:, 0], vec[:, 2]
    asc, desc = np.array([.058, -.033, -.019]), np.array([.0, .027, .025])     # E-062 section centroids (x, y, z)
    e2 = ev_ if ev_[0] > 0 else -ev_                                            # vertical, arch up
    e1 = np.cross(e2, n)                                                        # horizontal
    if (asc-desc)@e1 > 0:
        e1, n = -e1, -n
    q = (P-c0)@np.stack([e1, e2, n]).T
    lo, hi = q.min(0)-np.array([.012, .012, .006]), q.max(0)+np.array([.012, .012, .006])
    s1 = np.arange(lo[0], hi[0], STEP); s2 = np.arange(lo[1], hi[1], STEP); s3 = np.arange(lo[2], hi[2], DEPTH)
    S1, S2, S3 = np.meshgrid(s1, s2, s3, indexing="ij")
    pts = c0+S1.ravel()[:, None]*e1+S2.ravel()[:, None]*e2+S3.ravel()[:, None]*n
    shp = S1.shape
    R = filt(A.REFall[ba])
    inside = sample(np.ones(R.shape), pts, A).reshape(shp).mean(axis=2) > .3           # v3: column mostly inside the FOV (v2 min>.999 cut the vessel)
    mag = sample(np.abs(filt(A.REFall)).mean(axis=0), pts, A).reshape(shp).mean(axis=2)
    vmri = []
    for c in range(3):
        E = filt(A.REFall[ba]*np.exp(1j*A.vnall[ba][..., c]*np.pi/VENC_NOISE))
        prod = sample(E*R.conj(), pts, A).reshape(shp).sum(axis=2)
        vmri.append(np.angle(prod)*VENC_NOISE/np.pi)
    vmri = np.stack(vmri, -1)                                                    # (s1, s2, 3) MRI x, y, z (m/s)
    vview = [np.where(inside, vmri@ax_*100, np.nan) for ax_ in (n, e1, e2)]       # cm/s; outside the FOV black
    # Anatomy labels: project landmark points into the view.
    hiX = P[P[:, 0] >= np.percentile(P[:, 0], 97)]                               # top of the vessel tree (branches)
    archc = P[(P[:, 0] >= np.percentile(P[:, 0], 85)) & (P[:, 0] < np.percentile(P[:, 0], 93))].mean(0)
    asc_pts = P[np.linalg.norm(P[:, 1:]-asc[1:], axis=1) < .02]
    inlet = asc_pts[np.argmin((asc_pts-c0)@e2)] if len(asc_pts) else asc
    outlet = P[np.argmin(P[:, 0])]
    marks = {"Inlet": inlet, "Ascending": asc, "Arch": archc, "Branches": hiX.mean(0), "Descending": desc, "Outlet": outlet}
    proj = {k: ((v-c0)@e1*1000, (v-c0)@e2*1000) for k, v in marks.items()}
    extent = [s1[0]*1000, s1[-1]*1000, s2[0]*1000, s2[-1]*1000]
    plt.rcParams["font.size"] = 10*FS
    out = target.with_name(target.name+".building"); out.mkdir(parents=True)
    # v3: one row of four tall panels (v2's 2x2 was 4.6k px tall), one shared colour bar for the three phase panels.
    asp = (extent[3]-extent[2])/(extent[1]-extent[0])
    fig, axes = plt.subplots(1, 4, figsize=(26, 26/4*asp*1.05+2.2), layout="constrained")
    ax = axes[0]
    ax.imshow(mag.T, origin="lower", extent=extent, cmap="gray", vmin=0, vmax=np.percentile(mag, 99.7), aspect="equal", interpolation="bilinear")
    cx, cy = np.mean([v[0] for v in proj.values()]), np.mean([v[1] for v in proj.values()])
    for k, (x, y) in proj.items():   # labels pushed outward from the tree centre, with a leader line
        d = np.array([x-cx, y-cy]); d = d/max(np.linalg.norm(d), 1e-6)
        tx = float(np.clip(x+30*d[0], extent[0]+4, extent[1]-4)); ty = float(np.clip(y+30*d[1], extent[2]+8, extent[3]-8))   # v4: keep labels inside
        ax.annotate(k, (x, y), xytext=(tx, ty), color="#ffd24d", fontsize=9*FS, fontweight="bold",
                    ha="left" if tx >= x else "right", va="center", arrowprops=dict(arrowstyle="-|>", color="#ffd24d", lw=1.4))
    ax.set_title("Magnitude", fontsize=10*FS)
    labs = ("PC through-plane", "PC in-plane horizontal", "PC in-plane vertical")
    V = VENC_NOISE*100
    for k in range(3):
        ax = axes[k+1]
        im = ax.imshow(vview[k].T, origin="lower", extent=extent, cmap="gray", vmin=-V, vmax=V, aspect="equal", interpolation="nearest")
        ax.set_title(labs[k], fontsize=10*FS)
    cb = fig.colorbar(im, ax=axes[1:].tolist(), shrink=.6, ticks=[-V, 0, V]); cb.set_label(f"cm/s (VENC {V:.0f} cm/s)", fontsize=9*FS)
    for ax in axes:
        ax.set_xlabel("horizontal (mm)", fontsize=8*FS); ax.set_ylabel("vertical (mm)", fontsize=8*FS)
        ax.tick_params(labelsize=8*FS); ax.set_facecolor("black")
    fig.suptitle(f"Aorta (E-062) Koma | oblique 'candy-cane' view, thick slab | peak bin {ba+1}/20 | noise SNR 12, as for VENC {VENC_NOISE} m/s",
                 fontsize=12*FS)
    fig.supxlabel("View plane fitted to the vessel (PCA); slab through the whole vessel. Velocity rotated into the view frame. "
                  "Tukey k-space filter. Display noise only; Koma ran at VENC 2.5 m/s.", fontsize=8*FS)
    fig.savefig(out/"aorta_4dflow_candycane.png", dpi=110); plt.close(fig)
    (out/"MANIFEST.json").write_text(json.dumps(dict(created_utc=datetime.now(timezone.utc).isoformat(), status="PRESENTATION_DERIVATIVE_ONLY",
        script_sha256=sha(__file__), peak_bin=ba+1, view_normal_mri=n.tolist(), e1=e1.tolist(), e2=e2.tolist(), pca_eigenvalues=ev.tolist(),
        landmarks_mm={k: list(v) for k, v in proj.items()}, outputs={p.name: sha(p) for p in out.glob("*.png")}), indent=1)+"\n", encoding="utf-8")
    out.rename(target)
    print("DONE", n, ev)


if __name__ == "__main__":
    main(sys.argv[1])
