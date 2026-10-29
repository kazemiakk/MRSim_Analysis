"""E-064 S2: candy-cane-aligned aorta figure (flow + stationary tissue, exact superposition).

Images (bin, Z'=normal, Y'=horizontal, X'=vertical/readout). Orientation gate: the time-mean flow REF magnitude must
correlate best with the rotated-CFD occupancy without any axis flip. Tissue scale = 0.8 x blood/tissue spin density
(frame_v1). Display: flow phase re-encoded as for VENC 0.3 m/s (Koma ran at 2.5 m/s), complex noise SNR 12 (median
lumen |REF|), Tukey k-space filter. Row 1: slice perpendicular to the vertical axis through the ascending aorta level
(ascending + descending cross-sections). Row 2: 20 mm candy-cane slab centred on the plane through the descending axis
and the ascending centroid (magnitude-weighted complex sum). Presentation only.
"""
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import sys
import tomllib

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, "D:/MRSim/MRSIM/MRSimTracks")
from scripts.aorta_hemodynamic_reference import AortaGrid  # noqa: E402  (frozen, read-only)

ROOT = Path("D:/MRSim/outputs/aorta_aligned_e064")
CLOUD = Path("D:/MRSim/MRSIM/external_inputs/aorta_pointcloud/processed/dp2")
FS, SEED, VENC, VD = 1.5, 20261025, 2.5, 0.3


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def jld_image(path):
    with h5py.File(path) as f:
        d = f["image"][()]
    return (d["re"]+1j*d["im"]).astype(np.complex64)


def rel(a, b):
    return float(np.sqrt(np.sum(np.abs(a-b)**2)/np.sum(np.abs(b)**2)))


def tukey3(img, alpha=.5):
    w = np.ones(img.shape[-3:])
    for ax, n in enumerate(img.shape[-3:]):
        u = np.abs(np.fft.fftfreq(n))/.5
        wa = np.where(u <= 1-alpha, 1., .5*(1+np.cos(np.pi*(u-(1-alpha))/alpha)))
        sh = [1, 1, 1]; sh[ax] = n; w = w*wa.reshape(sh)
    return np.fft.ifftn(np.fft.fftn(img, axes=(-3, -2, -1))*w, axes=(-3, -2, -1))


def row(axes, mag, vel, title, ext, xl, yl):
    V = VD*100
    axes[0].imshow(mag, cmap="gray", vmin=0, vmax=np.percentile(mag, 99.7), extent=ext, origin="lower", interpolation="nearest")
    axes[0].set_title(f"{title}\nMagnitude", fontsize=9.5*FS)
    labels = ("Through-plane", "In-plane horizontal", "In-plane vertical")
    im = None
    for k in range(3):
        im = axes[k+1].imshow(vel[k], cmap="gray", vmin=-V, vmax=V, extent=ext, origin="lower", interpolation="nearest")
        axes[k+1].set_title(f"{labels[k]}\n±{V:.0f} cm/s", fontsize=9.5*FS)
    for ax in axes:
        ax.set_xlabel(xl, fontsize=8*FS); ax.set_ylabel(yl, fontsize=8*FS); ax.tick_params(labelsize=7*FS)
    return im


def main(target):
    target = Path(target)
    if target.exists():
        raise FileExistsError(target)
    fr = json.loads((ROOT/"frame_v1/frame.json").read_text())
    R, c = np.array(fr["rotation_rows_e2_e1_n"]), np.array(fr["centre_mri_m"])
    fov, mat = np.array(fr["fov_m"]), np.array(fr["matrix"]); vox = fov/mat
    fdir = ROOT/"flow_combined/simulation_motion_1m_20bin_candycane"; tdir = ROOT/"tissue_combined/simulation_static_tissue_20bin_candycane"
    for d in (fdir, tdir):
        if tomllib.loads((d/"simulation_report.toml").read_text())["status"] != "PASS":
            raise ValueError("simulation not passed: "+str(d))
    F = {e: jld_image(fdir/f"{e.lower()}_result.jld2") for e in ("REF", "X", "Y", "Z")}
    S = {e: jld_image(tdir/f"{e.lower()}_result.jld2") for e in ("REF", "X", "Y", "Z")}
    rep = dict(experiment="E-064", stage="S2_figure", utc=datetime.now(timezone.utc).isoformat(), script_sha256=sha(__file__),
               image_shape=list(F["REF"].shape))
    mref = np.abs(S["REF"]); big = mref > .2*mref.max()
    rep["tissue_encoding_vs_REF"] = {e: dict(rel_L2=rel(S[e], S["REF"]), phase_p99_rad=float(np.percentile(np.abs(np.angle(S[e]*S["REF"].conj())[big]), 99)))
                                     for e in ("X", "Y", "Z")}
    if any(v["rel_L2"] > 1e-2 or v["phase_p99_rad"] > 1e-2 for v in rep["tissue_encoding_vs_REF"].values()):
        raise ValueError("tissue encodings differ from REF (E-063 A1 gate)")
    # Orientation gate against the rotated CFD occupancy.
    g = AortaGrid(); order, cen = np.array(g.order), np.array(g.center)
    src = np.load(CLOUD/"point_coordinates.npy", allow_pickle=False).astype(np.float64)
    q = ((src[:, order]-cen[order])-c)@R.T                                       # (x', y', z')
    idx = [np.clip(np.round(q[:, k]/vox[k]+mat[k]//2).astype(int), 0, mat[k]-1) for k in range(3)]
    occ = np.zeros(F["REF"].shape[1:], float); np.add.at(occ, (idx[2], idx[1], idx[0]), 1.0)
    occ = occ > 0
    mflow = np.abs(F["REF"]).mean(axis=0)
    corr = {}
    for flips in itertools.product((False, True), repeat=3):
        m = mflow
        for ax, f in enumerate(flips):
            m = np.flip(m, axis=ax) if f else m
        corr[str(flips)] = float(np.corrcoef(m.ravel(), occ.ravel())[0, 1])
    rep["orientation_correlation"] = corr
    if max(corr, key=corr.get) != str((False, False, False)):
        raise ValueError("image orientation differs from the rotated geometry: "+max(corr, key=corr.get))
    a = fr["tissue_scale_factor"]
    tot = {"REF": F["REF"]+a*S["REF"]}
    for e in ("X", "Y", "Z"):
        phi = np.angle(F[e]*F["REF"].conj())
        tot[e] = F[e]*np.exp(1j*(phi*VENC/VD-phi))+a*S[e]
    lumen = mflow >= .15*np.percentile(mflow, 99.9)
    sigma = float(np.median(np.abs(tot["REF"]).mean(axis=0)[lumen]))/12.0
    rng = np.random.default_rng(SEED)
    noisy = {e: tukey3(v+sigma*(rng.standard_normal(v.shape)+1j*rng.standard_normal(v.shape))) for e, v in tot.items()}
    peak = int(np.argmax([np.abs(np.angle(F["X"][b]*F["REF"][b].conj())[lumen]).mean() for b in range(F["REF"].shape[0])]))
    mag = np.abs(noisy["REF"]).mean(axis=0)
    C = {e: noisy[e][peak]*noisy["REF"][peak].conj() for e in ("X", "Y", "Z")}
    vel = lambda cc: np.angle(cc)*VD*100/np.pi  # noqa: E731
    xs = (np.arange(mat[0])-mat[0]//2)*vox[0]*1e3; ys = (np.arange(mat[1])-mat[1]//2)*vox[1]*1e3; zs = (np.arange(mat[2])-mat[2]//2)*vox[2]*1e3
    asc = (np.array([.058, -.033, -.019])-c)@R.T*1e3                             # ascending centroid in (x', y', z') mm
    ix = int(np.argmin(abs(xs-asc[0])))
    sl = slice(int(np.argmin(abs(zs-(asc[2]-10)))), int(np.argmin(abs(zs-(asc[2]+10))))+1)
    plt.rcParams["font.size"] = 10*FS
    stage = target.with_name(target.name+".building"); stage.mkdir()
    fig, axes = plt.subplots(2, 4, figsize=(26, 17), layout="constrained", gridspec_kw=dict(height_ratios=[1, 2.2]))
    # Row 1: slice perpendicular to the vertical (readout) axis -> image (Z' rows, Y' cols).
    # v2: outer voxel cropped on every edge (the tissue block stops 2 mm inside the FOV, which drew a frame).
    cr = lambda a: a[1:-1, 1:-1]  # noqa: E731
    ext1 = [ys[1]-2, ys[-2]+2, zs[1]-2, zs[-2]+2]
    row(axes[0], cr(mag[:, :, ix]), [cr(vel(C["X"][:, :, ix])), cr(vel(C["Y"][:, :, ix])), cr(vel(C["Z"][:, :, ix]))],
        f"Slice across the ascending and descending aorta (vertical = {xs[ix]:.0f} mm)", ext1, "horizontal (mm)", "depth (mm)")
    for ax in axes[0][1:]:
        ax.set_title(ax.get_title().replace("Through-plane", "Through-plane (vertical flow)").replace("In-plane vertical", "In-plane depth"), fontsize=9.5*FS)
    # Row 2: candy-cane slab -> image (X' rows = vertical, Y' cols = horizontal).
    ext2 = [ys[1]-2, ys[-2]+2, xs[1]-2, xs[-2]+2]
    cs = {e: cr(C[e][sl].sum(axis=0).T) for e in C}
    im = row(axes[1], cr(mag[sl].mean(axis=0).T), [vel(cs["Z"]), vel(cs["Y"]), vel(cs["X"])], f"Candy-cane slab, {(sl.stop-sl.start)*vox[2]*1e3:.0f} mm",
             ext2, "horizontal (mm)", "vertical (mm)")
    # v2: anatomy labels from the rotated CFD nodes (x' vertical, y' horizontal, mm).
    qm = q*1e3
    top = qm[qm[:, 0] >= np.percentile(qm[:, 0], 98)]
    arch = qm[(qm[:, 0] >= np.percentile(qm[:, 0], 88)) & (qm[:, 0] < np.percentile(qm[:, 0], 94))]
    near_asc = qm[(np.abs(qm[:, 1]-asc[1]) < 20) & (qm[:, 0] < asc[0])]
    inlet = near_asc[np.argmin(near_asc[:, 0])]
    desc = qm[np.abs(qm[:, 0]+60) < 3]; desc = desc[desc[:, 1] > asc[1]+20]
    marks = {"Branches": (np.median(top[:, 1]), np.median(top[:, 0])), "Arch": (np.median(arch[:, 1]), np.median(arch[:, 0])),
             "Ascending": (asc[1], asc[0]), "Inlet (sparse spins)": (inlet[1], inlet[0]+4),
             "Descending": (np.median(desc[:, 1]), -60.0)}
    off = {"Branches": (25, 18), "Arch": (30, 15), "Ascending": (-12, 30), "Inlet (sparse spins)": (35, -40), "Descending": (-35, -30)}
    for k, (xh, yv) in marks.items():
        axes[1][0].annotate(k, (xh, yv), xytext=(xh+off[k][0], yv+off[k][1]), color="#ffd24d", fontsize=8.5*FS, fontweight="bold",
                            ha="center", arrowprops=dict(arrowstyle="-|>", color="#ffd24d", lw=1.4))
    rep["anatomy_marks_mm"] = {k: [float(v) for v in xy] for k, xy in marks.items()}
    fig.colorbar(im, ax=axes[:, 1:].ravel().tolist(), shrink=.5, label=f"cm/s (display VENC {VD*100:.0f} cm/s)")
    fig.suptitle(f"Aorta Koma, acquired in the candy-cane orientation (E-064) | peak bin {peak+1}/20 | noise SNR 12", fontsize=12*FS)
    fig.supxlabel("Scan volume aligned with the aorta (readout along the descending aorta). Stationary tissue block (synthetic; T1 0.9 s, T2 45 ms, PD 0.8 x blood) "
                  "added by exact superposition.\nFlow phase shown as for VENC 0.3 m/s (Koma ran at 2.5 m/s; display re-encoding). Tukey k-space filter.", fontsize=8*FS)
    fig.savefig(stage/"aorta_candycane_acquired_v2.png", dpi=110); plt.close(fig)
    rep.update(tissue_scale=a, noise_sigma=sigma, peak_bin=peak+1, slab_partitions=[sl.start, sl.stop], slice_index_vertical=ix, status="PRESENTATION_DERIVATIVE_ONLY")
    (stage/"report.json").write_text(json.dumps(rep, indent=1)+chr(10), encoding="utf-8")
    (stage/"MANIFEST.json").write_text(json.dumps({p.name: sha(p) for p in sorted(stage.iterdir())}, indent=1)+chr(10), encoding="utf-8")
    stage.rename(target)
    print("DONE", peak+1, max(corr, key=corr.get), corr[str((False, False, False))])


if __name__ == "__main__":
    main(sys.argv[1])
