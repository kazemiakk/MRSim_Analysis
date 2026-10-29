"""Presentation: smooth-boundary versions of the phantom projection figures.

Same projections as plot_projection_triptych.py (v2 rule: per-bin 5 % all-encoding support
in the lumen, fraction-weighted mean through R; rotated 90 deg CCW then flipped L-R).
Display only: the native 1.25 x 2.5 mm projections are nearest-filled inside the vessel,
bilinearly interpolated to 0.25 mm, and clipped to a smooth wall |P| <= r(S), with
r(S) = sqrt(section area / pi) from the E-061 REF3 (4x4) section area fractions, cubic-
interpolated along S. S cropped to the 60 mm slab. Numbers are unchanged; native-pixel
figures are in projections_v2 / projections_allbins_v1.
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
from scipy.optimize import curve_fit  # noqa: E402

E060 = Path("D:/MRSim/outputs/scanner_geometry_e060")
E061 = Path("D:/MRSim/outputs/venc_unwrap_e061")
sys.path.insert(0, str(E060/"code"))
import plot_projection_triptych as tri  # noqa: E402

SCROP, PCROP, STEP = 30.0, 16.0, 0.25
PF = np.arange(-PCROP, PCROP+STEP/2, STEP)
SF = np.arange(-SCROP, SCROP+STEP/2, STEP)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


WALL_FIT = {}


def wall(z):
    """v2: Gaussian stenosis fit r(S) = r0 - a exp(-(S-s0)^2/(2 sigma^2)) to the section radii (v1 spline
    interpolation made a cusp at the throat partition, whose quadrature area reads ~6 % low)."""
    with h5py.File(E061/"references_v1/references.h5") as f:
        pf = f["plane_fraction"][:]
    r = np.sqrt(pf.sum(axis=(1, 2))*1.5625/np.pi)
    ok = (r > 0) & (np.abs(z) <= 37.5)
    g = lambda s, r0, a, s0, sig: r0-a*np.exp(-(s-s0)**2/(2*sig**2))
    par, _ = curve_fit(g, z[ok], r[ok], p0=(13, 8.5, 0, 12))
    WALL_FIT.update(r0_mm=float(par[0]), depth_mm=float(par[1]), centre_mm=float(par[2]), sigma_mm=float(par[3]),
                    rms_residual_mm=float(np.sqrt(np.mean((g(z[ok], *par)-r[ok])**2))))
    print("WALL_FIT", WALL_FIT, flush=True)
    return g(SF, *par)


def smooth(img, z, rS):
    """Native (S,P) projection -> fine (SF,PF) image clipped to |P| <= r(S)."""
    filled = img.copy()
    bad = ~np.isfinite(filled)
    if bad.all():
        return np.full((SF.size, PF.size), np.nan)
    idx = ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True)
    filled = filled[tuple(idx)]
    p_mm = (np.arange(img.shape[1])-32)*1.25
    rows = np.interp(SF, z, np.arange(z.size))
    cols = np.interp(PF, p_mm, np.arange(p_mm.size))
    R, C = np.meshgrid(rows, cols, indexing="ij")
    out = ndimage.map_coordinates(filled, [R, C], order=1, mode="nearest")
    return np.where(np.abs(PF)[None, :] <= rS[:, None], out, np.nan)


# v3: displayed rotated 90 deg clockwise from v2: S (flow) left to right, P vertical with +P down.
def outline(ax, rS):
    ax.plot(SF, rS, color="#3a3936", lw=.8); ax.plot(SF, -rS, color="#3a3936", lw=.8)


def show(ax, data, extent, **kw):
    im = ax.imshow(data.T, origin="lower", extent=[extent[2], extent[3], extent[0], extent[1]],
                   interpolation="nearest", aspect="equal", **kw)
    ax.set_ylim(extent[1], extent[0])  # +P down after the clockwise rotation
    return im


def main(target):
    target = Path(target)
    if target.exists():
        raise FileExistsError(target)
    with h5py.File(E060/"references_v2/references.h5") as f:
        ref, frac, z = f["fft_adc"][:], f["fraction_fft"][:], f["partition_z_m"][:]*1e3
    rS = wall(z)
    speed_ref = np.linalg.norm(ref.astype(float), axis=-1)
    extent = [PF[0]-STEP/2, PF[-1]+STEP/2, SF[0]-STEP/2, SF[-1]+STEP/2]
    stage = target.with_name(target.name+".building"); stage.mkdir()
    cmap = plt.get_cmap("viridis").copy(); cmap.set_bad("#eeeeee")
    dmap = plt.get_cmap("RdBu_r").copy(); dmap.set_bad("#eeeeee")
    meta = {}
    for name, (images, vencs) in tri.arms().items():
        field = np.stack([np.angle(images[e]*images["REF"].conj())*(vencs[i]/np.pi) for i, e in enumerate(tri.an.ENC[1:])], axis=-1)
        speed_k = np.linalg.norm(field, axis=-1)
        sup = np.ones(images["REF"].shape, bool)
        for im in images.values():
            m = np.abs(im); sup &= m >= .05*m.max(axis=(1, 2, 3), keepdims=True)
        cores = sup & (frac > 0)[None]
        nb = speed_ref.shape[0]
        proj = [(smooth(tri.project(speed_ref[b], cores[b], frac, 2), z, rS), smooth(tri.project(speed_k[b], cores[b], frac, 2), z, rS)) for b in range(nb)]
        vmax = float(max(np.nanmax(a) for a, _ in proj))
        for b in tri.BINS:
            a, k = proj[b-1]; d = k-a; dmax = float(np.nanmax(np.abs(d)))
            fig, axes = plt.subplots(3, 1, figsize=(9, 9.5), layout="constrained")
            for ax, data, lab, cm, lim in zip(axes, (a, k, d), ("CFD", "Koma", "Koma minus CFD"), (cmap, cmap, dmap), ((0, vmax), (0, vmax), (-dmax, dmax))):
                im = show(ax, data, extent, cmap=cm, vmin=lim[0], vmax=lim[1])
                outline(ax, rS); ax.axvline(0, color="#555555", lw=.6, ls=(0, (3, 3)))
                ax.set_facecolor("#eeeeee"); ax.set_title(lab, fontsize=10)
                ax.set(xlabel="S, along vessel (mm; flow left to right)", ylabel="P (mm)")
                fig.colorbar(im, ax=ax, label="m/s", shrink=.9)
            fig.suptitle(f"Phantom speed | {name.replace('_', ' ')} | R support-weighted projection | bin {b}/30", fontsize=13)
            fig.supxlabel("Display: bilinear 0.25 mm interpolation clipped to a Gaussian fit of the CFD wall; values unchanged. "
                          "Flow left to right (rotated 90° clockwise from v2); 60 mm slab. Dashed: throat.", fontsize=8)
            path = stage/f"phantom_speed_smooth_{name}_bin{b:02d}.png"; fig.savefig(path, dpi=150); plt.close(fig)
            meta[path.name] = dict(arm=name, bin=b); print(path, flush=True)
        rows = 10
        fig, axes = plt.subplots(rows, 6, figsize=(16, rows*1.55), layout="constrained")
        for b in range(nb):
            for j, (data, lab) in enumerate(zip(proj[b], ("CFD", "Koma"))):
                ax = axes[b//3, 2*(b % 3)+j]
                image = show(ax, data, extent, cmap=cmap, vmin=0, vmax=vmax)
                outline(ax, rS); ax.set_facecolor("#eeeeee"); ax.set_title(f"{lab} | bin {b+1}", fontsize=8); ax.tick_params(labelsize=6)
                if b % 3 == 0 and j == 0:
                    ax.set_ylabel("P (mm)", fontsize=7)
                if b//3 == rows-1:
                    ax.set_xlabel("S, flow left to right (mm)", fontsize=7)
        fig.colorbar(image, ax=axes.ravel().tolist(), label="Speed (m/s)", shrink=.5)
        fig.suptitle(f"Phantom | {name.replace('_', ' ')} | R support-weighted speed projections | all 30 discrete bins")
        fig.supxlabel("Common all-bin scale. Display: bilinear 0.25 mm interpolation clipped to a Gaussian fit of the CFD wall; values unchanged. "
                      "Flow left to right; 60 mm slab.", fontsize=9)
        path = stage/f"phantom_speed_smooth_{name}_all_bins.png"; fig.savefig(path, dpi=150); plt.close(fig)
        meta[path.name] = dict(arm=name, speed_max=vmax); print(path, flush=True)
    (stage/"MANIFEST.json").write_text(json.dumps(dict(created_utc=datetime.now(timezone.utc).isoformat(), status="PRESENTATION_DERIVATIVE_ONLY",
        script_sha256=sha(__file__), wall_source="Gaussian fit to E-061 references_v1 plane_fraction (4x4) radii", wall_fit=WALL_FIT, figures=meta,
        outputs={p.name: sha(p) for p in sorted(stage.glob("*.png"))}), indent=2)+"\n", encoding="utf-8")
    stage.rename(target)


if __name__ == "__main__":
    main(sys.argv[1])
