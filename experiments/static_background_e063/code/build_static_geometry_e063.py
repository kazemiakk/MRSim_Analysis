"""E-063 S0: stationary-spin geometry for the phantom (bottles from the scan) and the aorta (tissue block).

Writes geometry_v1/{phantom_static_spins.h5, aorta_static_spins.h5, report.json}. Positions in metres in each Koma
phantom frame. No scan identifiers are read or written (only the de-identified magnitude/velocity arrays).
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

ROOT = Path("D:/MRSim/outputs/static_background_e063")
E055 = Path("D:/MRSim/outputs/phantom_density_e055")
E059 = Path("D:/MRSim/outputs/measured_scan_e059")
PRIVATE = Path("D:/MRSim/private_inputs/phantom_scan_2025-08-28/measured_v1/measured_30phase.npz")
CLOUD = Path("D:/MRSim/MRSIM/external_inputs/aorta_pointcloud/processed/dp2")
PIX, SLAB_HALF, FOV_HALF = 1.25e-3, 0.036, 0.040


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def phantom():
    rep = json.loads((E059/"measured_v2/report.json").read_text())
    z = np.load(PRIVATE)
    v, mag = z["velocity_cm_s"], z["magnitude"]
    sign = rep["segmentation"]["0.25"]["vendor_to_forward_sign"]
    m = mag.mean(axis=1); thr = .25*np.percentile(m, 99.5)
    vessel_c, static = [], np.zeros(m.shape, bool)
    for s in range(m.shape[0]):
        lab, n = ndimage.label(m[s] >= thr)
        flows = [abs((sign*v[s, :, :, :, 2][:, lab == c]).sum(axis=1).mean()) for c in range(1, n+1)]
        ves = 1+int(np.argmax(flows))
        vessel_c.append(ndimage.center_of_mass(lab == ves))
        for c in range(1, n+1):
            if c != ves and (lab == c).sum() >= 20:
                static[s] |= lab == c
    rc, cc = np.mean(vessel_c, axis=0)
    mask2d = static.mean(axis=0) >= .5
    bottle_med = float(np.median(m[static])); vessel_med = float(np.median(np.concatenate(
        [m[s][ndimage.label(m[s] >= thr)[0] == ndimage.label(m[s] >= thr)[0][int(round(vessel_c[s][0])), int(round(vessel_c[s][1]))]]
         for s in range(m.shape[0])])))
    # Tube axis in the Koma frame: E-060 rotation (x, y, z)_new = (y, z, x - x_throat); E055 dx/dy/dz are absolute paths.
    with h5py.File(E055/"inputs_v1/case_inputs.h5") as f:
        dy0, dz0 = f["dy"][0, :], f["dz"][0, :]                            # (time, spin): t = 0 row
        rho = f["rho"][:]
    keep = rho > 0
    ax_x, ax_y = float(np.median(dy0[keep])), float(np.median(dz0[keep]))
    rows, cols = np.nonzero(mask2d)
    sub = (np.arange(2)+.5)/2-.5
    XY = np.stack(np.meshgrid(np.arange(len(cols)), np.arange(2), np.arange(2), indexing="ij"), -1).reshape(-1, 3)
    x = ax_x+((cols[XY[:, 0]]+sub[XY[:, 1]])-cc)*PIX
    y = ax_y+((rows[XY[:, 0]]+sub[XY[:, 2]])-rc)*PIX
    zc = (np.arange(30)-15)*2.5e-3
    zc = zc[np.abs(zc) <= SLAB_HALF]
    X = np.repeat(x, zc.size); Y = np.repeat(y, zc.size); Z = np.tile(zc, x.size)
    lim = FOV_HALF-PIX
    ok = (np.abs(X) <= lim) & (np.abs(Y) <= lim)
    X, Y, Z = X[ok], Y[ok], Z[ok]
    # Distance to the nearest blood spin (lumen exclusion check, t = 0 positions in the Koma frame).
    with h5py.File(E055/"inputs_v1/case_inputs.h5") as f:
        dx0 = f["dx"][0, :]
    from_info = json.loads((Path("D:/MRSim/outputs/scanner_geometry_e060")/"phantom_v1/report.json").read_text())
    blood = np.stack([dy0[keep], dz0[keep], dx0[keep]+from_info["transform"]["translation_new_z_m"]], -1)   # attempts 1-2: key path / syntax
    near = blood[np.abs(blood[:, 2]) <= SLAB_HALF+.005]
    dmin = float(cKDTree(near).query(np.stack([X, Y, Z], -1)[::50], k=1)[0].min())
    return dict(x=X, y=Y, z=Z, report=dict(spins=int(X.size), scan_vessel_centre_row_col=[float(rc), float(cc)],
                bottle_pixels_2d=int(mask2d.sum()), partitions=int(zc.size), koma_tube_axis_xy_m=[ax_x, ax_y],
                extent_m=[[float(X.min()), float(X.max())], [float(Y.min()), float(Y.max())], [float(Z.min()), float(Z.max())]],
                min_distance_to_blood_m=dmin, scan_bottle_to_vessel_magnitude_ratio=bottle_med/vessel_med,
                scan_bottle_median=bottle_med, scan_vessel_median=vessel_med, mapping="scan rows -> Koma y (P), columns -> Koma x (R)"))


def aorta():
    sys.path.insert(0, "D:/MRSim/MRSIM/MRSimTracks")
    from scripts.aorta_hemodynamic_reference import AortaGrid
    g = AortaGrid(); order, cen = np.array(g.order), np.array(g.center)
    src = np.load(CLOUD/"point_coordinates.npy", allow_pickle=False).astype(np.float64)
    nodes = src[:, order]-cen[order]; tree = cKDTree(nodes)
    gx, gy, gz = g.axes(); sp = g.spacing
    sub = (np.arange(2)+.5)/2-.5
    ax = [np.concatenate([ax_+s*d for s in sub]) for ax_, d in ((gx, sp[0]), (gy, sp[1]), (gz, sp[2]))]   # attempt 3: scalar concat
    ax = [np.sort(a) for a in ax]
    lo = [a.min()-d/2+2e-3 for a, d in zip((gx, gy, gz), sp)]; hi = [a.max()+d/2-2e-3 for a, d in zip((gx, gy, gz), sp)]
    ax = [a[(a >= l) & (a <= h)] for a, l, h in zip(ax, lo, hi)]
    P = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
    d, _ = tree.query(P, k=1, workers=-1, distance_upper_bound=2.5e-3)
    P = P[d > 2e-3]
    lumen_ml = 273.531                                                           # native lumen (energy_v1, d <= 1 mm)
    blood_density = 1_000_000/(lumen_ml*1e3)                                     # spins per mm^3
    tissue_density = 8/(np.prod(sp)*1e9)
    return dict(x=P[:, 0], y=P[:, 1], z=P[:, 2], report=dict(spins=int(len(P)), voxel_mm=(sp*1e3).tolist(),
                extent_m=[[float(P[:, i].min()), float(P[:, i].max())] for i in range(3)],
                blood_spin_density_per_mm3=blood_density, tissue_spin_density_per_mm3=tissue_density,
                pd_ratio_tissue_to_blood=0.8, scale_factor=0.8*blood_density/tissue_density, T1_s=0.9, T2_s=0.045))


def main(target):
    target = Path(target)
    if target.exists():
        raise FileExistsError(target)
    stage = target.with_name(target.name+".building"); stage.mkdir(parents=True)
    rep = dict(experiment="E-063", stage="S0_geometry", utc=datetime.now(timezone.utc).isoformat(), script_sha256=sha(__file__),
               protocol_sha256=sha(ROOT/"code/PROTOCOL.md"))
    for name, fn in (("phantom", phantom), ("aorta", aorta)):
        g = fn()
        with h5py.File(stage/f"{name}_static_spins.h5", "w") as f:
            for k in ("x", "y", "z"):
                f[k] = g[k].astype(np.float64)
        rep[name] = g["report"]; print(name, g["report"], flush=True)
    ph = rep["phantom"]; ao = rep["aorta"]
    rep["gates"] = dict(phantom_inside_fov=max(abs(v) for e in ph["extent_m"][:2] for v in e) <= FOV_HALF-PIX+1e-12,
                        phantom_clear_of_blood=ph["min_distance_to_blood_m"] > 1e-3, aorta_spins_positive=ao["spins"] > 0)
    rep["status"] = "PASS_E063_S0_GEOMETRY" if all(rep["gates"].values()) else "FAIL_E063_S0_GEOMETRY"
    (stage/"report.json").write_text(json.dumps(rep, indent=1)+"\n", encoding="utf-8")
    (stage/"MANIFEST.json").write_text(json.dumps({p.name: sha(p) for p in stage.iterdir()}, indent=1)+"\n", encoding="utf-8")
    stage.rename(target)
    print(rep["status"], rep["gates"])


if __name__ == "__main__":
    main(sys.argv[1])
