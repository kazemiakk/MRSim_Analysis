"""E-064 S0a: candy-cane frame, rotation, FOV margins and stationary-tissue spin positions (new scanner frame).

Writes frame_v1/{frame.json, tissue_spins.h5}. New coordinates: (x', y', z') = (e2, e1, n) . (r - c) with r in the
E-062 MRI frame (metres).
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, "D:/MRSim/MRSIM/MRSimTracks")
from scripts.aorta_hemodynamic_reference import AortaGrid  # noqa: E402  (frozen, read-only)

ROOT = Path("D:/MRSim/outputs/aorta_aligned_e064")
E063 = Path("D:/MRSim/outputs/static_background_e063")
CLOUD = Path("D:/MRSim/MRSIM/external_inputs/aorta_pointcloud/processed/dp2")
FOV = (0.288, 0.160, 0.096)
MATRIX = (72, 40, 24)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def main(target):
    target = Path(target)
    if target.exists():
        raise FileExistsError(target)
    ax = json.loads((E063/"candycane_v2/report.json").read_text())["candycane_axes"]
    e2, e1 = np.array(ax["vertical"]), np.array(ax["horizontal"])
    n = np.cross(e2, e1)                                                        # det[e2, e1, n] = +1
    R = np.stack([e2, e1, n])
    g = AortaGrid(); order, cen = np.array(g.order), np.array(g.center)
    src = np.load(CLOUD/"point_coordinates.npy", allow_pickle=False).astype(np.float64)
    P = src[:, order]-cen[order]
    q = P@R.T
    c = ((q.min(0)+q.max(0))/2)@R                                               # centre in the MRI frame
    qn = (P-c)@R.T
    half = np.array(FOV)/2
    margin = half-np.abs(qn).max(0)
    voxel = np.array(FOV)/np.array(MATRIX)
    # Stationary tissue on a 2x2x2 sub-voxel grid in the new frame (centred FFT grid), 2 mm inside the FOV,
    # excluded within 2 mm of any rotated CFD node.
    axes = []
    for k in range(3):
        cen_k = (np.arange(MATRIX[k])-MATRIX[k]//2)*voxel[k]
        a = np.sort(np.concatenate([cen_k+s*voxel[k] for s in (-.25, .25)]))
        lo, hi = cen_k[0]-voxel[k]/2+2e-3, cen_k[-1]+voxel[k]/2-2e-3
        axes.append(a[(a >= lo) & (a <= hi)])
    T = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    d, _ = cKDTree(qn).query(T, k=1, workers=-1, distance_upper_bound=2.5e-3)
    T = T[d > 2e-3]
    lumen_ml = 273.531
    blood_density = 1_000_000/(lumen_ml*1e3); tissue_density = 8/(np.prod(voxel)*1e9)
    stage = target.with_name(target.name+".building"); stage.mkdir(parents=True)
    with h5py.File(stage/"tissue_spins.h5", "w") as f:
        for i, k in enumerate("xyz"):
            f[k] = T[:, i]
    rep = dict(experiment="E-064", stage="S0a_frame", utc=datetime.now(timezone.utc).isoformat(), script_sha256=sha(__file__),
               protocol_sha256=sha(ROOT/"code/PROTOCOL.md"), rotation_rows_e2_e1_n=R.tolist(), det=float(np.linalg.det(R)),
               orthonormality_error=float(np.abs(R@R.T-np.eye(3)).max()), centre_mri_m=c.tolist(),
               fov_m=list(FOV), matrix=list(MATRIX), voxel_m=voxel.tolist(), node_extent_new_m=[qn.min(0).tolist(), qn.max(0).tolist()],
               half_margin_m=margin.tolist(), tissue_spins=int(len(T)), blood_spin_density_per_mm3=blood_density,
               tissue_spin_density_per_mm3=tissue_density, pd_ratio=0.8, tissue_scale_factor=0.8*blood_density/tissue_density)
    rep["gates"] = dict(proper_rotation=abs(rep["det"]-1) < 1e-9 and rep["orthonormality_error"] < 1e-9, node_margin_ge_8mm=bool((margin >= 8e-3).all()))
    rep["status"] = "PASS_E064_S0A_FRAME" if all(rep["gates"].values()) else "FAIL_E064_S0A_FRAME"
    (stage/"frame.json").write_text(json.dumps(rep, indent=1)+"\n", encoding="utf-8")
    stage.rename(target)
    print(rep["status"], rep["gates"], "margin mm", np.round(margin*1e3, 1), "tissue", len(T), "centre mm", np.round(c*1e3, 2))


if __name__ == "__main__":
    main(sys.argv[1])
