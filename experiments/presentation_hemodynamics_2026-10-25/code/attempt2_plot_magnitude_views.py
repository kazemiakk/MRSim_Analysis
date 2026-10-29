"""Koma reconstruction magnitude views (user request; layout of the dp150 'static KomaMRI reconstruction' figure).

Projections (mean |M| through the third axis) and centre slices in three orientations, grey scale.
Phantom: E-060 REF image (VENC 3 run), peak bin, grid S x P x R = 2.5 x 1.25 x 1.25 mm.
Aorta: E-062 reference magnitude (20 bins, 4 mm), peak bin. Display only.
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

E060 = Path("D:/MRSim/outputs/scanner_geometry_e060")
E062 = Path("D:/MRSim/outputs/aorta_20bin_e062")
sys.path.insert(0, str(E060/"code"))
FS = 1.5


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def views(fig, vol, axes_mm, names, title):
    """vol (a0, a1, a2); axes_mm coordinates per axis; names per axis."""
    a = [np.asarray(x) for x in axes_mm]
    ext = lambda i: [a[i][0], a[i][-1]]  # noqa: E731
    vmax = np.percentile(vol, 99.7)
    panels = [("Projection", vol.mean(axis=0), 2, 1), ("Projection", vol.mean(axis=1), 2, 0), ("Projection", vol.mean(axis=2), 1, 0),
              ("Centre slice", vol[vol.shape[0]//2], 2, 1), ("Centre slice", vol[:, vol.shape[1]//2], 2, 0), ("Centre slice", vol[:, :, vol.shape[2]//2], 1, 0)]
    panels = [p for i, p in enumerate(panels) if i % 3 != 1]   # v2: second column (axis-1 views) removed (user request)
    gs = fig.add_gridspec(2, 2)
    for k, (kind, img, h, v) in enumerate(panels):
        ax = fig.add_subplot(gs[k // 2, k % 2])
        pv = vmax if kind == "Centre slice" else np.percentile(img, 99.7)
        ax.imshow(img, origin="lower", cmap="gray", vmin=0, vmax=pv, extent=ext(h)+ext(v), aspect="equal", interpolation="nearest")
        ax.set_title(f"{kind} - {names[h][0]}{names[v][0]}", fontsize=10*FS)
        ax.set_xlabel(f"{names[h]} (mm)", fontsize=8*FS); ax.set_ylabel(f"{names[v]} (mm)", fontsize=8*FS); ax.tick_params(labelsize=8*FS)
    fig.suptitle(title, fontsize=13*FS)


def main(target):
    target = Path(target)
    if target.exists():
        raise FileExistsError(target)
    import plot_projection_triptych as tri
    images, _ = tri.arms()["E060_VENC3"]
    rep = json.loads((E060/"analysis_v1/report.json").read_text()); b = rep["peak_bin"]-1
    with h5py.File(E060/"references_v2/references.h5") as f:
        zz = f["partition_z_m"][:]*1e3
    ph = np.abs(images["REF"][b])                                                   # (S, P, R)
    pr = (np.arange(64)-32)*1.25
    plt.rcParams["font.size"] = 10*FS
    stage = target.with_name(target.name+".building"); stage.mkdir()
    fig = plt.figure(figsize=(17, 14), layout="constrained")
    views(fig, ph, (zz, pr, pr), ("S, slab (along vessel)", "P, phase", "R, readout"), f"Phantom (E-060) Koma reconstruction | REF magnitude | cardiac bin {b+1}/30")
    fig.savefig(stage/"phantom_e060_magnitude_views.png", dpi=110); plt.close(fig)
    with h5py.File(E062/"combined/simulation_motion_1m_20bin_4mm/pc_reconstruction.h5") as f:
        mag = f["reference_magnitude"][:]                                           # (bin, Z, Y, X)
    an = json.loads((E062/"analysis_v1/report.json").read_text())
    with h5py.File(E062/"analysis_v1/matched_reference_20bin.h5") as f:
        ref, core = f["adc_time_average"][:], f["masks/core15"][:].astype(bool)
    ba = int(np.argmax([np.linalg.norm(ref[k][core], axis=-1).mean() for k in range(mag.shape[0])]))
    nz, ny, nx = mag.shape[1:]
    ax_x = (np.arange(nx)-nx//2)*280/nx; ax_y = (np.arange(ny)-ny//2)*4.0; ax_z = (np.arange(nz)-nz//2)*4.0
    fig = plt.figure(figsize=(17, 13), layout="constrained")
    views(fig, mag[ba], (ax_z, ax_y, ax_x), ("Z, partition", "Y, phase", "X, readout"), f"Aorta (E-062) Koma reconstruction | REF magnitude | cardiac bin {ba+1}/20")
    fig.savefig(stage/"aorta_e062_magnitude_views.png", dpi=110); plt.close(fig)
    (stage/"MANIFEST.json").write_text(json.dumps(dict(created_utc=datetime.now(timezone.utc).isoformat(), status="PRESENTATION_DERIVATIVE_ONLY",
        script_sha256=sha(__file__), phantom_bin=b+1, aorta_bin=ba+1, outputs={p.name: sha(p) for p in stage.glob("*.png")}), indent=1)+"\n", encoding="utf-8")
    stage.rename(target)
    print("DONE", b+1, ba+1)


if __name__ == "__main__":
    main(sys.argv[1])
