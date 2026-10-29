"""Presentation: all-30-bin overview of support-weighted speed projections, CFD vs Koma,
phantom E-060 (and E-061 VENC4 when complete). Same projection, mask and orientation as
plot_projection_triptych.py (v2: per-bin 5 % support in lumen; rotated 90 deg CCW then
flipped L-R; P cropped to +/-16 mm) plus S cropped to the 60 mm excited slab. Common
all-bin scale per arm; native pixels; no interpolated cardiac phases.
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
sys.path.insert(0, str(E060/"code"))
import plot_projection_triptych as tri  # noqa: E402

SCROP = 31.25   # slab +/-30 mm to the partition edges
PAIRS_PER_ROW = 5


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(target):
    target = Path(target)
    if target.exists():
        raise FileExistsError(target)
    with h5py.File(E060/"references_v2/references.h5") as f:
        ref, frac, z = f["fft_adc"][:], f["fraction_fft"][:], f["partition_z_m"][:]*1e3
    speed_ref = np.linalg.norm(ref.astype(float), axis=-1)
    p_mm = (np.arange(64)-32)*1.25
    extent = [p_mm[0]-.625, p_mm[-1]+.625, z[0]-1.25, z[-1]+1.25]
    stage = target.with_name(target.name+".building"); stage.mkdir()
    meta = {}
    for name, (images, vencs) in tri.arms().items():
        field = np.stack([np.angle(images[e]*images["REF"].conj())*(vencs[i]/np.pi) for i, e in enumerate(tri.an.ENC[1:])], axis=-1)
        speed_k = np.linalg.norm(field, axis=-1)
        sup = np.ones(images["REF"].shape, bool)
        for im in images.values():
            m = np.abs(im); sup &= m >= .05*m.max(axis=(1, 2, 3), keepdims=True)
        cores = sup & (frac > 0)[None]
        nb = speed_ref.shape[0]
        proj = [(tri.project(speed_ref[b], cores[b], frac, 2), tri.project(speed_k[b], cores[b], frac, 2)) for b in range(nb)]
        inslab = np.abs(z) <= SCROP
        vmax = float(max(np.nanmax(np.where(inslab[:, None], a, np.nan)) for a, _ in proj))
        rows = int(np.ceil(nb/PAIRS_PER_ROW))
        fig, axes = plt.subplots(rows, 2*PAIRS_PER_ROW, figsize=(2*PAIRS_PER_ROW*1.55, rows*3.2), layout="constrained")
        cmap = plt.get_cmap("viridis").copy(); cmap.set_bad("#eeeeee")
        for b in range(nb):
            for j, (data, lab) in enumerate(zip(proj[b], ("CFD", "Koma"))):
                ax = axes[b//PAIRS_PER_ROW, 2*(b % PAIRS_PER_ROW)+j]
                image = ax.imshow(data, origin="lower", extent=extent, interpolation="nearest", cmap=cmap, vmin=0, vmax=vmax, aspect="equal")
                ax.set_facecolor("#eeeeee"); ax.set_xlim(-tri.PCROP, tri.PCROP); ax.set_ylim(-SCROP, SCROP)
                ax.set_title(f"{lab} | bin {b+1}", fontsize=8)
                ax.tick_params(labelsize=6)
                if b % PAIRS_PER_ROW == 0 and j == 0:
                    ax.set_ylabel("S, flow upward (mm)", fontsize=7)
                if b//PAIRS_PER_ROW == rows-1:
                    ax.set_xlabel("P (mm)", fontsize=7)
        fig.colorbar(image, ax=axes.ravel().tolist(), label="Speed (m/s)", shrink=.5)
        fig.suptitle(f"Phantom | {name.replace('_', ' ')} | R support-weighted speed projections | all 30 discrete bins")
        fig.supxlabel("Common all-bin scale; gray = excluded (per-bin 5% all-encoding support within lumen); no interpolated cardiac phases. "
                      "Rotated 90° CCW then flipped L-R; S cropped to the 60 mm slab.", fontsize=9)
        path = stage/f"phantom_speed_{name}_all_bins.png"
        fig.savefig(path, dpi=150); plt.close(fig)
        meta[path.name] = dict(arm=name, speed_max=vmax); print(path, flush=True)
    (stage/"MANIFEST.json").write_text(json.dumps(dict(created_utc=datetime.now(timezone.utc).isoformat(), status="PRESENTATION_DERIVATIVE_ONLY",
        script_sha256=sha(__file__), figures=meta, outputs={p.name: sha(p) for p in sorted(stage.glob("*.png"))}), indent=2)+"\n", encoding="utf-8")
    stage.rename(target)


if __name__ == "__main__":
    main(sys.argv[1])
