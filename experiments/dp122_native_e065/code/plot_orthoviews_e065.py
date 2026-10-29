"""E-065 magnitude orthoviews (layout of the historical dp150 static figure): projections and centre slices of the
dp122 REF magnitude at a given cardiac bin. Noiseless Koma reconstruction; no added noise, smoothing or mask."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tomllib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path("D:/MRSim/outputs/dp122_native_e065")
sys.path.insert(0, "D:/MRSim/outputs/scanner_geometry_e060/code")
import analyze_e060 as an  # noqa: E402

FS = 1.5


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(target, bins):
    target = Path(target); stage = target.with_name(target.name+".building")
    if target.exists() or stage.exists():
        raise FileExistsError(target)
    plt.rcParams["font.size"] = 10*FS
    control = tomllib.loads((ROOT/"acquisition_v1/control_report.toml").read_text())
    images, _ = an.load_images(ROOT/"acquisition_v1", control, ROOT/"sequences_v1/timing_e060.h5")
    mag_all = np.abs(images["REF"])                                   # (bin, S, P, R)
    s = (np.arange(30)-15)*2.5; p = (np.arange(64)-32)*1.25
    stage.mkdir(); out = {}
    for b in bins:
        m = mag_all[b-1]
        vmax = np.percentile(m, 99.9)
        views = [("Projection - S/R", m.max(axis=1).T, s, p, "along vessel S (mm)", "R (mm)"),
                 ("Projection - S/P", m.max(axis=2).T, s, p, "along vessel S (mm)", "P (mm)"),
                 ("Projection - R/P", m.max(axis=0), p, p, "R (mm)", "P (mm)"),
                 ("Centre slice - S/R", m[:, 32, :].T, s, p, "along vessel S (mm)", "R (mm)"),
                 ("Centre slice - S/P", m[:, :, 32].T, s, p, "along vessel S (mm)", "P (mm)"),
                 ("Throat slice - R/P", m[15], p, p, "R (mm)", "P (mm)")]
        fig, axes = plt.subplots(2, 3, figsize=(21, 13), layout="constrained")
        for ax, (title, img, x, y, xl, yl) in zip(axes.ravel(), views):
            ax.imshow(img, cmap="gray", vmin=0, vmax=vmax if "slice" in title else np.percentile(img, 99.9), origin="lower",
                      extent=[x[0]-(x[1]-x[0])/2, x[-1]+(x[1]-x[0])/2, y[0]-.625, y[-1]+.625], interpolation="nearest", aspect="equal")
            ax.set_title(title); ax.set_xlabel(xl); ax.set_ylabel(yl)
        fig.suptitle(f"dp122 moving-spin KomaMRI reconstruction (300k spins, native 0.72-s cycle) | cardiac bin {b}/30", fontsize=12*FS)
        fig.supxlabel("Scanner geometry: 80 x 80 mm in-plane, 75 mm along the vessel (throat at S = 0); 1.25 x 1.25 x 2.5 mm voxels. "
                      "No added noise, smoothing or mask.", fontsize=8*FS)
        name = f"dp122_orthoviews_bin{b:02d}.png"; fig.savefig(stage/name, dpi=110); plt.close(fig); out[name] = b
    (stage/"MANIFEST.json").write_text(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(), script_sha256=sha(__file__),
        status="PRESENTATION_DERIVATIVE_ONLY", figures={k: sha(stage/k) for k in out}), indent=1)+chr(10), encoding="utf-8")
    stage.rename(target); print("DONE", list(out))


if __name__ == "__main__":
    main(sys.argv[1], [int(x) for x in sys.argv[2].split(",")])
