"""Presentation: E-028-style support-weighted speed projections (CFD / Koma / Koma - CFD)
for the E-060/E-061 phantom, reoriented as requested: rotated 90 deg counter-clockwise
then flipped left-right, i.e. vessel vertical with flow upward, P horizontal.

Projection = fraction-weighted mean through R inside the fixed all-bin 15 % core mask
(same rule as scripts/plot_aorta_extended_comparison.py: project()). Native pixels, no
smoothing. Arms: E060 (VENC 3) and, when complete, E061 VENC4 (E-060 REF/X/Y + E-061 Z).
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tomllib

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

E060 = Path("D:/MRSim/outputs/scanner_geometry_e060")
E061 = Path("D:/MRSim/outputs/venc_unwrap_e061")
sys.path.insert(0, str(E060/"code"))
import analyze_e060 as an  # noqa: E402
from analyze_e060 import fixed_mask  # noqa: E402

BINS = (1, 8, 20)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def project(values, mask, fraction, axis):
    """Verbatim rule of plot_aorta_extended_comparison.project."""
    weights = np.where(mask, fraction, 0)
    denominator = weights.sum(axis=axis)
    numerator = (np.where(mask, values, 0)*weights).sum(axis=axis)
    return np.divide(numerator, denominator, out=np.full(denominator.shape, np.nan), where=denominator > 0)


def plot_map(ax, data, extent, title, limits, difference=False):
    cmap = plt.get_cmap("RdBu_r" if difference else "viridis").copy(); cmap.set_bad("#eeeeee")
    image = ax.imshow(data, origin="lower", extent=extent, interpolation="nearest", cmap=cmap,
                      vmin=limits[0], vmax=limits[1], aspect="equal")
    ax.set_title(title, fontsize=10); ax.set_facecolor("#eeeeee")
    return image


def arms():
    control = tomllib.loads((E060/"acquisition_v1/control_report.toml").read_text())
    images, _ = an.load_images(E060/"acquisition_v1", control, E060/"sequences_v2/timing_e060.h5")
    out = {"E060_VENC3": (images, (1.5, 1.5, 3.0))}
    ctrl = E061/"acquisition_v1/control_report.toml"
    if ctrl.exists() and tomllib.loads(ctrl.read_text()).get("status") == "COMPLETE_E061_Z_VENC4_ACQUISITION":
        c = tomllib.loads(ctrl.read_text())
        if sha(E061/"acquisition_v1/z_result.h5") != c["result_sha256"]["Z"]:
            raise ValueError("Z result changed")
        with h5py.File(E061/"acquisition_v1/z_result.h5") as f:
            z4 = f["image"][:]
        im4 = dict(images); im4["Z"] = z4
        out["E061_VENC4"] = (im4, (1.5, 1.5, 4.0))
    return out


def main(target):
    target = Path(target)
    if target.exists():
        raise FileExistsError(target)
    with h5py.File(E060/"references_v2/references.h5") as f:
        ref, frac, z = f["fft_adc"][:], f["fraction_fft"][:], f["partition_z_m"][:]*1e3
    speed_ref = np.linalg.norm(ref.astype(float), axis=-1)
    p_mm = (np.arange(64)-32)*1.25
    # Transpose (S,P) -> rows S, cols P: rotate 90 deg CCW then flip L-R of the S-horizontal view.
    extent = [p_mm[0]-.625, p_mm[-1]+.625, z[0]-1.25, z[-1]+1.25]
    stage = target.with_name(target.name+".building"); stage.mkdir()
    meta = {}
    for name, (images, vencs) in arms().items():
        field = np.stack([np.angle(images[e]*images["REF"].conj())*(vencs[i]/np.pi) for i, e in enumerate(an.ENC[1:])], axis=-1)
        speed_k = np.linalg.norm(field, axis=-1)
        core = fixed_mask(images, .15) & (frac > 0)          # (S, P, R) fixed all-bin 15 % core
        smax = float(np.nanmax([project(speed_ref[b], core, frac, 2) for b in range(30)]))
        for b in BINS:
            a = project(speed_ref[b-1], core, frac, 2)            # (S, P)
            k = project(speed_k[b-1], core, frac, 2)
            dmax = float(np.nanmax(np.abs(k-a)))
            fig, axes = plt.subplots(1, 3, figsize=(12, 7.5), layout="constrained")
            for ax, data, lab, lim, diff in zip(axes, (a, k, k-a), ("CFD", "Koma", "Koma minus CFD"),
                                                ((0, smax), (0, smax), (-dmax, dmax)), (False, False, True)):
                im = plot_map(ax, data, extent, lab, lim, diff)
                ax.set(xlabel="P (mm)", ylabel="S, along vessel (mm; flow upward)")
                ax.axhline(0, color="#555555", lw=.6, ls=(0, (3, 3)))
                fig.colorbar(im, ax=ax, label="m/s", shrink=.8)
            fig.suptitle(f"Phantom speed | {name.replace('_', ' ')} | R support-weighted projection | bin {b}/30", fontsize=13)
            fig.supxlabel("Fixed 15% core; numerical support only. Gray = excluded. Native pixels; no spatial smoothing. "
                          "Rotated 90° CCW then flipped L-R. Dashed: throat.", fontsize=9)
            path = stage/f"phantom_speed_{name}_bin{b:02d}.png"
            fig.savefig(path, dpi=150); plt.close(fig)
            meta[path.name] = dict(arm=name, bin=b, speed_max=smax, difference_max=dmax)
            print(path, flush=True)
    (stage/"MANIFEST.json").write_text(json.dumps(dict(created_utc=datetime.now(timezone.utc).isoformat(),
        status="PRESENTATION_DERIVATIVE_ONLY", script_sha256=sha(__file__), figures=meta,
        outputs={p.name: sha(p) for p in sorted(stage.glob("*.png"))}), indent=2)+"\n", encoding="utf-8")
    stage.rename(target)


if __name__ == "__main__":
    main(sys.argv[1])
