"""Publication 3x3 / 4x3 through-plane velocity contours (presentation only).

Rows: 4D-flow MRI (E-059 private arrays), CFD (E-060 references_v2), Koma VENC 3 m/s
(E-060 registered acquisition) and, when available, Koma VENC 4 m/s (E-061 Z with E-060
REF). Columns: proximal -25 mm, throat 0, distal +15 mm from each throat, each dataset at
its own peak throat-flow phase. One common circular mask per column: 0.9 x the CFD lumen
equivalent radius, centred on each vessel. Scan in-plane rotation aligned to the CFD by
the direction of the distal jet offset (peak-velocity centroid relative to the vessel
centroid); declared presentation alignment, not a registration. Bilinear sampling on a
0.25 mm grid. Data loading reuses plot_contours_circular.py unchanged.
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

E060 = Path("D:/MRSim/outputs/scanner_geometry_e060")
E061 = Path("D:/MRSim/outputs/venc_unwrap_e061")
sys.path.insert(0, str(E060/"code"))
import plot_contours_circular as base  # noqa: E402

SHRINK, HALF, STEP = 0.9, 13.0, 0.25
GRID = np.arange(-HALF, HALF+STEP/2, STEP)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sample(p, angle=0.0):
    """Image on a grid centred on the vessel, rotated by angle (rad) about the centre."""
    X, Y = np.meshgrid(GRID, GRID)
    c, s = np.cos(angle), np.sin(angle)
    xs, ys = c*X-s*Y+p["center"][0], s*X+c*Y+p["center"][1]
    rows, cols = ys/1.25+32, xs/1.25+32
    return ndimage.map_coordinates(np.nan_to_num(p["img"]), [rows, cols], order=1, mode="nearest")


def jet_angle(p):
    img = sample(p)
    X, Y = np.meshgrid(GRID, GRID)
    inside = X**2+Y**2 <= p["radius"]**2
    hot = inside & (img >= .7*np.nanmax(np.where(inside, img, np.nan)))
    dx, dy = X[hot].mean(), Y[hot].mean()
    return float(np.arctan2(dy, dx)), float(np.hypot(dx, dy))


def venc4_row(cfd):
    control = E061/"acquisition_v1/control_report.toml"
    if not control.exists():
        return None
    ctrl = __import__("tomllib").loads(control.read_text())
    if ctrl.get("status") != "COMPLETE_E061_Z_VENC4_ACQUISITION":
        return None
    path = E061/"acquisition_v1/z_result.h5"
    if sha(path) != ctrl["result_sha256"]["Z"]:
        raise ValueError("Z result changed")
    rep = json.loads((E060/"analysis_v1/report.json").read_text()); b = rep["peak_bin"]-1
    with h5py.File(E060/"acquisition_v1/ref_result.h5") as f:
        ref = f["image"][b]
    with h5py.File(path) as f:
        z4 = f["image"][b]
    v = np.angle(z4*ref.conj())*(4.0/np.pi)
    with h5py.File(E060/"references_v2/references.h5") as f:
        zz = f["partition_z_m"][:]*1e3
    out = []
    for (name, d), c in zip(base.COLUMNS, cfd):
        k = int(np.argmin(abs(zz-d)))
        out.append(dict(img=v[k], pos=float(zz[k]), center=c["center"], radius=c["radius"]))
    return out


def draw(rows, labels, norm, path):
    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(10.5, 3.4*n+.6), facecolor=base.SURFACE)
    data = [[(img, p) for img, p in r] for r in rows]
    X, Y = np.meshgrid(GRID, GRID)
    radii = [p["radius"]*SHRINK for _, p in data[1]]
    def masked(img, j):
        return np.where(X**2+Y**2 <= radii[j]**2, img, np.nan)
    scales = [np.nanmax(masked(r[1][0], 1)) if norm else 1.0 for r in data]
    vmax = 1.0 if norm else max(np.nanpercentile(masked(img, j), 99.5) for r in data for j, (img, _) in enumerate(r))
    levels = np.linspace(0, vmax, 11)
    th = np.linspace(0, 2*np.pi, 361)
    for i, (row, lab, sc) in enumerate(zip(data, labels, scales)):
        for j, (img, p) in enumerate(row):
            ax = axes[i, j]; ax.set_facecolor(base.SURFACE)
            m = np.clip(masked(img, j)/sc, 0, None)
            cf = ax.contourf(GRID, GRID, np.ma.masked_invalid(m), levels=levels, cmap=base.SEQ, extend="max")
            ax.contour(GRID, GRID, np.ma.masked_invalid(m), levels=levels[1::2], colors=base.INK, linewidths=.45)
            ax.plot(radii[j]*np.cos(th), radii[j]*np.sin(th), color=base.INK2, lw=1.1)
            ax.set_xlim(-HALF, HALF); ax.set_ylim(-HALF, HALF); ax.set_aspect("equal"); ax.axis("off")
            p99 = np.nanpercentile(masked(img, j), 99)
            ax.text(0, -HALF+.6, f"p99 {p99:.2f} m/s", ha="center", fontsize=9, color=base.INK2)
            if i == 0:
                ax.set_title(f"{base.COLUMNS[j][0]} ({base.COLUMNS[j][1]:+.0f} mm)", fontsize=12, color=base.INK)
            if j == 0:
                ax.text(-HALF-1, 0, lab, rotation=90, ha="right", va="center", fontsize=11, color=base.INK)
    ax = axes[-1, 0]
    ax.plot([-HALF+1, -HALF+6], [HALF-1]*2, color=base.INK, lw=2); ax.text(-HALF+3.5, HALF-.6, "5 mm", ha="center", fontsize=8, color=base.INK)
    cb = fig.colorbar(cf, ax=axes, shrink=.55, pad=.03)
    cb.set_label("through-plane velocity / own throat peak" if norm else "through-plane velocity (m/s)", fontsize=10, color=base.INK2)
    cb.ax.tick_params(labelsize=8, colors=base.INK2)
    fig.savefig(path, dpi=200, facecolor=base.SURFACE, bbox_inches="tight"); plt.close(fig)


def main(out):
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    scan, sphase, strig = base.scan_data()
    cfd, koma, kbin = base.sim_data()
    radius_scan = [dict(p, radius=c["radius"]) for p, c in zip(scan, cfd)]
    a_cfd, off_cfd = jet_angle(dict(cfd[2], radius=cfd[2]["radius"]*SHRINK))
    a_scan, off_scan = jet_angle(dict(radius_scan[2], radius=cfd[2]["radius"]*SHRINK))
    # v6 applied a 5.9 deg rotation from jet offsets of only 0.10/0.14 mm (noise). Rotate only
    # when both distal jets are offset by >= 1 mm; otherwise orientation is undetermined.
    rot = a_scan-a_cfd if min(off_scan, off_cfd) >= 1.0 else 0.0
    rows = [[(sample(p, rot), p) for p in radius_scan], [(sample(p), p) for p in cfd], [(sample(p), p) for p in koma]]
    labels = ["4D-flow MRI", "CFD", "Koma, VENC 3 m/s"]
    v4 = venc4_row(cfd)
    if v4 is not None:
        rows.append([(sample(p), p) for p in v4]); labels.append("Koma, VENC 4 m/s")
    stage = out.with_name(out.name+".building"); stage.mkdir()
    draw(rows, labels, False, stage/"contours_publication_absolute.png")
    draw(rows, labels, True, stage/"contours_publication_normalized.png")
    meta = dict(created_utc=datetime.now(timezone.utc).isoformat(), status="PRESENTATION_DERIVATIVE_ONLY",
                scan_phase=sphase, scan_trigger_ms=strig, sim_bin=kbin, mask_radius_fraction=SHRINK,
                mask_radii_mm=[c["radius"]*SHRINK for c in cfd],
                scan_rotation_deg=float(np.degrees(rot)), distal_jet_offset_mm=dict(scan=off_scan, cfd=off_cfd),
                orientation="aligned by distal jet offset" if rot else "undetermined (jets centred within 1 mm); not rotated",
                venc4_row=v4 is not None,
                inputs={str(p): sha(p) for p in (Path(__file__), E060/"code/plot_contours_circular.py")},
                outputs={p.name: sha(p) for p in sorted(stage.glob("*.png"))})
    (stage/"MANIFEST.json").write_text(json.dumps(meta, indent=2)+"\n", encoding="utf-8")
    stage.rename(out)
    print(json.dumps({k: meta[k] for k in ("scan_rotation_deg", "distal_jet_offset_mm", "venc4_row", "mask_radii_mm")}))


if __name__ == "__main__":
    main(sys.argv[1])
