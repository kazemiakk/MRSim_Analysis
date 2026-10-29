r"""Generate and validate volume-uniform dp150 seeds with MRSimTracks 0.2.1.

Outputs an exact-size seed array, JSON and Markdown reports, and a paper-style
QC figure. Run in an ephemeral, non-project environment with Python's isolated
mode to prevent checkout shadowing:

    uv run --isolated --no-project --with "mrsimtracks==0.2.1" `
        python -I scripts\validate_dp150_uniform_seeding.py
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv


DEFAULT_ROOT = Path("outputs/dp150_volume_pipeline")
EXPECTED_VERSION = "0.2.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--count", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--axial-bins", type=int, default=48)
    return parser.parse_args()


def package_provenance() -> dict[str, str]:
    version = importlib.metadata.version("mrsimtracks")
    import mrsimtracks

    module = Path(mrsimtracks.__file__).resolve()
    if version != EXPECTED_VERSION:
        raise RuntimeError(f"Expected MRSimTracks {EXPECTED_VERSION}, found {version}")
    if "site-packages" not in {part.lower() for part in module.parts}:
        raise RuntimeError("Run with Python -I; the checkout is shadowing the wheel")
    return {"distribution_version": version, "module_path": str(module)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_seed_mesh(mesh: pv.DataSet, count: int, rng: np.random.Generator) -> np.ndarray:
    from mrsimtracks.seeding import seed_mesh

    chunks: list[np.ndarray] = []
    total = 0
    attempts = 0
    while total < count:
        attempts += 1
        if attempts > 8:
            raise RuntimeError(f"Unable to generate {count:,} valid volume seeds")
        requested = max(1_000, count - total)
        chunk = np.asarray(seed_mesh(mesh, requested, rng=rng), dtype=np.float64)
        if chunk.ndim != 2 or chunk.shape[1] != 3 or chunk.shape[0] == 0:
            raise RuntimeError("MRSimTracks seed_mesh returned no usable points")
        chunks.append(chunk)
        total += chunk.shape[0]
    candidates = np.vstack(chunks)
    if candidates.shape[0] > count:
        keep = rng.choice(candidates.shape[0], size=count, replace=False)
        candidates = candidates[keep]
    return np.ascontiguousarray(candidates, dtype=np.float32)


def tet_volumes_and_centers(mesh: pv.UnstructuredGrid) -> tuple[np.ndarray, np.ndarray]:
    if not np.all(np.asarray(mesh.celltypes) == pv.CellType.TETRA):
        raise ValueError("Axial volume validation requires an all-tetrahedral mesh")
    conn = np.asarray(mesh.cells).reshape(-1, 5)[:, 1:]
    points = np.asarray(mesh.points, dtype=np.float64)
    vertices = points[conn]
    e0 = vertices[:, 0] - vertices[:, 3]
    e1 = vertices[:, 1] - vertices[:, 3]
    e2 = vertices[:, 2] - vertices[:, 3]
    volume = np.abs(np.einsum("ij,ij->i", e0, np.cross(e1, e2))) / 6.0
    centers = vertices.mean(axis=1)
    return volume, centers


def nominal_radial_mask(
    seeds: np.ndarray, wall: pv.PolyData, bins: int = 96
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    wall_points = np.asarray(wall.points, dtype=np.float64)
    radius = np.linalg.norm(wall_points[:, :2], axis=1)
    global_radius = float(radius.max())
    z_min, z_max = float(wall.bounds[4]), float(wall.bounds[5])
    edges = np.linspace(z_min, z_max, bins + 1)
    wall_bin = np.clip(np.searchsorted(edges, wall_points[:, 2], side="right") - 1, 0, bins - 1)
    profile = np.full(bins, np.nan)
    for index in range(bins):
        values = radius[wall_bin == index]
        if values.size:
            profile[index] = values.max()
    valid = np.isfinite(profile)
    if valid.sum() < bins // 2:
        raise RuntimeError("Wall surface is too sparse to estimate the radius profile")
    profile = np.interp(np.arange(bins), np.flatnonzero(valid), profile[valid])
    seed_bin = np.clip(np.searchsorted(edges, seeds[:, 2], side="right") - 1, 0, bins - 1)
    nominal = profile[seed_bin] >= 0.98 * global_radius
    normalized_radius = np.linalg.norm(seeds[:, :2], axis=1) / global_radius
    return nominal, normalized_radius, global_radius, edges, profile


def write_markdown(path: Path, report: dict[str, object]) -> None:
    radial = report["radial_uniformity"]
    axial = report["axial_volume_uniformity"]
    acceptance = report["acceptance"]
    lines = [
        "# dp150 Uniform-Seeding QC",
        "",
        f"- MRSimTracks: `{report['mrsimtracks']['distribution_version']}`",
        f"- Seeds: **{report['seeds']['count']:,}**",
        f"- Valid inside volume: **{report['seeds']['valid_fraction']:.6f}**",
        f"- RNG seed: `{report['seeds']['rng_seed']}`",
        "",
        "## Radial uniformity",
        "",
        "| Metric | Observed | Uniform-volume expectation | Absolute error | Pass |",
        "|---|---:|---:|---:|:---:|",
        (
            f"| Median r/R | {radial['median_r_over_R']:.6f} | "
            f"{radial['expected_median_r_over_R']:.6f} | "
            f"{radial['median_absolute_error']:.6f} | "
            f"{'yes' if acceptance['checks']['radial_median'] else 'no'} |"
        ),
        (
            f"| Fraction r/R > 0.90 | {radial['fraction_gt_0_90']:.6f} | "
            f"{radial['expected_fraction_gt_0_90']:.6f} | "
            f"{radial['fraction_gt_0_90_absolute_error']:.6f} | "
            f"{'yes' if acceptance['checks']['radial_gt_0_90'] else 'no'} |"
        ),
        (
            f"| Fraction r/R > 0.95 | {radial['fraction_gt_0_95']:.6f} | "
            f"{radial['expected_fraction_gt_0_95']:.6f} | "
            f"{radial['fraction_gt_0_95_absolute_error']:.6f} | "
            f"{'yes' if acceptance['checks']['radial_gt_0_95'] else 'no'} |"
        ),
        "",
        "## Axial volume uniformity",
        "",
        f"- Total-variation distance from tetrahedral volume fractions: **{axial['total_variation_distance']:.6f}**",
        f"- Acceptance limit: **{acceptance['limits']['axial_total_variation_max']:.3f}**",
        "",
        "## Conclusion",
        "",
        f"**{'PASS' if acceptance['passed'] else 'FAIL'}** - {acceptance['interpretation']}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def make_figure(
    path: Path,
    seeds: np.ndarray,
    wall: pv.PolyData,
    nominal: np.ndarray,
    normalized_radius: np.ndarray,
    axial_centers_mm: np.ndarray,
    observed_fraction: np.ndarray,
    expected_fraction: np.ndarray,
    report: dict[str, object],
    rng: np.random.Generator,
) -> None:
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10})
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 7.5), constrained_layout=True)
    sample_count = min(30_000, seeds.shape[0])
    sample = seeds[rng.choice(seeds.shape[0], size=sample_count, replace=False)]
    axes[0, 0].scatter(
        sample[:, 2] * 1e3,
        sample[:, 0] * 1e3,
        s=0.35,
        color="#124E78",
        alpha=0.32,
        linewidths=0,
        rasterized=True,
    )
    wall_points = np.asarray(wall.points)
    axes[0, 0].scatter(
        wall_points[:, 2] * 1e3,
        wall_points[:, 0] * 1e3,
        s=0.18,
        color="#4B5563",
        alpha=0.28,
        linewidths=0,
        rasterized=True,
    )
    axes[0, 0].set_title("Uniform volume seeds - xz projection")
    axes[0, 0].set_xlabel("z (mm)")
    axes[0, 0].set_ylabel("x (mm)")
    axes[0, 0].set_aspect("equal", adjustable="box")

    values = normalized_radius[nominal]
    grid = np.linspace(0, 1, 201)
    empirical = np.searchsorted(np.sort(values), grid, side="right") / values.size
    axes[0, 1].plot(grid, empirical, color="#124E78", lw=2, label="Observed")
    axes[0, 1].plot(grid, grid**2, color="#D1495B", lw=1.7, ls="--", label=r"Uniform disk: $F(r/R)=(r/R)^2$")
    axes[0, 1].set_xlim(0, 1)
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_title("Radial cumulative distribution")
    axes[0, 1].set_xlabel("r/R")
    axes[0, 1].set_ylabel("Cumulative fraction")
    axes[0, 1].legend(frameon=False, loc="upper left")
    axes[0, 1].grid(color="#D1D5DB", lw=0.5, alpha=0.7)

    axes[1, 0].plot(axial_centers_mm, expected_fraction, color="#D1495B", lw=1.8, label="Mesh volume")
    axes[1, 0].plot(axial_centers_mm, observed_fraction, color="#124E78", lw=1.4, label="Seed count")
    axes[1, 0].set_title("Axial occupancy follows lumen volume")
    axes[1, 0].set_xlabel("z (mm)")
    axes[1, 0].set_ylabel("Fraction per bin")
    axes[1, 0].legend(frameon=False)
    axes[1, 0].grid(color="#D1D5DB", lw=0.5, alpha=0.7)

    radial = report["radial_uniformity"]
    axial = report["axial_volume_uniformity"]
    acceptance = report["acceptance"]
    axes[1, 1].axis("off")
    summary = (
        f"SEEDING QC: {'PASS' if acceptance['passed'] else 'FAIL'}\n\n"
        f"Valid in volume       {report['seeds']['valid_fraction']:.6f}\n"
        f"Median r/R            {radial['median_r_over_R']:.4f}  (expected 0.7071)\n"
        f"Fraction > 0.90R      {radial['fraction_gt_0_90']:.4f}  (expected 0.1900)\n"
        f"Fraction > 0.95R      {radial['fraction_gt_0_95']:.4f}  (expected 0.0975)\n"
        f"Axial TV distance     {axial['total_variation_distance']:.4f}\n\n"
        "Generated by MRSimTracks 0.2.1 volume probing"
    )
    axes[1, 1].text(
        0.02,
        0.96,
        summary,
        va="top",
        ha="left",
        family="monospace",
        fontsize=11,
        color="#111827",
        bbox={"facecolor": "#F8FAFC", "edgecolor": "#CBD5E1", "boxstyle": "square,pad=0.8"},
    )
    fig.suptitle("dp150 volume-uniform particle seeding validation", fontsize=14)
    fig.savefig(path, dpi=220, facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.count < 1 or args.axial_bins < 8:
        raise ValueError("count must be positive and axial-bins must be at least 8")
    provenance = package_provenance()
    import mrsimtracks as mt

    root = args.root.resolve()
    pvd_path = root / "flow" / "dp150_flow.pvd"
    wall_path = root / "flow" / "wall.vtp"
    conversion_path = root / "conversion_report.json"
    for path in (pvd_path, wall_path, conversion_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    conversion = json.loads(conversion_path.read_text(encoding="utf-8"))
    if conversion["temporal"]["duration_s"] != 0.72:
        raise RuntimeError("The accepted converter output must span exactly 0.72 s")

    print(f"Loading flow: {pvd_path}", flush=True)
    flow = mt.load_flow(
        pvd_path,
        active_key="Velocity",
        precision="f32",
        time_interp="linear",
        pbar=False,
    )
    if len(flow.times) != 73 or not np.isclose(flow.tmax, 0.72, atol=1e-12):
        raise RuntimeError(f"Unexpected PVD clock: {len(flow.times)} frames, tmax={flow.tmax}")
    rng = np.random.default_rng(args.seed)
    print(f"Generating {args.count:,} volume seeds with MRSimTracks 0.2.1", flush=True)
    seeds = exact_seed_mesh(flow.active_mesh, args.count, rng)
    cells = flow._sampler.locate(np.ascontiguousarray(seeds), guess=None)
    valid = cells >= 0
    valid_fraction = float(valid.mean())
    if not valid.all():
        raise RuntimeError(f"{np.count_nonzero(~valid):,} generated seeds are outside the mesh")

    wall = pv.read(wall_path)
    nominal, normalized_radius, radius, _, _ = nominal_radial_mask(seeds, wall)
    radial_values = normalized_radius[nominal]
    if radial_values.size < args.count // 2:
        raise RuntimeError("Too few seeds in nominal-radius sections for radial QC")
    radial_median = float(np.median(radial_values))
    radial_gt90 = float(np.mean(radial_values > 0.90))
    radial_gt95 = float(np.mean(radial_values > 0.95))
    expected_median = float(np.sqrt(0.5))
    expected_gt90 = 1.0 - 0.90**2
    expected_gt95 = 1.0 - 0.95**2

    volumes, cell_centers = tet_volumes_and_centers(flow.active_mesh)
    z_min, z_max = float(flow.active_mesh.bounds[4]), float(flow.active_mesh.bounds[5])
    axial_edges = np.linspace(z_min, z_max, args.axial_bins + 1)
    volume_by_bin, _ = np.histogram(cell_centers[:, 2], bins=axial_edges, weights=volumes)
    seed_by_bin, _ = np.histogram(seeds[:, 2], bins=axial_edges)
    expected_fraction = volume_by_bin / volume_by_bin.sum()
    observed_fraction = seed_by_bin / seed_by_bin.sum()
    total_variation = float(0.5 * np.abs(observed_fraction - expected_fraction).sum())
    volume_centroid = np.average(cell_centers, axis=0, weights=volumes)
    seed_centroid = seeds.astype(np.float64).mean(axis=0)

    limits = {
        "valid_fraction_min": 1.0,
        "radial_median_absolute_error_max": 0.02,
        "radial_gt_0_90_absolute_error_max": 0.02,
        "radial_gt_0_95_absolute_error_max": 0.015,
        "axial_total_variation_max": 0.03,
    }
    checks = {
        "all_inside": valid_fraction >= limits["valid_fraction_min"],
        "radial_median": abs(radial_median - expected_median)
        <= limits["radial_median_absolute_error_max"],
        "radial_gt_0_90": abs(radial_gt90 - expected_gt90)
        <= limits["radial_gt_0_90_absolute_error_max"],
        "radial_gt_0_95": abs(radial_gt95 - expected_gt95)
        <= limits["radial_gt_0_95_absolute_error_max"],
        "axial_volume": total_variation <= limits["axial_total_variation_max"],
    }
    passed = bool(all(checks.values()))
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "python": {"executable": sys.executable, "version": platform.python_version()},
        "mrsimtracks": provenance,
        "inputs": {
            "pvd_file": str(pvd_path),
            "conversion_report": str(conversion_path),
            "conversion_report_sha256": sha256(conversion_path),
        },
        "seeds": {
            "count": int(seeds.shape[0]),
            "dtype": str(seeds.dtype),
            "rng_seed": args.seed,
            "valid_count": int(valid.sum()),
            "valid_fraction": valid_fraction,
            "centroid_m": seed_centroid.tolist(),
            "volume_centroid_m": volume_centroid.tolist(),
            "centroid_error_m": (seed_centroid - volume_centroid).tolist(),
        },
        "radial_uniformity": {
            "nominal_section_definition": "local wall radius >= 0.98 * global radius",
            "nominal_seed_count": int(nominal.sum()),
            "global_radius_m": radius,
            "median_r_over_R": radial_median,
            "expected_median_r_over_R": expected_median,
            "median_absolute_error": abs(radial_median - expected_median),
            "fraction_gt_0_90": radial_gt90,
            "expected_fraction_gt_0_90": expected_gt90,
            "fraction_gt_0_90_absolute_error": abs(radial_gt90 - expected_gt90),
            "fraction_gt_0_95": radial_gt95,
            "expected_fraction_gt_0_95": expected_gt95,
            "fraction_gt_0_95_absolute_error": abs(radial_gt95 - expected_gt95),
        },
        "axial_volume_uniformity": {
            "bins": args.axial_bins,
            "total_mesh_volume_m3": float(volumes.sum()),
            "total_variation_distance": total_variation,
            "bin_edges_m": axial_edges.tolist(),
            "observed_seed_fraction": observed_fraction.tolist(),
            "expected_volume_fraction": expected_fraction.tolist(),
        },
        "acceptance": {
            "passed": passed,
            "checks": checks,
            "limits": limits,
            "interpretation": (
                "All seeds are inside the volume and their radial and axial "
                "distributions agree with uniform-volume expectations."
                if passed
                else "At least one predeclared uniform-volume seeding check failed."
            ),
        },
    }

    seed_path = root / "dp150_uniform_seeds_100k.npy"
    report_path = root / "seeding_report.json"
    markdown_path = root / "seeding_report.md"
    figure_path = root / "seeding_qc.png"
    np.save(seed_path, seeds, allow_pickle=False)
    report["seeds"]["file"] = str(seed_path)
    report["seeds"]["file_sha256"] = sha256(seed_path)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(markdown_path, report)
    axial_centers_mm = 0.5 * (axial_edges[:-1] + axial_edges[1:]) * 1e3
    make_figure(
        figure_path,
        seeds,
        wall,
        nominal,
        normalized_radius,
        axial_centers_mm,
        observed_fraction,
        expected_fraction,
        report,
        rng,
    )
    print(f"Seeding QC: {'PASS' if passed else 'FAIL'}", flush=True)
    print(
        f"median r/R={radial_median:.5f}, >0.90R={radial_gt90:.5f}, "
        f">0.95R={radial_gt95:.5f}, axial TV={total_variation:.5f}",
        flush=True,
    )
    print(f"Wrote seeds: {seed_path}", flush=True)
    print(f"Wrote reports: {report_path}, {markdown_path}, {figure_path}", flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
