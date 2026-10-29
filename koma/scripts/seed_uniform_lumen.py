r"""Draw a uniform-density volumetric point cloud inside a CFD lumen.

Purpose
-------
The static phantom built by flattening a particle trajectory has spin density
proportional to residence time, i.e. ~1/speed. Measured on aorta dp2 over a 2 mm
grid that gives a 1665x density range (median 30 spins/voxel, max 49,954), which
is the hot spot and the streaks in the reference reconstruction. This script
replaces that with a genuinely uniform cloud so the static simulation tests the
sequence and recon rather than the tracking.

Method
------
No tetrahedral connectivity survives in the processed data (``geometry.pkl`` holds
only ``mesh_coords`` / ``edge_index``), so nodal volumes cannot be integrated
exactly. Instead each interior node is weighted by its local spacing cubed, which
approximates its Voronoi volume:

    node positions are distributed as rho(x)   [nodes per m^3]
    local spacing h ~ rho^(-1/3)   =>   weight h^3 ~ 1/rho
    sampling nodes with probability ~ 1/rho gives point density ~ rho * 1/rho = const

Drawn points are jittered inside a ball of ``jitter_frac`` times the distance to
their parent's nearest neighbour, which keeps each point inside its parent's
Voronoi cell and so cannot push it meaningfully through the wall.

Wall nodes are identified from ``wall_shear.npy`` being nonzero, which was verified
to agree with the zero-speed-at-all-times node set to 100.00% on dp150, and are
excluded from the sampling pool.

Example:
    python seed_uniform_lumen.py \
        --case "../../../PI-GNN(SR)/data/aortic arch/processed/dp2" \
        --n-points 2000000 --out ../../aorta_dp2_uniform_2M.h5
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        type=Path,
        required=True,
        help="Processed case directory holding geometry.pkl and wall_shear.npy.",
    )
    parser.add_argument("--n-points", type=int, default=2_000_000)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--neighbors",
        type=int,
        default=8,
        help="Neighbours used to estimate local spacing (default: 8).",
    )
    parser.add_argument(
        "--jitter-frac",
        type=float,
        default=0.35,
        help="Jitter radius as a fraction of the parent's nearest-neighbour "
        "distance; stays inside the parent Voronoi cell (default: 0.35).",
    )
    parser.add_argument(
        "--voxel-mm",
        type=float,
        default=2.0,
        help="Voxel size used for the density-uniformity report (default: 2.0).",
    )
    return parser.parse_args()


def load_case(case: Path):
    with (case / "geometry.pkl").open("rb") as stream:
        coords = np.asarray(pickle.load(stream)["mesh_coords"], dtype=np.float64)

    wall_shear_path = case / "wall_shear.npy"
    if wall_shear_path.is_file():
        # column 0 is wall-shear magnitude; nonzero only on no-slip wall nodes
        wall_shear = np.load(wall_shear_path, mmap_mode="r")
        is_wall = np.abs(np.asarray(wall_shear[:, 0, :])).max(axis=1) > 0
        source = "wall_shear.npy"
    else:
        velocities = np.load(case / "velocities.npy", mmap_mode="r")
        speed = np.linalg.norm(np.asarray(velocities), axis=1)
        is_wall = speed.max(axis=1) < 1e-6
        source = "velocities.npy (zero-speed fallback)"
    return coords, is_wall, source


def report_uniformity(points, voxel_m, label):
    mins = points.min(axis=0)
    index = np.floor((points - mins) / voxel_m).astype(np.int64)
    shape = index.max(axis=0) + 1
    counts = np.bincount(np.ravel_multi_index(index.T, shape))
    occupied = counts[counts > 0]
    print(f"  {label}: {len(occupied):,} occupied voxels of {np.prod(shape):,}")
    print(
        f"    points/voxel  mean {occupied.mean():.1f}  median {np.median(occupied):.0f}"
        f"  p01 {np.percentile(occupied, 1):.0f}  p99 {np.percentile(occupied, 99):.0f}"
        f"  max {occupied.max()}"
    )
    print(
        f"    max/median = {occupied.max() / np.median(occupied):.1f}x"
        f"    top voxel holds {100 * occupied.max() / len(points):.3f}% of all points"
    )
    print(
        f"    fraction of occupied voxels below 30 points: "
        f"{(occupied < 30).mean():.3f}"
    )
    return occupied


def main():
    args = parse_args()
    if not args.case.is_dir():
        print(f"ERROR: missing case directory: {args.case}")
        return 1

    print(f"loading {args.case}")
    coords, is_wall, wall_source = load_case(args.case)
    print(
        f"  {len(coords):,} nodes, {is_wall.sum():,} wall ({100 * is_wall.mean():.2f}%) "
        f"from {wall_source}"
    )
    interior = coords[~is_wall]
    if len(interior) < args.neighbors + 1:
        print("ERROR: too few interior nodes to estimate spacing")
        return 1

    print(f"estimating local spacing over {len(interior):,} interior nodes")
    tree = cKDTree(interior)
    distances, _ = tree.query(interior, k=args.neighbors + 1, workers=-1)
    nearest = distances[:, 1]
    spacing = distances[:, 1:].mean(axis=1)
    print(
        f"  spacing  min {spacing.min() * 1e3:.4f} mm  median "
        f"{np.median(spacing) * 1e3:.3f} mm  max {spacing.max() * 1e3:.3f} mm "
        f"({spacing.max() / spacing.min():.1f}x range)"
    )

    weights = spacing**3
    weights /= weights.sum()
    rng = np.random.default_rng(args.seed)
    print(f"drawing {args.n_points:,} volume-weighted samples")
    parents = rng.choice(len(interior), size=args.n_points, p=weights)

    # isotropic jitter inside a ball of jitter_frac * nearest-neighbour distance
    direction = rng.normal(size=(args.n_points, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    radius = args.jitter_frac * nearest[parents] * rng.random(args.n_points) ** (1 / 3)
    points = interior[parents] + direction * radius[:, None]

    voxel_m = args.voxel_mm * 1e-3
    print(f"density uniformity at {args.voxel_mm} mm voxels:")
    report_uniformity(coords[~is_wall], voxel_m, "raw interior nodes (before weighting)")
    report_uniformity(points, voxel_m, "sampled cloud")

    # how many samples ended up nearer a wall node than any interior node
    wall_nodes = coords[is_wall]
    if len(wall_nodes):
        wall_tree = cKDTree(wall_nodes)
        d_wall, _ = wall_tree.query(points, workers=-1)
        d_interior, _ = tree.query(points, workers=-1)
        outside = d_wall < d_interior
        print(
            f"  samples closer to a wall node than any interior node: "
            f"{outside.sum():,} ({100 * outside.mean():.3f}%)"
        )

    extent = points.max(axis=0) - points.min(axis=0)
    order = np.argsort(-extent)
    print(f"  extent {np.round(extent * 1e3, 1)} mm, largest-first axis order {order}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.out, "w") as out:
        out.create_dataset("points", data=points.astype(np.float32))
        out.attrs["n_points"] = len(points)
        out.attrs["case"] = str(args.case)
        out.attrs["seed"] = args.seed
        out.attrs["jitter_frac"] = args.jitter_frac
        out.attrs["units"] = "m"
    print(f"wrote {args.out}  ({points.shape}, float32)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
