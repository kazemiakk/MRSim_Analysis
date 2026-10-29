r"""Extract exactly one cardiac cycle from a multi-cycle MRSimTracks HDF5 file.

``motion_phantom.jl`` builds its ``FlowPath`` from the *entire* ``position``
array and wraps it in ``Periodic(period=RR_INTERVAL)``. Handing it a 10 s file
therefore compresses every stored cycle into one RR -- a silent N-fold speed-up
with no error. This script slices one cycle out first.

The slice is half-open: steps ``[start, start + cycle_steps)``. The wrap sample is
deliberately not duplicated, on the assumption that KomaMRI's periodic time curve
maps the stored samples across the period and returns to sample 0 itself.
NOTE: that assumption is unverified against KomaMRI internals -- confirm before
trusting a production run.

Also reports the wrap discontinuity ``|p[0] - p[-1]|``, which is what every
particle traverses at each period boundary. Large values there are physical
(particles genuinely flow downstream over a cycle) but they mean the periodic
phantom teleports spins, so the reset flags matter.

Example:
    python slice_one_cycle.py ../../phantom_dp150_tracks_10s.h5 \
        --cycle-seconds 1.0 --out ../../phantom_dp150_one_cycle.h5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--cycle-seconds",
        type=float,
        help="Duration of one source cycle, in seconds.",
    )
    group.add_argument(
        "--cycle-steps",
        type=int,
        help="Length of one source cycle, in stored steps.",
    )
    parser.add_argument(
        "--start-cycle",
        type=int,
        default=-1,
        help="Which whole cycle to extract; -1 selects the last complete one, "
        "which is the most settled (default: -1).",
    )
    parser.add_argument(
        "--float32",
        action="store_true",
        help="Downcast positions to float32. Halves the file and matches the "
        "Float32 arrays KomaMRI builds anyway.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.source.is_file():
        print(f"ERROR: missing source: {args.source}")
        return 1

    with h5py.File(args.source, "r") as handle:
        dt = float(handle.attrs["dt"])
        n_steps, n_particles, _ = handle["position"].shape

        cycle_steps = (
            args.cycle_steps
            if args.cycle_steps is not None
            else int(round(args.cycle_seconds / dt))
        )
        if cycle_steps < 2:
            print(f"ERROR: cycle is {cycle_steps} steps; need at least 2")
            return 1
        if cycle_steps > n_steps:
            print(f"ERROR: cycle is {cycle_steps} steps but file holds {n_steps}")
            return 1

        n_cycles = n_steps // cycle_steps
        index = args.start_cycle % n_cycles
        start = index * cycle_steps
        stop = start + cycle_steps

        print(
            f"source: {n_steps} steps @ dt={dt} s = {n_steps * dt:.4f} s "
            f"-> {n_cycles} whole cycles of {cycle_steps} steps "
            f"({cycle_steps * dt:.4f} s)"
        )
        if n_steps % cycle_steps:
            print(
                f"  note: {n_steps % cycle_steps} trailing steps are not a whole "
                "cycle and are discarded"
            )
        print(f"extracting cycle {index} -> steps [{start}, {stop})")

        position = handle["position"][start:stop]
        reset = handle["reset"][start:stop]

    if args.float32:
        position = position.astype(np.float32)

    wrap = np.linalg.norm(position[0] - position[-1], axis=-1)
    print(
        f"wrap |p[0]-p[-1]|: mean {wrap.mean() * 1e3:.3f} mm, "
        f"median {np.median(wrap) * 1e3:.3f} mm, max {wrap.max() * 1e3:.3f} mm"
    )
    print(f"resets inside the slice: {int(reset.sum())}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.out, "w") as out:
        out.create_dataset("position", data=position)
        out.create_dataset("reset", data=reset)
        out.attrs["dt"] = dt
        out.attrs["n_steps"] = position.shape[0]
        out.attrs["n_particles"] = position.shape[1]
        out.attrs["source_file"] = args.source.name
        out.attrs["source_cycle_index"] = index
    print(f"wrote {args.out}  ({position.shape}, {position.dtype})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
