r"""Add the ``n_steps`` / ``n_particles`` attributes KomaMRI phantom builders need.

``TrackResult.save()`` writes only ``dt`` (mrsimtracks/core.py:98-100); the
streaming writer used when ``track(..., output_path=)`` is given writes all three
(core.py:155-157). Files produced through ``.save()`` therefore fail immediately in
``static_phantom.jl`` / ``motion_phantom.jl``, both of which call
``read_attribute(file, "n_steps")``.

Attributes are derived from the actual ``position`` dataset shape and only ever
added, never overwritten. An existing attribute that disagrees with the data is
reported as an error rather than silently corrected.

Example:
    python fix_track_attrs.py ../aorta_dp2_full_cycle.h5 ../phantom_dp150_tracks_10s.h5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without writing.",
    )
    return parser.parse_args()


def fix_one(path: Path, dry_run: bool) -> bool:
    """Return True when the file ends up carrying consistent attributes."""
    mode = "r" if dry_run else "r+"
    with h5py.File(path, mode) as handle:
        if "position" not in handle:
            print(f"  ERROR {path.name}: no 'position' dataset")
            return False
        n_steps, n_particles, ncomp = handle["position"].shape
        if ncomp != 3:
            print(f"  ERROR {path.name}: position last axis is {ncomp}, expected 3")
            return False

        ok = True
        for name, value in (("n_steps", n_steps), ("n_particles", n_particles)):
            if name in handle.attrs:
                existing = int(handle.attrs[name])
                if existing != value:
                    print(
                        f"  ERROR {path.name}: {name}={existing} disagrees with "
                        f"position shape ({value}); not touching it"
                    )
                    ok = False
                else:
                    print(f"  ok    {path.name}: {name}={existing} already correct")
                continue
            if dry_run:
                print(f"  would add {path.name}: {name}={value}")
            else:
                handle.attrs[name] = value
                print(f"  added {path.name}: {name}={value}")

        if "dt" not in handle.attrs:
            print(f"  ERROR {path.name}: no 'dt' attribute; cannot infer it")
            ok = False
        else:
            dt = float(handle.attrs["dt"])
            print(
                f"  info  {path.name}: dt={dt} s, {n_steps} steps "
                f"= {n_steps * dt:.4f} s, {n_particles} particles"
            )
        return ok


def main():
    args = parse_args()
    all_ok = True
    for path in args.files:
        if not path.is_file():
            print(f"  ERROR missing file: {path}")
            all_ok = False
            continue
        all_ok &= fix_one(path, args.dry_run)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
