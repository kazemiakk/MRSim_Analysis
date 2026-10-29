"""E-060 S1: verify the rigid rotation of the E055 coupled 300k phantom into scanner geometry.

(x, y, z)_new = (y, z, x - x_throat), a proper rotation plus a translation, so the
cycle map, resets and spin identities are unchanged. The runner applies it in
memory; this stage only derives x_throat and checks the rotated paths against
the E-060 field of view. Writes nothing large.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tomllib

import h5py
import numpy as np

E055 = Path("D:/MRSim/outputs/phantom_density_e055")
E048 = Path("D:/MRSim/MRSIM/MRSimTracks/outputs/phantom_diagnosis_e048")
ROOT = Path("D:/MRSim/outputs/scanner_geometry_e060")
ROTATION = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
INPLANE_HALF_FOV = .04
SLAB_HALF = .03
PARTITION_HALF_FOV = .0375
CHECK_HALF = .05  # in-plane containment checked wherever a spin could carry slab signal
CHUNK = 64


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def main(output):
    output = Path(output)
    stage = output.with_name(output.name+".building")
    if output.exists() or stage.exists():
        raise FileExistsError(output)
    info = tomllib.loads((E055/"inputs_v1/case_info.toml").read_text())
    inputs = E055/"inputs_v1/case_inputs.h5"
    if sha(inputs) != info["neutral_input_sha256"] or sha(E055/"map_v1/cycle_map.h5") != info["map_sha256"]:
        raise ValueError("E055 phantom or map changed")
    local = E048/"local_v1/local_fields.h5"
    if sha(local) != "6c57a8708ea4a5aff09f61db232b0ba1528ae4d000bf60d9ca00524516a27885":
        raise ValueError("E048 local fields changed")
    with h5py.File(local) as f:
        source_z = f["source_z_fft_m"][:]
    n = source_z.size
    x_grid = (np.arange(n)-n//2)*.5/n  # E048 centred-FFT readout grid of the 400 x 1.25 mm FOV
    offset = source_z-x_grid
    if np.ptp(offset) > 1e-12:
        raise ValueError("source z is not a pure translation of phantom x")
    x_throat = -float(offset.mean())
    det = int(round(np.linalg.det(ROTATION)))
    if det != 1:
        raise ValueError("Not a proper rotation")
    stats = dict(z_min=np.inf, z_max=-np.inf, inplane_max_abs_x=0., inplane_max_abs_y=0., outside_inplane_samples=0,
                 checked_samples=0)
    slab_counts, partition_counts = [], []
    with h5py.File(inputs) as f:
        rho = f["rho"][:]
        if rho[-1] != 0 or not (f["x"][:] == 0).all() or not (f["y"][:] == 0).all() or not (f["z"][:] == 0).all():
            raise ValueError("Expected zero base positions and a final zero-density sentinel")
        physical = slice(0, rho.size-1)
        nt = f["dx"].shape[0]
        for t0 in range(0, nt, CHUNK):
            t1 = min(nt, t0+CHUNK)
            ox, oy, oz = (f[k][t0:t1, physical].astype(np.float64) for k in ("dx", "dy", "dz"))
            nx, ny, nz = oy, oz, ox-x_throat
            stats["z_min"] = min(stats["z_min"], float(nz.min())); stats["z_max"] = max(stats["z_max"], float(nz.max()))
            near = np.abs(nz) <= CHECK_HALF
            stats["checked_samples"] += int(near.sum())
            if near.any():
                stats["inplane_max_abs_x"] = max(stats["inplane_max_abs_x"], float(np.abs(nx[near]).max()))
                stats["inplane_max_abs_y"] = max(stats["inplane_max_abs_y"], float(np.abs(ny[near]).max()))
                stats["outside_inplane_samples"] += int(np.sum(near & ((np.abs(nx) >= INPLANE_HALF_FOV) | (np.abs(ny) >= INPLANE_HALF_FOV))))
            slab_counts += list(np.sum(np.abs(nz) <= SLAB_HALF, axis=1))
            partition_counts += list(np.sum(np.abs(nz) <= PARTITION_HALF_FOV, axis=1))
    slab_counts, partition_counts = np.array(slab_counts), np.array(partition_counts)
    passed = stats["outside_inplane_samples"] == 0
    report = dict(experiment="E-060", stage="S1_phantom_rotation", utc=datetime.now(timezone.utc).isoformat(),
        status="PASS_E060_PHANTOM_ROTATION" if passed else "FAIL_E060_INPLANE_ALIASING_RISK",
        transform=dict(rotation_new_from_old=ROTATION.tolist(), determinant=det, x_throat_m=x_throat,
                       translation_new_z_m=-x_throat, definition="(x,y,z)_new = (y, z, x - x_throat)",
                       x_throat_derivation="E048 source_z_fft_m minus the centred 400 x 1.25 mm readout grid; source z = 0 is the throat plane",
                       source_offset_peak_to_peak_m=float(np.ptp(offset))),
        runner_construction=dict(x="zeros", y="zeros", z="Float32(-x_throat) for all spins",
                                 FlowPath="(dy, dz, dx) of the E055 inputs with unchanged resets and cycle map",
                                 float32_translation_rounding_m=float(abs(np.float32(-x_throat)-(-x_throat)))),
        spins=dict(physical=int(rho.size-1), sentinel_index_one_based=int(rho.size), time_samples=int(nt)),
        rotated_extent_m=dict(z_min=stats["z_min"], z_max=stats["z_max"]),
        inplane_check=dict(half_window_z_m=CHECK_HALF, inplane_half_fov_m=INPLANE_HALF_FOV, samples_checked=stats["checked_samples"],
                           max_abs_x_m=stats["inplane_max_abs_x"], max_abs_y_m=stats["inplane_max_abs_y"],
                           samples_at_or_beyond_inplane_fov=stats["outside_inplane_samples"]),
        spins_in_slab_per_sample=dict(half_m=SLAB_HALF, min=int(slab_counts.min()), max=int(slab_counts.max()), mean=float(slab_counts.mean())),
        spins_in_partition_fov_per_sample=dict(half_m=PARTITION_HALF_FOV, min=int(partition_counts.min()), max=int(partition_counts.max())),
        inputs=dict(case_inputs_sha256=info["neutral_input_sha256"], cycle_map_sha256=info["map_sha256"],
                    e048_local_fields_sha256=sha(local), script_sha256=sha(Path(__file__))))
    stage.mkdir()
    (stage/"report.json").write_text(json.dumps(report, indent=1)+"\n", encoding="utf-8")
    stage.rename(output)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
