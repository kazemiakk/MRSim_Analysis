"""E-066 S1: slab-restricted coupled-phantom export of the dp122 1M tracking (E-065 layout, native period).

Keeps particles whose |S| (E-060 rotation: S = MRI x + translation) is <= LIMIT at any of the 1440 steps. With
--within300k the candidate set is the E-055/E-065 300k identity selection (gate G1 control); otherwise all 1M.
Cycle map recomputed within the subset with the same upstream mapping module (E-055 practice for subsets).
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np

REPO = Path("D:/MRSim/MRSIM/MRSimTracks")
E055 = Path("D:/MRSim/outputs/phantom_density_e055")
E060 = Path("D:/MRSim/outputs/scanner_geometry_e060")
SOURCE = Path("D:/MRSim/outputs/dp122_native_e065/mri_pipeline_v1/dp122_motion_1m_mri.h5")
PERIOD = float(np.float32(0.72))
LIMIT = 0.060


def sha(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def toml_write(path, values):
    def lit(v):
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, str):
            return json.dumps(v)
        if isinstance(v, list):
            return "[" + ", ".join(lit(x) for x in v) + "]"
        return repr(v)
    path.write_text("\n".join(f"{k} = {lit(v)}" for k, v in values.items()) + "\n", encoding="utf-8")


def main(run, within300k):
    inputs, maps = run/"inputs_v1", run/"map_v1"
    if inputs.exists() or maps.exists():
        raise FileExistsError(inputs)
    tz = np.float32(json.loads((E060/"phantom_v1/report.json").read_text())["transform"]["translation_new_z_m"])
    with h5py.File(SOURCE) as src:
        pos, rst = src["position"], src["reset"]
        if pos.shape != (1440, 1000000, 3):
            raise ValueError("unexpected source shape")
        zmin = np.full(1000000, np.inf, np.float32)
        init_s = np.abs(src["initial_position"][:, 0]+tz); np.minimum(zmin, init_s, out=zmin)
        for t in range(1440):
            np.minimum(zmin, np.abs(pos[t, :, 0]+tz), out=zmin)
        cand = np.load(E055/"inputs_v1/selected_identity_indices.npy") if within300k else np.arange(1000000)
        selected = np.sort(cand[zmin[cand] <= LIMIT]).astype("<i8"); n, nt = len(selected), 1441
        inputs.mkdir(parents=True)
        np.save(inputs/"selected_identity_indices.npy", selected, allow_pickle=False)
        reset_events = 0
        with h5py.File(inputs/"case_inputs.h5.partial", "w") as out:
            initial = src["initial_position"][:][selected]
            for key, value, extra in [("x", 0., 0.), ("y", 0., 0.), ("z", 0., 0.), ("rho", 1., 0.), ("T1", np.float32(1.3), 1.3),
                                      ("T2", np.float32(.1), .1), ("delta_w", 0., 0.)]:
                out[key] = np.r_[np.full(n, value, np.float32), np.float32(extra)]
            out["time"] = np.array([0., PERIOD], np.float32); out["time_unit"] = np.array([0., 1.], np.float32)
            paths = {d: out.create_dataset(d, shape=(nt, n+1), dtype="f4", chunks=(1, 16384)) for d in ("dx", "dy", "dz")}
            resets = out.create_dataset("reset", shape=(nt, n+1), dtype="u1", chunks=(1, 16384))
            for c, d in enumerate(("dx", "dy", "dz")):
                paths[d][0, :] = np.r_[initial[:, c], np.float32(0)]
            resets[0, :] = 0
            for t in range(1440):
                positions = pos[t][selected]; real_reset = rst[t][selected]
                if not np.isfinite(positions).all():
                    raise ValueError("non-finite position")
                reset_events += int(real_reset.sum())
                for c, d in enumerate(("dx", "dy", "dz")):
                    paths[d][t+1, :] = np.r_[positions[:, c], np.float32(0)]
                flags = real_reset.copy()
                if t == 1439:
                    flags[:] = True
                resets[t+1, :] = np.r_[flags, np.any(flags)].astype("u1")
                if t % 240 == 0:
                    print("E066 exported step", t+1, "/ 1440", flush=True)
            final, restored = positions.copy(), real_reset.copy()
            out.attrs["physical_particles"] = n; out.attrs["t_start_s"] = 0.0; out.attrs["t_end_s"] = PERIOD
    (inputs/"case_inputs.h5.partial").rename(inputs/"case_inputs.h5")
    mapping_path = REPO/"outputs/upstream_migration_e046/tracker_source/mrsimtracks/mapping.py"
    spec = importlib.util.spec_from_file_location("e066_upstream_mapping", mapping_path)
    mapping = importlib.util.module_from_spec(spec); spec.loader.exec_module(mapping)
    indices = mapping.periodic_mapping(initial, final)
    if len(indices) != n or not np.all((indices >= 1) & (indices <= n)):
        raise ValueError("bad cycle map")
    maps.mkdir()
    distance = np.linalg.norm(initial-final[indices-1], axis=1)
    with h5py.File(maps/"cycle_map.h5", "w") as f:
        f["cycle_map"] = np.r_[indices, n+1].astype("i8"); f["restored_endpoint_reset"] = np.r_[restored, np.any(restored)].astype("u1")
        f["distance_m"] = distance
    info = dict(case="phantom_dp122", experiment="E-066", status="PASS_E066_RESTRICTED_EXPORT", period_s=PERIOD, seam_time_s=0.0,
                t_start_s=0.0, t_end_s=PERIOD, B0_T=3.0, physical_particles=n, total_spins_with_sentinel=n+1,
                artificial_global_reset_columns_one_based=[1441], velocity_scale=1.0, restriction_abs_S_max_m=LIMIT,
                candidate_set="E-055 300k identities" if within300k else "all 1M", candidates=int(len(cand)),
                neutral_input_sha256=sha(inputs/"case_inputs.h5"), map_sha256=sha(maps/"cycle_map.h5"),
                tracking_source_hdf5=str(SOURCE), exporter_sha256=sha(Path(__file__)), upstream_mapping_sha256=sha(mapping_path),
                source_real_reset_events=reset_events, map_distance_median_m=float(np.median(distance)),
                map_distance_p95_m=float(np.quantile(distance, .95)), map_distance_max_m=float(distance.max()),
                created_utc=datetime.now(timezone.utc).isoformat())
    toml_write(inputs/"case_info.toml", info)
    print(json.dumps({k: info[k] for k in ("status", "candidate_set", "candidates", "physical_particles", "map_distance_median_m", "map_distance_p95_m")}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--run", type=Path, required=True); p.add_argument("--within300k", action="store_true")
    a = p.parse_args(); main(a.run, a.within300k)
