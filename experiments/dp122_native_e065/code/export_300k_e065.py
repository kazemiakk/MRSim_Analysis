"""E-065 S2a: nested 300k export of the dp122 1M tracking, in the E-055 coupled-phantom layout, at the native period.

Layout as E-055 prepare_300k.py (same 300k identity indices, sentinel spin, rho 1, T1 1.3 s, T2 0.1 s, absolute
paths in dx/dy/dz with zero base positions, reset row 1441 set for every spin, nearest-neighbour cycle map from the
same upstream mapping module). Differences: source = dp122 tracking; no dp150 baseline-identity checks; period =
Float32(0.72) s (native CFD cycle, velocity scale 1), time = [0, period] before seam alignment.
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
SOURCE = Path("D:/MRSim/outputs/dp122_native_e065/mri_pipeline_v1/dp122_motion_1m_mri.h5")
PERIOD = float(np.float32(0.72))


def sha(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def toml_write(path, values):
    def literal(v):
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, str):
            return json.dumps(v)
        if isinstance(v, list):
            return "[" + ", ".join(literal(x) for x in v) + "]"
        return repr(v)
    path.write_text("\n".join(f"{k} = {literal(v)}" for k, v in values.items()) + "\n", encoding="utf-8")


def main(run):
    inputs, maps = run/"inputs_v1", run/"map_v1"
    if inputs.exists() or maps.exists():
        raise FileExistsError(inputs)
    report = json.loads((SOURCE.parent/"motion_1m_report.json").read_text())
    if not report["quality_control"]["acceptance"]["passed"]:
        raise ValueError("dp122 tracking QC not passed")
    selected = np.load(E055/"inputs_v1/selected_identity_indices.npy")
    n, nt = len(selected), 1441
    if n != 300000 or len(np.unique(selected)) != n:
        raise ValueError("unexpected identity selection")
    inputs.mkdir()
    np.save(inputs/"selected_identity_indices.npy", selected, allow_pickle=False)
    reset_events = 0
    with h5py.File(SOURCE) as src, h5py.File(inputs/"case_inputs.h5.partial", "w") as out:
        if src["position"].shape != (1440, 1000000, 3) or src["reset"].shape != (1440, 1000000):
            raise ValueError("unexpected source shape")
        initial = src["initial_position"][:][selected]
        for key, value, extra in [("x", 0., 0.), ("y", 0., 0.), ("z", 0., 0.), ("rho", 1., 0.), ("T1", np.float32(1.3), 1.3),
                                  ("T2", np.float32(.1), .1), ("delta_w", 0., 0.)]:
            out[key] = np.r_[np.full(n, value, np.float32), np.float32(extra)]
        out["time"] = np.array([0., PERIOD], np.float32)
        out["time_unit"] = np.array([0., 1.], np.float32)
        paths = {d: out.create_dataset(d, shape=(nt, n+1), dtype="f4", chunks=(1, 16384)) for d in ("dx", "dy", "dz")}
        resets = out.create_dataset("reset", shape=(nt, n+1), dtype="u1", chunks=(1, 16384))
        for c, d in enumerate(("dx", "dy", "dz")):
            paths[d][0, :] = np.r_[initial[:, c], np.float32(0)]
        resets[0, :] = 0
        for t in range(1440):
            positions = src["position"][t][selected]
            real_reset = src["reset"][t][selected]
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
                print("E065 exported step", t+1, "/ 1440", flush=True)
        final, restored = positions.copy(), real_reset.copy()
        out.attrs["physical_particles"] = n; out.attrs["t_start_s"] = 0.0; out.attrs["t_end_s"] = PERIOD
    (inputs/"case_inputs.h5.partial").rename(inputs/"case_inputs.h5")
    mapping_path = REPO/"outputs/upstream_migration_e046/tracker_source/mrsimtracks/mapping.py"
    spec = importlib.util.spec_from_file_location("e065_upstream_mapping", mapping_path)
    mapping = importlib.util.module_from_spec(spec); spec.loader.exec_module(mapping)
    indices = mapping.periodic_mapping(initial, final)
    if len(indices) != n or not np.all((indices >= 1) & (indices <= n)):
        raise ValueError("bad cycle map")
    maps.mkdir()
    distance = np.linalg.norm(initial-final[indices-1], axis=1)
    with h5py.File(maps/"cycle_map.h5", "w") as f:
        f["cycle_map"] = np.r_[indices, n+1].astype("i8"); f["restored_endpoint_reset"] = np.r_[restored, np.any(restored)].astype("u1")
        f["distance_m"] = distance
    info = dict(case="phantom_dp122", experiment="E-065", status="PASS_E065_300K_SOURCE_EXPORT", period_s=PERIOD, seam_time_s=0.0,
                t_start_s=0.0, t_end_s=PERIOD, B0_T=3.0, physical_particles=n, total_spins_with_sentinel=n+1,
                artificial_global_reset_columns_one_based=[1441], velocity_scale=1.0,
                neutral_input_sha256=sha(inputs/"case_inputs.h5"), map_sha256=sha(maps/"cycle_map.h5"),
                tracking_source_hdf5=str(SOURCE), tracking_report_sha256=sha(SOURCE.parent/"motion_1m_report.json"),
                exporter_sha256=sha(Path(__file__)), upstream_mapping_sha256=sha(mapping_path),
                source_real_reset_events=reset_events, map_distance_median_m=float(np.median(distance)),
                map_distance_p95_m=float(np.quantile(distance, .95)), map_distance_max_m=float(distance.max()),
                unique_map_source_fraction=float(len(np.unique(indices))/n), created_utc=datetime.now(timezone.utc).isoformat())
    toml_write(inputs/"case_info.toml", info)
    print(json.dumps({k: info[k] for k in ("status", "period_s", "source_real_reset_events", "map_distance_median_m", "map_distance_p95_m")}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--run", type=Path, required=True)
    main(p.parse_args().run)
