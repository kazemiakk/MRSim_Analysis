"""E-065 S4a: actual-ADC averaged dp122 CFD references on the E-065 scanner grid (native 0.72-s period, scale 1).

As E-060 build_references_e060_v2.py (frozen E048/E022 helpers: centred-FFT grid, Catmull-Rom ADC weights, voxel
and S-normal section lookups), with the dp122 periodic flow, the E-065 sequences/seam and velocity_scale = 1. The
grid is the E-060 references_v2 grid (R/P/S = source x/y/z, throat at S = 0); dp122 has the same tube bounds.
"""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tomllib

import h5py
import numpy as np

REPO = Path("D:/MRSim/MRSIM/MRSimTracks")
sys.path.insert(0, str(REPO))
from scripts.audit_phantom_reference import CenteredFFTGrid  # noqa: E402
from scripts.dp150_hemodynamic_reference import load_periodic_reference, build_voxel_lookup  # noqa: E402

ROOT = Path("D:/MRSim/outputs/dp122_native_e065")
E060 = Path("D:/MRSim/outputs/scanner_geometry_e060")
PVD = ROOT/"mri_pipeline_v1/flow_periodic/dp122_periodic.pvd"
BINS, PROFILES, SAMPLES = 30, 64*30, 64
_spec = importlib.util.spec_from_file_location("e060_refs", E060/"code/build_references_e060_v2.py")
e060 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(e060)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def main(target):
    target = Path(target); stage = target.with_name(target.name+".building")
    if target.exists() or stage.exists():
        raise FileExistsError(target)
    s0 = tomllib.loads((ROOT/"sequences_v1/report.toml").read_text())
    seam = tomllib.loads((ROOT/"seam_v1/report.toml").read_text())
    info = tomllib.loads((ROOT/"inputs_v1/case_info.toml").read_text())
    timing = ROOT/"sequences_v1/timing_e060.h5"
    if s0["status"] != "PASS_E065_SEQUENCES_AND_AUDITS" or sha(timing) != s0["timing_sha256"] or seam["status"] != "PASS_E065_SEAM_ALIGNMENT":
        raise ValueError("E-065 sequences/seam missing or changed")
    if seam["period_s"] != info["period_s"]:
        raise ValueError("seam period differs from the exported period")
    g = json.loads((E060/"references_v2/report.json").read_text())["grid"]
    grid = CenteredFFTGrid(tuple(g["shape_spr"]), tuple(g["fov_rps_m"]), tuple(g["source_center_m"]), tuple(g["source_axis_order"]))
    print("E065 load dp122 periodic CFD", flush=True)
    flow = load_periodic_reference(PVD)
    volume = build_voxel_lookup(flow.active_mesh, grid, subsamples=2)
    plane = build_voxel_lookup(flow.active_mesh, grid, subsamples=(2, 2, 1))
    frames = [grid.velocity_to_mri(np.asarray(flow._frame_vel(f))) for f in range(72)]
    weights, _, metadata = e060.timing_weights_general(timing, info["period_s"], seam["seam_time_s"], BINS, PROFILES, SAMPLES)
    if not np.allclose(weights.sum(axis=1), 1, atol=1e-12, rtol=0):
        raise AssertionError("weights not unity")
    nodes = np.zeros((BINS, flow.active_mesh.n_points, 3))
    for f, v in enumerate(frames):
        nodes += weights[:, f, None, None]*v[None]          # velocity_scale = 1 (native period)
    shape = tuple(g["shape_spr"])
    fft_adc = np.empty((BINS, *shape, 3), np.float32); plane_ref = np.empty_like(fft_adc)
    for b in range(BINS):
        print("E065 reference bin", b+1, flush=True)
        fft_adc[b] = np.moveaxis(volume.average(nodes[b])["velocity_mri"], 0, -1)
        plane_ref[b] = np.moveaxis(plane.average(nodes[b])["velocity_mri"], 0, -1)
    fraction, plane_fraction = volume.fraction_grid(), plane.fraction_grid()
    z = (np.arange(shape[0])-shape[0]//2)*g["fov_rps_m"][2]/shape[0]
    q = e060.section_flows(plane_ref, plane_fraction*1.5625, axis=2)            # (bins, S) mL/s
    up = [k for k in range(len(z)) if -.030-1e-9 <= z[k] <= -.020+1e-9]
    rep = dict(experiment="E-065", stage="S4a_references", status="PASS_E065_REFERENCES", utc=datetime.now(timezone.utc).isoformat(),
               script_sha256=sha(__file__), grid=g, period_s=info["period_s"], seam_time_s=seam["seam_time_s"], velocity_scale=1.0,
               partition_z_m=z.tolist(), adc_timing=metadata, upstream_flow_ml_s=q[:, up].mean(axis=1).tolist(),
               throat_flow_ml_s=q[:, int(np.argmin(abs(z)))].tolist(), section_cycle_mean_flow_ml_s=q.mean(axis=0).tolist(),
               evidence_class="CFD comparator construction with frozen E048/E060 functions; not MRI data.")
    stage.mkdir()
    with h5py.File(stage/"references.h5", "w") as f:
        for k, v in (("fft_adc", fft_adc), ("plane_reference", plane_ref), ("fraction_fft", fraction), ("plane_fraction", plane_fraction)):
            f.create_dataset(k, data=v, compression="gzip", compression_opts=1)
        f["adc_weights"] = weights; f["partition_z_m"] = z
    (stage/"report.json").write_text(json.dumps(rep, indent=1)+chr(10), encoding="utf-8")
    stage.rename(target)
    print("PASS_E065_REFERENCES upstream peak", round(max(rep["upstream_flow_ml_s"]), 1), "mean", round(float(np.mean(rep["upstream_flow_ml_s"])), 1), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
