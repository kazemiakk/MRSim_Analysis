"""E-058 S1: actual-ADC averaged CFD references at each arm's own ADC timing.

Uses the frozen E048 functions unchanged. Identity gate first: the original legacy
timing must reproduce the saved E048 volume and plane references.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tomllib
import xml.etree.ElementTree as ET

import h5py
import numpy as np

SOURCE_REPO = Path("D:/MRSim/MRSIM/MRSimTracks")
sys.path.insert(0, str(SOURCE_REPO))
from scripts.audit_phantom_reference import CenteredFFTGrid, timing_weights  # noqa: E402
from scripts.dp150_hemodynamic_reference import (read_e022_contract, load_periodic_reference,  # noqa: E402
                                                 build_voxel_lookup, build_section_lookup)
from scripts.diagnose_phantom_signal import ROOT, PROD, E047  # noqa: E402
from scripts.build_aorta_inlet_picker import sha  # noqa: E402

E058 = Path("D:/MRSim/outputs/paired_acquisition_e058")
E048 = ROOT/"outputs/phantom_diagnosis_e048"
E055 = Path("D:/MRSim/outputs/phantom_density_e055")
TOLERANCE = 5e-5


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def identity(rebuilt, saved):
    same_gaps = bool(np.array_equal(np.isfinite(rebuilt), np.isfinite(saved)))
    finite = np.isfinite(rebuilt) & np.isfinite(saved)
    return same_gaps, float(np.max(np.abs(rebuilt[finite].astype(float)-saved[finite].astype(float))))


def main():
    target = E058/"references_v1"
    stage = target.with_name(target.name+".building")
    if target.exists() or stage.exists():
        raise FileExistsError(target)
    if SOURCE_REPO.resolve() != ROOT.resolve():
        raise ValueError("Frozen helper import resolved outside the reviewed repository")
    sequences = tomllib.loads((E058/"sequences_v1/report.toml").read_text(encoding="utf-8"))
    candidate_timing = E058/"sequences_v1/short_echo_timing.h5"
    if sequences["status"] != "PASS_E058_SEQUENCE_CERTIFICATION_AND_CANDIDATE_AUDIT" or sha(candidate_timing) != sequences["short_echo_timing_sha256"]:
        raise ValueError("S0 sequence certification missing or candidate timing changed")
    for name, folder, digest in [("reference_fields.h5", E048/"reference_v1", "f4c4ec3153d30a2a9dfd360d8ff1c21811e14e6f942697cdd990b669fec75694"),
                                 ("local_fields.h5", E048/"local_v1", "6c57a8708ea4a5aff09f61db232b0ba1528ae4d000bf60d9ca00524516a27885")]:
        manifest = json.loads((folder/"MANIFEST.json").read_text())["outputs"]
        if manifest[name] != digest or sha(folder/name) != digest:
            raise ValueError("Frozen E048 reference changed: "+name)
    legacy = E047/"phantom_inputs_v1/legacy_timing.h5"
    with h5py.File(legacy) as a, h5py.File(E055/"inputs_v1/legacy_timing.h5") as b:
        for e in ("REF", "X", "Y", "Z"):
            for k in ("adc_first_s", "dwell_s", "n", "labels"):
                if not np.array_equal(a[e][k][()], b[e][k][()]):
                    raise ValueError("E047 and E055 legacy timing differ")
    info = tomllib.loads((E047/"phantom_inputs_v1/case_info.toml").read_text())
    info55 = tomllib.loads((E055/"inputs_v1/case_info.toml").read_text())
    if info["period_s"] != info55["period_s"] or info["seam_time_s"] != info55["seam_time_s"]:
        raise ValueError("Period/seam contract differs from E055")
    pipe = ROOT/"outputs/dp150_mri_pipeline"
    support = PROD/"final_image_analysis/support_constrained_phase/dp150_seam_safe_support_phase.h5"
    if not support.exists():
        from scripts.compare_dp150_hemodynamics import H5
        support = H5
    grid, timing = read_e022_contract(pipe/"motion_1m_report.json", PROD/"simulation_report.toml", support)
    fftgrid = CenteredFFTGrid(grid.shape_spr, grid.fov_rps_m, grid.source_center_m, grid.source_axis_order)
    if not np.isclose(timing["velocity_scale"], .72/info["period_s"], rtol=1e-12, atol=0):
        raise ValueError("Volume and plane velocity scales differ")
    pvd = pipe/"flow_periodic/dp150_periodic.pvd"
    if sha(pvd) != "342c2cfd195048ee1a469df10d986a6219a8bbac6ea967c83cfff9b8436ae39a":
        raise ValueError("Wrong matched CFD")
    files = [pvd.parent/x.attrib["file"] for x in ET.parse(pvd).findall(".//DataSet")]
    print("E058 load matched periodic CFD", flush=True)
    flow = load_periodic_reference(pvd)
    print("E058 build centered-FFT volume and plane lookups", flush=True)
    volume = build_voxel_lookup(flow.active_mesh, fftgrid, subsamples=2)
    plane = build_section_lookup(flow.active_mesh, fftgrid, transverse_subsamples=2)
    frames = [grid.velocity_to_mri(np.asarray(flow._frame_vel(frame))) for frame in range(72)]
    results, weights_by_arm = {}, {}
    for arm, path in (("baseline", legacy), ("short_echo", candidate_timing)):
        weights, _, metadata = timing_weights(path, info["period_s"], info["seam_time_s"])
        if not np.allclose(weights.sum(axis=1), 1, atol=1e-12, rtol=0):
            raise AssertionError("Weights not unity")
        nodes_volume = np.zeros((15, flow.active_mesh.n_points, 3), np.float64)
        nodes_plane = np.zeros_like(nodes_volume)
        for frame, velocity in enumerate(frames):
            nodes_volume += weights[:, frame, None, None]*(velocity*timing["velocity_scale"])[None]
            nodes_plane += weights[:, frame, None, None]*(velocity*(.72/info["period_s"]))[None]
        fft_adc = np.empty((15, *grid.shape_spr, 3), np.float32)
        for b in range(15):
            print("E058 reference", arm, "bin", b+1, flush=True)
            fft_adc[b] = np.moveaxis(volume.average(nodes_volume[b])["velocity_mri"], 0, -1)
        plane_ref = np.stack([np.moveaxis(plane.average(node)["velocity_mri"], 0, -1) for node in nodes_plane])
        results[arm] = (fft_adc, plane_ref, metadata)
        weights_by_arm[arm] = weights
    with h5py.File(E048/"reference_v1/reference_fields.h5") as f:
        saved_volume, fraction = f["fft_adc"][:], f["fraction_fft"][:]
        saved_weights = f["adc_weights"][:]
    with h5py.File(E048/"local_v1/local_fields.h5") as f:
        saved_plane, plane_fraction = f["plane_reference"][:], f["plane_fraction"][:]
    gaps_v, err_v = identity(results["baseline"][0], saved_volume)
    gaps_p, err_p = identity(results["baseline"][1], saved_plane)
    weights_identical = bool(np.array_equal(weights_by_arm["baseline"], saved_weights))
    if not (gaps_v and gaps_p and err_v <= TOLERANCE and err_p <= TOLERANCE and weights_identical):
        raise AssertionError(f"E048 reference not reproduced: volume {err_v} plane {err_p} gaps {gaps_v}/{gaps_p} weights {weights_identical}")
    eligible = fraction > 0
    delta = results["short_echo"][0].astype(float)-results["baseline"][0].astype(float)
    finite = np.isfinite(delta).all(axis=-1) & eligible[None]
    vector = np.linalg.norm(delta, axis=-1)[finite]
    report = dict(experiment="E-058", stage="S1_references", status="PASS_E058_REFERENCE_IDENTITY_AND_CANDIDATE_TIMING",
        utc=datetime.now(timezone.utc).isoformat(),
        identity=dict(volume_max_abs_m_s=err_v, plane_max_abs_m_s=err_p, gap_patterns_identical=gaps_v and gaps_p,
                      adc_weights_identical=weights_identical, tolerance_m_s=TOLERANCE),
        timing_sensitivity=dict(description="short_echo own-ADC reference minus baseline-ADC reference, eligible voxels, all bins",
            vector_rms_m_s=float(np.sqrt(np.mean(vector**2))), vector_max_m_s=float(vector.max())),
        baseline_timing=results["baseline"][2], short_echo_timing=results["short_echo"][2],
        evidence_class="CFD comparator construction with frozen E048 functions; not MRI data.", new_1m=False)
    stage.mkdir()
    with h5py.File(stage/"references.h5", "w") as f:
        for arm, (volume_ref, plane_ref, _) in results.items():
            f.create_dataset("fft_adc_"+arm, data=volume_ref, compression="gzip", compression_opts=1)
            f.create_dataset("plane_reference_"+arm, data=plane_ref, compression="gzip", compression_opts=1)
            f.create_dataset("adc_weights_"+arm, data=weights_by_arm[arm])
        f.create_dataset("fraction_fft", data=fraction, compression="gzip", compression_opts=1)
        f.create_dataset("plane_fraction", data=plane_fraction, compression="gzip", compression_opts=1)
    dump(stage/"report.json", report)
    inputs = [Path(__file__), E058/"code/PROTOCOL.md", E058/"sequences_v1/report.toml", candidate_timing, legacy,
              E048/"reference_v1/reference_fields.h5", E048/"local_v1/local_fields.h5", pvd, *files]
    dump(stage/"MANIFEST.json", dict(inputs={str(p): sha(p) for p in inputs},
                                     outputs={p.name: sha(p) for p in stage.iterdir() if p.is_file()}))
    stage.rename(target)
    print(json.dumps(dict(status=report["status"], identity=report["identity"], timing_sensitivity=report["timing_sensitivity"])), flush=True)


if __name__ == "__main__":
    main()
