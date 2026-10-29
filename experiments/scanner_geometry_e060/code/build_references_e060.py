"""E-060 S4a: actual-ADC averaged CFD references on the rotated scanner grid.

Frozen E048/E022 functions (centred-FFT grid, Catmull-Rom ADC weights, voxel and
section lookups) with the E-060 grid: R/P/S = source (x, y, z) about the E022
centre, translated so the throat (source z = 0) is the partition centre, exactly
as the S1 phantom. Sections are planes normal to S (the vessel), sampled at the
partition centres. Gates: the generalized ADC-weight function must reproduce the
frozen one on the E055 legacy timing; the cycle-mean throat and upstream flows
must agree with the frozen E048 plane reference within 5% (mass conservation
makes the cycle-mean flow section-independent; residual is discretization).
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
from scripts.audit_phantom_reference import CenteredFFTGrid, timing_weights, catmull_weights  # noqa: E402
from scripts.dp150_hemodynamic_reference import read_e022_contract, load_periodic_reference, build_voxel_lookup  # noqa: E402
from scripts.diagnose_phantom_signal import ROOT, PROD  # noqa: E402
from scripts.build_aorta_inlet_picker import sha  # noqa: E402

E060 = Path("D:/MRSim/outputs/scanner_geometry_e060")
E055 = Path("D:/MRSim/outputs/phantom_density_e055")
E048 = ROOT/"outputs/phantom_diagnosis_e048"
SHAPE_SPR = (30, 64, 64)
FOV_RPS = (.08, .08, .075)
BINS, PROFILES, SAMPLES = 30, 64*30, 64
FLOW_GATE = .05
UPSTREAM_Z = (-.030, -.020)   # E-059 upstream slices: 20-30 mm upstream of the throat


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def timing_weights_general(path, period, seam, bins, profiles, samples):
    """Frozen timing_weights with the bin/profile/sample counts as parameters (body otherwise identical)."""
    with h5py.File(path) as f:
        g = f['REF']; starts = g['adc_first_s'][:]; dwell = g['dwell_s'][:]; n = g['n'][:]; labels = g['labels'][:]
        for e in ('X', 'Y', 'Z'):
            if not np.array_equal(labels, f[e]['labels'][:]) or not np.allclose(starts, f[e]['adc_first_s'][:], atol=1e-9, rtol=0):
                raise ValueError('Encoding timing differs')
    weights = []; weights32 = []; metadata = []
    for b in range(bins):
        ids = np.flatnonzero(labels[:, 2] == b)
        if len(ids) != profiles or not np.all(n[ids] == samples):
            raise ValueError('Missing bin encoding labels')
        times = (starts[ids, None]+np.arange(samples)[None, :]*dwell[ids, None]).ravel()
        phase = np.mod(times-seam, period)/period
        phase32 = (np.mod(times.astype(np.float32)-np.float32(seam), np.float32(period))/np.float32(period)).astype(float)
        weights.append(catmull_weights(phase)); weights32.append(catmull_weights(phase32))
        midpoint = (b+.5)/bins
        error = ((phase-midpoint+.5) % 1-.5)*period
        metadata.append(dict(bin=b+1, ADC_samples=len(times), nominal_source_phase_s=midpoint*.72,
                             mean_offset_from_midpoint_s=float(error.mean()), minimum_offset_s=float(error.min()), maximum_offset_s=float(error.max()),
                             float32_vs_float64_phase_rms_s=float(np.sqrt(np.mean((((phase32-phase+.5) % 1-.5)*period)**2)))))
    return np.array(weights), np.array(weights32), metadata


def section_flows(plane_ref, area_mm2, axis):
    """Per-bin, per-plane flow in mL/s (m/s x mm^2); plane_ref (bins, S, P, R, 3), area on the plane axis."""
    v = np.nan_to_num(plane_ref[..., axis].astype(np.float64))
    if axis == 2:
        return np.einsum("bspr,spr->bs", v, area_mm2)
    return np.einsum("bspr,spr->br", v, area_mm2)


def main():
    target = E060/"references_v1"
    stage = target.with_name(target.name+".building")
    if target.exists() or stage.exists():
        raise FileExistsError(target)
    if SOURCE_REPO.resolve() != ROOT.resolve():
        raise ValueError("Frozen helper import resolved outside the reviewed repository")
    s0 = tomllib.loads((E060/"sequences_v2/report.toml").read_text(encoding="utf-8"))  # amendment A2 (TBP 12)
    timing_file = E060/"sequences_v2/timing_e060.h5"
    if s0["status"] != "PASS_E060_SEQUENCES_AND_AUDITS" or s0["hardware"]["TBP"] != 12 or sha(timing_file) != s0["timing_sha256"]:
        raise ValueError("E-060 S0 v2 missing or timing changed")
    s1 = json.loads((E060/"phantom_v1/report.json").read_text())
    if s1["status"] != "PASS_E060_PHANTOM_ROTATION":
        raise ValueError("S1 rotation missing")
    translation = s1["transform"]["translation_new_z_m"]
    for name, folder, digest in [("reference_fields.h5", E048/"reference_v1", "f4c4ec3153d30a2a9dfd360d8ff1c21811e14e6f942697cdd990b669fec75694"),
                                 ("local_fields.h5", E048/"local_v1", "6c57a8708ea4a5aff09f61db232b0ba1528ae4d000bf60d9ca00524516a27885")]:
        if sha(folder/name) != digest:
            raise ValueError("Frozen E048 reference changed: "+name)
    info = tomllib.loads((E055/"inputs_v1/case_info.toml").read_text())
    legacy = E055/"inputs_v1/legacy_timing.h5"
    # Gate 1: generalized weights reproduce the frozen function exactly on the 15-bin legacy timing.
    frozen = timing_weights(legacy, info["period_s"], info["seam_time_s"])
    general = timing_weights_general(legacy, info["period_s"], info["seam_time_s"], 15, 2048, 400)
    weights_identity = bool(np.array_equal(frozen[0], general[0]) and np.array_equal(frozen[1], general[1]) and frozen[2] == general[2])
    if not weights_identity:
        raise AssertionError("Generalized ADC weights differ from the frozen function")
    pipe = ROOT/"outputs/dp150_mri_pipeline"
    support = PROD/"final_image_analysis/support_constrained_phase/dp150_seam_safe_support_phase.h5"
    if not support.exists():
        from scripts.compare_dp150_hemodynamics import H5
        support = H5
    old_grid, timing = read_e022_contract(pipe/"motion_1m_report.json", PROD/"simulation_report.toml", support)
    if tuple(old_grid.source_axis_order) != (2, 0, 1):
        raise ValueError("Unexpected E022 axis order")
    cx, cy, cz = old_grid.source_center_m
    # Old Koma x = source z - cz; S1 sets new z = old x + translation, so the new S axis is centred at cz - translation.
    grid = CenteredFFTGrid(SHAPE_SPR, FOV_RPS, (cx, cy, cz-translation), (0, 1, 2))
    if abs(cz-translation) > 1e-8:
        raise ValueError("E022 centre and S1 translation disagree on the throat position")
    if not np.isclose(timing["velocity_scale"], .72/info["period_s"], rtol=1e-12, atol=0):
        raise ValueError("Velocity scale differs")
    pvd = pipe/"flow_periodic/dp150_periodic.pvd"
    if sha(pvd) != "342c2cfd195048ee1a469df10d986a6219a8bbac6ea967c83cfff9b8436ae39a":
        raise ValueError("Wrong matched CFD")
    files = [pvd.parent/x.attrib["file"] for x in ET.parse(pvd).findall(".//DataSet")]
    print("E060 load matched periodic CFD", flush=True)
    flow = load_periodic_reference(pvd)
    print("E060 build volume and S-normal section lookups", flush=True)
    volume = build_voxel_lookup(flow.active_mesh, grid, subsamples=2)
    plane = build_voxel_lookup(flow.active_mesh, grid, subsamples=(2, 2, 1))
    frames = [grid.velocity_to_mri(np.asarray(flow._frame_vel(frame))) for frame in range(72)]
    weights, _, metadata = timing_weights_general(timing_file, info["period_s"], info["seam_time_s"], BINS, PROFILES, SAMPLES)
    if not np.allclose(weights.sum(axis=1), 1, atol=1e-12, rtol=0):
        raise AssertionError("Weights not unity")
    nodes = np.zeros((BINS, flow.active_mesh.n_points, 3), np.float64)
    for frame, velocity in enumerate(frames):
        nodes += weights[:, frame, None, None]*(velocity*timing["velocity_scale"])[None]
    fft_adc = np.empty((BINS, *SHAPE_SPR, 3), np.float32)
    plane_ref = np.empty_like(fft_adc)
    for b in range(BINS):
        print("E060 reference bin", b+1, flush=True)
        fft_adc[b] = np.moveaxis(volume.average(nodes[b])["velocity_mri"], 0, -1)
        plane_ref[b] = np.moveaxis(plane.average(nodes[b])["velocity_mri"], 0, -1)
    fraction, plane_fraction = volume.fraction_grid(), plane.fraction_grid()
    z = (np.arange(SHAPE_SPR[0])-SHAPE_SPR[0]//2)*FOV_RPS[2]/SHAPE_SPR[0]
    q_new = section_flows(plane_ref, plane_fraction*1.5625, axis=2).mean(axis=0)
    # Frozen E048 plane reference (axial = R, area fraction x dP*dS = 3.125 mm^2) at source z.
    with h5py.File(E048/"local_v1/local_fields.h5") as f:
        old_plane, old_fraction, source_z = f["plane_reference"][:], f["plane_fraction"][:], f["source_z_fft_m"][:]
    q_old = section_flows(old_plane, old_fraction*3.125, axis=0).mean(axis=0)
    throat_new, throat_old = int(np.argmin(abs(z))), int(np.argmin(abs(source_z)))
    up_new = [k for k in range(len(z)) if UPSTREAM_Z[0]-1e-9 <= z[k] <= UPSTREAM_Z[1]+1e-9]
    up_old = [i for i in range(len(source_z)) if UPSTREAM_Z[0]-1e-9 <= source_z[i] <= UPSTREAM_Z[1]+1e-9]
    checks = dict(
        throat_cycle_mean_flow_ml_s=dict(E060=float(q_new[throat_new]), E048=float(q_old[throat_old]),
                                         relative=float(q_new[throat_new]/q_old[throat_old]-1)),
        upstream_cycle_mean_flow_ml_s=dict(E060=float(q_new[up_new].mean()), E048=float(q_old[up_old].mean()),
                                           relative=float(q_new[up_new].mean()/q_old[up_old].mean()-1)),
        E060_slab_section_cycle_mean_cv=float(np.std(q_new[np.abs(z) <= .03])/np.mean(q_new[np.abs(z) <= .03])),
        throat_partition_index=throat_new, upstream_partition_indices=up_new)
    passed = abs(checks["throat_cycle_mean_flow_ml_s"]["relative"]) <= FLOW_GATE and abs(checks["upstream_cycle_mean_flow_ml_s"]["relative"]) <= FLOW_GATE
    report = dict(experiment="E-060", stage="S4a_references",
        status="PASS_E060_REFERENCES" if passed else "FAIL_E060_REFERENCE_FLOW_CROSSCHECK",
        utc=datetime.now(timezone.utc).isoformat(), weights_identity_with_frozen_function=weights_identity,
        grid=dict(shape_spr=SHAPE_SPR, fov_rps_m=FOV_RPS, source_center_m=list(grid.source_center_m), source_axis_order=[0, 1, 2],
                  convention="centred FFT (i - n/2) * d; R/P/S = source x/y/z; S = vessel axis"),
        partition_z_m=z.tolist(), flow_crosscheck=checks, flow_gate=FLOW_GATE, adc_timing=metadata,
        section_cycle_mean_flow_ml_s=q_new.tolist(),
        evidence_class="CFD comparator construction with frozen E048 functions; not MRI data.", new_1m=False)
    stage.mkdir()
    with h5py.File(stage/"references.h5", "w") as f:
        f.create_dataset("fft_adc", data=fft_adc, compression="gzip", compression_opts=1)
        f.create_dataset("plane_reference", data=plane_ref, compression="gzip", compression_opts=1)
        f.create_dataset("adc_weights", data=weights)
        f.create_dataset("fraction_fft", data=fraction, compression="gzip", compression_opts=1)
        f.create_dataset("plane_fraction", data=plane_fraction, compression="gzip", compression_opts=1)
        f.create_dataset("partition_z_m", data=z)
    dump(stage/"report.json", report)
    inputs = [Path(__file__), E060/"code/PROTOCOL.md", E060/"code/AMENDMENT_A2_SLAB_GPU.md", E060/"sequences_v2/report.toml", timing_file, legacy,
              E060/"phantom_v1/report.json", E048/"local_v1/local_fields.h5", pvd, *files]
    dump(stage/"MANIFEST.json", dict(inputs={str(p): sha(p) for p in inputs},
                                     outputs={p.name: sha(p) for p in stage.iterdir() if p.is_file()}))
    if not passed:
        stage.rename(target.with_name(target.name+".failed"))
        raise AssertionError("Reference flow cross-check failed: "+json.dumps(checks))
    stage.rename(target)
    print(json.dumps(dict(status=report["status"], checks=checks)), flush=True)


if __name__ == "__main__":
    main()
