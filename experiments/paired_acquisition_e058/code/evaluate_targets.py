"""E-058 amendment A1: evaluate the user's throat targets per arm.

Definitions are fixed in code/TARGETS_AMENDMENT.md (registered before any E-058
analysis). Reuses the frozen E050/E055 helpers; no fitting, filling or tuning.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tomllib

import h5py
import numpy as np

SOURCE_REPO = Path("D:/MRSim/MRSIM/MRSimTracks")
sys.path.insert(0, str(SOURCE_REPO))
from scripts.analyze_phantom_precision import load_arm, bind_reference, coverage  # noqa: E402
from scripts.analyze_remapped_confirmation import fixed_mask, velocities  # noqa: E402
from scripts.build_aorta_inlet_picker import sha  # noqa: E402
from scripts.compare_dp150_hemodynamics import PLANES, REGIONS  # noqa: E402

E055 = Path("D:/MRSim/outputs/phantom_density_e055")
E058 = Path("D:/MRSim/outputs/paired_acquisition_e058")
E048 = SOURCE_REPO/"outputs/phantom_diagnosis_e048"
ARMS = dict(
    baseline_coarse=dict(folder=E055/"mapped_f64_v1", status="COMPLETE_E055_F64_300K_MAPPED_ACQUISITIONS", timing="baseline"),
    baseline_fine=dict(folder=E058/"acquisition_baseline_v1", status="COMPLETE_E058_F64_300K_BASELINE_FINE_ACQUISITIONS", timing="baseline"),
    short_echo_fine=dict(folder=E058/"acquisition_short_echo_v1", status="COMPLETE_E058_F64_300K_SHORT_ECHO_FINE_ACQUISITIONS", timing="short_echo"))
TARGET, COVERAGE, THRESHOLD = .08, .90, .05


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def per_bin_support(images):
    support = None
    for image in images.values():
        magnitude = np.abs(image)
        ok = (magnitude >= THRESHOLD*magnitude.max(axis=(1, 2, 3), keepdims=True)) & (magnitude > 0)
        support = ok if support is None else support & ok
    return support


def evaluate(field, support, volume_ref, plane_ref, fraction, area, region, idx, peak_bin):
    """support is (bins, slice, phase, readout); plane_ref is (bins, slice, phase) at idx."""
    full = area > 0
    eligible = fraction > 0
    bins, koma_sum, same_sum, full_sum = [], 0., 0., 0.
    peak_flow = max(abs(float(plane_ref[b][full]@area[full])) for b in range(len(plane_ref)))
    for b in range(len(plane_ref)):
        keep = support[b][:, :, idx] & full
        if not np.isfinite(plane_ref[b][full]).all():
            raise ValueError("Nonfinite CFD inside the throat section")
        cfd_same = float(plane_ref[b][keep]@area[keep]) if keep.any() else 0.
        cfd_full = float(plane_ref[b][full]@area[full])
        koma = float(field[b, :, :, idx, 0][keep]@area[keep]) if keep.any() else 0.
        koma_sum += koma; same_sum += cfd_same; full_sum += cfd_full
        bins.append(dict(bin=b+1, section_area_coverage=float(np.sum(area[keep], dtype=float)/np.sum(area[full], dtype=float)),
            Koma_ml_s=koma, CFD_same_area_ml_s=cfd_same, CFD_full_section_ml_s=cfd_full,
            same_area_relative_error=None if abs(cfd_same) < .1*peak_flow else koma/cfd_same-1,
            # Same frozen helper (float64 sums) that produced E055's published bin_coverage.
            whole_lumen_coverage=coverage(fraction, support[b])))
    rel = [r["same_area_relative_error"] for r in bins if r["same_area_relative_error"] is not None]
    t1 = dict(cycle_same_area_error=koma_sum/same_sum-1, cycle_vs_full_section_error=koma_sum/full_sum-1,
              peak_bin_same_area_error=bins[peak_bin]["same_area_relative_error"],
              worst_bin_same_area_error=max(rel, key=abs) if rel else None,
              min_section_area_coverage=min(r["section_area_coverage"] for r in bins), bins=bins)
    reg_b = region & support[peak_bin]
    axial_k, axial_c = field[peak_bin][..., 0], volume_ref[peak_bin][..., 0]
    t2 = dict(peak_bin=peak_bin+1, supported_voxels=int(reg_b.sum()), region_voxels=int(region.sum()),
              Koma_max_m_s=float(axial_k[reg_b].max()) if reg_b.any() else None,
              CFD_region_max_m_s=float(np.nanmax(axial_c[region])),
              CFD_same_support_max_m_s=float(axial_c[reg_b].max()) if reg_b.any() else None,
              Koma_p99_m_s=float(np.percentile(axial_k[reg_b], 99)) if reg_b.any() else None,
              CFD_region_p99_m_s=float(np.nanpercentile(axial_c[region], 99)))
    t2["max_error"] = None if t2["Koma_max_m_s"] is None else t2["Koma_max_m_s"]/t2["CFD_region_max_m_s"]-1
    t2["max_error_same_support"] = None if t2["Koma_max_m_s"] is None else t2["Koma_max_m_s"]/t2["CFD_same_support_max_m_s"]-1
    t2["p99_error"] = None if t2["Koma_p99_m_s"] is None else t2["Koma_p99_m_s"]/t2["CFD_region_p99_m_s"]-1
    t3 = dict(min_section_area_coverage=t1["min_section_area_coverage"],
              throat_region_coverage_peak_bin=float(np.sum(fraction[reg_b], dtype=float)/np.sum(fraction[region], dtype=float)),
              mean_whole_lumen_coverage=float(np.mean([r["whole_lumen_coverage"] for r in bins])))
    passes = dict(T1=bool(abs(t1["cycle_same_area_error"]) <= TARGET and t1["min_section_area_coverage"] >= COVERAGE),
                  T2=bool(t2["max_error"] is not None and abs(t2["max_error"]) <= TARGET),
                  T3=bool(t3["min_section_area_coverage"] >= COVERAGE and t3["throat_region_coverage_peak_bin"] >= COVERAGE))
    passes["ALL"] = all(passes.values())
    return dict(T1_throat_flow=t1, T2_peak_throat_velocity=t2, T3_coverage=t3, passes=passes)


def main(selection):
    output = E058/("targets_e055_v1" if selection == "e055" else "targets_v1")
    stage = output.with_name(output.name+".building")
    if output.exists() or stage.exists():
        raise FileExistsError(output)
    arms = ["baseline_coarse"] if selection == "e055" else list(ARMS)
    info = tomllib.loads((E055/"inputs_v1/case_info.toml").read_text())
    ref48 = bind_reference(E048/"reference_v1", "reference_fields.h5", "f4c4ec3153d30a2a9dfd360d8ff1c21811e14e6f942697cdd990b669fec75694")
    local48 = bind_reference(E048/"local_v1", "local_fields.h5", "6c57a8708ea4a5aff09f61db232b0ba1528ae4d000bf60d9ca00524516a27885")
    with h5py.File(ref48) as f:
        fraction = f["fraction_fft"][:]
    with h5py.File(local48) as f:
        source_z = f["source_z_fft_m"][:]
        idx = int(np.argmin(abs(source_z-PLANES["throat"])))
        area = f["plane_fraction"][:, :, idx]*3.125
        plane_baseline = f["plane_reference"][:, :, :, idx, 0]
    low, high = REGIONS["throat"]
    region = (fraction > 0) & ((source_z >= low) & (source_z < high))[None, None, :]
    full = area > 0
    peak_bin = int(np.argmax([plane_baseline[b][full]@area[full] for b in range(15)]))
    results, inputs = {}, [Path(__file__), E058/"code/TARGETS_AMENDMENT.md", ref48, local48]
    for arm in arms:
        spec = ARMS[arm]
        report = tomllib.loads((spec["folder"]/"control_report.toml").read_text())
        if report["status"] != spec["status"]:
            raise ValueError("Incomplete arm: "+arm)
        inputs.append(spec["folder"]/"control_report.toml")
        print("E058_TARGETS", arm, flush=True)
        images, _ = load_arm(spec["folder"], report, candidate=True)
        field = velocities(images, info["Venc_m_s"])
        if spec["timing"] == "baseline":
            with h5py.File(ref48) as f:
                volume_ref = f["fft_adc"][:]
            plane_ref = plane_baseline
        else:
            own = E058/"references_v1/references.h5"
            manifest = json.loads((E058/"references_v1/MANIFEST.json").read_text())["outputs"]
            if sha(own) != manifest["references.h5"]:
                raise ValueError("S1 references changed")
            inputs.append(own)
            with h5py.File(own) as f:
                volume_ref = f["fft_adc_short_echo"][:]
                plane_ref = f["plane_reference_short_echo"][:, :, :, idx, 0]
        support = per_bin_support(images)
        fixed = np.broadcast_to(fixed_mask(images, THRESHOLD), support.shape)
        results[arm] = dict(per_bin_support=evaluate(field, support, volume_ref, plane_ref, fraction, area, region, idx, peak_bin),
                            fixed_mask_sensitivity=evaluate(field, fixed, volume_ref, plane_ref, fraction, area, region, idx, peak_bin))
        del images, field, volume_ref
    # Identity checks against numbers E055 already published with the same helpers.
    checks = {}
    if "baseline_coarse" in results:
        e055 = json.loads((E055/"analysis_v2/report.json").read_text())
        published = [r["coverage_fraction"] for r in e055["bin_coverage"] if r["arm"] == "candidate300k" and r["threshold"] == .05]
        mine = [r["whole_lumen_coverage"] for r in results["baseline_coarse"]["per_bin_support"]["T1_throat_flow"]["bins"]]
        checks["per_bin_coverage_max_abs_diff"] = float(np.max(np.abs(np.array(published)-np.array(mine))))
        own = next(r for r in e055["sections"] if r["section"] == "throat" and r["threshold"] == .05
                   and r["arm"] == "candidate300k" and r["support"] == "own")
        mine_flow = results["baseline_coarse"]["fixed_mask_sensitivity"]["T1_throat_flow"]["cycle_same_area_error"]
        checks["fixed_mask_throat_cycle_error_abs_diff"] = abs(mine_flow-own["metrics"]["cycle_volume_error_fraction"])
        if checks["per_bin_coverage_max_abs_diff"] > 1e-9 or checks["fixed_mask_throat_cycle_error_abs_diff"] > 1e-6:
            raise AssertionError("Target evaluation does not reproduce E055 published values: "+json.dumps(checks))
    status = "COMPLETE_TARGET_EVALUATION"
    report = dict(status=status, experiment="E-058", amendment="A1 user targets", utc=datetime.now(timezone.utc).isoformat(),
        targets=dict(throat_flow_abs_error=TARGET, peak_throat_velocity_abs_error=TARGET, coverage_min=COVERAGE),
        throat_section_index=idx, throat_section_source_z_m=float(source_z[idx]), peak_bin=peak_bin+1,
        identity_checks=checks, arms=results,
        evidence_class="Same-source internal consistency, noiseless, single spin sample; not independent validation or SNR.")
    stage.mkdir()
    dump(stage/"report.json", report)
    dump(stage/"MANIFEST.json", dict(inputs={str(p): sha(p) for p in inputs}, report_sha256=sha(stage/"report.json")))
    stage.rename(output)
    summary = {a: dict(passes=r["per_bin_support"]["passes"],
                       T1=r["per_bin_support"]["T1_throat_flow"]["cycle_same_area_error"],
                       T2=r["per_bin_support"]["T2_peak_throat_velocity"]["max_error"],
                       T3=r["per_bin_support"]["T3_coverage"]) for a, r in results.items()}
    print(json.dumps(dict(status=status, identity_checks=checks, summary=summary), indent=1), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("selection", choices=["e055", "all"])
    main(p.parse_args().selection)
