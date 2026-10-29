"""E-060 S4b: Koma in the measured scanner geometry versus its own CFD and the scan.

Definitions are fixed in code/S4_ANALYSIS.md, registered before any E-060 signal
existed. Reuses E-058's frozen target functions and E-059's frozen scan analysis
unchanged; arrays are only transposed so the vessel axis is last.
"""
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
import tomllib

import h5py
import numpy as np

SOURCE_REPO = Path("D:/MRSim/MRSIM/MRSimTracks")
E058_CODE = Path("D:/MRSim/outputs/paired_acquisition_e058/code")
E059 = Path("D:/MRSim/outputs/measured_scan_e059")
E060 = Path("D:/MRSim/outputs/scanner_geometry_e060")
sys.path.insert(0, str(SOURCE_REPO))
sys.path.insert(0, str(E058_CODE))
from scripts.analyze_phantom_precision import fft_image, sort_profiles  # noqa: E402
from scripts.analyze_remapped_confirmation import fixed_mask  # noqa: E402
from scripts.build_aorta_inlet_picker import sha  # noqa: E402
import evaluate_targets as et  # noqa: E402

_spec = importlib.util.spec_from_file_location("measured_s1_v2", E059/"code/measured_s1_v2.py")
scan_method = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scan_method)

ENC = ("REF", "X", "Y", "Z")
VENC = (1.5, 1.5, 3.0)          # R, P, S
SHAPE = (30, 30, 64, 64)        # bin, S, P, R
THROAT_K = 15
UPSTREAM_K = range(3, 8)        # z -30 to -20 mm
CENTRAL_K = range(3, 27)        # scan slices 0-23
AREA_MM2 = 1.5625
LOW, HIGH = et.REGIONS["throat"]


def load_images(folder, report, timing):
    images, audits = {}, []
    with h5py.File(timing) as t:
        for e in ENC:
            path = folder/(e.lower()+"_result.h5")
            if sha(path) != report["result_sha256"][e]:
                raise ValueError("Result changed: "+str(path))
            with h5py.File(path) as f:
                k, image, raw = f["rawsort"][:], f["image"][:], f["raw_signal_f64"][:]
            if k.shape != SHAPE or image.shape != SHAPE:
                raise ValueError("Unexpected result shape "+str(k.shape))
            if k.dtype != np.complex64 or image.dtype != np.complex64 or raw.dtype != np.complex128:
                raise ValueError("Unexpected result dtype")
            if not (np.isfinite(k).all() and np.isfinite(image).all() and np.isfinite(raw).all()):
                raise ValueError("Nonfinite result")
            fft_error = float(abs(image-fft_image(k)).max()/max(float(abs(image).max()), 1e-30))
            k64 = sort_profiles(raw, t[e]["labels"][:], SHAPE)
            sort_error = float(abs(k.astype(np.complex128)-k64).max()/max(float(abs(k64).max()), 1e-30))
            if fft_error > 3e-6 or sort_error > 1e-6:
                raise ValueError(f"Reconstruction audit failed {e}: fft {fft_error} sort {sort_error}")
            images[e] = image
            audits.append(dict(encoding=e, independent_fft_relative_max=fft_error, independent_sort_relative_max=sort_error))
    return images, audits


def to_e058(a, vector=False):
    """(bin, S, P, R[, 3 as R,P,S]) -> (bin, R, P, S[, 3 as S,P,R])."""
    a = np.asarray(a)
    if vector:
        return np.ascontiguousarray(a.transpose(0, 3, 2, 1, 4)[..., ::-1])
    return np.ascontiguousarray(a.transpose(0, 3, 2, 1))


def scan_style(field, images, z_mm, fraction):
    v = np.ascontiguousarray(field[:, list(CENTRAL_K)].transpose(1, 0, 2, 3, 4))*100
    mag = np.ascontiguousarray(np.abs(images["REF"][:, list(CENTRAL_K)]).transpose(1, 0, 2, 3))
    pos = z_mm[list(CENTRAL_K)]
    out = {}
    for frac in (.20, .25, .30):
        sign, rows = scan_method.analyse(v, mag, frac)
        out[f"{frac:.2f}"] = dict(sign=sign, rows=rows)
    rows = out[f"{fraction:.2f}"]["rows"]
    diam = np.array([r["diameter_mm"] for r in rows])
    throat = int(np.argmin(diam))
    mean_q = np.array([r["mean_flow_ml_s"] for r in rows])
    peak_q = np.array([r["peak_flow_ml_s"] for r in rows])
    upstream = [s for s in range(len(rows)) if pos[s] <= pos[throat]-20]
    summary = dict(throat_slice=throat, throat_position_mm=float(pos[throat]), throat_diameter_mm=float(diam[throat]),
        upstream_slices=upstream, throat_vs_upstream_mean_flow=float(mean_q[throat]/mean_q[upstream].mean()-1),
        throat_vs_upstream_peak_flow=float(peak_q[throat]/peak_q[upstream].mean()-1),
        flow_conservation_cv_mean_flow=float(mean_q.std()/mean_q.mean()),
        peak_forward_velocity_cm_s=float(max(r["peak_forward_velocity_cm_s"] for r in rows)),
        mean_flow_ml_s=mean_q.tolist(), flow_ml_s=[r["flow_ml_s"] for r in rows], diameter_mm=diam.tolist(),
        sensitivity={k: dict(throat_vs_upstream_mean_flow=float(
            np.array([r["mean_flow_ml_s"] for r in o["rows"]])[throat]/np.array([r["mean_flow_ml_s"] for r in o["rows"]])[upstream].mean()-1))
            for k, o in out.items()})
    return summary


def main():
    target = E060/"analysis_v1"
    stage = target.with_name(target.name+".building")
    if target.exists() or stage.exists():
        raise FileExistsError(target)
    control = tomllib.loads((E060/"acquisition_v1/control_report.toml").read_text())
    if control["status"] != "COMPLETE_E060_F64_300K_SCANNER_GEOMETRY_ACQUISITIONS":
        raise ValueError("Acquisition incomplete")
    refs = json.loads((E060/"references_v2/report.json").read_text())  # amendment A3: E-060 seam
    if refs["status"] != "PASS_E060_REFERENCES":
        raise ValueError("References missing")
    ref_h5 = E060/"references_v2/references.h5"
    if sha(ref_h5) != json.loads((E060/"references_v2/MANIFEST.json").read_text())["outputs"]["references.h5"]:
        raise ValueError("References changed")
    s0 = tomllib.loads((E060/"sequences_v2/report.toml").read_text())  # amendment A2 (TBP 12)
    timing = E060/"sequences_v2/timing_e060.h5"
    if sha(timing) != s0["timing_sha256"]:
        raise ValueError("Timing changed")
    images, audits = load_images(E060/"acquisition_v1", control, timing)
    field = np.stack([np.angle(images[e]*images["REF"].conj())*(VENC[i]/np.pi) for i, e in enumerate(ENC[1:])], axis=-1)
    with h5py.File(ref_h5) as f:
        fft_adc, plane, fraction, plane_fraction, z = (f[k][:] for k in ("fft_adc", "plane_reference", "fraction_fft", "plane_fraction", "partition_z_m"))
    if abs(z[THROAT_K]) > 1e-12:
        raise ValueError("Throat partition is not at z = 0")
    field_t, volume_t, plane_t = to_e058(field, True), to_e058(fft_adc, True), to_e058(plane, True)
    fraction_t, pfrac_t = fraction.transpose(2, 1, 0), plane_fraction.transpose(2, 1, 0)
    images_t = {e: to_e058(v) for e, v in images.items()}
    region = (fraction_t > 0) & ((z >= LOW) & (z < HIGH))[None, None, :]
    support = et.per_bin_support(images_t)
    fixed = np.broadcast_to(fixed_mask(images_t, et.THRESHOLD), support.shape)

    def at(k):
        area = pfrac_t[:, :, k]*AREA_MM2
        return area, plane_t[:, :, :, k, 0]
    area, plane_k = at(THROAT_K)
    full = area > 0
    peak_bin = int(np.argmax([plane_k[b][full]@area[full] for b in range(SHAPE[0])]))
    primary = et.evaluate(field_t, support, volume_t, plane_k, fraction_t, area, region, THROAT_K, peak_bin)
    fixed_sens = et.evaluate(field_t, fixed, volume_t, plane_k, fraction_t, area, region, THROAT_K, peak_bin)
    interior = region & (fraction_t >= 1)
    reg_b = interior & support[peak_bin]
    axial_k, axial_c = field_t[peak_bin][..., 0], volume_t[peak_bin][..., 0]
    interior_t2 = dict(voxels=int(reg_b.sum()), Koma_max_m_s=float(axial_k[reg_b].max()), CFD_max_m_s=float(np.nanmax(axial_c[interior])),
                       max_error=float(axial_k[reg_b].max()/np.nanmax(axial_c[interior])-1))
    upstream = {}
    for k in UPSTREAM_K:
        a, p = at(k)
        t1 = et.evaluate(field_t, support, volume_t, p, fraction_t, a, region, k, peak_bin)["T1_throat_flow"]
        upstream[f"{z[k]*1e3:.1f}"] = dict(partition=k, cycle_same_area_error=t1["cycle_same_area_error"],
                                           min_section_area_coverage=t1["min_section_area_coverage"])
    scan = json.loads((E059/"measured_v2/report.json").read_text())
    koma_scan = scan_style(field, images, z*1e3, .25)
    t1_ok = abs(primary["T1_throat_flow"]["cycle_same_area_error"]) <= et.TARGET
    conserve_ok = abs(koma_scan["throat_vs_upstream_mean_flow"]) <= et.TARGET
    report = dict(experiment="E-060", stage="S4b_analysis", utc=datetime.now(timezone.utc).isoformat(),
        status="COMPLETE_E060_ANALYSIS", peak_bin=peak_bin+1, reconstruction_audits=audits,
        targets_per_bin_support=primary, targets_fixed_mask_sensitivity=fixed_sens, T2_interior_only_sensitivity=interior_t2,
        upstream_same_area_flow=upstream,
        scan_method=dict(koma=koma_scan, measured=dict((k, scan[k]) for k in (
            "throat_slice", "throat_position_mm", "throat_diameter_mm", "upstream_slices", "throat_vs_upstream_mean_flow",
            "throat_vs_upstream_peak_flow", "flow_conservation_cv_mean_flow", "peak_forward_velocity_cm_s"))),
        hypothesis=dict(rule="supported only if |scan-method throat vs upstream mean flow| <= 8% and |T1| <= 8%",
                        scan_method_conservation_within_8pct=conserve_ok, T1_within_8pct=t1_ok,
                        verdict="SUPPORTED_IN_SIMULATION" if (t1_ok and conserve_ok) else "NOT_SUPPORTED"),
        evidence_class="Same-source internal consistency (noiseless, one 300k spin sample) plus a qualitative comparison with the independent scan; not physical validation.")
    stage.mkdir()
    et.dump(stage/"report.json", report)
    inputs = [Path(__file__), E060/"code/S4_ANALYSIS.md", E058_CODE/"evaluate_targets.py", E059/"code/measured_s1_v2.py",
              E059/"measured_v2/report.json", ref_h5, timing, E060/"acquisition_v1/control_report.toml"]
    et.dump(stage/"MANIFEST.json", dict(inputs={str(p): sha(p) for p in inputs}, report_sha256=sha(stage/"report.json")))
    stage.rename(target)
    p = primary["passes"]
    print(json.dumps(dict(peak_bin=peak_bin+1, passes=p, T1=primary["T1_throat_flow"]["cycle_same_area_error"],
                          T2=primary["T2_peak_throat_velocity"]["max_error"], scan_method=koma_scan["throat_vs_upstream_mean_flow"],
                          verdict=report["hypothesis"]["verdict"])), flush=True)


if __name__ == "__main__":
    main()
