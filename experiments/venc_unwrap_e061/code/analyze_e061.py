"""E-061 analysis: arms E060 / E060_UNWRAP / VENC4 / VENC4_UNWRAP against references
E-060 v2 (2x sub-samples) and E-061 REF3 (4x). Definitions per code/PROTOCOL.md; the
A1 targets use E-058's frozen evaluate() through the E-060 analysis helpers.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tomllib

import h5py
import numpy as np

E060 = Path("D:/MRSim/outputs/scanner_geometry_e060")
E061 = Path("D:/MRSim/outputs/venc_unwrap_e061")
sys.path.insert(0, str(E060/"code"))
import analyze_e060 as an  # noqa: E402
from analyze_e060 import et, fft_image, sort_profiles, fixed_mask, sha  # noqa: E402

REGIONS = dict(upstream=(-30, -10), throat=(-10, 10), downstream=(10, 30), slab=(-30, 30))


def load_z4():
    control = tomllib.loads((E061/"acquisition_v1/control_report.toml").read_text())
    if control["status"] != "COMPLETE_E061_Z_VENC4_ACQUISITION":
        raise ValueError("VENC4 Z acquisition incomplete")
    path = E061/"acquisition_v1/z_result.h5"
    if sha(path) != control["result_sha256"]["Z"]:
        raise ValueError("Z result changed")
    with h5py.File(path) as f:
        k, image, raw = f["rawsort"][:], f["image"][:], f["raw_signal_f64"][:]
    if k.shape != an.SHAPE or image.shape != an.SHAPE or not (np.isfinite(k).all() and np.isfinite(raw).all()):
        raise ValueError("Bad Z result")
    fft_error = float(abs(image-fft_image(k)).max()/abs(image).max())
    with h5py.File(E061/"sequences_v1/timing_e060.h5") as t:
        k64 = sort_profiles(raw, t["Z"]["labels"][:], an.SHAPE)
    sort_error = float(abs(k.astype(np.complex128)-k64).max()/abs(k64).max())
    if fft_error > 3e-6 or sort_error > 1e-6:
        raise ValueError("Z reconstruction audit failed")
    return image, dict(fft=fft_error, sort=sort_error)


def velocity(images, vencs, unwrap):
    out = []
    for i, e in enumerate(an.ENC[1:]):
        phi = np.angle(images[e]*images["REF"].conj()).astype(np.float64)
        if unwrap:
            phi = np.unwrap(phi, axis=0)   # temporal unwrapping along the 30 bins, from bin 1
        out.append(phi*vencs[i]/np.pi)
    return np.stack(out, axis=-1)


def evaluate_arm(images, field, ref_h5):
    with h5py.File(ref_h5) as f:
        fft_adc, plane, frac, pfrac, z = (f[k][:] for k in ("fft_adc", "plane_reference", "fraction_fft", "plane_fraction", "partition_z_m"))
    ft, vt, pt = an.to_e058(field, True), an.to_e058(fft_adc, True), an.to_e058(plane, True)
    fr, pf = frac.transpose(2, 1, 0), pfrac.transpose(2, 1, 0)
    it = {e: an.to_e058(v) for e, v in images.items()}
    region = (fr > 0) & ((z >= an.LOW) & (z < an.HIGH))[None, None, :]
    support = et.per_bin_support(it)
    fixed = np.broadcast_to(fixed_mask(it, et.THRESHOLD), support.shape)
    area = pf[:, :, an.THROAT_K]*an.AREA_MM2; pk = pt[:, :, :, an.THROAT_K, 0]; full = area > 0
    peak = int(np.argmax([pk[b][full]@area[full] for b in range(an.SHAPE[0])]))
    prim = et.evaluate(ft, support, vt, pk, fr, area, region, an.THROAT_K, peak)
    fix = et.evaluate(ft, fixed, vt, pk, fr, area, region, an.THROAT_K, peak)
    sup = np.ones(field.shape[:4], bool)
    for im in images.values():
        m = np.abs(im); sup &= m >= .05*m.max(axis=(1, 2, 3), keepdims=True)
    lum = frac > 0; zmm = z*1e3
    wrap = sup & lum[None] & (fft_adc[..., 2] > 1.5) & (field[..., 2] < 0)
    regional = {}
    for name, (lo, hi) in REGIONS.items():
        keep = sup & (lum & ((zmm >= lo) & (zmm < hi))[:, None, None])[None]
        a, b = fft_adc[..., 2][keep].astype(float), field[..., 2][keep]
        rm = float(np.sqrt(np.mean((b-a)**2)))
        regional[name] = dict(axial_rmse_m_s=rm, axial_bias_m_s=float(np.mean(b-a)), axial_nrmse=rm/float(a.max()-a.min()),
                              axial_r=float(np.corrcoef(a, b)[0, 1]), wrapped_samples=int((keep & wrap).sum()))
    sections = []
    for k in range(3, 27):
        a_k = pf[:, :, k]*an.AREA_MM2
        t1 = et.evaluate(ft, support, vt, pt[:, :, :, k, 0], fr, a_k, region, k, peak)["T1_throat_flow"]
        sections.append(dict(partition=k, z_mm=float(zmm[k]), cycle_same_area_error=t1["cycle_same_area_error"],
                             min_section_area_coverage=t1["min_section_area_coverage"]))
    return dict(peak_bin=peak+1, targets_per_bin_support=prim, targets_fixed_mask_sensitivity=fix, regional_axial=regional,
                wrapped_samples_total=int(wrap.sum()), section_flow=sections)


def main():
    target = E061/"analysis_v1"
    stage = target.with_name(target.name+".building")
    if target.exists() or stage.exists():
        raise FileExistsError(target)
    control = tomllib.loads((E060/"acquisition_v1/control_report.toml").read_text())
    images, audits = an.load_images(E060/"acquisition_v1", control, E060/"sequences_v2/timing_e060.h5")
    z4, z4_audit = load_z4()
    images4 = dict(images); images4["Z"] = z4
    refs = dict(REF2=E060/"references_v2/references.h5", REF3=E061/"references_v1/references.h5")
    for name, folder in (("REF2", E060/"references_v2"), ("REF3", E061/"references_v1")):
        if json.loads((folder/"report.json").read_text())["status"] != "PASS_E060_REFERENCES" or \
                sha(refs[name]) != json.loads((folder/"MANIFEST.json").read_text())["outputs"]["references.h5"]:
            raise ValueError("Reference missing or changed: "+name)
    arms = dict(E060=(images, (1.5, 1.5, 3.0), False), E060_UNWRAP=(images, (1.5, 1.5, 3.0), True),
                VENC4=(images4, (1.5, 1.5, 4.0), False), VENC4_UNWRAP=(images4, (1.5, 1.5, 4.0), True))
    results = {}
    for arm, (ims, vencs, unwrap) in arms.items():
        field = velocity(ims, vencs, unwrap)
        for rname, path in refs.items():
            print("E061_EVAL", arm, rname, flush=True)
            results[f"{arm}|{rname}"] = evaluate_arm(ims, field, path)
    summary = {k: dict(T1=v["targets_per_bin_support"]["T1_throat_flow"]["cycle_same_area_error"],
                       T2=v["targets_per_bin_support"]["T2_peak_throat_velocity"]["max_error"],
                       T3_section=v["targets_per_bin_support"]["T3_coverage"]["min_section_area_coverage"],
                       ALL=v["targets_per_bin_support"]["passes"]["ALL"], wrapped=v["wrapped_samples_total"],
                       throat_axial_rmse=v["regional_axial"]["throat"]["axial_rmse_m_s"],
                       worst_bin_T1=v["targets_per_bin_support"]["T1_throat_flow"]["worst_bin_same_area_error"]) for k, v in results.items()}
    report = dict(experiment="E-061", stage="analysis", utc=datetime.now(timezone.utc).isoformat(), status="COMPLETE_E061_ANALYSIS",
                  e060_audits=audits, z_venc4_audit=z4_audit, unwrap_method="numpy.unwrap of the per-voxel phase difference along the 30 bins (axis 0), no CFD input",
                  summary=summary, arms=results,
                  evidence_class="Same-source internal consistency, noiseless, one 300k seed; VENC4 noise penalty not modelled.")
    stage.mkdir()
    et.dump(stage/"report.json", report)
    inputs = [Path(__file__), E061/"code/PROTOCOL.md", E060/"code/analyze_e060.py", refs["REF2"], refs["REF3"],
              E060/"acquisition_v1/control_report.toml", E061/"acquisition_v1/control_report.toml"]
    et.dump(stage/"MANIFEST.json", dict(inputs={str(p): sha(p) for p in inputs}, report_sha256=sha(stage/"report.json")))
    stage.rename(target)
    for k, v in summary.items():
        print(k, json.dumps(v), flush=True)


if __name__ == "__main__":
    main()
