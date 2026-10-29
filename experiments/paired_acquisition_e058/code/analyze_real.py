"""E-058 S4: paired real-CFD comparison of baseline and short-echo acquisitions.

Mirrors E055 analyze_300k_v2.py and reuses the frozen E050/E055 helpers unchanged.
Each arm is compared with the CFD averaged at its OWN actual ADC times; errors use
identical paired support across arms; own coverage is reported beside them.
No fitting of gain, shift, phase, mask or waveform. Never promotes 1M.
"""
import csv
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import sys
import tomllib

import h5py
import numpy as np

SOURCE_REPO = Path("D:/MRSim/MRSIM/MRSimTracks")
sys.path.insert(0, str(SOURCE_REPO))
from scripts.analyze_phantom_precision import load_arm, bind_reference, region_rows, coverage, partial_flow  # noqa: E402
from scripts.analyze_remapped_confirmation import fixed_mask, velocities  # noqa: E402
from scripts.build_aorta_inlet_picker import sha  # noqa: E402
from scripts.compare_dp150_hemodynamics import PLANES, REGIONS  # noqa: E402

E055 = Path("D:/MRSim/outputs/phantom_density_e055")
E058 = Path("D:/MRSim/outputs/paired_acquisition_e058")
E048 = SOURCE_REPO/"outputs/phantom_diagnosis_e048"
ARMS = ("baseline_coarse", "baseline_fine", "short_echo_fine")
FOLDERS = dict(baseline_coarse=E055/"mapped_f64_v1", baseline_fine=E058/"acquisition_baseline_v1",
               short_echo_fine=E058/"acquisition_short_echo_v1")
STATUS = dict(baseline_coarse="COMPLETE_E055_F64_300K_MAPPED_ACQUISITIONS",
              baseline_fine="COMPLETE_E058_F64_300K_BASELINE_FINE_ACQUISITIONS",
              short_echo_fine="COMPLETE_E058_F64_300K_SHORT_ECHO_FINE_ACQUISITIONS")
THRESHOLDS = (.05, .10, .15, .20)


def quantile(value, weight, p):
    order = np.argsort(value); x = value[order]; w = weight[order]
    return float(x[min(np.searchsorted(np.cumsum(w), p*w.sum()), len(x)-1)])


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def finite_list(values):
    return [float(v) if np.isfinite(v) else None for v in np.asarray(values)]


def main():
    final = E058/"analysis_v1"; stage = final.with_name(final.name+".building")
    if final.exists() or stage.exists():
        raise FileExistsError(final)
    reports = {a: tomllib.loads((FOLDERS[a]/"control_report.toml").read_text()) for a in ARMS}
    for a in ARMS:
        if reports[a]["status"] != STATUS[a]:
            raise ValueError("Incomplete arm: "+a)
    for key in ("input_sha256", "map_sha256", "manifest_sha256", "helper_sha256"):
        if not (reports[ARMS[0]][key] == reports[ARMS[1]][key] == reports[ARMS[2]][key]):
            raise ValueError("Arms differ in source identity: "+key)
    if reports["baseline_fine"]["free_precession_step_s"] != reports["short_echo_fine"]["free_precession_step_s"]:
        raise ValueError("Fine arms use different precession steps")
    info = tomllib.loads((E055/"inputs_v1/case_info.toml").read_text())
    ref48 = bind_reference(E048/"reference_v1", "reference_fields.h5", "f4c4ec3153d30a2a9dfd360d8ff1c21811e14e6f942697cdd990b669fec75694")
    local48 = bind_reference(E048/"local_v1", "local_fields.h5", "6c57a8708ea4a5aff09f61db232b0ba1528ae4d000bf60d9ca00524516a27885")
    s1 = json.loads((E058/"references_v1/report.json").read_text())
    s1_manifest = json.loads((E058/"references_v1/MANIFEST.json").read_text())["outputs"]
    own_refs = E058/"references_v1/references.h5"
    if s1["status"] != "PASS_E058_REFERENCE_IDENTITY_AND_CANDIDATE_TIMING" or sha(own_refs) != s1_manifest["references.h5"]:
        raise ValueError("S1 references missing or changed")
    with h5py.File(ref48) as f:
        baseline_volume, fraction = f["fft_adc"][:], f["fraction_fft"][:]
    with h5py.File(local48) as f:
        source_z = f["source_z_fft_m"][:]
    with h5py.File(own_refs) as f:
        candidate_volume = f["fft_adc_short_echo"][:]
    volume_ref = dict(baseline_coarse=baseline_volume, baseline_fine=baseline_volume, short_echo_fine=candidate_volume)
    images, fields, audits = {}, {}, {}
    for arm in ARMS:
        print("E058 ANALYZE", arm, flush=True)
        images[arm], audits[arm] = load_arm(FOLDERS[arm], reports[arm], candidate=True)
        fields[arm] = velocities(images[arm], info["Venc_m_s"])
    pairings = dict(all_three=ARMS, fine_pair=("baseline_fine", "short_echo_fine"))
    regions, masks_by_threshold, bin_coverage = [], {}, []
    for threshold in THRESHOLDS:
        masks = {a: fixed_mask(images[a], threshold) for a in ARMS}
        masks_by_threshold[threshold] = masks
        for pairing, members in pairings.items():
            for arm in members:
                rows = region_rows(volume_ref[arm], {arm: fields[arm]}, {a: masks[a] for a in members}, fraction, source_z, threshold)
                regions += [dict(pairing=pairing, reference="own_actual_ADC", **r) for r in rows]
        # Disclosed sensitivity: candidate against the baseline-timing reference.
        rows = region_rows(baseline_volume, {"short_echo_fine": fields["short_echo_fine"]}, {a: masks[a] for a in ARMS}, fraction, source_z, threshold)
        regions += [dict(pairing="all_three", reference="baseline_ADC_sensitivity", **r) for r in rows]
        for arm in ARMS:
            per_bin = np.ones(baseline_volume.shape[:-1], bool)
            for im in images[arm].values():
                m = abs(im); per_bin &= (m >= threshold*m.max(axis=(1, 2, 3), keepdims=True)) & (m > 0)
            bin_coverage += [dict(arm=arm, threshold=threshold, bin=b+1, coverage_fraction=coverage(fraction, per_bin[b])) for b in range(15)]
    paired = np.logical_and.reduce([masks_by_threshold[.05][a] for a in ARMS])
    norm_mag = {a: float(np.mean(abs(images[a]["REF"])[:, paired])/300000) if paired.any() else None for a in ARMS}
    del images; gc.collect()
    eligible = fraction > 0
    keep = paired & eligible
    numerics = dict(description="baseline_fine minus baseline_coarse velocity on all-three paired support, all bins",
                    vector_rms_m_s=float(np.sqrt(np.mean(np.sum((fields["baseline_fine"][:, keep]-fields["baseline_coarse"][:, keep])**2, axis=-1)))),
                    vector_max_m_s=float(np.max(np.linalg.norm(fields["baseline_fine"][:, keep]-fields["baseline_coarse"][:, keep], axis=-1))))
    sections, profiles = [], []
    with h5py.File(local48) as f48, h5py.File(own_refs) as fown:
        for section, target in PLANES.items():
            idx = int(np.argmin(abs(source_z-target))); area = f48["plane_fraction"][:, :, idx]*3.125
            plane_ref = dict(baseline_coarse=f48["plane_reference"][:, :, :, idx, 0], baseline_fine=f48["plane_reference"][:, :, :, idx, 0],
                             short_echo_fine=fown["plane_reference_short_echo"][:, :, :, idx, 0])
            for threshold, masks in masks_by_threshold.items():
                for pairing, members in pairings.items():
                    common = np.logical_and.reduce([masks[a][:, :, idx] for a in members])
                    for arm in members:
                        for policy, support in (("own", masks[arm][:, :, idx]), ("paired", common)):
                            sections.append(dict(pairing=pairing, section=section, index=idx, source_z_m=float(source_z[idx]), threshold=threshold,
                                arm=arm, support=policy, **partial_flow(plane_ref[arm], fields[arm][:, :, :, idx, 0], area, support, info["period_s"])))
            if section == "throat":
                full_q = plane_ref["baseline_fine"][:, area > 0]@area[area > 0]
                for phase, b in (("peak", int(np.argmax(full_q))), ("low", int(np.argmin(full_q)))):
                    for orientation in ("transverse", "longitudinal"):
                        if orientation == "transverse":
                            line = paired[16, :, idx] & (fraction[16, :, idx] > 0)
                            refs = {a: volume_ref[a][b, 16, :, idx, 0] for a in ARMS}
                            est = {a: fields[a][b, 16, :, idx, 0] for a in ARMS}; coordinate = (np.arange(64)-32)*1.25
                        else:
                            line = paired[16, 32, :] & (fraction[16, 32, :] > 0)
                            refs = {a: volume_ref[a][b, 16, 32, :, 0] for a in ARMS}
                            est = {a: fields[a][b, 16, 32, :, 0] for a in ARMS}; coordinate = source_z*1000
                        profiles.append(dict(phase=phase, bin=b+1, orientation=orientation, coordinate_mm=coordinate.tolist(),
                            paired_retained=line.tolist(), **{a+"_CFD_axial_m_s": finite_list(refs[a]) for a in ARMS},
                            **{a+"_axial_m_s": est[a].tolist() for a in ARMS},
                            **{a+"_paired_RMSE_m_s": float(np.sqrt(np.mean((est[a][line]-refs[a][line])**2))) if line.any() else None for a in ARMS}))
    percentiles = []
    for region, (low, high) in REGIONS.items():
        region_eligible = eligible & ((source_z >= low) & (source_z < high))[None, None, :]
        support = paired & region_eligible
        weights = np.tile(fraction[support].astype(float), 15)
        for arm in ARMS:
            speed = np.linalg.norm(fields[arm][:, support], axis=-1).ravel()
            ref_speed = np.linalg.norm(volume_ref[arm][:, support], axis=-1).ravel()
            percentiles.append(dict(region=region, arm=arm, paired_coverage_fraction=coverage(fraction, support, region_eligible),
                speed_p95_m_s=quantile(speed, weights, .95) if support.any() else None,
                speed_p99_m_s=quantile(speed, weights, .99) if support.any() else None,
                reference_p95_m_s=quantile(ref_speed, weights, .95) if support.any() else None,
                reference_p99_m_s=quantile(ref_speed, weights, .99) if support.any() else None))

    def region(arm, name, support, pairing="all_three"):
        return next(r for r in regions if r["threshold"] == .05 and r["pairing"] == pairing and r["reference"] == "own_actual_ADC"
                    and r["arm"] == arm and r["region"] == name and r["support"] == support)

    def flow(arm, name, pairing="all_three"):
        return next(r for r in sections if r["threshold"] == .05 and r["pairing"] == pairing and r["arm"] == arm
                    and r["section"] == name and r["support"] == "paired")

    rmse = lambda arm, name: (region(arm, name, "paired")["metrics"] or {}).get("vector_rmse_m_s")
    base, cand = "baseline_fine", "short_echo_fine"
    throat_reduction = None if None in (rmse(base, "throat"), rmse(cand, "throat")) else 1-rmse(cand, "throat")/rmse(base, "throat")
    coverage_change = region(cand, "whole", "own")["coverage_fraction"]-region(base, "whole", "own")["coverage_fraction"]
    flow_no_worse = {}
    for name in ("throat", "upstream"):
        a, b = flow(base, name)["metrics"], flow(cand, name)["metrics"]
        flow_no_worse[name] = bool(a is not None and b is not None and b["rmse_ml_s"] <= a["rmse_ml_s"])
    criteria = dict(throat_paired_RMSE_reduction_fraction=throat_reduction,
                    throat_reduction_at_least_10pct=bool(throat_reduction is not None and throat_reduction >= .10),
                    whole_paired_RMSE_not_worse=bool(rmse(cand, "whole") is not None and rmse(cand, "whole") <= rmse(base, "whole")),
                    own_whole_coverage_change_fraction=coverage_change,
                    own_coverage_loss_at_most_1pp=bool(coverage_change >= -.01),
                    same_area_flow_RMSE_not_worse=flow_no_worse)
    supported = (criteria["throat_reduction_at_least_10pct"] and criteria["whole_paired_RMSE_not_worse"]
                 and criteria["own_coverage_loss_at_most_1pp"] and all(flow_no_worse.values()))
    primary = [r for r in regions if r["threshold"] == .05 and r["region"] in ("whole", "throat") and r["reference"] == "own_actual_ADC"]
    status = "CANDIDATE_SUPPORTED_ON_REAL_CASE" if supported else "CANDIDATE_NOT_SUPPORTED_ON_REAL_CASE"
    report = dict(status=status, experiment="E-058", created_utc=datetime.now(timezone.utc).isoformat(),
        evidence_class="Same-source internal consistency, noiseless; not independent physical validation or an SNR study",
        decision_criteria=criteria, numerics_only_effect=numerics, primary=primary, regions=regions, sections=sections,
        profiles=profiles, regional_percentiles=percentiles, bin_coverage=bin_coverage, reconstruction_audits=audits,
        amplitude_mean_REF_divided_by_particle_count=norm_mag, reference_timing_sensitivity=s1["timing_sensitivity"],
        free_precession_step_s=reports[base]["free_precession_step_s"], new_1m_started=False,
        limitations=["Own coverage and identical-paired error populations have different denominators.",
                     "Noiseless: the 700 Hz/pixel readout's roughly sqrt(2) thermal-noise penalty per voxel is not modeled.",
                     "The candidate changes echo and readout timing together; not a TE-only effect.",
                     "Single deterministic 300k spin sample; seed variability unmeasured.",
                     "Encoded complex-image phase and CFD mean velocity are different observables under acceleration and mixing.",
                     "No fitting, phase filling, threshold relaxation, TKE/wall/pressure validation or automatic 1M."])
    stage.mkdir()
    dump(stage/"report.json", report)
    with (stage/"regions.csv").open("w", newline="", encoding="utf-8") as f:
        values = [{k: r[k] for k in ("threshold", "pairing", "reference", "region", "arm", "support", "coverage_fraction")} |
                  {k: (r["metrics"] or {}).get(k) for k in ("voxels", "vector_rmse_m_s", "mean_speed_bias_m_s")} for r in regions]
        writer = csv.DictWriter(f, fieldnames=list(values[0])); writer.writeheader(); writer.writerows(values)
    with h5py.File(stage/"paired_fields.h5", "w") as f:
        f["paired_fixed5_mask_all_three"] = paired
        for a in ARMS:
            f.create_dataset(a, data=fields[a], compression="gzip", compression_opts=1)
            f[a+"_fixed5_mask"] = masks_by_threshold[.05][a]
    inputs = [Path(__file__), E058/"code/PROTOCOL.md", ref48, local48, own_refs, E058/"references_v1/report.json",
              *[FOLDERS[a]/"control_report.toml" for a in ARMS]]
    dump(stage/"MANIFEST.json", dict(inputs={str(p): sha(p) for p in inputs},
                                     outputs={p.name: sha(p) for p in stage.iterdir() if p.is_file()}))
    stage.rename(final)
    print(json.dumps(dict(status=status, criteria=criteria, numerics_only_effect=numerics)), flush=True)


if __name__ == "__main__":
    main()
