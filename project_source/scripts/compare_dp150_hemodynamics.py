"""E-025: immutable-input, matched-support CFD/Koma hemodynamic comparison.

Run from the repository with ``python -m scripts.compare_dp150_hemodynamics``.
Outputs are internal consistency evidence, not independent validation.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import xml.etree.ElementTree as ET

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_erosion, gaussian_filter

from mrsimtracks.hemodynamics import (
    differential_metrics, jet_metrics, midpoint_cycle_integrals,
    region_kinetic_energy, section_flow, velocity_gradient, weighted_error_metrics,
)
from scripts.dp150_hemodynamic_reference import (
    bin_sample_times, build_voxel_lookup, load_periodic_reference,
    native_tet_moments, read_e022_contract, transformed_nodal_velocity,
)

ROOT = Path(__file__).resolve().parents[1]
PIPE = ROOT / "outputs/dp150_mri_pipeline"
PRODUCTION = PIPE / "koma/simulation_motion_1m_scanner_matched_venc5_seam_safe_v1"
H5 = PRODUCTION / "final_image_analysis/support_constrained_phase/dp150_seam_safe_support_phase.h5"
OUT = ROOT / "outputs/dp150_hemodynamic_comparison/e025_v1/results"
RHO, MU = 1060.0, 0.0035
PLANES = dict(upstream=-0.035, throat=0.0, near_downstream=0.030,
              mid_downstream=0.080, far_downstream=0.180)
REGIONS = dict(whole=(-np.inf, np.inf), upstream=(-.080, -.020),
               throat=(-.010, .010), near_downstream=(.010, .060),
               mid_downstream=(.060, .140), far_downstream=(.140, .300))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def safe_json(value):
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return safe_json(value.tolist())
    if isinstance(value, np.generic):
        return safe_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def paired_velocity_metrics(ref, est, weight, mask):
    result = {axis: weighted_error_metrics(ref[..., c], est[..., c], weight, mask)
              for c, axis in enumerate(("X", "Y", "Z"))}
    result["speed"] = weighted_error_metrics(np.linalg.norm(ref, axis=-1),
                                             np.linalg.norm(est, axis=-1), weight, mask)
    keep = mask & np.isfinite(ref).all(-1) & np.isfinite(est).all(-1) & (weight > 0)
    if keep.any():
        w = weight[keep]
        error = np.sum((est[keep] - ref[keep])**2, axis=-1)
        den = np.sum(w * np.sum(ref[keep]**2, axis=-1))
        result["vector_rmse_m_s"] = float(np.sqrt(np.sum(w * error) / w.sum()))
        result["vector_nrmse_reference_rms"] = float(np.sqrt(np.sum(w * error) / den)) if den else np.nan
    else:
        result["vector_rmse_m_s"] = result["vector_nrmse_reference_rms"] = np.nan
    return result


def finite_summary(values, weights, valid):
    keep = valid & np.isfinite(values) & (weights > 0)
    if not keep.any():
        return {"mean": np.nan, "max": np.nan, "p99": np.nan, "weight": 0.0}
    v, w = values[keep], weights[keep]
    order = np.argsort(v)
    p99 = np.interp(.99, (np.cumsum(w[order]) - .5*w[order]) / w.sum(), v[order])
    return dict(mean=float(np.average(v, weights=w)), max=float(v.max()),
                p99=float(p99), weight=float(w.sum()))


def combine_error_summaries(items):
    """Pool sufficient statistics; never average percentages or bin RMSEs."""
    valid = [x for x in items if x["weight_sum"] > 0 and np.isfinite(x["rmse"])]
    if not valid:
        return dict(bias=np.nan,mae=np.nan,rmse=np.nan,nrmse_reference_rms=np.nan,weight_sum=0.)
    w = np.array([x["weight_sum"] for x in valid])
    mean = lambda key: float(np.average([x[key] for x in valid], weights=w))
    rmse = float(np.sqrt(np.average([x["rmse"]**2 for x in valid],weights=w)))
    rms = float(np.sqrt(np.average([x["reference_rms"]**2 for x in valid],weights=w)))
    return dict(bias=mean("bias"),mae=mean("mae"),rmse=rmse,
                reference_rms=rms,nrmse_reference_rms=rmse/rms if rms>0 else np.nan,
                reference_mean=mean("reference_mean"),estimate_mean=mean("estimate_mean"),
                weight_sum=float(w.sum()))


def matched_smooth(u, valid, spacing, sigma_m=.001):
    """Identical normalized in-fluid smoothing; no background zero velocity."""
    sigma = sigma_m / np.asarray(spacing)[::-1]
    den = gaussian_filter(valid.astype(float), sigma, mode="constant", cval=0)
    result = np.full_like(u, np.nan, dtype=float)
    for c in range(3):
        num = gaussian_filter(np.where(valid, u[..., c], 0), sigma, mode="constant", cval=0)
        np.divide(num, den, out=result[..., c], where=valid & (den > 0))
    return result


def provenance():
    support_report = H5.with_name("dp150_seam_safe_support_phase_report.json")
    j = json.loads(support_report.read_text(encoding="utf-8"))
    if j.get("status") != "PASS_RECONSTRUCTION_DERIVATIVE":
        raise ValueError("Not an accepted reconstruction derivative")
    if digest(H5) != j["outputs"]["reconstruction_h5"]["sha256"]:
        raise ValueError("Reconstruction hash mismatch")
    periodic = PIPE / "periodic_flow_report.json"
    tracking = PIPE / "motion_1m_report.json"
    pvd = PIPE / "flow_periodic/dp150_periodic.pvd"
    pj = json.loads(periodic.read_text(encoding="utf-8"))
    tj = json.loads(tracking.read_text(encoding="utf-8"))
    if digest(pvd) != pj["output"]["pvd_sha256"] or digest(pvd) != tj["inputs"]["periodic_pvd_sha256"]:
        raise ValueError("Periodic source linkage mismatch")
    files = [H5, support_report, periodic, tracking, pvd, PRODUCTION / "simulation_report.toml",
             ROOT / "outputs/dp150_hemodynamic_comparison/e025_v1/ANALYSIS_PROTOCOL.md",
             Path(__file__), ROOT / "scripts/dp150_hemodynamic_reference.py",
             ROOT / "mrsimtracks/hemodynamics.py"]
    frames = [pvd.parent / ds.attrib["file"] for ds in ET.parse(pvd).iter("DataSet")]
    if len(frames) != 73:
        raise ValueError("Expected 73 source snapshots")
    if sum(f.stat().st_size for f in frames) != pj["output"]["frame_bytes"]:
        raise ValueError("Periodic frame byte count mismatch")
    return [{"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size,
             "sha256": digest(p)} for p in files + frames]


def e024_regression(corrected, support, period):
    old = json.loads((PRODUCTION / "final_image_analysis/cfd_koma_flow_waveform_e024/dp150_cfd_koma_flow_waveform_report.json").read_text())
    expected = np.array([x["koma_reconstructed_m3_per_s"] for x in old["binwise_values"]])
    actual = np.array([np.sum(u[:, :, 70, 0][support[:, :, 70]], dtype=np.float64) * 1.25e-3 * 2.5e-3
                       for u in corrected])
    if not np.allclose(actual, expected, rtol=1e-7, atol=1e-12):
        raise ValueError("E-024 binary-support waveform regression failed")
    return dict(passed=True, max_abs_difference_m3_s=float(np.max(np.abs(actual-expected))),
                reproduced_stroke_volume_ml=midpoint_cycle_integrals(actual, period)["net_m3"]*1e6,
                legacy_values=old["comparison_metrics"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".e025-stage-", dir=output.parent))
    start = time.perf_counter()
    print(f"Staging in {stage}", flush=True)
    inputs = provenance()
    grid, timing = read_e022_contract(PIPE / "motion_1m_report.json", PRODUCTION / "simulation_report.toml", H5)
    period = timing["effective_motion_period_s"]
    times = bin_sample_times(15, period)[:, 0]
    temporal_times = bin_sample_times(15, period, samples_per_bin=3)
    with h5py.File(H5, "r") as source:
        support = np.asarray(source["support_mask"], bool)
        direct = np.asarray(source["direct_mask"], bool)
        regularized = np.asarray(source["regularized_mask"], bool)
        # Array order becomes (bin,slice,phase,readout,component).
        corrected = np.moveaxis(source["velocity_corrected_m_per_s"][...], 0, -1)
        raw = np.moveaxis(source["velocity_raw_m_per_s"][...], 0, -1)
    regression = e024_regression(corrected, support, period)
    print("Production provenance and E-024 regression passed. Loading CFD.", flush=True)
    flow = load_periodic_reference(PIPE / "flow_periodic/dp150_periodic.pvd")
    lookup = build_voxel_lookup(flow.active_mesh, grid, subsamples=2)
    print("2x2x2 geometry lookup complete.", flush=True)
    lookup3 = build_voxel_lookup(flow.active_mesh, grid, subsamples=3)
    plane_lookup = build_voxel_lookup(flow.active_mesh, grid, subsamples=(1, 2, 2))
    fraction, fraction3 = lookup.fraction_grid(), lookup3.fraction_grid()
    area_fraction = plane_lookup.fraction_grid()
    dv = grid.voxel_volume_m3
    volume, volume3 = fraction.astype(float)*dv, fraction3.astype(float)*dv
    common = support & (fraction > 0)
    direct_common = direct & common
    interior = common & (fraction >= .999999)
    eroded = binary_erosion(interior)
    masks = dict(hybrid_common=common, direct_only=direct_common, regularized_only=regularized & common,
                 geometric_interior=interior, eroded_interior=eroded)
    r, p, s = grid.axis_centers_m()
    source_z = r + grid.source_center_m[2]
    indices = {name: int(np.argmin(np.abs(source_z - z))) for name, z in PLANES.items()}
    plane_area = grid.spacing_rps_m[1] * grid.spacing_rps_m[2]
    ss, pp = np.meshgrid(s, p, indexing="ij")
    regions = {name: common & ((source_z >= low) & (source_z < high))[None,None,:]
               for name, (low, high) in REGIONS.items()}
    rows, sections, jets, derivatives, sensitivity, native = [], [], [], [], [], []
    region_errors, profiles, axial_profiles, jet_extent = [], [], [], []
    errors_by_bin = []
    all_ref = []
    paired_ids = np.flatnonzero(common)
    weights = volume.ravel()[paired_ids]
    fields_path = stage / "matched_reference.h5"
    print("All geometry lookups complete. Evaluating 15 bins.", flush=True)
    with h5py.File(fields_path, "w") as fields:
        fields.attrs.update(evidence_class="internal consistency", velocity_units="m/s",
                            axis_order="bin,slice,phase,readout,component", density_kg_m3=RHO,
                            viscosity_pa_s=MU, effective_period_s=period)
        fields.create_dataset("volume_fraction_2", data=fraction, compression="gzip")
        fields.create_dataset("volume_fraction_3", data=fraction3, compression="gzip")
        fields.create_dataset("plane_area_fraction_2", data=area_fraction, compression="gzip")
        fields.create_dataset("common_mask", data=common, compression="gzip")
        fields.create_dataset("direct_common_mask", data=direct_common, compression="gzip")
        for name in ("cfd_voxel_midpoint", "cfd_voxel_three_sample_mean", "cfd_voxel_3x3x3"):
            fields.create_dataset(name, (15, *grid.shape_spr, 3), dtype="f4", compression="gzip", compression_opts=1,
                                  chunks=(1, 8, 16, 100, 3))
        for b, t in enumerate(times):
            u_nodes = transformed_nodal_velocity(flow, grid, t, period)
            sampled = lookup.average(u_nodes)
            ref = np.moveaxis(sampled["velocity_mri"], 0, -1)
            ref3 = np.moveaxis(lookup3.average(u_nodes)["velocity_mri"], 0, -1)
            plane_ref = np.moveaxis(plane_lookup.average(u_nodes)["velocity_mri"], 0, -1)
            # Middle quadrature sample is exactly the primary midpoint.
            ref_avg = ref.astype(float).copy()
            for tsub in temporal_times[b, [0,2]]:
                values = transformed_nodal_velocity(flow, grid, tsub, period)
                ref_avg += np.moveaxis(lookup.average(values)["velocity_mri"], 0, -1)
            ref_avg /= 3
            fields["cfd_voxel_midpoint"][b] = ref
            fields["cfd_voxel_three_sample_mean"][b] = ref_avg
            fields["cfd_voxel_3x3x3"][b] = ref3
            all_ref.append(ref.reshape(-1,3)[paired_ids])
            errors_by_bin.append({"bin": b+1, "time_s": float(t),
                "synthetic_tail": bool((b+.5)*.72/15 >= .64),
                "masks": {name: paired_velocity_metrics(ref, raw[b] if name == "direct_only" else corrected[b], volume, mask)
                          for name, mask in masks.items()}})
            moments = native_tet_moments(flow.active_mesh, u_nodes, grid=grid)
            native.append(dict(bin=b+1, total_volume_m3=moments["total_volume_m3"],
                kinetic_energy_j=.5*RHO*moments["integral_speed_squared_m5_s2"],
                source_definition="exact linear-tet converted and retimed CFD; not original Fluent cells"))
            for name, mask in regions.items():
                region_errors.append(dict(bin=b+1,region=name,
                    metrics=paired_velocity_metrics(ref,corrected[b],volume,mask)))
                for dataset, u in (("cfd_voxel", ref), ("koma_hybrid", corrected[b])):
                    energy = region_kinetic_energy(u, volume, mask, RHO)
                    speed = finite_summary(np.linalg.norm(u, axis=-1), volume, mask)
                    rows.append(dict(bin=b+1,time_s=t,region=name,dataset=dataset,
                                     kinetic_energy_j=energy["energy_j"],speed_mean_m_s=speed["mean"],
                                     speed_p99_m_s=speed["p99"],speed_max_m_s=speed["max"],
                                     volume_m3=energy["sampled_volume_m3"]))
            # Subvoxel kinetic energy retained as a separate moment, never labelled TKE.
            native[-1]["quadrature_unaveraged_ke_common_j"] = .5*RHO*float(np.nansum(sampled["mean_speed_squared"][common]*volume[common]))
            sensitivity.append(dict(bin=b+1, quadrature_3_vs_2=paired_velocity_metrics(ref,ref3,volume,common & (fraction3>0)),
                temporal_3_vs_midpoint=paired_velocity_metrics(ref,ref_avg,volume,common),
                koma_vs_temporal_3=paired_velocity_metrics(ref_avg,corrected[b],volume,common),
                koma_vs_spatial_3=paired_velocity_metrics(ref3,corrected[b],volume3,support & (fraction3>0))))
            throat = indices["throat"]
            throat_p99 = finite_summary(np.maximum(plane_ref[:,:,throat,0],0),
                                       area_fraction[:,:,throat]*plane_area, common[:,:,throat])["p99"]
            for dataset,u in (("cfd",ref),("koma",corrected[b])):
                axial_p99=[]
                for idx in range(grid.shape_spr[2]):
                    stats=finite_summary(u[:,:,idx,0],area_fraction[:,:,idx]*plane_area,common[:,:,idx])
                    axial_p99.append(stats["p99"])
                    axial_profiles.append(dict(bin=b+1,dataset=dataset,source_z_mm=source_z[idx]*1e3,
                                                axial_velocity_p99_m_s=stats["p99"],axial_velocity_max_m_s=stats["max"]))
                axial_p99=np.asarray(axial_p99)
                downstream=source_z>=source_z[throat]
                for threshold_fraction in (.25,.5,.75):
                    exceed=downstream & (axial_p99>=throat_p99*threshold_fraction)
                    length=float((source_z[exceed].max()-source_z[throat])*1e3) if exceed.any() else np.nan
                    jet_extent.append(dict(bin=b+1,dataset=dataset,threshold_fraction=threshold_fraction,
                        furthest_downstream_exceedance_mm=length,
                        definition="furthest section with axial p99 >= fraction of CFD throat p99; not connected jet length"))
            for name, idx in indices.items():
                area = area_fraction[:,:,idx] * plane_area
                cm = common[:,:,idx]
                position = np.stack((np.full_like(pp,r[idx]),pp,ss),axis=-1)
                for dataset,u in (("cfd",ref),("koma",corrected[b])):
                    for axis,axis_centers,values,selected in (
                        ("MRI_Y",p,u[grid.shape_spr[0]//2,:,idx,0],cm[grid.shape_spr[0]//2]),
                        ("MRI_Z",s,u[:,grid.shape_spr[1]//2,idx,0],cm[:,grid.shape_spr[1]//2])):
                        for coord,value,keep in zip(axis_centers,values,selected):
                            profiles.append(dict(bin=b+1,section=name,dataset=dataset,profile_axis=axis,
                                coordinate_mm=coord*1e3,axial_velocity_m_s=float(value) if keep else np.nan,
                                in_common_support=bool(keep)))
                for dataset,u,mask in (("cfd_plane_full",plane_ref[:,:,idx],area>0),
                        ("cfd_plane_common",plane_ref[:,:,idx],cm),
                        ("cfd_voxel_common",ref[:,:,idx],cm),
                        ("koma_hybrid_common",corrected[b,:,:,idx],cm),
                        ("cfd_direct_partial",ref[:,:,idx],direct_common[:,:,idx]),
                        ("koma_direct_partial",raw[b,:,:,idx],direct_common[:,:,idx]),
                        ("koma_hybrid_binary",corrected[b,:,:,idx],support[:,:,idx])):
                    chosen_area = plane_area if dataset.endswith("binary") else area
                    q = section_flow(u, chosen_area, valid=mask)
                    sections.append(dict(bin=b+1,time_s=t,section=name,readout_index=idx,
                        source_z_mm=source_z[idx]*1e3,dataset=dataset,**q))
                for threshold_fraction in (.25,.5,.75):
                    for dataset,u in (("cfd",ref[:,:,idx]),("koma",corrected[b,:,:,idx])):
                        jet = jet_metrics(u,position,area,valid=cm,
                                          minimum_forward_speed_m_s=throat_p99*threshold_fraction)
                        jets.append(dict(bin=b+1,section=name,dataset=dataset,
                            throat_p99_m_s=throat_p99,threshold_fraction=threshold_fraction,**jet))
            for maskname,mask in (("interior",interior),("direct_stencil",interior & direct),
                                  ("extra_eroded",eroded),("matched_smoothing_1mm",interior)):
                a,c = ref,corrected[b]
                if maskname == "matched_smoothing_1mm":
                    a,c = matched_smooth(a,mask,grid.spacing_rps_m),matched_smooth(c,mask,grid.spacing_rps_m)
                ga,va = velocity_gradient(a,grid.spacing_rps_m,mask)
                gc,vc = velocity_gradient(c,grid.spacing_rps_m,mask)
                valid = va & vc
                da,dc = differential_metrics(ga,a,MU),differential_metrics(gc,c,MU)
                comparison = {key: weighted_error_metrics(da[key],dc[key],volume,valid)
                              for key in ("vorticity_magnitude_s_inv","helicity_m_s_inv2",
                                          "divergence_s_inv","dissipation_w_m3")}
                comparison["strain_frobenius_s_inv"] = weighted_error_metrics(
                    np.linalg.norm(da["strain_rate_s_inv"],axis=(-2,-1)),
                    np.linalg.norm(dc["strain_rate_s_inv"],axis=(-2,-1)),volume,valid)
                derivatives.append(dict(bin=b+1,mask=maskname,n_valid=int(valid.sum()),
                    volume_m3=float(volume[valid].sum()),metrics=comparison,
                    cfd_resolved_dissipation_w=float(np.nansum(da["dissipation_w_m3"][valid]*volume[valid])),
                    koma_resolved_dissipation_w=float(np.nansum(dc["dissipation_w_m3"][valid]*volume[valid]))))
            print(f"Bin {b+1}/15 completed ({time.perf_counter()-start:.1f}s)",flush=True)
        all_ref = np.asarray(all_ref)
        all_est = corrected.reshape(15,-1,3)[:,paired_ids]
        all_weight = np.broadcast_to(weights,all_ref.shape[:-1])
        aggregate = paired_velocity_metrics(all_ref,all_est,all_weight,np.ones(all_weight.shape,bool))
        aggregate_no_tail = paired_velocity_metrics(all_ref[:13],all_est[:13],all_weight[:13],np.ones(all_weight[:13].shape,bool))
        fields.create_dataset("source_z_m",data=source_z)
        fields.create_dataset("time_s",data=times)
    flow_summary = {}
    for name in indices:
        series = {dataset: np.array([x["net_m3_s"] for x in sections if x["section"]==name and x["dataset"]==dataset])
                  for dataset in ("cfd_plane_full","cfd_plane_common","cfd_voxel_common","koma_hybrid_common",
                                  "cfd_direct_partial","koma_direct_partial","koma_hybrid_binary")}
        refq,komq = series["cfd_voxel_common"],series["koma_hybrid_common"]
        flow_summary[name] = dict(readout_index=indices[name],source_z_mm=source_z[indices[name]]*1e3,
            errors=weighted_error_metrics(refq,komq),
            cycles={k:midpoint_cycle_integrals(v,period) for k,v in series.items()},
            cfd_peak_bin=int(np.argmax(refq)+1),koma_peak_bin=int(np.argmax(komq)+1),
            peak_ratio=float(komq.max()/refq.max()),
            cycle_volume_ratio=float(komq.sum()/refq.sum()))
        flow_summary[name]["cycle_partition_definition"] = "cycles forward/reverse partition net flow over time; spatial_transport separately includes simultaneous opposing velocities"
        flow_summary[name]["spatial_transport"] = {
            dataset:dict(forward_m3=sum(x["forward_m3_s"] for x in sections if x["section"]==name and x["dataset"]==dataset)*period/15,
                         reverse_m3=sum(x["reverse_m3_s"] for x in sections if x["section"]==name and x["dataset"]==dataset)*period/15)
            for dataset in series}
    geometry = dict(cfd_quadrature_volume_m3=float(volume.sum()),
        cfd_quadrature3_volume_m3=float(volume3.sum()), native_mesh_volume_m3=native[0]["total_volume_m3"],
        common_volume_m3=float(volume[common].sum()),
        geometry_coverage_by_mri_support=float(volume[common].sum()/volume.sum()),
        direct_common_volume_fraction=float(volume[direct_common].sum()/volume[common].sum()),
        support_voxels=int(support.sum()),common_voxels=int(common.sum()),
        support_voxels_without_cfd_subsamples=int((support & (fraction==0)).sum()))
    report = dict(schema_version=1,experiment="E-025",created_utc=datetime.now(timezone.utc).isoformat(),
        status="COMPLETED_OBSERVABLE_METRICS_CONDITIONAL_METRICS_GATED",
        evidence_class="descriptive internal consistency; not independent validation",
        inputs=inputs,grid=asdict(grid),timing=timing,properties=dict(density_kg_m3=RHO,viscosity_pa_s=MU),
        geometry=geometry,regression_e024=regression,aggregate=aggregate,
        aggregate_excluding_synthetic_tail=aggregate_no_tail,flow_summary=flow_summary,
        errors_by_bin=errors_by_bin,section_values=sections,regional_values=rows,
        region_errors=region_errors,jet_extent=jet_extent,
        regional_error_summary={name:{key:combine_error_summaries([x["metrics"][key] for x in region_errors if x["region"]==name])
                                      for key in ("X","Y","Z","speed")} for name in REGIONS},
        derivative_error_summary={mask:{key:combine_error_summaries([x["metrics"][key] for x in derivatives if x["mask"]==mask])
                                       for key in derivatives[0]["metrics"]} for mask in ("interior","direct_stencil","extra_eroded","matched_smoothing_1mm")},
        jet_values=jets,derivative_values=derivatives,sensitivity=sensitivity,native_values=native,
        conditional_metrics=dict(wss_tawss_osi="Gated: wall estimator/resolution validity not established; see input audit",
            pressure="Gated: original Fluent pressure is incompatible with retimed/closed velocity without a consistent reference",
            tke="Unavailable from mean velocity or cardiac-bin variance; no fluctuation-sensitive acquisition",
            independent_validation="Awaiting measured waveform and independently registered velocity"),
        limitations=["Kinematic voxel and box-time averages are not the MRI complex-signal transfer function.",
            "Native reference is converted wall-zeroed nodal CFD, not original Fluent cell velocities.",
            "Native mesh whole-domain KE and common-support KE have different domains; do not interpret ratio solely as resolution loss.",
            "Volume/area fractions use midpoint quadrature, not exact clipping; 2 versus 3 is sensitivity, not convergence proof.",
            "Derivative integrals exclude walls and invalid stencils; molecular resolved dissipation is not total turbulent energy loss.",
            "Direct-only values are partial-domain integrals. Regularized values are estimates.",
            "Bins14/15 include synthetic closure; near-zero references make relative errors unstable.",
            "No confidence intervals from correlated voxels and no fitted alignment or composite score."],
        runtime_seconds=time.perf_counter()-start)
    for filename,data in (("sections.csv",sections),("regions.csv",rows),("profiles.csv",profiles),("axial_profiles.csv",axial_profiles)):
        with (stage/filename).open("w",newline="",encoding="utf-8") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(data[0]));writer.writeheader();writer.writerows(safe_json(data))
    make_figures(stage,report,corrected,grid)
    (stage/"report.json").write_text(json.dumps(safe_json(report),indent=2,allow_nan=False)+"\n",encoding="utf-8")
    (stage/"REPORT.md").write_text(markdown_report(report),encoding="utf-8")
    manifest=[dict(file=p.name,bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(stage.iterdir()) if p.is_file()]
    (stage/"MANIFEST.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    if output.exists():
        raise FileExistsError(output)
    os.rename(stage,output)
    print(f"Published {output}; runtime {report['runtime_seconds']:.1f}s",flush=True)


def make_figures(stage,report,corrected,grid):
    plt.rcParams.update({"font.size":10,"figure.dpi":140})
    sections=report["section_values"]
    fig,axes=plt.subplots(2,3,figsize=(14,7),constrained_layout=True)
    for ax,name in zip(axes.flat,PLANES):
        for dataset,label in (("cfd_voxel_common","CFD: MRI-scale"),("koma_hybrid_common","Koma: hybrid"),
                              ("cfd_plane_full","CFD: full geometric plane")):
            series=[x for x in sections if x["section"]==name and x["dataset"]==dataset]
            ax.plot([x["time_s"] for x in series],[x["net_m3_s"]*1e6 for x in series],label=label)
        ax.set(title=name.replace("_"," "),xlabel="Motion-cycle time (s)",ylabel="Flow (mL/s)")
        ax.axvspan(.64/.72*report["timing"]["effective_motion_period_s"],report["timing"]["effective_motion_period_s"],color="gray",alpha=.15)
    axes.flat[0].legend(fontsize=8);axes.flat[-1].axis("off")
    axes.flat[-1].text(0,.9,"Internal consistency only\n\nSame support for paired curves\nGray: synthetic closure tail\nNo fitted scale or time lag",va="top")
    fig.suptitle("E-025 | Five-section flow comparison");fig.savefig(stage/"flow_waveforms.png");plt.close(fig)
    rows=report["regional_values"]
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for dataset,label in (("cfd_voxel","CFD: MRI-scale"),("koma_hybrid","Koma: hybrid")):
        series=[x for x in rows if x["region"]=="whole" and x["dataset"]==dataset]
        axes[0].plot([x["time_s"] for x in series],[x["kinetic_energy_j"]*1e3 for x in series],label=label)
        axes[1].plot([x["time_s"] for x in series],[x["speed_p99_m_s"] for x in series],label=label)
    axes[0].set(xlabel="Time (s)",ylabel="Resolved KE (mJ)",title="Common-domain kinetic energy")
    axes[1].set(xlabel="Time (s)",ylabel="Speed (m/s)",title="Volume-weighted p99 speed")
    axes[0].legend();fig.savefig(stage/"energy_speed.png");plt.close(fig)
    fig,axes=plt.subplots(1,5,figsize=(16,3.5),constrained_layout=True)
    # Center-plane profile locations are actual voxel centers, not assumed y=z=0.
    with (stage/"profiles.csv").open(newline="",encoding="utf-8") as stream:
        profile_rows=list(csv.DictReader(stream))
    for ax,name in zip(axes,PLANES):
        for dataset,label in (("cfd","CFD"),("koma","Koma")):
            selected=[x for x in profile_rows if x["bin"]=="4" and x["section"]==name and x["dataset"]==dataset and x["profile_axis"]=="MRI_Y" and x["axial_velocity_m_s"]]
            ax.plot([float(x["coordinate_mm"]) for x in selected],[float(x["axial_velocity_m_s"]) for x in selected],label=label)
        ax.set(title=name.replace("_"," "),xlabel="MRI Y (mm)",ylabel="Axial velocity (m/s)")
    axes[0].legend();fig.suptitle("Bin 4 | center-slice velocity profiles")
    fig.savefig(stage/"velocity_profiles_bin04.png");plt.close(fig)
    with h5py.File(stage/"matched_reference.h5","r") as source:
        ref=source["cfd_voxel_midpoint"][3]
        mask=source["common_mask"][...]
    # A true center slice, never a projection or a phase/velocity conflation.
    mid=grid.shape_spr[0]//2
    a=np.linalg.norm(ref[mid],axis=-1);b=np.linalg.norm(corrected[3,mid],axis=-1)
    a=np.where(mask[mid],a,np.nan);b=np.where(mask[mid],b,np.nan)
    fig,axes=plt.subplots(3,1,figsize=(14,6),constrained_layout=True)
    vmax=max(np.nanpercentile(a,99.5),np.nanpercentile(b,99.5));lim=np.nanpercentile(np.abs(b-a),99.5)
    extent=(-250,250,-40,40)
    for ax,data,title in zip(axes,(a,b,b-a),("CFD voxel speed","Koma hybrid speed","Koma minus CFD")):
        diff=title.startswith("Koma minus")
        im=ax.imshow(data,origin="lower",extent=extent,aspect="equal",cmap="coolwarm" if diff else "viridis",
                     vmin=-lim if diff else 0,vmax=lim if diff else vmax,interpolation="nearest")
        ax.set(title=title+" | bin 4 | center slice",ylabel="MRI Y (mm)")
        fig.colorbar(im,ax=ax,label="m/s",shrink=.7)
    axes[-1].set_xlabel("MRI X/readout (mm)")
    fig.savefig(stage/"velocity_center_slice_bin04.png");plt.close(fig)


def markdown_report(report):
    a,g=report["aggregate"],report["geometry"]
    lines=["# E-025: CFD versus reconstructed Koma hemodynamics","",
        "**Evidence: internal consistency, not independent experimental validation.**","",
        f"Whole common-domain vector RMSE: {a['vector_rmse_m_s']:.5f} m/s; reference-RMS-normalized vector error: {100*a['vector_nrmse_reference_rms']:.2f}%.",
        f"Speed bias: {a['speed']['bias']:.5f} m/s; speed RMSE: {a['speed']['rmse']:.5f} m/s.",
        f"MRI support covers {100*g['geometry_coverage_by_mri_support']:.4f}% of the quadrature CFD volume; {100*g['direct_common_volume_fraction']:.2f}% of the common volume is direct.","",
        "## Section flow","","Both paired curves use MRI-scale velocity and the same geometric area/support. Native-plane and binary-support alternatives are retained in JSON/CSV.","",
        "| Section | Source z (mm) | Flow RMSE (mL/s) | Cycle-volume ratio | Peak ratio |",
        "|---|---:|---:|---:|---:|"]
    for name,x in report["flow_summary"].items():
        lines.append(f"| {name} | {x['source_z_mm']:.3f} | {x['errors']['rmse']*1e6:.3f} | {x['cycle_volume_ratio']:.4f} | {x['peak_ratio']:.4f} |")
    lines += ["","## Gradient-derived metrics","",
        "These errors use only common valid interior derivative stencils; they are not wall-shear measurements.","",
        "| Quantity | RMSE | Reference-RMS-normalized error |","|---|---:|---:|"]
    for name,x in report["derivative_error_summary"]["interior"].items():
        lines.append(f"| {name} | {x['rmse']:.4g} | {100*x['nrmse_reference_rms']:.1f}% |")
    lines += ["","## Included analyses","",
        "Component/vector/speed errors; five section waveforms, forward/reverse flux and cycle volumes; six regional speed/energy summaries; converted native-mesh energy and subvoxel second moments; threshold-sensitive jet centroid/width/direction and downstream exceedance extent; vorticity, helicity, divergence, strain and resolved molecular dissipation on valid common stencils.","",
        "Cycle forward/reverse values partition net section flow in time. The separate spatial_transport fields integrate simultaneous forward and reverse flux. Downstream extent is the furthest axial-p99 threshold exceedance, not a connected or experimentally established jet boundary.","",
        "Sensitivities: 2³ versus 3³ quadrature, midpoint versus three-sample box-bin reference, direct/regularized/interior masks, eroded derivative stencils, identical 1-mm smoothing, and exclusion of synthetic-tail bins. Values and coverage are in report.json; section/regional series are CSV; matched reference fields are HDF5.","",
        f"E-024 regression: PASS; binary-support stroke volume reproduced as {report['regression_e024']['reproduced_stroke_volume_ml']:.6f} mL.","",
        "## Gated results","",*[f"- {k}: {v}" for k,v in report["conditional_metrics"].items()],"",
        "## Interpretation limits","",*[f"- {x}" for x in report["limitations"]],"",
        "## Files","","- flow_waveforms.png\n- energy_speed.png\n- velocity_center_slice_bin04.png\n- velocity_profiles_bin04.png\n- report.json\n- sections.csv\n- regions.csv\n- profiles.csv\n- axial_profiles.csv\n- matched_reference.h5\n- MANIFEST.json","",
        "Material properties verified from all 73 Fluent snapshots: density 1060 kg/m³, molecular viscosity 0.0035 Pa·s. Production reconstruction is unchanged. Report inputs include SHA-256 hashes for source frames and analysis code.",""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
