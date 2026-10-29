"""Create a reviewed Data-app snapshot from an existing E-025 numerical report.

This script does not run simulations, recompute CFD metrics, build a UI, launch a
server, or publish data. It selects existing report values and converts display
units. All source paths in the snapshot are repository-relative.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    "upstream": "Upstream",
    "throat": "Stenosis throat",
    "near_downstream": "Near downstream",
    "mid_downstream": "Mid downstream",
    "far_downstream": "Far downstream",
    "whole": "Whole lumen",
}
DATASETS = {
    "cfd_plane_full": "CFD plane: full geometry",
    "cfd_plane_common": "CFD plane: common support",
    "cfd_voxel_common": "CFD voxel mean: common support",
    "koma_hybrid_common": "Koma hybrid: common support",
    "cfd_direct_partial": "CFD: direct-only partial area",
    "koma_direct_partial": "Koma: direct-only partial area",
    "koma_hybrid_binary": "Koma: legacy binary support",
    "cfd_voxel": "CFD voxel mean",
    "koma_hybrid": "Koma hybrid",
}


def relative_path(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def scaled(value, factor):
    return None if value is None else value * factor


def make_snapshot(report_path: Path, exact_audit_path: Path | None = None) -> dict:
    raw = report_path.read_bytes()
    report = json.loads(raw)
    if report.get("experiment") != "E-025" or report.get("schema_version") != 1:
        raise ValueError("Expected an E-025 schema-version-1 numerical report")
    if not report.get("regression_e024", {}).get("passed"):
        raise ValueError("E-024 regression check must pass before report packaging")
    source_file = relative_path(report_path)
    source_hash = hashlib.sha256(raw).hexdigest()
    queries = {}

    def add(name, rows, definition, component_ids):
        queries[name] = {
            "rows": rows,
            "source": {
                "label": "E-025 local simulated phantom comparison",
                "files": [source_file],
                "executedAt": report["created_utc"],
                "evidenceFlow": [
                    {"title": "Numerical report", "detail": f"{source_file}; SHA256 {source_hash}"},
                    {"title": "Extraction", "detail": f"scripts/build_dp150_hemodynamic_report_app.py selects {name}; no source metric is fitted or recomputed."},
                    {"title": "Evidence class", "detail": report["evidence_class"]},
                ],
                "metricDefinitions": [{"label": name.replace("_", " "), "definition": definition, "componentIds": component_ids,
                                       "sourceLineage": [{"files": [source_file]}]}],
            },
            "methods": [{"language": "text", "code": definition}],
        }

    flow = []
    for row in report["section_values"]:
        flow.append({"bin": row["bin"], "time_s": row["time_s"], "section": LABELS[row["section"]],
                     "dataset": DATASETS[row["dataset"]], "dataset_key": row["dataset"],
                     "flow_ml_s": row["net_m3_s"] * 1e6, "forward_ml_s": row["forward_m3_s"] * 1e6,
                     "reverse_ml_s": row["reverse_m3_s"] * 1e6, "source_z_mm": row["source_z_mm"],
                     "sampled_area_mm2": row["sampled_area_m2"] * 1e6,
                     "sampled_area_fraction": row["sampled_area_fraction"]})
    add("flow", flow, "Signed normal-velocity section integrals; m3/s multiplied by 1e6 for mL/s. CFD voxel mean and Koma hybrid use common area support. Direct-only quantities are partial-area integrals; legacy binary support is a distinct sensitivity definition.",
        ["flow-waveform", "flow-summary", "report-summary"])
    flow_summary = []
    for key, row in report["flow_summary"].items():
        flow_summary.append({"section": LABELS[key], "source_z_mm": row["source_z_mm"],
                             "cfd_volume_ml": row["cycles"]["cfd_voxel_common"]["net_m3"] * 1e6,
                             "koma_volume_ml": row["cycles"]["koma_hybrid_common"]["net_m3"] * 1e6,
                             "volume_error_pct": 100 * (row["cycle_volume_ratio"] - 1),
                             "flow_rmse_ml_s": row["errors"]["rmse"] * 1e6,
                             "correlation": row["errors"]["pearson_r"], "peak_ratio": row["peak_ratio"],
                             "cfd_peak_bin": row["cfd_peak_bin"], "koma_peak_bin": row["koma_peak_bin"]})
    add("flow_summary", flow_summary, "Full-cycle midpoint net volumes on common area support, m3 multiplied by 1e6 to mL. Volume error = 100*(Koma/CFD-1). Correlation and RMSE compare 15-bin flow waveforms; cardiac bins are not independent subjects.",
        ["flow-summary", "report-summary"])
    energy = [{"bin": row["bin"], "time_s": row["time_s"], "region": LABELS[row["region"]],
               "dataset": DATASETS[row["dataset"]], "energy_mj": row["kinetic_energy_j"] * 1e3,
               "mean_speed_m_s": row["speed_mean_m_s"], "p99_speed_m_s": row["speed_p99_m_s"],
               "max_speed_m_s": row["speed_max_m_s"], "volume_ml": row["volume_m3"] * 1e6}
              for row in report["regional_values"]]
    add("energy", energy, "Resolved kinetic energy from voxel mean velocities: one-half density times volume-weighted squared speed; J multiplied by 1000 for mJ. Each paired region uses common support. This is not turbulent kinetic energy.",
        ["energy-waveform", "energy-table", "regional-speed", "report-summary"])
    aggregate = []
    for scope, values in [("All 15 bins", report["aggregate"]), ("Excluding synthetic tail", report["aggregate_excluding_synthetic_tail"])]:
        for name in ("X", "Y", "Z", "speed"):
            row = values[name]
            aggregate.append({"scope": scope, "quantity": name, "bias_m_s": row["bias"], "mae_m_s": row["mae"],
                              "rmse_m_s": row["rmse"], "nrmse_reference_rms_pct": row["nrmse_reference_rms"] * 100,
                              "correlation": row["pearson_r"]})
        aggregate.append({"scope": scope, "quantity": "vector", "bias_m_s": None, "mae_m_s": None,
                          "rmse_m_s": values["vector_rmse_m_s"],
                          "nrmse_reference_rms_pct": values["vector_nrmse_reference_rms"] * 100,
                          "correlation": None})
    add("velocity_summary", aggregate, "Volume-weighted paired voxel velocity/speed errors. NRMSE uses CFD root-mean-square velocity as denominator, not its range. Synthetic-tail exclusion is a sensitivity subset, not a full cycle.",
        ["velocity-summary", "tail-sensitivity", "report-summary"])
    errors = []
    for row in report["errors_by_bin"]:
        for mask, metrics in row["masks"].items():
            errors.append({"bin": row["bin"], "time_s": row["time_s"], "synthetic_tail": row["synthetic_tail"],
                           "mask": mask, "vector_rmse_m_s": metrics["vector_rmse_m_s"],
                           "vector_nrmse_reference_rms_pct": metrics["vector_nrmse_reference_rms"] * 100,
                           "volume_ml": metrics["X"]["weight_sum"] * 1e6})
    add("velocity_by_mask", errors, "Per-bin vector RMSE on declared masks. Interior, direct and regularized masks select different populations; differences are descriptive and do not isolate causal effects of regularization.",
        ["velocity-error-waveform", "mask-sensitivity"])
    derivatives = []
    for row in report["derivative_values"]:
        for metric, values in row["metrics"].items():
            derivatives.append({"bin": row["bin"], "mask": row["mask"], "metric": metric,
                                "bias": values["bias"], "rmse": values["rmse"],
                                "nrmse_reference_rms_pct": values["nrmse_reference_rms"] * 100,
                                "cfd_mean": values["reference_mean"], "koma_mean": values["estimate_mean"],
                                "volume_ml": row["volume_m3"] * 1e6,
                                "cfd_resolved_dissipation_mw": row["cfd_resolved_dissipation_w"] * 1e3,
                                "koma_resolved_dissipation_mw": row["koma_resolved_dissipation_w"] * 1e3})
    add("derivatives", derivatives, "Matched spatial derivatives on valid interior stencils, with direct-stencil, extra-eroded and matched-1-mm-smoothing sensitivity. Vorticity/divergence/strain units 1/s, helicity m/s2, dissipation density W/m3. Integrated molecular resolved dissipation uses W*1000=mW and excludes walls; it is not total turbulent loss.",
        ["derivative-error", "dissipation-waveform", "derivative-table"])
    derivative_summary = []
    for mask, metrics in report.get("derivative_error_summary", {}).items():
        for metric, values in metrics.items():
            derivative_summary.append({"mask": mask, "metric": metric, "bias": values["bias"], "rmse": values["rmse"],
                                       "nrmse_reference_rms_pct": values["nrmse_reference_rms"] * 100})
    add("derivative_summary", derivative_summary,
        "Pooled weighted raw moments across bins, not an average of per-bin normalized errors. Derivative units vary by metric; the dimensionless RMS-normalized percentage is comparable as a relative error only.", ["derivative-summary"])
    sensitivities = []
    for row in report["sensitivity"]:
        for name, values in row.items():
            if name != "bin":
                sensitivities.append({"bin": row["bin"], "comparison": name, "vector_rmse_m_s": values["vector_rmse_m_s"],
                                      "nrmse_reference_rms_pct": values["vector_nrmse_reference_rms"] * 100})
    add("sensitivities", sensitivities, "Existing per-bin spatial-quadrature and three-time-point sensitivity metrics. Quadrature differences are not numerical convergence proof, and temporal box averaging is not an MRI signal transfer function.",
        ["sampling-sensitivity"])
    jets = [{"bin": row["bin"], "section": LABELS[row["section"]], "dataset": row["dataset"],
             "threshold_fraction": row["threshold_fraction"], "rms_radius_mm": scaled(row["rms_radius_m"], 1000),
             "angle_deg": row["angle_to_normal_deg"], "forward_flux_ml_s": row["forward_flux_m3_s"] * 1e6,
             "centroid_readout_mm": scaled(row["centroid_m"][0], 1000),
             "centroid_phase_mm": scaled(row["centroid_m"][1], 1000),
             "centroid_slice_mm": scaled(row["centroid_m"][2], 1000)} for row in report["jet_values"]]
    add("jets", jets, "Forward-flux-weighted jet centroid, RMS radius and direction above fixed fractions of the reference throat p99 speed. Empty thresholded regions remain null, not zero-width jets. This is a threshold-dependent descriptive jet measure.",
        ["jet-width", "jet-table"])
    add("gates", [{"parameter": key, "status": value} for key, value in report["conditional_metrics"].items()],
        "Explicit unavailable or scientifically gated comparison endpoints; no values are invented.", ["gated-metrics"])
    add("limitations", [{"limitation": value} for value in report["limitations"]],
        "Limitations recorded by the numerical analysis.", ["methods-limitations"])
    add("geometry", [{"quantity": key, "value": value} for key, value in report["geometry"].items()],
        "Geometry and coverage statistics from the numerical report; volume quantities are stored in m3, fractions are unitless, counts are voxel counts.", ["geometry-coverage"])
    snapshot = {
        "surface": "report", "title": "CFD versus Koma: hemodynamic agreement depends on the measurement",
        "generatedAt": report["created_utc"], "status": "simulated-internal-comparison", "buildStatus": "creating",
        "filters": [], "queries": queries,
        "report": {"asOf": report["created_utc"][:10], "experiment": "E-025", "sourceReport": source_file,
                   "sourceReportSha256": source_hash, "evidenceClass": report["evidence_class"],
                   "timing": report["timing"], "properties": report["properties"],
                   "regressionPassed": report["regression_e024"]["passed"],
                   "vectorRmseMS": report["aggregate"]["vector_rmse_m_s"],
                   "vectorNrmsePct": report["aggregate"]["vector_nrmse_reference_rms"] * 100,
                   "geometry": report["geometry"]},
    }
    if exact_audit_path:
        exact_raw = exact_audit_path.read_bytes()
        audit = json.loads(exact_raw)
        if audit.get("input_report", {}).get("sha256") != source_hash:
            raise ValueError("Exact-plane audit does not match the selected numerical report")
        audit_rows = [{"section": LABELS[key], "bin": 4, "exact_flow_ml_s": value,
                       "maximum_plane_quadrature_difference_ml_s": audit["sections"][key]["cfd_plane_full"]["max_absolute_difference_ml_s"]}
                      for key, value in audit["bin4_exact_ml_s"].items()]
        queries["exact_section_audit"] = {
            "rows": audit_rows,
            "source": {"label": "Exact linear-tetrahedron converted-CFD flux audit", "files": [relative_path(exact_audit_path)],
                       "executedAt": audit["created_utc"],
                       "evidenceFlow": [{"title": "Exact-plane audit", "detail": f"{relative_path(exact_audit_path)}; SHA256 {hashlib.sha256(exact_raw).hexdigest()}"}],
                       "metricDefinitions": [{"label": "Converted CFD flux", "definition": audit["method"] + " " + audit["quadrature"], "componentIds": ["exact-section-audit", "exact-audit-context"]}]},
            "methods": [{"language": "text", "code": "\n".join(audit["limitations"])}],
        }
        snapshot["report"]["exactAudit"] = {
            "bin4VariationPct": next(row["range_over_mean_absolute"] * 100 for row in audit["exact_between_plane_variation"] if row["bin"] == 4),
            "maximumQuadratureDifferenceMLS": audit["maximum_plane_quadrature_difference_ml_s"],
        }
    encoded = json.dumps(snapshot, allow_nan=False)
    if re.search(r"[A-Za-z]:[\\/]", encoded) or "C:\\\\Users" in encoded:
        raise ValueError("Absolute private path detected in report snapshot")
    return snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replace", action="store_true", help="Replace this script's own reviewed snapshot")
    parser.add_argument("--app-id-from", type=Path, help="Preserve the prepared report's stable artifact identity")
    parser.add_argument("--complete", action="store_true", help="Mark finished content authoring complete")
    parser.add_argument("--exact-audit", type=Path, help="Include checked exact-section flux addendum")
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(ROOT)
    if output.exists() and not args.replace:
        raise FileExistsError(output)
    snapshot = make_snapshot(args.report, args.exact_audit)
    if args.app_id_from:
        identity = json.loads(args.app_id_from.read_text(encoding="utf-8"))
        for key in ("id", "appId"):
            if key in identity:
                snapshot[key] = identity[key]
    if args.complete:
        snapshot["buildStatus"] = "complete"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(snapshot, stream, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(output)
    print(json.dumps({"output": relative_path(output), "queries": len(snapshot["queries"]),
                      "rows": sum(len(q["rows"]) for q in snapshot["queries"].values()),
                      "sourceReportSha256": snapshot["report"]["sourceReportSha256"]}))


if __name__ == "__main__":
    main()
