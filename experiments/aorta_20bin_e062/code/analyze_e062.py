"""E-062 S2: E-027-style CFD-vs-Koma comparison at 20 cardiac bins (see code/PROTOCOL.md).

Uses the frozen E-027 functions read-only. Regression gate: the same code applied to the
8-bin production inputs must reproduce the E-027 section_series net flows (relative 1e-6).
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tomllib

import h5py
import numpy as np
from scipy.spatial import cKDTree

from mrsimtracks.hemodynamics import midpoint_cycle_integrals, section_flow, weighted_error_metrics
from scripts.aorta_hemodynamic_reference import AortaGrid, build_lookup, sequence_timing, temporal_node_values
from scripts.compare_aorta_hemodynamics import masks_for, section_components
from scripts.compare_dp150_hemodynamics import paired_velocity_metrics, safe_json

ROOT = Path("D:/MRSim/outputs/aorta_20bin_e062")
DATA = Path("D:/MRSim/MRSIM/external_inputs/aorta_pointcloud/processed/dp2")
MULTI = Path("D:/MRSim/MRSIM/MRSimTracks/outputs/aorta_dp2_mri_pipeline/koma/multibin_8x4mm")
E027 = Path("D:/MRSim/MRSIM/MRSimTracks/outputs/aorta_hemodynamic_comparison/e027_v1/results_v2")
RECON8 = MULTI/"simulation_motion_1m_8bin_4mm/pc_reconstruction.h5"
RECON20 = ROOT/"combined/simulation_motion_1m_20bin_4mm/pc_reconstruction.h5"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def analyse(recon, seq_dir, period, grid, lookups, velocities, common_override=None):
    with h5py.File(recon, "r") as f:
        magnitude = f["reference_magnitude"][...].astype(float)
        stored = f["mask"][...].astype(bool)
        est = np.moveaxis(f["velocity"][...], 0, -1).astype(float)
    weights = {}
    for e in ("REF", "X", "Y", "Z"):
        _, weights[e], _ = sequence_timing(seq_dir/f"{e.lower()}.seq", period)
    for e in ("X", "Y", "Z"):
        if not np.allclose(weights["REF"], weights[e], atol=1e-8):
            raise ValueError("Encoding time distributions differ")
    nb = len(weights["REF"])
    nodes = temporal_node_values(velocities, weights["REF"], order=grid.order, scale=.72/period)
    ref = lookups["primary"].evaluate(nodes); del nodes
    fraction, plane_fraction = lookups["primary"].fraction, lookups["plane"].fraction
    common = masks_for(magnitude, stored, fraction)["core15"] if common_override is None else common_override
    if not np.all(np.isfinite(ref).all(axis=(0, 4))[common]):
        raise ValueError("Unpaired primary reference support")
    spacing = grid.spacing
    out = dict(bins=nb, core15_voxels=int(common.sum()),
               pooled=paired_velocity_metrics(ref, est, np.broadcast_to(fraction, ref.shape[:-1]), np.broadcast_to(common, ref.shape[:-1])),
               section_series=[], section_summary=[])
    for section in section_components(plane_fraction, grid):
        i = section["readout_index"]; roi = section["roi"]
        valid = roi & common[:, :, i] & (plane_fraction[:, :, i] > 0)
        area = plane_fraction[:, :, i]*spacing[1]*spacing[2]
        q = {}
        for method, field in (("CFD", ref), ("Koma", est)):
            rows = []
            for b in range(nb):
                flow = section_flow(field[b, :, :, i], area, valid=valid)
                out["section_series"].append(dict(section=section["id"], bin=b+1, method=method, x_m=section["x_m"],
                                                  net_ml_s=flow["net_m3_s"]*1e6, sampled_area_mm2=flow["sampled_area_m2"]*1e6))
                rows.append(flow["net_m3_s"])
            q[method] = np.array(rows)
        cyc = {m: midpoint_cycle_integrals(q[m], period) for m in q}
        den = cyc["CFD"]["net_m3"]
        out["section_summary"].append(dict(
            section=section["id"], x_m=section["x_m"], paired_pixels=int(valid.sum()),
            flow_error=weighted_error_metrics(q["CFD"], q["Koma"]),
            net_cycle_volume_ratio=cyc["Koma"]["net_m3"]/den if abs(den) > 1e-12 else None,
            peak_absolute_flow_ratio=float(np.max(abs(q["Koma"]))/np.max(abs(q["CFD"]))),
            temporal_correlation=float(np.corrcoef(q["CFD"], q["Koma"])[0, 1]),
            peak_bin_CFD=int(np.argmax(abs(q["CFD"])))+1, peak_bin_Koma=int(np.argmax(abs(q["Koma"])))+1))
    return out, ref, est, common, weights["REF"], fraction, plane_fraction


def main():
    target = ROOT/"analysis_v1"
    stage = target.with_name(target.name+".building")
    if target.exists() or stage.exists():
        raise FileExistsError(target)
    s1 = tomllib.loads((ROOT/"combined/simulation_motion_1m_20bin_4mm/simulation_report.toml").read_text())
    if s1["status"] != "PASS" or s1["sequence"]["REF"]["cardiac_bins"] != 20:
        raise ValueError("S1 not passed")
    rep027 = json.loads((E027/"report.json").read_text())
    for r in rep027["inputs"]:
        if Path(r["path"]).name in ("velocities.npy", "point_coordinates.npy") and sha(DATA/Path(r["path"]).name) != r["sha256"]:
            raise ValueError("CFD point cloud changed")
    motion = tomllib.loads((MULTI/"motion_phantom_report.toml").read_text())
    period = motion["realized_sequence_rr_s"]
    grid = AortaGrid(center=tuple(motion["source_coordinate_center_m"]),
                     order=tuple(np.array(motion["source_axis_order_largest_to_smallest_one_based"])-1))
    coords = np.load(DATA/"point_coordinates.npy", mmap_mode="r", allow_pickle=False)
    velocities = np.load(DATA/"velocities.npy", mmap_mode="r", allow_pickle=False)
    tree = cKDTree(np.asarray(coords, dtype=np.float32), copy_data=False)
    lookups = {"primary": build_lookup(tree, grid, (2, 2, 2)), "plane": build_lookup(tree, grid, (1, 2, 2))}
    print("lookups built", flush=True)

    # Regression gate on the 8-bin production inputs.
    r8, ref8, est8, common8, w8, fraction, plane_fraction = analyse(RECON8, MULTI/"sequences", period, grid, lookups, velocities)
    old = {(r["section"], r["bin"], r["method"]): r["net_ml_s"] for r in rep027["section_series"]}
    new = {(r["section"], r["bin"], r["method"]): r["net_ml_s"] for r in r8["section_series"]}
    scale = max(abs(v) for v in old.values())
    regression = dict(keys_equal=old.keys() == new.keys(),
                      max_abs_diff_ml_s=float(max(abs(old[k]-new[k]) for k in old)) if old.keys() == new.keys() else None)
    regression["pass"] = bool(regression["keys_equal"] and regression["max_abs_diff_ml_s"] <= 1e-6*scale)
    print("regression", regression, flush=True)
    if not regression["pass"]:
        raise ValueError("8-bin regression failed")

    r20, ref20, est20, common20, w20, _, _ = analyse(RECON20, ROOT/"sequences_v1", period, grid, lookups, velocities)
    r20_mask8, *_ = analyse(RECON20, ROOT/"sequences_v1", period, grid, lookups, velocities, common_override=common8)
    report = dict(experiment="E-062", stage="S2_analysis", utc=datetime.now(timezone.utc).isoformat(), status="COMPLETE_E062_ANALYSIS",
                  evidence_class="Same-source internal consistency; noiseless; one 1M-spin phantom; approximate point-cloud support",
                  period_s=period, source_velocity_time_scale=.72/period, regression_8bin_vs_e027=regression,
                  primary_20bin=r20, sensitivity_20bin_with_8bin_core15=r20_mask8, reproduced_8bin=r8,
                  inputs={str(p): sha(p) for p in (Path(__file__), ROOT/"code/PROTOCOL.md", RECON8, RECON20, E027/"report.json")})
    stage.mkdir()
    (stage/"report.json").write_text(json.dumps(safe_json(report), indent=1)+"\n", encoding="utf-8")
    with h5py.File(stage/"matched_reference_20bin.h5", "w") as f:
        f["adc_time_average"] = ref20.astype(np.float32); f["Koma"] = est20; f["fraction"] = fraction
        f["plane_fraction"] = plane_fraction; f["masks/core15"] = common20; f["frame_weights"] = w20
        f.attrs["dimension_order"] = "bin,slice,phase,readout,component"
        for s in section_components(plane_fraction, grid):
            f["sections/"+s["id"]] = s["roi"]
    (stage/"MANIFEST.json").write_text(json.dumps({p.name: sha(p) for p in sorted(stage.iterdir())}, indent=1)+"\n", encoding="utf-8")
    stage.rename(target)
    for s in r20["section_summary"]:
        print(s["section"], "vol ratio %.4f peak ratio %.4f r %.4f" % (s["net_cycle_volume_ratio"], s["peak_absolute_flow_ratio"], s["temporal_correlation"]), flush=True)


if __name__ == "__main__":
    sys.exit(main())
