"""E-057 C0c: inspect Koma's actual exported free-precession grid.

Diagnostic only. It does not replace the exact E056 oracle or any acceptance gate.
The unchanged E056 reference module is imported read-only; only its phase-moment
quadrature is swapped, in this process, for a trapezoid on Koma's own nodes.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tomllib

import h5py
import numpy as np

E056 = Path("D:/MRSim/outputs/acquisition_physics_e056")
ROOT = Path("D:/MRSim/outputs/step_convergence_e057")
sys.path.insert(0, str(E056/"code"))
import analyze_physics as reference  # noqa: E402

ARM = "bw700_te274"
TOL = 1e-11


def sha(path):
    return reference.sha(path)


def manifest_digest(manifest, name):
    hits = [v["sha256"] for k, v in manifest["inputs"].items() if Path(k).name == name]
    if len(hits) != 1:
        raise ValueError("Manifest entry not unique: "+name)
    return hits[0]


def main():
    target = ROOT/"grid_certification_v1"
    stage = target.with_name(target.name+".building")
    if target.exists() or stage.exists():
        raise FileExistsError(target)
    preflight = tomllib.loads((ROOT/"grid_preflight_v1/report.toml").read_text(encoding="utf-8"))
    grid_path = ROOT/"grid_preflight_v1/primary_grid.h5"
    if preflight["status"] != "PASS_E057_GRID_CERTIFICATION" or sha(grid_path) != preflight["primary_grid_sha256"]:
        raise ValueError("Stage 0 grid receipt missing or changed")
    acquisition = tomllib.loads((E056/"acquisition_v1/report.toml").read_text(encoding="utf-8"))
    contract = json.loads((E056/"reference_contract_v2/registration.json").read_text(encoding="utf-8"))
    analysis_manifest = json.loads((E056/"analysis_v1/MANIFEST.json").read_text(encoding="utf-8"))
    wave = E056/"reference_contract_v2/waveforms.h5"
    if sha(wave) != contract["waveforms_sha256"]:
        raise ValueError("Reference waveform changed")
    if sha(E056/"code/analyze_physics.py") != manifest_digest(analysis_manifest, "analyze_physics.py"):
        raise ValueError("E056 reference implementation changed")
    controls = E056/"inputs_v1/controls.h5"
    if sha(controls) != acquisition["input_sha256"]:
        raise ValueError("Control input changed")
    with h5py.File(controls) as f:
        positions = np.asarray(f["positions"], float)
        if positions.shape[1] != 3 and positions.shape[0] == 3:
            positions = positions.T
        rho = np.asarray(f["rho"], float).ravel()
        u = np.asarray(f["u_uniform_osc"], float).ravel()
        t1, t2 = float(f["T1"][0]), float(f["T2"][0])
        times = np.asarray(f["path_time_s"], float)
        factor = np.asarray(f["path_basis_s"], float)
        period = float(f["motion_period_s"][()])
    rows = []
    with h5py.File(wave) as f, h5py.File(grid_path) as g, h5py.File(E056/"analysis_v1/independent_fields.h5") as oracle:
        for enc in ("REF", "X"):
            meta = reference.load_group(f[ARM][enc], 80, 16, 32)
            meta["gamma"] = float(f["gamma_Hz_T"][()])
            T = np.asarray(g[enc]["t_s"], float)
            G = np.column_stack([np.asarray(g[enc][k], float) for k in ("Gx_T_m", "Gy_T_m", "Gz_T_m")])
            stats = dict(windows=0, identical_to_diagnostic_knots=0, missing_nodes=0, extra_nodes=0,
                         max_first_node_offset_s=0., max_waveform_difference_T_m=0., longest_interval_s=0.)

            def koma_grid_moments(template, rf_start, rf_end, path_time, path_factor, period_s):
                adc = template["adc"]; finish = float(adc[-1])
                lo = np.searchsorted(T, rf_start+rf_end-TOL)
                hi = np.searchsorted(T, rf_start+finish+TOL, side="right")
                tk = T[lo:hi]-rf_start; gk = G[lo:hi]
                if not len(tk) or abs(tk[0]-rf_end) > TOL or abs(tk[-1]-finish) > TOL:
                    raise AssertionError("Koma window does not start at RF end and stop at last ADC")
                distinct = np.r_[True, np.diff(tk) > TOL]
                nodes = tk[distinct]
                cuts = np.unique(np.r_[rf_end, finish, adc, template["t"]])
                cuts = cuts[(cuts >= rf_end-1e-15) & (cuts <= finish+1e-15)]
                distance = np.abs(nodes[:, None]-cuts[None, :])
                missing = int(np.sum(distance.min(axis=0) > TOL))
                extra = int(np.sum(distance.min(axis=1) > TOL))
                stats["windows"] += 1
                stats["identical_to_diagnostic_knots"] += int(missing == 0 and extra == 0 and len(nodes) == len(cuts))
                stats["missing_nodes"] += int(missing); stats["extra_nodes"] += int(extra)
                stats["max_first_node_offset_s"] = max(stats["max_first_node_offset_s"], abs(float(tk[0]-rf_end)))
                # Compare waveform values only where Koma holds no left/right discontinuity pair.
                gap = np.diff(tk) > TOL
                single = np.r_[True, gap] & np.r_[gap, True]
                expected = reference.interpolate(tk[single], template["t"], template["g"])
                stats["max_waveform_difference_T_m"] = max(stats["max_waveform_difference_T_m"],
                                                           float(np.max(np.abs(gk[single]-expected))))
                dt = np.diff(tk)
                stats["longest_interval_s"] = max(stats["longest_interval_s"], float(dt.max()))
                unit = np.interp(np.remainder(rf_start+tk, period_s), path_time, path_factor)
                ordinary = np.vstack([np.zeros(3), np.cumsum((gk[:-1]+gk[1:])*dt[:, None]/2, axis=0)])
                product = gk[:, 0]*unit
                moving = np.r_[0., np.cumsum((product[:-1]+product[1:])*dt/2)]
                idx = np.searchsorted(tk, adc-TOL)
                if np.max(np.abs(tk[idx]-adc)) > TOL:
                    raise AssertionError("ADC node missing from Koma grid")
                return ordinary[idx], moving[idx]

            original = reference.phase_moments
            try:
                reference.phase_moments = koma_grid_moments
                predicted, _ = reference.predict(meta, positions, rho, u, t1, t2, times, factor, period, (80, 16, 32, 15))
            finally:
                reference.phase_moments = original
            result = E056/"acquisition_v1"/f"uniform_osc_{ARM}_{enc.lower()}.h5"
            if sha(result) != acquisition["result_sha256"][f"uniform_osc_{ARM}_{enc.lower()}"]:
                raise ValueError("E056 Koma result changed")
            with h5py.File(result) as r:
                actual = reference.complex_array(r["raw_signal_f64"]).ravel()
            exact = np.asarray(oracle[f"uniform_osc/{ARM}/{enc}/raw_signal_independent"][()]).ravel()
            rows.append(dict(encoding=enc, **stats,
                Koma_vs_exact_oracle_relative_L2=float(np.linalg.norm(actual-exact)/np.linalg.norm(exact)),
                Koma_vs_trapezoid_on_Koma_nodes_relative_L2=float(np.linalg.norm(actual-predicted)/np.linalg.norm(predicted)),
                trapezoid_on_Koma_nodes_vs_exact_oracle_relative_L2=float(np.linalg.norm(predicted-exact)/np.linalg.norm(exact))))
            print("E057_C0C", json.dumps(rows[-1]), flush=True)
    report = dict(experiment="E-057", stage="C0c_actual_grid_inspection", utc=datetime.now(timezone.utc).isoformat(),
        status="COMPLETE_DIAGNOSTIC_NOT_A_GATE", rows=rows, acceptance_reference_changed=False, new_1m=False,
        interpretation="Koma's exported free-precession nodes are compared with the E056 diagnostic knot grid, "
                       "and the independent free-precession phase is recomputed by trapezoid on those exact nodes. "
                       "Residual Koma-versus-this-trapezoid difference measures what remains outside free-precession quadrature.")
    stage.mkdir()
    reference.write_json(stage/"report.json", report)
    inputs = [grid_path, ROOT/"grid_preflight_v1/report.toml", wave, controls, E056/"code/analyze_physics.py",
              E056/"analysis_v1/independent_fields.h5", Path(__file__)]
    inputs += [E056/"acquisition_v1"/f"uniform_osc_{ARM}_{e}.h5" for e in ("ref", "x")]
    reference.write_json(stage/"MANIFEST.json", dict(inputs={str(p): sha(p) for p in inputs},
                                                     report_sha256=sha(stage/"report.json")))
    stage.rename(target)
    print(report["status"], flush=True)


if __name__ == "__main__":
    main()
