"""E-057 Stage 1 analysis: Koma step ladder against the UNCHANGED saved E056 oracle.

Gates are copied from code/PROTOCOL.md and fixed before any E-057 signal.
No gain, phase, time shift, velocity scale or mask is fitted.
"""
from datetime import datetime, timezone
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
VENC = 5.0
LATE_FRACTION = 0.10


def relative(a, b):
    return float(np.linalg.norm(a-b)/np.linalg.norm(b))


def axial(x_image, ref_image):
    return np.angle(x_image*ref_image.conj())*VENC/np.pi


def output_digest(manifest, name):
    return manifest["outputs"][name]["sha256"]


def main():
    target = ROOT/"ladder_analysis_v1"
    stage = target.with_name(target.name+".building")
    if target.exists() or stage.exists():
        raise FileExistsError(target)
    run = ROOT/"ladder_v1"
    receipt = tomllib.loads((run/"report.toml").read_text(encoding="utf-8"))
    if receipt["status"] != "COMPLETE_E057_STAGE1_ACQUISITIONS":
        raise ValueError("Stage 1 acquisitions incomplete")
    preflight = tomllib.loads((ROOT/"grid_preflight_v1/report.toml").read_text(encoding="utf-8"))
    acquisition = tomllib.loads((E056/"acquisition_v1/report.toml").read_text(encoding="utf-8"))
    analysis_manifest = json.loads((E056/"analysis_v1/MANIFEST.json").read_text(encoding="utf-8"))
    fields_path = E056/"analysis_v1/independent_fields.h5"
    if reference.sha(fields_path) != output_digest(analysis_manifest, "independent_fields.h5"):
        raise ValueError("Saved E056 oracle changed")
    ladder = sorted((float(x) for x in preflight["ladder_retained_s"]), reverse=True)
    labels = ["inf"]+[f"{round(x*1e6)}us" for x in ladder]
    finest = labels[-1]

    def koma(model, enc, label):
        key = f"{model}_{ARM}_{enc.lower()}_{label}"
        path = run/f"{key}.h5"
        if reference.sha(path) != receipt["result_sha256"][key]:
            raise ValueError("E057 result changed: "+key)
        with h5py.File(path) as f:
            return reference.complex_array(f["raw_signal_f64"]).ravel(), reference.complex_array(f["image"])

    def e056_raw(model, enc):
        key = f"{model}_{ARM}_{enc.lower()}"
        path = E056/"acquisition_v1"/f"{key}.h5"
        if reference.sha(path) != acquisition["result_sha256"][key]:
            raise ValueError("E056 result changed: "+key)
        with h5py.File(path) as f:
            return reference.complex_array(f["raw_signal_f64"]).ravel()

    checks, failures = 0, []

    def gate(name, passed):
        nonlocal checks
        checks += 1
        if not passed:
            failures.append(name)
        return bool(passed)

    with h5py.File(fields_path) as oracle, h5py.File(E056/"reference_contract_v2/waveforms.h5") as wave:
        exact = {enc: (np.asarray(oracle[f"uniform_osc/{ARM}/{enc}/raw_signal_independent"][()]).ravel(),
                       np.asarray(oracle[f"uniform_osc/{ARM}/{enc}/image_independent"][()])) for enc in ("REF", "X")}
        core = oracle[f"uniform_osc/{ARM}/analysis/paired_core"][()].astype(bool)
        analytic = oracle[f"uniform_osc/{ARM}/analysis/velocity_instantaneous_analytic"][()][..., 0]
        static_core = oracle[f"static/{ARM}/analysis/paired_core"][()].astype(bool)
        lab = np.asarray(wave[ARM]["REF"]["labels0"][()], int)
        if lab.shape[1] != 3 and lab.shape[0] == 3:
            lab = lab.T
    oracle_velocity = axial(exact["X"][1], exact["REF"][1])

    rows, signals = [], {}
    for label in labels:
        row = dict(dt_prec=label)
        images = {}
        for enc in ("REF", "X"):
            raw, image = koma("uniform_osc", enc, label)
            signals[(enc, label)] = raw
            images[enc] = image
            k = np.zeros((15, 32, 16, 80), np.complex128)
            k[lab[:, 2], lab[:, 1], lab[:, 0]] = raw.reshape(-1, 80)
            fft = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(k, axes=(1, 2, 3)), axes=(1, 2, 3)), axes=(1, 2, 3))
            fft_error = float(np.max(abs(fft-image))/np.max(abs(fft)))
            gate(f"independent_fft_{enc}_{label}", fft_error <= 3e-6)
            late = slice(int(round((1-LATE_FRACTION)*7680))*80, None)
            row[enc] = dict(raw_relative_L2=relative(raw, exact[enc][0]),
                            image_relative_L2=relative(image, exact[enc][1]),
                            late_raw_relative_L2=relative(raw[late], exact[enc][0][late]),
                            independent_fft_relative_max=fft_error)
        velocity = axial(images["X"], images["REF"])
        row["Koma_vs_oracle_axial_RMSE_m_s"] = float(np.sqrt(np.mean((velocity[:, core]-oracle_velocity[:, core])**2)))
        row["Koma_vs_instantaneous_analytic_axial_RMSE_m_s"] = float(np.sqrt(np.mean((velocity[:, core]-analytic[:, core])**2)))
        rows.append(row)

    # R0: the custom rule at Inf must reproduce the saved E056 signal.
    reproduction = {enc: relative(signals[(enc, "inf")], e056_raw("uniform_osc", enc)) for enc in ("REF", "X")}
    for enc, value in reproduction.items():
        gate(f"R0_reproduce_E056_{enc}", value <= 1e-12)

    # Acceptance at the finest retained step.
    best = rows[-1]
    for enc in ("REF", "X"):
        gate(f"raw_L2_{enc}_{finest}", best[enc]["raw_relative_L2"] <= .01)
        gate(f"image_L2_{enc}_{finest}", best[enc]["image_relative_L2"] <= .01)
        gate(f"late_clock_L2_{enc}_{finest}", best[enc]["late_raw_relative_L2"] <= .01)
    gate(f"Koma_vs_oracle_axial_RMSE_{finest}", best["Koma_vs_oracle_axial_RMSE_m_s"] <= .01)

    # Convergence: oracle error nonincreasing over the last three retained steps.
    convergence = {}
    for enc in ("REF", "X"):
        errors = [r[enc]["raw_relative_L2"] for r in rows[1:]]
        tail = errors[-3:]
        gate(f"nonincreasing_last_three_{enc}", all(b <= a+1e-4 for a, b in zip(tail[:-1], tail[1:])))
        differences = [relative(signals[(enc, a)], signals[(enc, b)]) for a, b in zip(labels[1:-1], labels[2:])]
        orders = [float(np.log2(a/b)) if b > 0 else None for a, b in zip(differences[:-1], differences[1:])]
        convergence[enc] = dict(steps=labels[1:], oracle_relative_L2=errors,
                                successive_self_difference_relative_L2=differences, observed_order=orders)

    # Static control: exact trapezoid identity and zero velocity.
    static = {}
    static_images = {}
    for enc in ("REF", "X"):
        raw, image = koma("static", enc, finest)
        static_images[enc] = image
        static[enc] = dict(relative_L2_to_E056_static=relative(raw, e056_raw("static", enc)))
        gate(f"static_identity_{enc}", static[enc]["relative_L2_to_E056_static"] <= 1e-9)
    static_velocity = axial(static_images["X"], static_images["REF"])
    static["core_axial_RMSE_m_s"] = float(np.sqrt(np.mean(static_velocity[:, static_core]**2)))
    gate("static_velocity", static["core_axial_RMSE_m_s"] <= .001)

    status = "PASS_E057_STAGE1_STEP_CONVERGENCE" if not failures else "FAIL_E057_STAGE1_STEP_CONVERGENCE"
    e056_primary = json.loads((E056/"analysis_v1/report.json").read_text(encoding="utf-8"))
    e056_rows = {r["encoding"]: r for r in e056_primary["numerical_checks"] if r["model"] == "uniform_osc" and r["arm"] == ARM}
    report = dict(status=status, experiment="E-057", stage="1_uniform_fast_readout_step_ladder",
        utc=datetime.now(timezone.utc).isoformat(), checks=checks, failed_gates=failures,
        finest_step=finest, rows=rows, convergence=convergence, R0_reproduction_relative_L2=reproduction,
        static_control=static, paired_core_voxels=int(core.sum()), static_core_voxels=int(static_core.sum()),
        E056_primary_raw_relative_L2={k: v["raw_relative_l2"] for k, v in e056_rows.items()},
        acceptance_reference_changed=False, fitted_parameters=None, new_1m=False,
        evidence_class="Numerical verification against an independent implementation sharing exported waveforms; not physical validation.",
        limitation="Two uniform-motion encodings of one arm; the full 36-encoding suite is Stage 2. Synthetic stroboscopic control, not CFD-driven stenosis accuracy.")
    stage.mkdir()
    reference.write_json(stage/"report.json", report)
    inputs = [fields_path, E056/"analysis_v1/MANIFEST.json", E056/"reference_contract_v2/waveforms.h5",
              run/"report.toml", ROOT/"grid_preflight_v1/report.toml", Path(__file__), E056/"code/analyze_physics.py"]
    reference.write_json(stage/"MANIFEST.json", dict(inputs={str(p): reference.sha(p) for p in inputs},
        e057_results=receipt["result_sha256"], report_sha256=reference.sha(stage/"report.json")))
    stage.rename(target)
    print(json.dumps(dict(status=status, checks=checks, failed_gates=failures)), flush=True)


if __name__ == "__main__":
    main()
