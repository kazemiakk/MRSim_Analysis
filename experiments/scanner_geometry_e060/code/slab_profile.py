"""E-060 S0b: independent finite-RF Bloch slab profile and fold-in gate (no Koma code).

Uses the exported first-shot excitation (B1 and slab gradient) and E056's RK4
`rf_affine`. Static spins along z, Mz = 1 before the pulse.
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
ROOT = Path("D:/MRSim/outputs/scanner_geometry_e060")
sys.path.insert(0, str(E056/"code"))
import analyze_physics as reference  # noqa: E402

FOV_EDGE = .0375
SLAB_HALF = .03
GATE = .02


def main():
    target = ROOT/"slab_profile_v1"
    if target.exists():
        raise FileExistsError(target)
    s0 = tomllib.loads((ROOT/"sequences_v1/report.toml").read_text(encoding="utf-8"))
    timing = ROOT/"sequences_v1/timing_e060.h5"
    if s0["status"] != "PASS_E060_SEQUENCES_AND_AUDITS" or reference.sha(timing) != s0["timing_sha256"]:
        raise ValueError("S0 sequences missing or changed")
    with h5py.File(timing) as f:
        g = f["excitation_REF_first_shot"]
        grad = np.asarray(g["g_T_m"][()], float)
        if grad.shape[1] != 3 and grad.shape[0] == 3:
            grad = grad.T
        template = dict(t=np.asarray(g["t_s"][()], float), g=grad, b1=reference.complex_array(g["b1_T"]),
                        frequency=np.asarray(g["rf_frequency_Hz"][()], float))
    z = np.linspace(-.15, .15, 1201)
    op = reference.rf_affine(template, z, 1.0, 1.3, .1, 42.577478518e6)
    mxy = np.abs(op["axy"]+op["bxy"])
    profile = mxy/mxy[np.argmin(np.abs(z))]
    outside = np.abs(z) >= FOV_EDGE
    half = profile >= .5
    report = dict(experiment="E-060", stage="S0b_slab_profile", utc=datetime.now(timezone.utc).isoformat(),
        max_excitation_beyond_fov_edge=float(profile[outside].max()),
        excitation_at_slab_edges=[float(np.interp(-SLAB_HALF, z, profile)), float(np.interp(SLAB_HALF, z, profile))],
        fwhm_mm=float((z[half].max()-z[half].min())*1e3), rf_duration_s=float(template["t"][np.flatnonzero(np.abs(template["b1"]) > 0)[-1]]
                                                                               - template["t"][np.flatnonzero(np.abs(template["b1"]) > 0)[0]]),
        gate=GATE, status="PASS_E060_SLAB_FOLD_IN_GATE" if profile[outside].max() <= GATE else "FAIL_E060_SLAB_FOLD_IN_GATE",
        method="Independent RK4 finite-RF Bloch (E056 rf_affine) on the exported REF first-shot excitation; static spins; Mz0=1.")
    target.mkdir()
    (target/"report.json").write_text(json.dumps(report, indent=1)+"\n", encoding="utf-8")
    np.savetxt(target/"profile.csv", np.column_stack([z*1e3, profile]), delimiter=",", header="z_mm,relative_Mxy", comments="")
    (target/"MANIFEST.json").write_text(json.dumps(dict(script_sha256=reference.sha(Path(__file__)), timing_sha256=reference.sha(timing),
        report_sha256=reference.sha(target/"report.json")), indent=1)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
