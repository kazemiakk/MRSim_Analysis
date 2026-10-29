"""E-064 G0d (copy of E-062 audit_sequences_e062.py; 960 lines per bin; the shot-relative ADC gate is informational
here because the FOV changed): parse the E-062 and production 8-bin sequences with the frozen E-027
sequence_timing(); check exact heartbeat periodicity, shot-relative ADC timing equal to the
8-bin set, identical REF/X/Y/Z time weights; report bin windows and seam crossings."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tomllib

import numpy as np

from scripts.aorta_hemodynamic_reference import sequence_timing  # frozen, read-only

ROOT = Path("D:/MRSim/outputs/aorta_aligned_e064")
MULTI = Path("D:/MRSim/MRSIM/MRSimTracks/outputs/aorta_dp2_mri_pipeline/koma/multibin_8x4mm")
LINES = 11


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    target = ROOT/"sequence_audit_v1"
    if target.exists():
        raise FileExistsError(target)
    julia = tomllib.loads((ROOT/"sequences_v1/report_julia.toml").read_text())
    if julia["status"] != "PASS_E064_S0_JULIA_GATES":
        raise ValueError("Julia gates did not pass")
    period = tomllib.loads((MULTI/"motion_phantom_report.toml").read_text())["realized_sequence_rr_s"]
    out = dict(experiment="E-064", stage="S0_G0d", utc=datetime.now(timezone.utc).isoformat(), period_s=period,
               script_sha256=sha(__file__), encodings={})
    weights, te = {}, {}
    for enc in ("REF", "X", "Y", "Z"):
        p20, p8 = ROOT/f"sequences_v1/{enc.lower()}.seq", MULTI/f"sequences/{enc.lower()}.seq"
        if sha(p20) != julia["sequences_sha256"][enc]:
            raise ValueError("sequence changed after S0: "+enc)
        rep20, w20, rec20 = sequence_timing(p20, period)
        rep8, _, rec8 = sequence_timing(p8, period)
        worst = 0.0
        for b in range(1, 21):
            t = np.array([r["adc_first_s"] for r in rec20 if r["bin"] == b])
            t.sort()
            worst = max(worst, float(np.max(np.abs(t[LINES:]-t[:-LINES]-julia["G0c"][enc]["RR_s"]))))
        rel20 = {round(r["adc_first_s"]-r["rf_start_s"], 12) for r in rec20}
        rel8 = {round(r["adc_first_s"]-r["rf_start_s"], 12) for r in rec8}
        nd20 = {(r["n"], round(r["dwell_s"], 15)) for r in rec20}
        nd8 = {(r["n"], round(r["dwell_s"], 15)) for r in rec8}
        weights[enc] = w20; te[enc] = (rel20, rel8)
        out["encodings"][enc] = dict(profiles=rep20["profiles"], bins=len(rep20["bin_timing"]), sequence_duration_s=rep20["sequence_duration_s"],
                                     max_beat_periodicity_error_s=worst, rf_to_first_adc_s_e062=sorted(rel20), rf_to_first_adc_s_8bin=sorted(rel8),
                                     adc_n_dwell_e062=sorted(nd20), adc_n_dwell_8bin=sorted(nd8),
                                     seam_crossings=rep20["RF_to_ADC_seam_crossings"], seam_crossings_8bin=rep8["RF_to_ADC_seam_crossings"],
                                     bin_timing=rep20["bin_timing"])
    e = out["encodings"]
    gates = dict(
        profiles=all(v["profiles"] == 19200 and v["bins"] == 20 for v in e.values()),
        periodicity=all(v["max_beat_periodicity_error_s"] < 1e-9 for v in e.values()),
        weights_identical=all(np.allclose(weights["REF"], weights[k], atol=1e-8) for k in ("X", "Y", "Z")))
    out["gates"] = gates
    out["status"] = "PASS_E064_S0_G0D" if all(gates.values()) else "FAIL_E064_S0_G0D"
    stage = target.with_name(target.name+".building"); stage.mkdir()
    np.save(stage/"frame_weights_20bin.npy", weights["REF"])
    (stage/"report.json").write_text(json.dumps(out, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))+"\n", encoding="utf-8")
    stage.rename(target)
    print(out["status"], json.dumps(gates), {k: (v["max_beat_periodicity_error_s"], v["seam_crossings"], v["seam_crossings_8bin"]) for k, v in e.items()})


if __name__ == "__main__":
    sys.exit(main())
