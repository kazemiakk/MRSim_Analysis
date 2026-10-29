"""Phantom (dp150) CFD inlet flow waveform, native and as simulated in Koma (user: "the phantom CFD peak flow is 175 mL/s
but it is not in the data; have you changed the data?").

Read-only. Flow = integral of the axial velocity over planar sections of each CFD frame (pyvista slice), at the inlet
cap (+0.5 mm) and 30 mm upstream of the throat. Two CFD copies: the converted Fluent frames (dp150_volume_pipeline,
73 frames, 10 ms, 0.72 s) and the periodic-closed copy used for tracking (last 80 ms blended). Koma replays the
0.72 s trajectories over the scan RR 1.012 s, so its velocities and flows are the CFD x 0.72/1.012 (E-055/E-060
velocity_scale); panel b shows that retimed curve with the Koma 30-bin upstream flows and the 30-bin CFD reference.
Integrity: periodic files are compared with the E-060 references_v2 MANIFEST hashes.
"""
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyvista as pv  # noqa: E402

REPO = Path("D:/MRSim/MRSIM/MRSimTracks/outputs")
E060 = Path("D:/MRSim/outputs/scanner_geometry_e060")
ORIG = REPO/"dp150_volume_pipeline/flow/dp150_flow.pvd"
PERI = REPO/"dp150_mri_pipeline/flow_periodic/dp150_periodic.pvd"
Z_IN = -0.10610323399305344+5e-4
Z_UP = -0.030
RR, T_CFD = 1.012, 0.72
FS = 1.5


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def section_flows(pvd):
    ds = ET.parse(pvd).findall(".//DataSet")
    t = np.array([float(d.attrib["timestep"]) for d in ds])
    q = {"inlet": [], "up30": []}
    for d in ds:
        m = pv.read(pvd.parent/d.attrib["file"])
        for k, z in (("inlet", Z_IN), ("up30", Z_UP)):
            s = m.slice(normal=(0, 0, 1), origin=(0, 0, z))
            s.point_data["w"] = np.asarray(s.point_data["Velocity"])[:, 2]
            q[k].append(float(s.integrate_data().point_data["w"][0])*1e6)
    return t, {k: np.array(v) for k, v in q.items()}


def main(target):
    target = Path(target)
    if target.exists():
        raise FileExistsError(target)
    stage = target.with_name(target.name+".building"); stage.mkdir()
    plt.rcParams["font.size"] = 10*FS
    rep = dict(figure="dp150 CFD inlet flow waveform", utc=datetime.now(timezone.utc).isoformat(), script_sha256=sha(__file__))
    man = json.loads((E060/"references_v2/MANIFEST.json").read_text())["inputs"]
    reg = {k: v for k, v in man.items() if "dp150_periodic" in k}
    same = sum(sha(k) == v for k, v in reg.items())
    if same != len(reg):
        raise ValueError("periodic CFD files differ from the E-060 registration")
    rep["periodic_cfd_files_identical_to_E060_registration"] = f"{same}/{len(reg)}"
    rep["original_pvd_sha256"] = sha(ORIG)
    to, qo = section_flows(ORIG)
    tp, qp = section_flows(PERI)
    s = T_CFD/RR
    # Koma 30-bin upstream flows (E-060 scan-method, noiseless) and the 30-bin CFD reference at the same slices/timing.
    km = json.loads((E060/"analysis_v1/report.json").read_text())["scan_method"]["koma"]
    q_koma = np.array(km["flow_ml_s"])[km["upstream_slices"]].mean(axis=0)
    with h5py.File(E060/"references_v2/references.h5") as f:
        pref, pfr, z = f["plane_reference"][:], f["plane_fraction"][:], f["partition_z_m"][:]
    up = [k for k in range(len(z)) if -.030-1e-9 <= z[k] <= -.020+1e-9]
    q_ref = np.einsum("bspr,spr->bs", np.nan_to_num(pref[..., 2].astype(np.float64)), pfr*1.5625)[:, up].mean(axis=1)
    tb = (np.arange(30)+.5)/30*RR
    i = int(np.argmax(qo["inlet"]))
    rep.update(native=dict(cycle_s=T_CFD, inlet_peak_ml_s=float(qo["inlet"][i]), inlet_peak_time_s=float(to[i]),
                           inlet_mean_ml_s=float(qo["inlet"][:-1].mean()), up30_peak_ml_s=float(qo["up30"].max()), up30_mean_ml_s=float(qo["up30"][:-1].mean())),
               periodic=dict(inlet_peak_ml_s=float(qp["inlet"].max()), inlet_mean_ml_s=float(qp["inlet"][:-1].mean()), synthetic_tail_s=[0.64, 0.72]),
               koma_retimed=dict(rr_s=RR, velocity_scale=s, inlet_peak_ml_s=float(qp["inlet"].max()*s), inlet_mean_ml_s=float(qp["inlet"][:-1].mean()*s),
                                 koma_30bin_upstream_peak_ml_s=float(q_koma.max()), koma_peak_bin=int(np.argmax(q_koma))+1,
                                 cfd_ref_30bin_upstream_peak_ml_s=float(q_ref.max())),
               note_174="~174 mL/s (journal, E-059 entry) = retimed CFD at the throat section in the 15-bin E048 reference (173.5); upstream 181.1.")
    with open(stage/"cfd_inlet_flow.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cfd_time_s", "inlet_flow_native_ml_s", "inlet_flow_periodic_ml_s", "up30mm_flow_native_ml_s",
                    "koma_time_s", "koma_inlet_flow_retimed_ml_s"])
        for k in range(len(to)):
            w.writerow([f"{to[k]:.3f}", f"{qo['inlet'][k]:.2f}", f"{qp['inlet'][k]:.2f}", f"{qo['up30'][k]:.2f}", f"{tp[k]/s:.4f}", f"{qp['inlet'][k]*s:.2f}"])
    C = dict(cfd="#1f5fa8", tail="#7aa6d6", koma="#d9822b", ref="#555555")
    fig, ax = plt.subplots(1, 2, figsize=(22, 8), layout="constrained", sharey=True)
    a = ax[0]
    a.axvspan(.64, .72, color="#e8e8e8", lw=0)
    a.plot(tp, qp["inlet"], color=C["tail"], lw=2, ls="--", label="Periodic copy used for tracking (tail blended)")
    a.plot(to, qo["inlet"], color=C["cfd"], lw=2.4, label="Inlet")
    a.plot(to, qo["up30"], color=C["cfd"], lw=1.4, ls=":", label="30 mm upstream of throat")
    a.annotate(f"peak {qo['inlet'][i]:.0f} mL/s at {to[i]*1e3:.0f} ms", (to[i], qo["inlet"][i]), xytext=(to[i]+.08, qo["inlet"][i]+6),
               arrowprops=dict(arrowstyle="-", color="#888"), fontsize=9*FS)
    a.text(.68, 10, "synthetic\ntail", ha="center", fontsize=8*FS, color="#666")
    a.set_title(f"(a) CFD (Fluent dp150), {T_CFD:.2f} s cycle | mean {qo['inlet'][:-1].mean():.0f} mL/s", fontsize=10*FS)
    a.set_xlabel("CFD time (s)"); a.set_ylabel("Flow (mL/s)"); a.set_xlim(0, T_CFD)
    b = ax[1]
    b.plot(tp/s, qp["inlet"]*s, color=C["cfd"], lw=2.4, label=f"CFD x {s:.3f} (retimed {T_CFD:.2f} -> {RR:.3f} s)")
    b.plot(tb, q_ref, "o", mfc="none", mec=C["ref"], ms=8, mew=1.6, label="CFD reference, 30 bins, upstream slices")
    b.plot(tb, q_koma, "s", color=C["koma"], ms=7, label="Koma, 30 bins, upstream slices")
    b.annotate(f"peak {qp['inlet'].max()*s:.0f} mL/s", (to[i]/s, qp["inlet"].max()*s), xytext=(to[i]/s+.1, qp["inlet"].max()*s+6),
               arrowprops=dict(arrowstyle="-", color="#888"), fontsize=9*FS)
    b.set_title(f"(b) As simulated in Koma, scan RR {RR:.3f} s | mean {qp['inlet'][:-1].mean()*s:.0f} mL/s", fontsize=10*FS)
    b.set_xlabel("Cardiac time (s)"); b.set_xlim(0, RR)
    for x in ax:
        x.grid(color="#dddddd", lw=.8); x.set_ylim(-10, 290); x.legend(fontsize=8*FS, loc="upper right", frameon=False)
        x.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Phantom CFD inlet flow: native vs as simulated in Koma", fontsize=12*FS)
    fig.supxlabel(f"Koma replays the {T_CFD:.2f} s CFD particle paths over the scan RR {RR:.3f} s, so all velocities and flows are x {s:.3f} "
                  "(same paths, slower).", fontsize=8*FS)
    p = stage/"cfd_inlet_waveform.png"; fig.savefig(p, dpi=110); plt.close(fig)
    rep["status"] = "READ_ONLY_CFD_AUDIT"
    (stage/"report.json").write_text(json.dumps(rep, indent=1)+chr(10), encoding="utf-8")
    (stage/"MANIFEST.json").write_text(json.dumps({q.name: sha(q) for q in sorted(stage.iterdir())}, indent=1)+chr(10), encoding="utf-8")
    stage.rename(target)
    print(json.dumps({k: rep[k] for k in ("native", "koma_retimed", "periodic_cfd_files_identical_to_E060_registration")}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
