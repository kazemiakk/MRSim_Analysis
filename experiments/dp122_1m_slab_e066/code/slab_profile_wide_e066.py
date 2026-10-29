"""E-066 diagnostic: E-060 slab_profile_v2 method (independent RK4 finite-RF Bloch, E056 rf_affine) over |S| <= 500 mm
on the E-065 REF first-shot excitation, to look for distant excitation (replica bands) after gate G1 failed."""
import json, sys
from pathlib import Path
import h5py, numpy as np
E056 = Path("D:/MRSim/outputs/acquisition_physics_e056")
sys.path.insert(0, str(E056/"code"))
import analyze_physics as reference  # noqa: E402
out = Path(sys.argv[1]); assert not out.exists()
with h5py.File("D:/MRSim/outputs/dp122_native_e065/sequences_v1/timing_e060.h5") as f:
    g = f["excitation_REF_first_shot"]; grad = np.asarray(g["g_T_m"][()], float)
    grad = grad.T if grad.shape[1] != 3 and grad.shape[0] == 3 else grad
    tpl = dict(t=np.asarray(g["t_s"][()], float), g=grad, b1=reference.complex_array(g["b1_T"]), frequency=np.asarray(g["rf_frequency_Hz"][()], float))
z = np.linspace(-.5, .5, 8001)
op = reference.rf_affine(tpl, z, 1.0, 1.3, .1, 42.577478518e6)
mxy = np.abs(op["axy"]+op["bxy"]); prof = mxy/mxy[np.argmin(np.abs(z))]
dt = np.diff(tpl["t"]); gz = np.abs(tpl["g"][:, 2]).max()
rep = dict(max_beyond_37p5mm=float(prof[np.abs(z) >= .0375].max()), max_beyond_60mm=float(prof[np.abs(z) >= .06].max()),
           max_beyond_100mm=float(prof[np.abs(z) >= .1].max()), max_beyond_150mm=float(prof[np.abs(z) >= .15].max()),
           mean_60_to_400mm=float(prof[(np.abs(z) >= .06) & (np.abs(z) <= .4)].mean()),
           top_bands_mm=[[float(z[i]*1e3), float(prof[i])] for i in np.argsort(-np.where(np.abs(z) >= .06, prof, 0))[:8]],
           waveform_dt_s=[float(dt.min()), float(np.median(dt))], slab_gradient_max_T_m=float(gz))
out.mkdir(); (out/"report.json").write_text(json.dumps(rep, indent=1))
np.savetxt(out/"profile.csv", np.column_stack([z*1e3, prof]), delimiter=",", header="z_mm,relative_Mxy", comments="")
print(json.dumps(rep, indent=1))
