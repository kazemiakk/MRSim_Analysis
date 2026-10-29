"""E-060 S0b probe: independent RK4 slab profiles (E056 rf_affine) for probe excitations."""
from pathlib import Path
import sys

import h5py
import numpy as np

sys.path.insert(0, "D:/MRSim/outputs/acquisition_physics_e056/code")
import analyze_physics as reference  # noqa: E402

folder = Path(sys.argv[1])
z = np.linspace(-.15, .15, 1201)
for tbp in (4, 6, 8, 10, 12, 16):
    with h5py.File(folder/f"tbp{tbp}.h5") as f:
        grad = np.asarray(f["g_T_m"][()], float)
        grad = grad.T if grad.shape[0] == 3 and grad.shape[1] != 3 else grad
        template = dict(t=np.asarray(f["t_s"][()], float), g=grad, b1=reference.complex_array(f["b1_T"]),
                        frequency=np.asarray(f["rf_frequency_Hz"][()], float))
    op = reference.rf_affine(template, z, 1.0, 1.3, .1, 42.577478518e6)
    p = np.abs(op["axy"]+op["bxy"])
    p = p/p[np.argmin(np.abs(z))]
    nz = np.flatnonzero(np.abs(template["b1"]) > 0)
    at = lambda zz: float(np.interp(zz, z, p))
    print(f"TBP{tbp:2d} rf_us={1e6*(template['t'][nz[-1]]-template['t'][nz[0]]):6.0f} "
          f"beyond37.5={p[np.abs(z) >= .0375].max():.4f} beyond45={p[np.abs(z) >= .045].max():.4f} "
          f"at25={at(.025):.3f} at30={at(.03):.3f} at35={at(.035):.3f} min|z|<=25={p[np.abs(z) <= .025].min():.3f}")
