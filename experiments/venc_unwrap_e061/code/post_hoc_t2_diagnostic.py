import sys, json, tomllib
from pathlib import Path
import numpy as np, h5py
E060 = Path("D:/MRSim/outputs/scanner_geometry_e060"); E061 = Path("D:/MRSim/outputs/venc_unwrap_e061")
sys.path.insert(0, str(E061/"code")); sys.path.insert(0, str(E060/"code"))
import analyze_e061 as a61
an = a61.an
control = tomllib.loads((E060/"acquisition_v1/control_report.toml").read_text())
images, _ = an.load_images(E060/"acquisition_v1", control, E060/"sequences_v2/timing_e060.h5")
z4, _ = a61.load_z4()
with h5py.File(E060/"references_v2/references.h5") as f:
    fft_adc, frac, z = f["fft_adc"][:], f["fraction_fft"][:], f["partition_z_m"][:]
zmm = z*1e3
b = 7
thr = (frac > 0) & ((zmm >= -10) & (zmm < 10))[:, None, None]
ref = fft_adc[b][..., 2]
lum_med = np.median(np.abs(images["REF"][b])[frac > 0])
for name, zim, venc in (("VENC3", images["Z"], 3.0), ("VENC4", z4, 4.0)):
    phi = np.angle(zim[b]*images["REF"][b].conj()); v = phi*venc/np.pi
    sel = np.argsort(np.where(thr, v, -9).ravel())[::-1][:8]
    print("==", name)
    for i in sel:
        k = np.unravel_index(i, v.shape)
        print(k, "v %.2f ref %.2f |phi|/pi %.2f |REF|/med %.2f |Z|/med %.2f frac %.2f" % (v[k], ref[k], abs(phi[k])/np.pi, abs(images["REF"][b][k])/lum_med, abs(zim[b][k])/lum_med, frac[k]))
    big = thr & (np.abs(phi) > .85*np.pi)
    print("throat voxels |phi|>0.85pi:", int(big.sum()), "of", int(thr.sum()), "median |REF|/med there", float(np.median(np.abs(images["REF"][b][big])/lum_med)) if big.any() else None)
print("== post hoc magnitude-filtered peak (diagnostic only)")
cfd_max = float(ref[thr].max()); cfd_p99 = float(np.percentile(ref[thr], 99))
for name, zim, venc in (("VENC3", images["Z"], 3.0), ("VENC4", z4, 4.0)):
    v = np.angle(zim[b]*images["REF"][b].conj())*venc/np.pi
    for cut in (0.0, 0.3, 0.5):
        ok = thr & (np.abs(zim[b])/lum_med >= cut) & (np.abs(images["REF"][b])/lum_med >= cut)
        print(name, "cut", cut, "voxels", int(ok.sum()), "max err %+.3f p99 err %+.3f" % (v[ok].max()/cfd_max-1, np.percentile(v[ok], 99)/cfd_p99-1),
              "core |v-ref| median %.3f" % float(np.median(np.abs(v-ref)[ok & (ref > 2.0)])))
