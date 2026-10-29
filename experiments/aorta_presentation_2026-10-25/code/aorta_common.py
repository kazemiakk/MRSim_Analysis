"""Shared aorta data and geometry for the presentation analyses (secondary flow, energy, Bland-Altman).

Same inputs and definitions as plot_aorta_20bin.py v21/v22: E-062 20-bin reference and Koma, the native
CFD point cloud (scaled by 0.72/RR), planes A/D perpendicular to the cycle-mean flow, and the v22 display
noise realisation (SNR 12 as for VENC 0.3 m/s, seed 20261025). Read-only; nothing here is registered.
"""
import json
from pathlib import Path
import sys

import h5py
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
from plot_aorta_highres import CLOUD, E027, PC, PLANE, R_IN, SEED, idw, sha  # noqa: E402
from scripts.aorta_hemodynamic_reference import AortaGrid  # noqa: E402  (frozen, read-only)

ROOT = Path("D:/MRSim/outputs/aorta_presentation_2026-10-25")
E062 = Path("D:/MRSim/outputs/aorta_20bin_e062")
REC20 = E062/"combined/simulation_motion_1m_20bin_4mm/pc_reconstruction.h5"
SECTIONS = dict(asc="X+058_C1", desc="X+000_C1")
SNR, VENC_NOISE = 12.0, 0.3


class Aorta:
    def __init__(self):
        self.ana = json.loads((E062/"analysis_v1/report.json").read_text())
        man = json.loads((E062/"analysis_v1/MANIFEST.json").read_text())
        if sha(E062/"analysis_v1/matched_reference_20bin.h5") != man["matched_reference_20bin.h5"]:
            raise ValueError("E-062 matched reference changed")
        self.audit = json.loads((E062/"sequence_audit_v1/report.json").read_text())
        self.rep027 = json.loads((E027/"report.json").read_text())
        want = {Path(r["path"]).name: r["sha256"] for r in self.rep027["inputs"]}
        for name in ("point_coordinates.npy", "velocities.npy"):
            if sha(CLOUD/name) != want[name]:
                raise ValueError("CFD point cloud changed: "+name)
        with h5py.File(E062/"analysis_v1/matched_reference_20bin.h5") as f:
            self.ref, self.koma, self.frac = f["adc_time_average"][:], f["Koma"][:], f["fraction"][:]
            self.core = f["masks/core15"][:].astype(bool); self.fw = f["frame_weights"][:]
            self.secs = {k: f["sections"][k][:].astype(bool) for k in f["sections"]}
            self.pfrac = f["plane_fraction"][:]
        with h5py.File(E027/"matched_reference.h5") as f:
            self.koma8, self.frac8, self.ref8 = f["Koma"][:], f["fraction"][:], f["adc_time_average"][:]
            self.core8 = f["masks/core15"][:].astype(bool)
        with h5py.File(PC) as f:
            self.mag8 = f["reference_magnitude"][:]
        with h5py.File(REC20) as f:
            self.mag = f["reference_magnitude"][:]
            self.vel = np.moveaxis(f["velocity"][:], 0, -1)
        if np.nanmax(np.abs(self.vel[:, self.core]-self.koma[:, self.core])) > 1e-4:
            raise ValueError("reconstruction does not match the E-062 analysis field")
        self.nb = self.mag.shape[0]
        self.rr, self.vscale = float(self.ana["period_s"]), float(self.ana["source_velocity_time_scale"])
        self.grid = AortaGrid()
        self.order, self.cen = np.array(self.grid.order), np.array(self.grid.center)
        self.gx, self.gy, self.gz = self.grid.axes()
        self.spacing = self.grid.spacing                                        # (dx, dy, dz) m
        self.tc20 = np.array([b["phase_circular_mean_s"] for b in self.audit["encodings"]["REF"]["bin_timing"]])
        self.tc8 = np.array([b["phase_circular_mean_s"] for b in self.rep027["timing_audit"]["REF"]["bin_timing"]])
        self.t0 = float(json.loads((ROOT/"figure_v22/MANIFEST.json").read_text())["cycle_start_phase_s"])
        # v22 display-noise realisation (same generator, call order and shapes).
        self.sigma = float(np.median(self.mag.mean(axis=0)[self.core]))/SNR
        rng = np.random.default_rng(SEED)
        def noisy(z):
            return z+self.sigma*(rng.standard_normal(z.shape)+1j*rng.standard_normal(z.shape))
        self.REFall = noisy(self.mag.astype(complex))
        self.vnall = np.stack([np.angle(noisy(self.mag*np.exp(1j*self.vel[..., c]*np.pi/VENC_NOISE))*self.REFall.conj())*VENC_NOISE/np.pi
                               for c in range(3)], axis=-1)
        src = np.load(CLOUD/"point_coordinates.npy", allow_pickle=False).astype(np.float64)
        self.vel_src = np.load(CLOUD/"velocities.npy", mmap_mode="r", allow_pickle=False)
        self.nodes = src[:, self.order]-self.cen[self.order]
        self.tree = cKDTree(self.nodes)

    def tau_ms(self, t):
        return ((np.asarray(t)-self.t0) % self.rr)*1000

    def native(self, pts):
        """Native CFD at points: IDW of the scaled source velocities, MRI components, (points, 3, 72)."""
        _, nq, wq = idw(self.tree, pts)
        used = np.unique(nq)
        vp = np.asarray(self.vel_src[used], dtype=np.float32)[:, self.order, :]*self.vscale
        return np.einsum("qk,qkcf->qcf", wq, vp[np.searchsorted(used, nq)])

    def native_bins(self, pts):
        """Native CFD averaged over each Koma bin's ADC window: (bins, points, 3)."""
        return np.einsum("qcf,bf->bqc", self.native(pts), self.fw)

    def interp4(self, field, w, pts):
        """4 mm field at points, weight-normalised trilinear (v21: weight = voxel lumen fraction)."""
        kk = [(pts[:, a]-g[0])/(g[1]-g[0]) for a, g in enumerate((self.gx, self.gy, self.gz))]
        out = np.zeros((len(field), len(pts), 3))
        for k in range(len(field)):
            wk = (w[k] if w.ndim == 4 else w).astype(float)
            den = ndimage.map_coordinates(wk, [kk[2], kk[1], kk[0]], order=1, mode="constant", cval=0.0)
            for c in range(3):
                num = ndimage.map_coordinates(np.nan_to_num(field[k][..., c])*wk, [kk[2], kk[1], kk[0]], order=1, mode="constant", cval=0.0)
                out[k, :, c] = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-9*wk.max())
        return out

    def _lumen_on_plane(self, c0, n, half=0.035):
        e1 = np.array([0.0, 1.0, 0.0])-n[1]*n; e1 /= np.linalg.norm(e1); e2 = np.cross(n, e1)
        s = np.arange(-half, half+1e-9, PLANE)
        SS, TT = np.meshgrid(s, s, indexing="ij")
        pts = c0+SS.ravel()[:, None]*e1+TT.ravel()[:, None]*e2
        d1, _ = self.tree.query(pts, k=1, workers=-1)
        lab, _ = ndimage.label((d1 <= R_IN).reshape(SS.shape))
        ci = lab[SS.shape[0]//2, SS.shape[1]//2]
        if ci == 0:
            raise ValueError("plane centre is outside the lumen")
        m = lab == ci
        if m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any():
            raise ValueError("lumen touches the plane sampling border")
        return pts[m.ravel()], e1, e2, np.argwhere(m), SS.shape

    def plane(self, key):
        """v21 construction: X-plane section -> plane perpendicular to the cycle-mean flow (4 re-centrings)."""
        name = SECTIONS[key]; sec = self.secs[name]
        xsec = float([r for r in self.ana["primary_20bin"]["section_series"] if r["section"] == name][0]["x_m"])
        zz, yy = np.where(sec); gy, gz = self.gy, self.gz
        py = np.arange(gy[yy].min()-.006, gy[yy].max()+.006, PLANE); pz = np.arange(gz[zz].min()-.006, gz[zz].max()+.006, PLANE)
        YY, ZZ = np.meshgrid(py, pz, indexing="ij")
        P0 = np.stack([np.full(YY.size, xsec), YY.ravel(), ZZ.ravel()], -1)
        d1, _ = self.tree.query(P0, k=1, workers=-1)
        lab2, n2 = ndimage.label((d1 <= R_IN).reshape(YY.shape))
        jy = np.clip(np.round((YY-gy[0])/(gy[1]-gy[0])).astype(int), 0, len(gy)-1)
        jz = np.clip(np.round((ZZ-gz[0])/(gz[1]-gz[0])).astype(int), 0, len(gz)-1)
        hit = np.bincount(lab2[sec[jz, jy] & (lab2 > 0)], minlength=n2+1); hit[0] = 0
        xpts = P0[(lab2 == int(np.argmax(hit))).ravel()]
        vx = self.native(xpts)
        c0, vm = xpts.mean(axis=0), vx.mean(axis=0).mean(axis=1)
        n = vm/np.linalg.norm(vm)
        for _ in range(4):
            pts, e1, e2, ij, shp = self._lumen_on_plane(c0, n)
            vm = self.native(pts).mean(axis=0).mean(axis=1)
            c0, n = pts.mean(axis=0), vm/np.linalg.norm(vm)
        pts, e1, e2, ij, shp = self._lumen_on_plane(c0, n)
        return dict(name=name, pts=pts, n=n, e1=e1, e2=e2, c0=c0, ij=ij, shape=shp, basis=np.stack([n, e1, e2]))
