import numpy as np
import pyvista as pv
from vtkmodules.vtkCommonDataModel import vtkStaticCellLocator

import mrsimtracks as mt
from mrsimtracks.sampler import _TetSampler, _condition_mesh
from mrsimtracks.seeding import seed_mesh

def extract_steady_caps(mesh_path, out_path, active_key, vmag_thresh=0.01, min_faces=20):
    """Extract inlet/outlet caps from a steady-state velocity field."""
    full = pv.read(mesh_path)
    surf = full.extract_surface(algorithm="dataset_surface").triangulate()
    orig = surf.point_data["vtkOriginalPointIds"]
    v = full.point_data[active_key][orig]
    vmax = np.linalg.norm(v, axis=1)
    
    faces = surf.faces.reshape(-1, 4)[:, 1:]
    cap_node = vmax > vmag_thresh
    cap_face = cap_node[faces].any(axis=1)
    print(f"{surf.n_cells} boundary faces -> {cap_face.sum()} cap faces")
    
    caps = surf.extract_cells(np.where(cap_face)[0]).extract_surface(algorithm="dataset_surface")
    caps = caps.connectivity("all")
    region = np.asarray(caps.cell_data["RegionId"])
    
    keep_ids, counts = np.unique(region, return_counts=True)
    keep_ids = keep_ids[counts >= min_faces]
    mask = np.isin(region, keep_ids)
    caps = caps.extract_cells(np.where(mask)[0]).extract_surface(algorithm="dataset_surface")
    
    region = np.asarray(caps.cell_data["RegionId"])
    _, region = np.unique(region, return_inverse=True)
    caps.cell_data["region_id"] = region.astype(np.int32)
    if "RegionId" in caps.cell_data:
        del caps.cell_data["RegionId"]
        
    caps.save(out_path)
    print(f"Saved {out_path} with {region.max()+1} caps.")
    return caps


class SteadyFlow:
    """Adapter for MRSimTracks to use a single steady-state velocity field."""
    def __init__(self, mesh_path, active_key):
        self.mesh = pv.read(mesh_path)
        self.active_key = active_key
        
        arr = self.mesh.point_data[active_key]
        if arr.dtype != np.float64:
            self.mesh.point_data[active_key] = arr.astype(np.float64)
            
        arr = np.asarray(self.mesh.point_data[active_key])
        self.active_mesh = _condition_mesh(self.mesh)
        self.times = np.array([0.0, 100.0])
        self.times_shift_s = np.array([0.0, 100.0])
        self.fields = [arr, arr]
        self.tmax = 100.0  # arbitrary large number since it's steady
        self.locator = vtkStaticCellLocator()
        self.locator.SetDataSet(self.active_mesh)
        self.locator.BuildLocator()
        self._sampler = _TetSampler(self.active_mesh, dtype=np.float64)

    def set_active_time(self, time):
        pass

    def get_mesh(self, time):
        return self.active_mesh

    def sample_v(self, points_xyz, time, guess=None):
        vel = np.asarray(self.active_mesh.point_data[self.active_key])
        return self._sampler.sample(points_xyz, vel, guess=guess)


def main():
    mesh_path = r"d:\aneurysm\From Laurel MD patient\data_acom_DM\data\aneurysm_mesh.vtu"
    caps_path = "aneurysm_caps.vtp"
    active_key = "Velocity_High"
    
    print("1. Extracting caps...")
    # Use a small threshold to find boundaries where velocity is not identically zero (i.e. not no-slip wall)
    extract_steady_caps(mesh_path, caps_path, active_key, vmag_thresh=0.01)
    
    print("\n2. Loading steady flow field...")
    flow = SteadyFlow(mesh_path, active_key)
    
    print("\n3. Setting up reseeder and tracking...")
    # Setting dt to None or a small value for reseeder. Since it's steady, a time step dt=0.002 is good.
    # The reseeder spreads particles at boundaries.
    reseeder = mt.BoundaryReseeder(caps_path, flow, dt=0.002)
    
    # Seed 1000 particles across the mesh volume
    seeds = seed_mesh(flow.active_mesh, 1_000, rng=np.random.default_rng(42))
    
    # Track for a duration of tmax=0.5 simulated seconds with dt=0.002
    result = mt.track(flow, seeds=seeds, dt=0.002, tmax=0.5, reseeder=reseeder, pbar=True)
    
    print(f"\npositions {result.positions.shape}  (n_steps, n_particles, 3)")
    print(f"total resets: {int(result.reset.sum())}")
    
    out_file = "aneurysm_tracks_high.h5"
    result.save(out_file)
    print(f"saved {out_file}")

if __name__ == "__main__":
    main()
