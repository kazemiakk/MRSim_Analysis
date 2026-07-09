import sys
sys.path.append(r"C:\Users\uoflm\Desktop\lambda 4 code\codes\MRSimTracks")
import pickle
import numpy as np
import scipy.spatial
import mrsimtracks as mt
from mrsimtracks.io import _interp_time

class PointCloudFlow:
    """Adapter for MRSimTracks to track particles over a point cloud."""
    def __init__(self, geometry_path, velocity_path, dt_cycle=1.0, oob_threshold=0.015):
        print("Loading point cloud geometry...")
        with open(geometry_path, "rb") as f:
            data = pickle.load(f)
        self.coords = data["mesh_coords"]
        
        print("Loading velocities...")
        self.velocities = np.load(velocity_path)  # (N, 3, 72)
        
        self.n_frames = self.velocities.shape[2]
        self.dt = dt_cycle / self.n_frames
        self.times = np.arange(self.n_frames) * self.dt
        self.times_shift_s = self.times.copy()
        self.tmax = dt_cycle
        
        print("Building spatial KDTree...")
        self.kdtree = scipy.spatial.cKDTree(self.coords)
        self.oob_threshold = oob_threshold
        
    def _frame_vel(self, index):
        return self.velocities[:, :, index]
        
    def sample_v(self, points_xyz, time, guess=None):
        # 1. Temporal interpolation (linear blend of all nodes at current time)
        # MRSimTracks provides _interp_time for linear/cubic periodic blending.
        vel_t = _interp_time(
            self.times_shift_s, self.tmax, self.n_frames, 
            self._frame_vel, time, mode="linear"
        )
        
        # 2. Spatial interpolation (IDW from k nearest neighbors)
        k = 4
        dists, indices = self.kdtree.query(points_xyz, k=k)
        
        # Avoid division by zero
        dists = np.maximum(dists, 1e-8)
        weights = 1.0 / dists
        weights_sum = weights.sum(axis=1, keepdims=True)
        
        # Node velocities for the neighbors
        v_neighbors = vel_t[indices]  # (n_particles, k, 3)
        weights_expanded = np.expand_dims(weights, axis=2) # (n_particles, k, 1)
        
        # IDW weighted average
        interpolated_v = (v_neighbors * weights_expanded).sum(axis=1) / weights_sum
        
        # 3. Validity mask: if a particle is too far from any point, mark invalid
        min_dists = dists[:, 0]
        valid = min_dists < self.oob_threshold
        
        # Return signature expected by tracker: (velocities, validity_mask, cells (dummy))
        return interpolated_v, valid, np.zeros(len(points_xyz), dtype=np.int32)

def main():
    dp_name = "dp150"
    base_dir = rf"C:\Users\uoflm\Desktop\lambda 4 code\codes\PI-GNN(SR)\data\phantom\processed\{dp_name}"
    geometry_path = rf"{base_dir}\geometry.pkl"
    velocity_path = rf"{base_dir}\velocities.npy"
    
    flow = PointCloudFlow(geometry_path, velocity_path, dt_cycle=1.0, oob_threshold=0.005)
    
    print("\nSeeding particles...")
    # Since we don't have a volume mesh to randomly sample inside, we will pick 
    # random points from the point cloud itself as our starting seeds.
    n_particles = 1000
    rng = np.random.default_rng(42)
    seed_indices = rng.choice(len(flow.coords), size=n_particles, replace=False)
    seeds = flow.coords[seed_indices]
    
    print("Tracking particles over 10 cardiac cycles (10 seconds)...")
    result = mt.track(flow, seeds=seeds, inlet=seeds, dt=0.001, tmax=10.0, pbar=True)
    
    print(f"\npositions {result.positions.shape}  (n_steps, n_particles, 3)")
    print(f"total resets: {int(result.reset.sum())}")
    
    out_file = f"phantom_{dp_name}_tracks_10s.h5"
    result.save(out_file)
    print(f"Saved {out_file}")

if __name__ == "__main__":
    main()
