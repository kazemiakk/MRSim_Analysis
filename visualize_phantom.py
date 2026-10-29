import sys
sys.path.append(r"C:\Users\uoflm\Desktop\lambda 4 code\codes\MRSimTracks")
import pickle
import h5py
import numpy as np
import pyvista as pv

def load_tracks(path, time_stride=1):
    with h5py.File(path, "r") as h5:
        position_data = h5["position"]
        reset_data = h5["reset"]
        n_steps = position_data.shape[0]
        n_windows = n_steps // time_stride
        stop = n_windows * time_stride
        time_indices = np.arange(time_stride - 1, stop, time_stride, dtype=np.int64)
        
        positions = position_data[time_indices, :, :]
        reset_window = reset_data[:stop, :]
        reset = reset_window.reshape(n_windows, time_stride, reset_window.shape[1]).any(axis=1)
        dt = float(h5.attrs["dt"]) * time_stride
        return positions, reset, dt

def trajectory_segments(positions, reset):
    n_steps, n_particles, _ = positions.shape
    if n_steps < 2: return None

    points = positions.reshape(-1, 3)
    lines = []
    for step in range(1, n_steps):
        previous_offset = (step - 1) * n_particles
        current_offset = step * n_particles
        for particle in np.flatnonzero(~reset[step]):
            lines.extend([2, previous_offset + int(particle), current_offset + int(particle)])

    if not lines: return None
    tracks = pv.PolyData(points)
    tracks.lines = np.asarray(lines, dtype=np.int64)
    return tracks

def main():
    dp_name = "dp150"
    geometry_path = rf"C:\Users\uoflm\Desktop\lambda 4 code\codes\PI-GNN(SR)\data\phantom\processed\{dp_name}\geometry.pkl"
    tracks_path = f"phantom_{dp_name}_tracks_10s.h5"
    
    print("Loading point cloud geometry and tracks...")
    with open(geometry_path, "rb") as f:
        data = pickle.load(f)
    coords = data["mesh_coords"]
    
    # Reconstruct a solid surface mesh from the point cloud
    max_points = 500_000
    stride = max(1, len(coords) // max_points)
    vessel_pts = pv.PolyData(coords[::stride])
    print("Reconstructing solid surface mesh... (this takes ~5-10 seconds)")
    vessel = vessel_pts.delaunay_3d(alpha=0.005).extract_surface()
    
    positions, reset, dt = load_tracks(tracks_path)
    
    speeds = np.zeros(positions.shape[:2], dtype=float)
    if len(positions) > 1:
        speeds[1:] = np.linalg.norm(np.diff(positions, axis=0), axis=2) / dt
        speeds[reset] = np.nan
        
    finite_speeds = speeds[np.isfinite(speeds)]
    speed_max = float(np.percentile(finite_speeds, 99)) if finite_speeds.size else 1.0

    final_step = len(positions) - 1
    particles = pv.PolyData(positions[final_step])
    particles["speed"] = speeds[final_step]

    plotter = pv.Plotter(window_size=(1400, 900))
    plotter.set_background("white")
    plotter.enable_anti_aliasing("ssaa")  # High resolution anti-aliasing
    
    plotter.add_mesh(vessel, color="silver", opacity=0.2, specular=0.5, show_edges=False, label="Phantom Wall")
    
    tracks = trajectory_segments(positions, reset)
    if tracks is not None:
        plotter.add_mesh(tracks, color="#1f77b4", opacity=0.8, line_width=2, label="Particle paths")
        
    sphere_geom = pv.Sphere(radius=0.0006, theta_resolution=16, phi_resolution=16)
    initial_glyphs = particles.glyph(geom=sphere_geom, scale=False, orient=False)
    
    plotter.add_mesh(initial_glyphs, scalars="speed", cmap="turbo", clim=(0.0, speed_max),
                     name="particles", label="Particles",
                     scalar_bar_args={"title": "Particle speed", "color": "black", "vertical": True, "position_x": 0.05, "position_y": 0.27})

    time_label = plotter.add_text(f"Step {final_step + 1}/{len(positions)}   t={(final_step + 1) * dt:.4f} s", position="upper_left", font_size=12, color="black")

    def set_step(value):
        step = int(np.clip(round(value), 0, len(positions) - 1))
        particles.points = positions[step]
        particles["speed"] = speeds[step]
        new_glyphs = particles.glyph(geom=sphere_geom, scale=False, orient=False)
        plotter.add_mesh(new_glyphs, scalars="speed", cmap="turbo", clim=(0.0, speed_max), name="particles", show_scalar_bar=False)
        time_label.SetText(2, f"Step {step + 1}/{len(positions)}   t={(step + 1) * dt:.4f} s")
        plotter.render()

    if len(positions) > 1:
        plotter.add_slider_widget(set_step, rng=(0, len(positions) - 1), value=final_step, title="Timestep",
                                  fmt="%.0f", pointa=(0.30, 0.08), pointb=(0.70, 0.08), color="#1f77b4", title_color="black", interaction_event="always")

    plotter.add_legend(bcolor="white", border=True)
    plotter.add_axes()
    plotter.view_zx()  # Set Z as horizontal axis and X as vertical axis
    plotter.enable_parallel_projection()  # Orthographic projection for scientific look
    plotter.reset_camera()
    
    # Pan camera to the left (negative Z direction) to center on the stenosis and inlet
    fp = plotter.camera.focal_point
    pos = plotter.camera.position
    shift_z = -0.127  # Shift center from 0.127 to 0.0
    plotter.camera.focal_point = (fp[0], fp[1], fp[2] + shift_z)
    plotter.camera.position = (pos[0], pos[1], pos[2] + shift_z)
    
    plotter.camera.zoom(2.5)  # Zoom in to focus more on the stenosis in the center
    print("Close the PyVista window to exit.")
    plotter.show()

if __name__ == "__main__":
    main()
