import sys
sys.path.append(r"C:\Users\uoflm\Desktop\lambda 4 code\codes\MRSimTracks")
import pickle
import h5py
import numpy as np
import pyvista as pv
from visualize_phantom import load_tracks, trajectory_segments

def main():
    dp_name = "dp150"
    geometry_path = rf"C:\Users\uoflm\Desktop\lambda 4 code\codes\PI-GNN(SR)\data\phantom\processed\{dp_name}\geometry.pkl"
    
    # We will animate the 10-second high-resolution tracks
    tracks_path = f"phantom_{dp_name}_tracks_10s.h5"
    out_gif = f"phantom_{dp_name}_animation.gif"
    
    print("Loading point cloud geometry and tracks...")
    with open(geometry_path, "rb") as f:
        data = pickle.load(f)
    coords = data["mesh_coords"]
    
    # Subsample point cloud for performance
    max_points = 100_000
    stride_points = max(1, len(coords) // max_points)
    vessel = pv.PolyData(coords[::stride_points])
    
    positions, reset, dt = load_tracks(tracks_path)
    
    speeds = np.zeros(positions.shape[:2], dtype=float)
    if len(positions) > 1:
        speeds[1:] = np.linalg.norm(np.diff(positions, axis=0), axis=2) / dt
        speeds[reset] = np.nan
        
    finite_speeds = speeds[np.isfinite(speeds)]
    speed_max = float(np.percentile(finite_speeds, 99)) if finite_speeds.size else 1.0

    # Start at step 0
    particles = pv.PolyData(positions[0])
    particles["speed"] = speeds[0]

    # Use off_screen=True to render without popping up a window
    plotter = pv.Plotter(window_size=(1400, 900), off_screen=True)
    plotter.set_background("#10151c", top="#263342")
    
    plotter.add_mesh(vessel, color="#d6dde5", opacity=0.13, point_size=2, render_points_as_spheres=False)
    
    tracks = trajectory_segments(positions, reset)
    if tracks is not None:
        plotter.add_mesh(tracks, color="#4db6ff", opacity=0.55, line_width=2)
        
    plotter.add_mesh(particles, scalars="speed", cmap="turbo", clim=(0.0, speed_max),
                     point_size=5, render_points_as_spheres=True,
                     scalar_bar_args={"title": "Particle speed", "color": "white", "vertical": True, "position_x": 0.05, "position_y": 0.27})

    time_label = plotter.add_text(f"Step 1/{len(positions)}   t={dt:.4f} s", position="upper_left", font_size=12, color="white")

    plotter.view_isometric()
    plotter.reset_camera()
    plotter.camera.roll -= 90  # Rotate scene 90 degrees clockwise

    # Since the 10-second simulation has 10,000 frames, saving them all would create a massive GIF!
    # We will skip every 20 frames to create a smooth ~16-second GIF with 500 frames.
    frame_stride = 20
    frames_to_render = range(0, len(positions), frame_stride)
    
    print(f"\nRendering {len(frames_to_render)} frames to {out_gif}...")
    plotter.open_gif(out_gif)

    for i, step in enumerate(frames_to_render):
        # Update particle positions and colors
        particles.points = positions[step]
        particles["speed"] = speeds[step]
        
        # Update the text label
        time_label.SetText(2, f"Step {step + 1}/{len(positions)}   t={(step + 1) * dt:.4f} s")
        
        # Write the frame to the GIF
        plotter.write_frame()
        
        if i % 50 == 0:
            print(f"Rendered frame {i}/{len(frames_to_render)}")

    plotter.close()
    print(f"\nAnimation successfully saved to: {out_gif}")

if __name__ == "__main__":
    main()
