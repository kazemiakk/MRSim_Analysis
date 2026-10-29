import sys
sys.path.append(r"C:\Users\uoflm\Desktop\lambda 4 code\codes\MRSimTracks")
import pickle
import h5py
import numpy as np
import pyvista as pv
from visualize_phantom import load_tracks, trajectory_segments
from PIL import Image

def save_animation(
    path,
    frames,
    duration_ms,
    scale=1.0,
    webp_quality=80,
):
    if scale != 1.0:
        resized = []
        for frame in frames:
            width = max(1, int(round(frame.width * scale)))
            height = max(1, int(round(frame.height * scale)))
            resized.append(frame.resize((width, height), Image.Resampling.LANCZOS))
        frames = resized
        
    frames = [frame.convert("RGB") for frame in frames]
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        format="WEBP",
        quality=webp_quality,
        method=6,
        kmin=1,
        kmax=1,
    )

def main():
    dp_name = "dp150"
    geometry_path = rf"C:\Users\uoflm\Desktop\lambda 4 code\codes\PI-GNN(SR)\data\phantom\processed\{dp_name}\geometry.pkl"
    
    # We will animate the 10-second high-resolution tracks
    tracks_path = f"phantom_{dp_name}_tracks_10s.h5"
    out_webp = f"phantom_{dp_name}_animation.webp"
    
    print("Loading point cloud geometry and tracks...")
    with open(geometry_path, "rb") as f:
        data = pickle.load(f)
    coords = data["mesh_coords"]
    
    # Reconstruct a solid surface mesh from the point cloud
    max_points = 500_000
    stride_points = max(1, len(coords) // max_points)
    vessel_pts = pv.PolyData(coords[::stride_points])
    print("Reconstructing solid surface mesh... (this takes ~5-10 seconds)")
    vessel = vessel_pts.delaunay_3d(alpha=0.005).extract_surface()
    
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
    plotter.set_background("white")
    plotter.enable_anti_aliasing("ssaa")
    
    plotter.add_mesh(vessel, color="silver", opacity=0.2, specular=0.5, show_edges=False)
    
    tracks = trajectory_segments(positions, reset)
    if tracks is not None:
        plotter.add_mesh(tracks, color="#1f77b4", opacity=0.8, line_width=2)
        
    sphere_geom = pv.Sphere(radius=0.0006, theta_resolution=16, phi_resolution=16)
    initial_glyphs = particles.glyph(geom=sphere_geom, scale=False, orient=False)
    
    plotter.add_mesh(initial_glyphs, scalars="speed", cmap="turbo", clim=(0.0, speed_max),
                     name="particles",
                     scalar_bar_args={"title": "Particle speed", "color": "black", "vertical": True, "position_x": 0.05, "position_y": 0.27})

    time_label = plotter.add_text(f"Step 1/{len(positions)}   t={dt:.4f} s", position="upper_left", font_size=12, color="black")

    plotter.view_zx()
    plotter.enable_parallel_projection()  # Orthographic projection for scientific look
    plotter.reset_camera()
    
    # Pan camera to the left (negative Z direction) to center on the stenosis and inlet
    fp = plotter.camera.focal_point
    pos = plotter.camera.position
    shift_z = -0.127  # Shift center from 0.127 to 0.0
    plotter.camera.focal_point = (fp[0], fp[1], fp[2] + shift_z)
    plotter.camera.position = (pos[0], pos[1], pos[2] + shift_z)
    
    plotter.camera.zoom(2.5)

    # Since the 10-second simulation has 10,000 frames, saving them all would create a massive file!
    # We will skip every 20 frames to create a smooth ~16-second animation with 500 frames.
    frame_stride = 20
    frames_to_render = range(0, len(positions), frame_stride)
    
    print(f"\nRendering {len(frames_to_render)} frames to WebP...")
    
    captured_frames = []
    # Calculate a fixed mask for cropping over all frames to avoid jitter
    global_mask = None
    pad = 18
    threshold = 248

    for i, step in enumerate(frames_to_render):
        # Update particle positions and colors
        particles.points = positions[step]
        particles["speed"] = speeds[step]
        new_glyphs = particles.glyph(geom=sphere_geom, scale=False, orient=False)
        plotter.add_mesh(new_glyphs, scalars="speed", cmap="turbo", clim=(0.0, speed_max), name="particles", show_scalar_bar=False)
        
        # Update the text label
        time_label.SetText(2, f"Step {step + 1}/{len(positions)}   t={(step + 1) * dt:.4f} s")
        
        # Get screenshot
        img_arr = plotter.screenshot(transparent_background=False, return_img=True)
        img = Image.fromarray(img_arr).convert("RGB")
        captured_frames.append(img)
        
        # Build cropping mask
        arr = np.asarray(img)
        current = np.any(arr < threshold, axis=2)
        global_mask = current if global_mask is None else global_mask | current
        
        if i % 50 == 0:
            print(f"Rendered frame {i}/{len(frames_to_render)}")

    plotter.close()
    
    print("Applying auto-crop and saving WebP...")
    if global_mask is not None and global_mask.any():
        ys, xs = np.where(global_mask)
        box = (
            max(int(xs.min()) - pad, 0),
            max(int(ys.min()) - pad, 0),
            min(int(xs.max()) + pad + 1, captured_frames[0].width),
            min(int(ys.max()) + pad + 1, captured_frames[0].height),
        )
        captured_frames = [frame.crop(box) for frame in captured_frames]
        
    save_animation(
        out_webp,
        captured_frames,
        duration_ms=40, # 25 FPS
        scale=0.55,
        webp_quality=80
    )
    print(f"\nAnimation successfully saved to: {out_webp}")

if __name__ == "__main__":
    main()
