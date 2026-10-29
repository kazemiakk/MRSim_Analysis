"""Source-verify three user rim annotations and preview their plane (E-029).

This defines an annotation plane, not a lumen contour, area or inlet flux.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

import numpy as np

from scripts.build_aorta_inlet_picker import (
    SOURCE_SHA, WALL_SHA, sha, source_to_mri_mm,
)


def fit_rim_plane(points_mm):
    points = np.asarray(points_mm, dtype=float)
    if points.shape != (3, 3) or not np.isfinite(points).all():
        raise ValueError('Three finite XYZ points required')
    ab, ac = points[1] - points[0], points[2] - points[0]
    edges = np.array([np.linalg.norm(ab), np.linalg.norm(ac),
                      np.linalg.norm(points[2] - points[1])])
    cross = np.cross(ab, ac)
    double_area = np.linalg.norm(cross)
    # Dimensionless, invariant to units and rigid transforms. An annotation
    # triangle below this conservative shape threshold is not usable here.
    quality = double_area / max(float(edges.max() ** 2), np.finfo(float).tiny)
    if edges.min() < .001 or quality < .01:
        raise ValueError('Duplicate or nearly collinear rim points')
    normal = cross / double_area
    u = ab / np.linalg.norm(ab)
    v = np.cross(normal, u)
    return dict(origin_mm=points.mean(axis=0).tolist(),
                normal_ABC=normal.tolist(), basis_u=u.tolist(), basis_v=v.tolist(),
                edge_lengths_mm=edges.tolist(), triangle_area_mm2=float(double_area / 2),
                triangle_shape_quality=float(quality),
                rejection_shape_threshold=.01,
                max_point_plane_residual_mm=float(np.max(np.abs((points-points.mean(axis=0))@normal))))


def verify_selection(record, samples, original, rows):
    if record.get('purpose') != 'user_marked_inlet_location':
        raise ValueError('Not a real user annotation')
    if record.get('source_coordinates_sha256') != SOURCE_SHA or record.get('wall_samples_sha256') != WALL_SHA:
        raise ValueError('Annotation source identity mismatch')
    index = record['wall_associated_sample_index']
    row = record['original_coordinate_row']
    if type(index) is not int or not 0 <= index < len(samples):
        raise ValueError('Invalid sample index')
    if type(row) is not int or not 0 <= row < len(original) or int(rows[index]) != row:
        raise ValueError('Invalid original row')
    point = np.asarray(record['source_xyz_m'], dtype=float)
    if not np.array_equal(point, samples[index]) or not np.array_equal(point, original[row]):
        raise ValueError('Annotation coordinate mismatch')
    if not np.allclose(record['source_xyz_mm'], point*1000, atol=1e-12, rtol=0):
        raise ValueError('Annotation unit mismatch')
    if not np.allclose(record['mri_xyz_mm'], source_to_mri_mm(point), atol=1e-12, rtol=0):
        raise ValueError('Annotation transform mismatch')
    return point*1000


def preview(path, wall_mm, points, plane):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    center = np.asarray(plane['origin_mm'])
    n, u, v = (np.asarray(plane[k]) for k in ('normal_ABC', 'basis_u', 'basis_v'))
    fig = plt.figure(figsize=(15, 12))
    patch_extent = 17  # Display extent only; not an assumed lumen radius.
    g = np.array([[-1,-1], [1,-1], [1,1], [-1,1], [-1,-1]])*patch_extent
    patch = center + g[:, :1]*u + g[:, 1:]*v
    labels = ['A', 'B', 'C']
    for panel, angles in enumerate([(18, -65), (30, 25)], 1):
        ax = fig.add_subplot(2, 2, panel, projection='3d')
        ax.scatter(*wall_mm[::3].T, s=.15, c='#667f92', alpha=.22, rasterized=True)
        ax.plot(*patch.T, color='#e8a130', linewidth=1.5)
        ax.scatter(*points.T, s=35, c=['#d34242', '#27815b', '#704abc'], depthshade=False)
        for p, label in zip(points, labels):
            ax.text(*p, label, fontsize=9)
        ax.quiver(*center, *n, length=20, color='#e8a130')
        ax.set(xlabel='Source X (mm)', ylabel='Source Y (mm)', zlabel='Source Z (mm)',
               title='Original point geometry + user-defined plane')
        bounds = np.ptp(wall_mm, axis=0)
        ax.set_box_aspect(bounds)
        ax.view_init(*angles)
    local = wall_mm[np.linalg.norm(wall_mm-center, axis=1) < 25] - center
    p = points-center
    ax = fig.add_subplot(2, 2, 3)
    ax.scatter(local@u, local@v, s=1, c='#a7b6c1', alpha=.3, label='Wall samples within 25 mm')
    slab = abs(local@n) <= 1
    ax.scatter(local[slab]@u, local[slab]@v, s=3, c='#395e77', label='Samples within +/-1 mm of plane')
    for q, label, color in zip(p, labels, ['#d34242', '#27815b', '#704abc']):
        ax.scatter(q@u, q@v, s=40, c=color)
        ax.annotate(label, (q@u, q@v), xytext=(5, 5), textcoords='offset points')
    ax.set(xlabel='Plane u (mm)', ylabel='Plane v (mm)', title='Looking perpendicular to the selected plane', aspect='equal')
    ax.legend(fontsize=8, loc='lower left')
    ax = fig.add_subplot(2, 2, 4)
    ax.scatter(local@u, local@n, s=1, c='#667f92', alpha=.4)
    ax.axhline(0, color='#e8a130', label='Selected plane')
    ax.scatter(p@u, p@n, c=['#d34242', '#27815b', '#704abc'], s=40)
    ax.set(xlabel='Plane u (mm)', ylabel='Signed plane-normal distance (mm)',
           title='Local edge-on view', aspect='equal')
    ax.legend(fontsize=8)
    fig.suptitle('E-029 | Three-point inlet-plane annotation review', fontsize=15)
    fig.text(.5, .015, 'Orange rectangle is a display guide, NOT a cap boundary. ABC normal sign is arbitrary. No flow or physical area calculated.', ha='center', fontsize=10)
    # Reserve explicit space between 3D axes and lower panels: tight_layout
    # underestimates the projected 3D tick/axis-label bounds.
    fig.subplots_adjust(left=.07, right=.96, bottom=.09, top=.92, hspace=.42, wspace=.3)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selections', nargs=3, type=Path, required=True, metavar=('A','B','C'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    picker = root/'outputs/aorta_inlet_picker/e029_v1'
    source = root.parent/'external_inputs/aorta_pointcloud/processed/dp2/point_coordinates.npy'
    wall = root/'outputs/aorta_dp2_mri_pipeline/visualization_branch_corrected/aorta_wall_points_from_wss.npy'
    if args.output.exists():
        raise FileExistsError(args.output)
    if sha(source) != SOURCE_SHA or sha(wall) != WALL_SHA:
        raise ValueError('Source geometry identity mismatch')
    samples = np.load(picker/'source_samples.npy', allow_pickle=False)
    rows = np.load(picker/'original_coordinate_rows.npy', allow_pickle=False)
    original = np.load(source, mmap_mode='r', allow_pickle=False)
    if not np.array_equal(samples, np.load(wall, allow_pickle=False)):
        raise ValueError('Picker samples mismatch')
    records = [json.loads(p.read_text(encoding='utf-8')) for p in args.selections]
    points = np.array([verify_selection(r, samples, original, rows) for r in records])
    plane = fit_rim_plane(points)
    mri_points = source_to_mri_mm(points/1000)
    report = dict(experiment='E-029', created_utc=datetime.now(timezone.utc).isoformat(),
                  status='THREE_POINT_ANNOTATION_PLANE_DEFINED_CAP_UNVERIFIED',
                  selections=[dict(role=role, path=str(p), sha256=sha(p), record=r)
                              for role,p,r in zip('ABC', args.selections, records)],
                  label_interpretation='Third saved label rim bc interpreted as Rim C in the requested three-point sequence',
                  source_coordinates_sha256=SOURCE_SHA, wall_samples_sha256=WALL_SHA,
                  source_plane=plane, mri_plane=fit_rim_plane(mri_points),
                  source_points_mm=points.tolist(), mri_points_mm=mri_points.tolist(),
                  origin_definition='Arithmetic centroid of the three annotations; NOT a certified cap center',
                  normal_definition='Unit cross(B-A,C-A); sign NOT established as inward/outward',
                  limitations=['Zero fit residual is by construction with three points, not an independent plane check',
                               'No certified cap contour, physical area, boundary normal, or anatomical label',
                               'Point-cloud source remains approximate; no wall-patch PCA normal used',
                               'No inlet flow calculated and no original CFD or Koma artifact changed'],
                  builder_sha256=sha(__file__))
    temporary = args.output.with_name(args.output.name+'.building-'+uuid.uuid4().hex[:8])
    temporary.mkdir(parents=True)
    for role, record in zip('ABC', records):
        (temporary/f'rim_{role}.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    preview(temporary/'rim_plane_preview.png', samples.astype(float)*1000, points, plane)
    (temporary/'plane_review.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    manifest = dict(experiment='E-029', files={p.name:dict(bytes=p.stat().st_size, sha256=sha(p)) for p in temporary.iterdir()})
    (temporary/'MANIFEST.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    temporary.rename(args.output)
    print(json.dumps(dict(output=str(args.output), status=report['status'], source_plane=plane), indent=2))


if __name__ == '__main__':
    main()
