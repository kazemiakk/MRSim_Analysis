"""Build a local, self-contained 3D point-cloud inlet-location picker (E-029)."""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import uuid

import numpy as np
from scipy.spatial import cKDTree


SOURCE_SHA = '21af8e15f9542add12320352706ba1d250e84b2c49f4a0035b34647b3e9f0197'
WALL_SHA = 'eb195abe0a0a3a604648b9eaf933066af5a178a37c77f00bdb48ff5f9f9983dd'
CENTER_M = np.array([.038389891386032104, .015447176992893219, -.023723918944597244])
AXIS_ORDER = np.array([2, 0, 1])


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def source_to_mri_mm(points):
    return (np.asarray(points, dtype=float) - CENTER_M)[..., AXIS_ORDER] * 1000


def make_selection(index, points, node_ids, metadata, *, test_mode=False, note=''):
    if type(index) is not int or not 0 <= index < len(points):
        raise ValueError('Select a valid displayed sample index')
    if not isinstance(note, str) or len(note) > 500:
        raise ValueError('Note must be text of at most 500 characters')
    point = np.asarray(points[index], dtype=float)
    return dict(schema='mrsim-inlet-location-v1',
        purpose='browser_test_not_user_inlet' if test_mode else 'user_marked_inlet_location',
        created_utc=datetime.now(timezone.utc).isoformat(),
        wall_associated_sample_index=index, original_coordinate_row=int(node_ids[index]),
        source_xyz_m=point.tolist(), source_xyz_mm=(point*1000).tolist(),
        mri_xyz_mm=source_to_mri_mm(point).tolist(),
        source_coordinates_sha256=metadata['source_coordinates_sha256'],
        wall_samples_sha256=metadata['wall_samples_sha256'], note=note,
        classification='User-marked inlet neighborhood; not a verified cap, center, plane or normal',
        plane_normal=None, inlet_cap_verified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = root.parent/'external_inputs/aorta_pointcloud/processed/dp2/point_coordinates.npy'
    wall = root/'outputs/aorta_dp2_mri_pipeline/visualization_branch_corrected/aorta_wall_points_from_wss.npy'
    template = root/'scripts/aorta_inlet_picker/viewer.html'
    if args.output.exists():
        raise FileExistsError(args.output)
    if sha(source) != SOURCE_SHA or sha(wall) != WALL_SHA:
        raise ValueError('Geometry identity mismatch')
    points = np.load(wall, allow_pickle=False)
    native = np.load(source, mmap_mode='r', allow_pickle=False)
    if points.shape != (119097, 3) or not np.isfinite(points).all():
        raise ValueError('Unexpected wall-associated point array')
    distances, node_ids = cKDTree(native).query(points, workers=-1)
    if np.max(distances) != 0:
        raise ValueError('Display samples do not exactly match restored CFD coordinates')
    positions = source_to_mri_mm(points).astype('<f4')
    metadata = dict(experiment='E-029', display_kind='Native wall-associated point samples; no connected wall/caps',
        point_count=len(points), source_coordinates_sha256=SOURCE_SHA, wall_samples_sha256=WALL_SHA,
        maximum_nearest_source_distance_m=float(distances.max()),
        source_center_m=CENTER_M.tolist(), mri_source_axis_order_zero_based=AXIS_ORDER.tolist(),
        positions_dtype='little-endian float32', positions_units='MRI mm',
        positions_base64=base64.b64encode(positions.tobytes()).decode('ascii'),
        source_bounds_m=[points.min(axis=0).tolist(), points.max(axis=0).tolist()],
        mri_bounds_mm=[positions.min(axis=0).tolist(), positions.max(axis=0).tolist()],
        geometry_certification='Not a verified physical wall, cap, anatomical segmentation or inlet label')
    temporary = args.output.with_name(args.output.name+'.building-'+uuid.uuid4().hex[:8])
    temporary.mkdir(parents=True, exist_ok=False)
    np.save(temporary/'source_samples.npy', points, allow_pickle=False)
    np.save(temporary/'original_coordinate_rows.npy', node_ids, allow_pickle=False)
    raw = json.dumps(metadata, separators=(',', ':'), allow_nan=False)
    (temporary/'geometry.json').write_text(raw, encoding='utf-8')
    html = template.read_text(encoding='utf-8').replace('__GEOMETRY_JSON__', raw)
    if '__GEOMETRY_JSON__' in html or 'const LOCAL_RUNTIME = null; // SERVER_RUNTIME' not in html:
        raise ValueError('Viewer template is incomplete')
    (temporary/'index.html').write_text(html, encoding='utf-8')
    (temporary/'README.md').write_text(
        '# E-029: aortic inlet-location picker\n\n'
        'Open http://127.0.0.1:4175/ while the local service is running.\n'
        'Drag to rotate; wheel or +/- to zoom; shift/right-drag to pan. Click the wall/rim near the inlet. '
        'The selected location is a real displayed sample; it is not a cap center or a plane normal.\n\n'
        'Click **Save selection for analysis**, then tell Codex "done". '
        'User choices are stored in `user_selections/`, with a `latest_selection.json` pointer. '
        'Browser tests use a different folder and are tagged as tests.\n\n'
        'The HTML also opens offline: rotate/click and use Copy coordinates or Download JSON. '
        'Offline mode cannot save directly for Codex. No external scripts, hosting, upload or login.\n\n'
        'Geometry comprises all 119,097 preserved branch-corrected native wall-associated samples, '
        'each matching restored CFD coordinates exactly. No closed shell, invented cap or branch identity. '
        'Coordinate precision does not imply physical boundary accuracy. Original source/Koma data are unchanged.\n\n'
        'Start service from the repository root:\n\n'
        '```powershell\npython -m scripts.serve_aorta_inlet_picker --root outputs/aorta_inlet_picker/e029_v1 --port 4175\n```\n', encoding='utf-8')
    files = {p.name: dict(bytes=p.stat().st_size, sha256=sha(p)) for p in temporary.iterdir() if p.is_file()}
    manifest = dict(experiment='E-029', created_utc=datetime.now(timezone.utc).isoformat(),
        source_coordinates_sha256=SOURCE_SHA, wall_samples_sha256=WALL_SHA,
        point_count=len(points), nearest_source_max_m=float(distances.max()),
        builder_sha256=sha(__file__), template_sha256=sha(template), files=files)
    (temporary/'MANIFEST.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    temporary.rename(args.output)
    print(json.dumps(dict(output=str(args.output), point_count=len(points), source_match_max_m=float(distances.max())), indent=2))


if __name__ == '__main__':
    main()
