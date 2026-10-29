"""E-030: inspect a native planar cap candidate near the user-confirmed origin.

This fits a dense planar subset of original CFD coordinates, NOT a wall-patch
PCA normal. A point-cloud face/hull is not a certified connected cap mesh.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

import numpy as np
from scipy.spatial import ConvexHull

from scripts.build_aorta_inlet_picker import SOURCE_SHA, WALL_SHA, sha, source_to_mri_mm


def planar_candidate(points_mm, *, iterations=500, seed=2901, tolerance_mm=.0001):
    points = np.asarray(points_mm, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('Finite XYZ coordinates required')
    if len(points) < 100 or tolerance_mm <= 0:
        raise ValueError('Insufficient candidate points or invalid tolerance')
    rng = np.random.default_rng(seed)
    best_count, best_mask = 0, None
    for _ in range(iterations):
        a,b,c = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(b-a, c-a)
        norm = np.linalg.norm(normal)
        if norm < 1:
            continue
        normal /= norm
        if abs(normal[2]) < .95:
            continue
        mask = abs((points-a)@normal) <= tolerance_mm
        if mask.sum() > best_count:
            best_count, best_mask = int(mask.sum()), mask
    if best_count < 100:
        raise ValueError('No dense near-XY planar candidate detected')
    # Refine only the robustly selected planar-face nodes, not surrounding wall.
    for _ in range(3):
        selected = points[best_mask]
        center = selected.mean(axis=0)
        _, values, axes = np.linalg.svd(selected-center, full_matrices=False)
        normal = axes[-1]
        if normal[2] < 0:
            normal = -normal
        if values[1] < 1:
            raise ValueError('Planar candidate is effectively a line')
        best_mask = abs((points-center)@normal) <= tolerance_mm
    selected = points[best_mask]
    return normal, float(center@normal), best_mask


def preview(path, wall, cap, normal, offset, old_points):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    u=np.array([1.,0.,0.]); u-=normal*(u@normal); u/=np.linalg.norm(u)
    v=np.cross(normal,u)
    uv=np.column_stack((cap@u,cap@v))
    hull=ConvexHull(uv); rim=np.r_[hull.vertices,hull.vertices[:1]]
    fig=plt.figure(figsize=(14,11))
    ax=fig.add_subplot(2,2,1,projection='3d')
    ax.scatter(*wall[::3].T,s=.2,c='#8ba0b0',alpha=.25,rasterized=True)
    ax.scatter(*cap.T,s=1,c='#19836d',alpha=.8,rasterized=True)
    ax.scatter(*old_points.T,s=25,c='#b3549e',label='Earlier rejected points')
    ax.scatter(0,0,0,s=65,c='#db5b35',marker='*',label='ANSYS origin / inlet reference')
    ax.set(xlabel='Source X (mm)',ylabel='Source Y (mm)',zlabel='Source Z (mm)',title='Corrected inlet location in original coordinates')
    ax.set_box_aspect(np.ptp(wall,axis=0));ax.view_init(20,-65);ax.legend(fontsize=8,loc='upper left')
    local=wall[np.linalg.norm(wall,axis=1)<25]
    ax=fig.add_subplot(2,2,2,projection='3d')
    ax.scatter(*local.T,s=2,c='#8ba0b0',alpha=.35)
    ax.scatter(*cap.T,s=2,c='#19836d',alpha=.75)
    ax.scatter(0,0,0,s=90,c='#db5b35',marker='*')
    ax.quiver(0,0,0,*normal,length=12,color='#db5b35')
    ax.set(xlabel='Source X (mm)',ylabel='Source Y (mm)',zlabel='Source Z (mm)',title='Native planar face nodes and adjoining wall')
    ax.set_box_aspect([1,1,.7]);ax.view_init(20,-65)
    ax=fig.add_subplot(2,2,3)
    ax.scatter(*uv.T,s=2,c='#19836d',label='Detected coplanar CFD nodes')
    ax.plot(*uv[rim].T,c='#4e6272',lw=1,label='Convex-hull guide (not native connectivity)')
    ax.scatter(0,0,s=90,c='#db5b35',marker='*',label='Projected ANSYS origin')
    ax.set(xlabel='Plane u (mm)',ylabel='Plane v (mm)',title='Complete detected face-point extent',aspect='equal')
    ax.legend(fontsize=8,loc='lower left')
    ax=fig.add_subplot(2,2,4)
    ax.scatter(local@u,local@normal-offset,s=2,c='#8ba0b0',alpha=.45,label='Nearby wall samples')
    ax.scatter(cap@u,cap@normal-offset,s=2,c='#19836d',label='Detected face nodes')
    ax.axhline(0,c='#19836d',lw=1)
    ax.scatter(0,-offset,c='#db5b35',marker='*',s=90)
    ax.set(xlabel='Plane u (mm)',ylabel='Signed distance from candidate face (mm)',title='Edge-on: geometry continues into +normal side')
    ax.legend(fontsize=8)
    fig.suptitle('E-030 | Origin-anchored inlet geometry review',fontsize=15)
    fig.text(.5,.025,'Native coordinate evidence only: no certified cap connectivity, physical-area validation, or inlet flow measurement.',ha='center',fontsize=10)
    fig.subplots_adjust(left=.07,right=.96,top=.92,bottom=.1,hspace=.48,wspace=.3)
    fig.savefig(path,dpi=180);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root=Path(__file__).resolve().parents[1]
    source=root.parent/'external_inputs/aorta_pointcloud/processed/dp2/point_coordinates.npy'
    wall_path=root/'outputs/aorta_dp2_mri_pipeline/visualization_branch_corrected/aorta_wall_points_from_wss.npy'
    old_path=root/'outputs/aorta_inlet_picker/e029_v1/rim_plane_review_v2/plane_review.json'
    if sha(source)!=SOURCE_SHA or sha(wall_path)!=WALL_SHA:
        raise ValueError('Native geometry hash mismatch')
    points=np.load(source,allow_pickle=False).astype(float)*1000
    wall=np.load(wall_path,allow_pickle=False).astype(float)*1000
    local_ids=np.flatnonzero(np.linalg.norm(points,axis=1)<25)
    local=points[local_ids]
    # Exploratory low-Z search is based on inspected local bounds, not velocities
    # or CFD/Koma agreement. Record its orientation restriction explicitly.
    candidates=local[local[:,2]<1]
    normal,offset,mask=planar_candidate(candidates)
    distance=local@normal-offset
    face_ids=local_ids[abs(distance)<=.0001]
    cap=points[face_ids]
    u=np.array([1.,0.,0.]);u-=normal*(u@normal);u/=np.linalg.norm(u)
    v=np.cross(normal,u)
    uv=np.column_stack((cap@u,cap@v));hull=ConvexHull(uv)
    old=json.loads(old_path.read_text(encoding='utf-8'))
    old_plane=old['source_plane']
    origin_distance=abs(offset)
    one_side=float(np.mean(distance>=-.01))
    if len(cap)<100 or origin_distance>.1 or one_side<.99:
        raise ValueError('Candidate does not pass origin/planarity/one-sided geometry checks')
    report=dict(experiment='E-030',created_utc=datetime.now(timezone.utc).isoformat(),
        status='ORIGIN_PLANAR_FACE_CANDIDATE_IDENTIFIED_NATIVE_CAP_UNCERTIFIED',
        evidence_class='Exploratory native-coordinate geometry identification; not independent validation',
        user_reference=dict(source_xyz_mm=[0,0,0],meaning='User confirms original CFD/ANSYS origin lies in inlet plane; not necessarily cap center',
                            mri_xyz_mm=source_to_mri_mm(np.zeros(3)).tolist()),
        inputs=dict(source_coordinates_sha256=SOURCE_SHA,wall_samples_sha256=WALL_SHA,
                    rejected_three_point_review_sha256=sha(old_path),script_sha256=sha(__file__)),
        search=dict(local_radius_mm=25,local_nodes=len(local),low_source_z_candidate_limit_mm=1,
                    low_z_nodes=len(candidates),iterations=500,seed=2901,plane_distance_tolerance_mm=.0001,
                    normal_search_constraint='abs(nZ)>=0.95; exploratory after local geometry inspection'),
        candidate=dict(normal_source_xyz=normal.tolist(),plane_equation='normal dot source_xyz_mm = offset_mm',
                       offset_mm=offset,normal_mri_xyz=normal[[2,0,1]].tolist(),
                       origin_distance_mm=origin_distance,normal_sign='Positive source Z; local CFD geometry continues on +normal side',
                       detected_face_nodes=len(cap),bounds_source_mm=[cap.min(0).tolist(),cap.max(0).tolist()],
                       centroid_source_mm=cap.mean(0).tolist(),
                       max_detected_node_residual_mm=float(np.max(abs(cap@normal-offset))),
                       local_fraction_on_positive_side_with_10micron_tolerance=one_side,
                       convex_hull_area_mm2=float(hull.volume),
                       hull_area_classification='Geometric convex-hull approximation, NOT native mesh cap area',
                       basis_u=u.tolist(),basis_v=v.tolist()),
        previous_plane=dict(status='REJECTED_FOR_INLET_ANALYSIS_BY_USER',
                            origin_distance_mm=float(abs(np.dot(old_plane['origin_mm'],old_plane['normal_ABC'])))),
        limitations=['Detected coplanar points do not supply certified native cap faces or boundary-zone labels',
                     'Convex hull can bridge concavities; do not promote its area to physical cap area',
                     'No inlet flux or new CFD/Koma simulation calculated',
                     'Earlier annotations and production evidence retained unchanged'])
    temporary=args.output.with_name(args.output.name+'.building-'+uuid.uuid4().hex[:8])
    temporary.mkdir(parents=True)
    np.save(temporary/'candidate_face_original_rows.npy',face_ids,allow_pickle=False)
    np.save(temporary/'candidate_face_source_mm.npy',cap,allow_pickle=False)
    preview(temporary/'origin_inlet_preview.png',wall,cap,normal,offset,np.array(old['source_points_mm']))
    (temporary/'origin_inlet_review.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    manifest=dict(experiment='E-030',files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in temporary.iterdir()})
    (temporary/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    temporary.rename(args.output)
    print(json.dumps(dict(output=str(args.output),candidate=report['candidate']),indent=2))


if __name__=='__main__':
    main()
