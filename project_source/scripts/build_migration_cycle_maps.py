"""E-046: exact upstream NN maps with original endpoint inlet-reset restoration."""
import argparse
import importlib.util
import json
from pathlib import Path
import tomllib
import uuid
import h5py
import numpy as np
from scripts.build_aorta_inlet_picker import sha


def build(case_dir, output):
    if output.exists():
        raise FileExistsError(output)
    root = Path(__file__).resolve().parents[1]
    info = tomllib.loads((case_dir/'case_info.toml').read_text())
    data = case_dir/'case_inputs.h5'
    if sha(data) != info['neutral_input_sha256']:
        raise ValueError('Neutral control input changed')
    module_path = root/'outputs/upstream_migration_e046/tracker_source/mrsimtracks/mapping.py'
    spec = importlib.util.spec_from_file_location('e046_upstream_mapping', module_path)
    upstream = importlib.util.module_from_spec(spec); spec.loader.exec_module(upstream)
    # Analytical direction/control checks before evaluating the physical cases.
    points = np.array([[0,0,0],[1,0,0],[2,0,0]], dtype=float)
    control = upstream.periodic_mapping(points, points[[2,0,1]])
    if not np.array_equal(control, [2,3,1]):
        raise AssertionError('Wrong destination-to-source or index convention')
    with h5py.File(data, 'r') as f:
        n = int(f.attrs['physical_particles'])
        base = np.column_stack([f[c][:n] for c in ['x','y','z']])
        initial = base+np.column_stack([f[c][0,:n] for c in ['dx','dy','dz']])
        final = base+np.column_stack([f[c][-1,:n] for c in ['dx','dy','dz']])
    if info['case'] == 'aorta':
        track = root/'outputs/aorta_inlet_replenishment/e045_native0_100k_tracking_v1/tracks.h5'
        source_report = tomllib.loads((Path(info['source_output'])/'phantom_report.toml').read_text())
        if sha(track) != source_report['tracks_sha256']:
            raise ValueError('Aortic endpoint-reset source changed')
        with h5py.File(track,'r') as f:
            original_initial=f['initial_position'][:n]
            original_final=f['position'][-1,:n]
            restored=f['reset'][-1,:n].astype(bool)
            union=bool(f['reset'][-1].any())
        center=np.array([.038389891386032104,.015447176992893219,-.023723918944597244],dtype=np.float32)
        expected_initial=(original_initial-center)[:,[2,0,1]]
        expected_final=(original_final-center)[:,[2,0,1]]
    else:
        track=root/'outputs/dp150_mri_pipeline/dp150_motion_100k_mri.h5'
        source_report=json.loads((root/'outputs/dp150_mri_pipeline/koma/pilot_100k/PHANTOM_REPRODUCIBILITY_MANIFEST.json').read_text())
        if sha(track) != source_report['subset_hdf5']['sha256']:
            raise ValueError('Phantom endpoint-reset source changed')
        with h5py.File(track,'r') as f:
            expected_initial=f['initial_position'][:n]
            expected_final=f['position'][-1,:n]
            restored=f['reset'][-1,:n].astype(bool)
            union=bool(f['reset'][-1].any())
    if not np.allclose(initial,expected_initial,rtol=0,atol=1e-7) or not np.allclose(final,expected_final,rtol=0,atol=1e-7):
        raise ValueError('Endpoint identity/axis/coordinate mismatch')
    mapping=upstream.periodic_mapping(initial,final)
    distance=np.linalg.norm(initial-final[mapping-1],axis=1)
    counts=np.bincount(mapping-1,minlength=n)
    # The zero-density timing sentinel never participates in physical NN selection.
    full=np.concatenate([mapping,[n+1]]).astype(np.int64)
    staging=output.with_name(output.name+'.building-'+uuid.uuid4().hex[:8]);staging.mkdir(parents=True)
    with h5py.File(staging/'cycle_map.h5','w') as f:
        f['cycle_map']=full;f['restored_endpoint_reset']=np.append(restored,union).astype(np.uint8)
        f['distance_m']=distance
    report=dict(status='PASS_UPSTREAM_MAP_AND_ENDPOINT_IDENTITY',case=info['case'],physical_particles=n,
        total_spins_with_sentinel=n+1,upstream_commit='200c87be5541ffe36c31bad504b6cad81d9da693',
        code_sha256=sha(__file__),upstream_mapping_sha256=sha(module_path),input_sha256=sha(data),
        endpoint_source=str(track),endpoint_source_sha256=sha(track),map_sha256=sha(staging/'cycle_map.h5'),
        endpoint_coordinate_identity_checked=True,sentinel_self_map=True,map_is_one_based_destination_to_source=True,
        map_distance_m=dict(median=float(np.median(distance)),p95=float(np.quantile(distance,.95)),maximum=float(distance.max())),
        unique_source_fraction=float(np.mean(counts>0)),maximum_source_multiplicity=int(counts.max()),
        restored_real_endpoint_inlet_resets=int(restored.sum()),analytical_direction_control_passed=True,
        limitation='Nearest-neighbor ensemble-state approximation; not one-to-one or exact conservation')
    (staging/'map_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    staging.rename(output)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();build(a.case_dir,a.output)
