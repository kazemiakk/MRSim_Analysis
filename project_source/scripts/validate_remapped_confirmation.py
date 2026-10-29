"""Independent E047 arithmetic, map, source and export checks; no source writes."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import tomllib

import h5py
import numpy as np
from scipy.spatial import cKDTree
from PIL import Image
from scripts.build_aorta_inlet_picker import sha

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/remapped_confirmation_e047'


def validate(output):
    if output.exists():raise FileExistsError(output)
    checks=[]
    def check(name,passed):
        if not bool(passed):raise AssertionError(name)
        checks.append(name)
    frozen=json.loads((BASE/'source_registration.json').read_text())
    for name,item in frozen['inputs'].items():
        check('frozen:'+name,sha(ROOT/name)==item['sha256'])
    old=json.loads((ROOT/'outputs/aorta_inlet_replenishment/e046_legacy_snapshot_v1/registration.json').read_text())
    for name,item in old['inputs'].items():
        check('original39:'+name,sha(ROOT/name)==item['sha256'])
    for case in ('aorta','phantom'):
        directory=BASE/f'{case}_analysis_v1'
        report=json.loads((directory/'report.json').read_text())
        manifest=json.loads((directory/'MANIFEST.json').read_text())
        for name,value in manifest['input_identities'].items():check(case+':input:'+name,sha(name)==value['sha256'])
        for name,value in manifest['outputs'].items():check(case+':output:'+name,sha(directory/name)==value)
        with h5py.File(directory/'paired_fields.h5') as f:
            a=f['reference'][:];fraction=f['numerical_volume_fraction'][:]
            fields={arm:f[arm][:] for arm in ('legacy100k','map20k','map100k')}
        images={}
        for arm in fields:
            folder=BASE/f'{case}_inputs_v1' if arm=='legacy100k' else (
                ROOT/f'outputs/upstream_migration_e046/{case}_updated_map_v1' if arm=='map20k' else BASE/f'{case}_mapped_v1')
            images[arm]={}
            for e in ('REF','X','Y','Z'):
                with h5py.File(folder/(e.lower()+('_legacy_result.h5' if arm=='legacy100k' else '_result.h5'))) as f:
                    images[arm][e]=f['image'][:]
        for row in report['thresholds']:
            threshold=row['signal_threshold']
            for arm,b in fields.items():
                mask=fraction>0
                for e in ('REF','X','Y','Z'):
                    magnitude=np.abs(images[arm][e])
                    maximum=np.max(magnitude.reshape(len(a),-1),axis=1)
                    mask=mask & np.all((magnitude>=maximum[:,None,None,None]*threshold)&(magnitude>0),axis=0)
                wanted=row['volume']['own'][arm]
                check(f'{case}:{threshold}:{arm}:voxel_count',int(mask.sum())==wanted['voxels'])
                coverage=float(np.sum(fraction[mask],dtype=np.float64)/np.sum(fraction,dtype=np.float64))
                check(f'{case}:{threshold}:{arm}:volume_coverage',np.isclose(coverage,wanted['coverage_fraction'],rtol=1e-6,atol=1e-8))
                weights=np.broadcast_to(fraction[mask],(len(a),int(mask.sum()))).astype(float)
                if mask.any():
                    squared=np.sum((b[:,mask].astype(float)-a[:,mask].astype(float))**2,axis=-1)
                    expected=float(np.sqrt(np.average(squared,weights=weights)))
                    check(f'{case}:{threshold}:{arm}:vector_rmse',np.isclose(expected,wanted['metrics']['vector_rmse_m_s'],rtol=1e-9,atol=1e-10))
                else:check(f'{case}:{threshold}:{arm}:null_error',wanted['metrics'] is None)
            flow=row['flow']
            for arm in fields:
                own=flow['own'][arm]
                check(f'{case}:{threshold}:{arm}:area_coverage',np.isclose(own['coverage_fraction'],own['area_mm2']/flow['whole_section_area_mm2'],atol=1e-12))
                if own['metrics'] is None:continue
                cfd=np.array(own['CFD_ml_s']);koma=np.array(own['Koma_ml_s'])
                error=float(np.linalg.norm(koma-cfd)/np.sqrt(len(cfd)))
                check(f'{case}:{threshold}:{arm}:flow_rmse',np.isclose(error,own['metrics']['rmse_ml_s'],atol=1e-12))
                check(f'{case}:{threshold}:{arm}:cycle_ratio',np.isclose(koma.sum()/cfd.sum()-1,own['metrics']['cycle_volume_error_fraction'],atol=1e-12))
            check(f'{case}:{threshold}:shared_area',np.isclose(flow['three_arm_area_mm2']/flow['whole_section_area_mm2'],flow['three_arm_coverage_fraction'],atol=1e-12))
        info=tomllib.loads((BASE/f'{case}_inputs_v1/case_info.toml').read_text())
        with h5py.File(BASE/f'{case}_inputs_v1/case_inputs.h5') as f:
            base=np.column_stack([f[c][:100000] for c in ('x','y','z')])
            first=base+np.column_stack([f[c][0,:100000] for c in ('dx','dy','dz')])
            last=base+np.column_stack([f[c][-1,:100000] for c in ('dx','dy','dz')])
        with h5py.File(BASE/f'{case}_map_v1/cycle_map.h5') as f:
            mapping=f['cycle_map'][:];distance=f['distance_m'][:]
        mapreport=json.loads((BASE/f'{case}_map_v1/map_report.json').read_text())
        check(case+':sentinel_and_bounds',mapping[-1]==100001 and (mapping[:-1]>=1).all() and (mapping[:-1]<=100000).all())
        independent_distance,independent_index=cKDTree(last).query(first,k=1)
        check(case+':all100k_NN_distances',np.allclose(independent_distance,distance,atol=1e-9,rtol=1e-5))
        check(case+':all100k_NN_indices',np.array_equal(independent_index+1,mapping[:-1]))
        counts=np.bincount(mapping[:-1]-1,minlength=100000)
        check(case+':multiplicity',counts.max()==mapreport['maximum_source_multiplicity'])
        check(case+':unique_fraction',np.isclose(np.mean(counts>0),mapreport['unique_source_fraction']))
        run=tomllib.loads((BASE/f'{case}_mapped_v1/control_report.toml').read_text())
        check(case+':worker_binding',run['script_sha256']==sha(ROOT/'koma_migration_e046/run_100k_e047.jl'))
        check(case+':input_binding',run['input_sha256']==info['neutral_input_sha256'])
        check(case+':map_binding',run['map_sha256']==sha(BASE/f'{case}_map_v1/cycle_map.h5'))
        check(case+':no_independent_spin_batching',run['coupled_spin_partitioning'] is False)
        check(case+':counts_timing',run['physical_particles']==100000 and run['total_spins']==100001 and run['actual_ADC_timing_and_labels_match_legacy'])
        for p in directory.glob('*.png'):
            with Image.open(p) as im:
                im.load();check(case+':PNG:'+p.name,im.width>1500 and im.height>1200)
        for p in directory.glob('*.pdf'):check(case+':PDF_signature:'+p.name,p.read_bytes().startswith(b'%PDF-'))
    result=dict(status='PASS_INDEPENDENT_E047_CHECKS',created_utc=datetime.now(timezone.utc).isoformat(),
                checks_passed=len(checks),checks=checks,legacy_identities_unchanged=len(old['inputs']),
                output_sha256={str(p.relative_to(ROOT)):sha(p) for case in ('aorta','phantom') for p in (BASE/f'{case}_analysis_v1').glob('*') if p.is_file()},
                limitation='Numerical/provenance checks, not physical validation. PNG decode checked; separate visual review required. PDFs not rendered.')
    output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=result['status'],checks_passed=len(checks),legacy_identities_unchanged=len(old['inputs']))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    validate(p.parse_args().output)
