"""Create a nested 300k export and the unchanged upstream map outside the repo."""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tomllib
import h5py
import numpy as np


def sha(path):
    with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def toml_write(path,values):
    scalar={k:v for k,v in values.items() if not isinstance(v,dict)}
    lines=[]
    def literal(v):
        if isinstance(v,bool):return 'true' if v else 'false'
        if isinstance(v,str):return json.dumps(v)
        if isinstance(v,list):return '['+', '.join(literal(x) for x in v)+']'
        if isinstance(v,(float,int)):return repr(v)
        raise TypeError(type(v))
    for k,v in scalar.items():lines.append(k+' = '+literal(v))
    for k,v in values.items():
        if isinstance(v,dict):
            lines.append('\n['+k+']')
            for key,value in v.items():lines.append(json.dumps(key)+' = '+literal(value))
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main(repo,run):
    inputs=run/'inputs_v1'
    if inputs.exists():raise FileExistsError(inputs)
    control=run.parent/'stenosis_controls_e054'
    gate=json.loads((control/'validation_v1.json').read_text())
    report_path=control/'analysis_v1/report.json';report=json.loads(report_path.read_text())
    assert gate['status']=='PASS_INDEPENDENT_SAVED_SIGNAL_AND_PATH_CHECKS'
    assert gate['report_sha256']==sha(report_path)=='c88a592791b0536f190c3507f18b0b54fadeb00866fcc19cfe5e0f7c4a8eb5f0'
    assert report['status']=='PASS_SMALL_ACCELERATION_SHEAR_CONTROLS' and all(g['passed'] for g in report['gates'])
    old=repo/'outputs/remapped_confirmation_e047/phantom_inputs_v1'
    info=tomllib.loads((old/'case_info.toml').read_text())
    assert sha(old/'case_inputs.h5')==info['neutral_input_sha256']
    source=repo/'outputs/dp150_mri_pipeline/dp150_motion_1m_mri.h5'
    assert shutil.disk_usage(run).free>20_000_000_000
    source_digest=sha(source)
    assert source_digest=='f904e81e3626baf26007f3a20858ef7d9e15816b96978f64b4d0553fd505978c'
    baseline=np.sort(np.random.default_rng(20260831).choice(1_000_000,100_000,replace=False)).astype('<i8')
    assert hashlib.sha256(baseline.tobytes()).hexdigest()=='6ed2102040d4147471da1eb3f41d2317d08d1f48979f6ffbc6fd381e6439317e'
    remaining=np.setdiff1d(np.arange(1_000_000),baseline,assume_unique=True)
    added=np.random.default_rng(20261024).choice(remaining,200_000,replace=False)
    selected=np.sort(np.r_[baseline,added]).astype('<i8');baseline_in_new=np.searchsorted(selected,baseline)
    assert len(np.unique(selected))==300000 and np.array_equal(selected[baseline_in_new],baseline)
    inputs.mkdir();n=len(selected);nt=1441
    np.save(inputs/'selected_identity_indices.npy',selected,allow_pickle=False)
    checks=0;reset_events=0
    with h5py.File(source) as src,h5py.File(old/'case_inputs.h5') as oldf,h5py.File(inputs/'case_inputs.h5.partial','w') as out:
        assert src['position'].shape==(1440,1000000,3) and src['reset'].shape==(1440,1000000)
        initial=src['initial_position'][:][selected]
        for key,value in [('x',0.),('y',0.),('z',0.),('rho',1.),('T1',float(np.float32(1.3))),('T2',float(np.float32(.1))),('delta_w',0.)]:
            assert np.all(oldf[key][:100000]==np.float32(value));checks+=1
            extra=1.3 if key=='T1' else (.1 if key=='T2' else 0.)
            out[key]=np.r_[np.full(n,value,np.float32),np.float32(extra)]
        for key in ['time','time_unit']:out[key]=oldf[key][:]
        paths={d:out.create_dataset(d,shape=(nt,n+1),dtype='f4',chunks=(1,16384)) for d in ['dx','dy','dz']}
        resets=out.create_dataset('reset',shape=(nt,n+1),dtype='u1',chunks=(1,16384))
        for c,d in enumerate(['dx','dy','dz']):
            paths[d][0,:]=np.r_[initial[:,c],np.float32(0)]
            assert np.array_equal(initial[baseline_in_new,c],oldf[d][0,:100000]);checks+=1
        resets[0,:]=0
        for t in range(1440):
            positions=src['position'][t][selected]
            real_reset=src['reset'][t][selected]
            assert np.isfinite(positions).all()
            reset_events+=int(real_reset.sum())
            for c,d in enumerate(['dx','dy','dz']):
                paths[d][t+1,:]=np.r_[positions[:,c],np.float32(0)]
                assert np.array_equal(positions[baseline_in_new,c],oldf[d][t+1,:100000]);checks+=1
            flags=real_reset.copy()
            if t==1439:flags[:]=True
            resets[t+1,:]=np.r_[flags,np.any(flags)].astype('u1')
            assert np.array_equal(flags[baseline_in_new],oldf['reset'][t+1,:100000].astype(bool));checks+=1
            if t%240==0:print('E055 exported source step',t+1,'/',1440,flush=True)
        final=positions.copy();restored=real_reset.copy()
        out.attrs['physical_particles']=n;out.attrs['t_start_s']=info['t_start_s'];out.attrs['t_end_s']=info['t_end_s']
    (inputs/'case_inputs.h5.partial').rename(inputs/'case_inputs.h5')
    mapping_path=repo/'outputs/upstream_migration_e046/tracker_source/mrsimtracks/mapping.py'
    spec=importlib.util.spec_from_file_location('e055_same_upstream_mapping',mapping_path);mapping=importlib.util.module_from_spec(spec);spec.loader.exec_module(mapping)
    probe=np.array([[0,0,0],[1,0,0],[2,0,0]],float)
    assert np.array_equal(mapping.periodic_mapping(probe,probe[[2,0,1]]),[2,3,1]);checks+=1
    indices=mapping.periodic_mapping(initial,final)
    assert len(indices)==n and np.all((indices>=1)&(indices<=n))
    maps=run/'map_v1';maps.mkdir()
    with h5py.File(maps/'cycle_map.h5','w') as f:
        f['cycle_map']=np.r_[indices,n+1].astype('i8');f['restored_endpoint_reset']=np.r_[restored,np.any(restored)].astype('u1')
        distance=np.linalg.norm(initial-final[indices-1],axis=1);f['distance_m']=distance
    shutil.copy2(old/'legacy_timing.h5',inputs/'legacy_timing.h5')
    old_info_sha=sha(old/'case_info.toml')
    info={k:info[k] for k in ['case','sequence_directory','period_s','seam_time_s','B0_T','Venc_m_s',
                              't_start_s','t_end_s','sequence_sha256','artificial_global_reset_columns_one_based']}
    info.update(experiment='E-055',status='PASS_NESTED_300K_SOURCE_EXPORT',physical_particles=n,total_spins_with_sentinel=n+1,
                neutral_input_sha256=sha(inputs/'case_inputs.h5'),legacy_timing_sha256=sha(inputs/'legacy_timing.h5'),
                map_sha256=sha(maps/'cycle_map.h5'),e054_report_sha256=sha(report_path),e054_validation_sha256=sha(control/'validation_v1.json'),
                tracking_source_hdf5=str(source),tracking_source_hdf5_sha256=source_digest,exporter_sha256=sha(Path(__file__)))
    # Same sequence bytes, physically read from D: rather than the frozen C: path.
    info['sequence_directory']=str(repo/'koma_dp150/pc_3d_scanner_matched_venc5')
    for e,digest in info['sequence_sha256'].items():assert sha(Path(info['sequence_directory'])/(e.lower()+'.seq'))==digest
    toml_write(inputs/'case_info.toml',info)
    registration=dict(experiment='E-055',decision='D-058',status='PASS_NESTED_SOURCE_AND_MAP_BEFORE_SIGNAL',
        created_utc=datetime.now(timezone.utc).isoformat(),source=str(source),source_sha256=source_digest,
        baseline_case_info_sha256=old_info_sha,physical_particles=n,baseline_identities_included=100000,
        all_baseline_path_reset_property_checks=checks,selected_indices_sha256=sha(inputs/'selected_identity_indices.npy'),
        case_input_sha256=info['neutral_input_sha256'],map_sha256=info['map_sha256'],case_info_sha256=sha(inputs/'case_info.toml'),
        upstream_mapping_sha256=sha(mapping_path),source_real_reset_events=reset_events,
        map_distance_m=dict(median=float(np.median(distance)),p95=float(np.quantile(distance,.95)),maximum=float(distance.max())),
        unique_map_source_fraction=float(len(np.unique(indices))/n),new_1m_started=False,
        limitation='Same-source nested density confirmation; nearest-neighbor map changes naturally with population size. Not independent physical validation.')
    (run/'source_registration.json').write_text(json.dumps(registration,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:registration[k] for k in ['status','physical_particles','all_baseline_path_reset_property_checks','map_distance_m']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();main(a.repo,a.run)
