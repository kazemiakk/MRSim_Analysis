"""E049 independently verify phase controls, clock failure and source integrity."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import tomllib

import h5py
import numpy as np
from scripts.build_aorta_inlet_picker import sha
from scripts.diagnose_phantom_signal import ROOT
from scripts.prepare_phantom_encoding import BASE


def validate(output):
    if output.exists():raise FileExistsError(output)
    checks=[];hashes={};frozen_bindings=[]
    snapshots=list(BASE.glob('*_frozen_v*.*'))
    def digest(path):
        path=Path(path).resolve()
        if path not in hashes:hashes[path]=sha(path)
        return hashes[path]
    def check(name,passed):
        if not passed:raise AssertionError(name)
        checks.append(name)
    def close(name,a,b,tol=2e-7):
        check(name,np.allclose(a,b,atol=tol,rtol=2e-6))
    def identity(path,expected,label):
        path=Path(path)
        if not path.is_absolute():path=ROOT/path
        if path.exists() and digest(path)==expected:
            check(label,True);return
        # Explicitly retained versions, not weaker substitute data. Current
        # orchestration sources may evolve; exact historical source is required.
        matches=[p for p in snapshots if digest(p)==expected]
        check(label+':exact frozen version',len(matches)>0)
        frozen_bindings.append(dict(original=str(path),snapshot=str(matches[0]),sha256=expected))
    def entries(data,parent,label):
        for key,value in data.items():
            path=Path(key)
            if not path.is_absolute():path=parent/path
            identity(path,value['sha256'] if isinstance(value,dict) else value,label+':'+key)
    # Preserve all scientific E048 stages; intentionally updated governance is
    # not asserted equal to a historical whole-delivery document snapshot.
    previous=ROOT/'outputs/phantom_diagnosis_e048'
    for name in ('basic_v1','reference_v1','local_v1','kernel_v2','figures_v1'):
        folder=previous/name;manifest=json.loads((folder/'MANIFEST.json').read_text())
        entries(manifest.get('inputs',{}),ROOT,'E048:'+name+':input')
        entries(manifest['outputs'],folder,'E048:'+name+':output')
        reg=folder/'registration.json'
        if reg.exists():
            r=json.loads(reg.read_text());entries(r.get('inputs',r.get('input_sha256',{})),ROOT,'E048:'+name+':registration')
    legacy=ROOT/'outputs/aorta_inlet_replenishment/e046_legacy_snapshot_v1/registration.json'
    source=json.loads(legacy.read_text())['inputs']
    check('39 preserved legacy identities declared',len(source)==39)
    entries(source,ROOT,'legacy')
    for folder in sorted(BASE.iterdir()):
        if not folder.is_dir() or '.building' in folder.name:continue
        manifest=folder/'MANIFEST.json'
        if manifest.exists():
            m=json.loads(manifest.read_text())
            if 'outputs' in m:
                entries(m.get('inputs',{}),ROOT,folder.name+':input');entries(m['outputs'],folder,folder.name+':output')
            else:entries(m,folder,folder.name+':files')
        registration=folder/'registration.json'
        if registration.exists():
            r=json.loads(registration.read_text());entries(r.get('inputs',{}),ROOT,folder.name+':registration')
    physical=[]
    for input_name in ('inputs_v2','inputs_v3','inputs_v4'):
        with h5py.File(BASE/input_name/'controls.h5') as f:
            physical.append({k:f[k][:] for k in ('time_s','adc_time_s')})
    for p in physical[1:]:
        for k in physical[0]:check('identical physical controls '+k,np.array_equal(physical[0][k],p[k]))
    for analysis,cpu,input_name in [('analysis_v1','cpu_v3','inputs_v3'),('analysis_production_v1','cpu_production_v1','inputs_v4')]:
        a=json.loads((BASE/analysis/'report.json').read_text())
        c=tomllib.loads((BASE/cpu/'report.toml').read_text())
        identity(BASE/cpu/'signals.h5',c['signals_sha256'],cpu+':signals')
        identity(BASE/input_name/'controls.h5',c['input_sha256'],cpu+':inputs')
        identity(ROOT/'koma_migration_e046/encoding_controls_e049.jl',c['script_sha256'],cpu+':code')
        with h5py.File(BASE/cpu/'signals.h5') as f:
            for i,row in enumerate(a['cases']):
                reference=f[f'case_{i:03}/REF'][:]
                value=[]
                for e in ('X','Y','Z'):
                    product=f[f'case_{i:03}/'+e][200]*reference[200].conj()
                    value.append(float(np.arctan2(product.imag,product.real)*5/np.pi))
                close(analysis+':raw velocity:'+row['case'],value,row['Koma_velocity_m_s'])
                error=max(abs(np.array(value)-row['gradient_history_velocity_m_s']))
                close(analysis+':phase error:'+row['case'],error,row['maximum_gradient_history_error_m_s'])
                gate=next(g for g in a['gates'] if g['case']==row['case'])
                check(analysis+':registered numerical gate:'+row['case'],gate['passed'])
    clock_reports={}
    for name in ('clock_v1','clock_v2','clock_gpu_v1','clock_inrun_cpu_v1','clock_inrun_gpu_v1'):
        r=tomllib.loads((BASE/name/'report.toml').read_text());clock_reports[name]=r
        identity(BASE/name/'clock_signals.h5',r['signals_sha256'],name+':signals')
        identity(ROOT/'koma_migration_e046/test_encoding_clock_e049.jl',r['script_sha256'],name+':code')
        with h5py.File(BASE/name/'clock_signals.h5') as f:
            for row in r['rows']:
                key=f"{row['case']}_{row['mode']}_{row['cycles']}"
                product=f[key+'/X'][200]*f[key+'/REF'][200].conj()
                v=float(np.arctan2(product.imag,product.real)*5/np.pi)
                close(name+':raw phase:'+key,v,row['velocity_m_s'])
    cpu=clock_reports['clock_inrun_cpu_v1'];gpu=clock_reports['clock_inrun_gpu_v1']
    lookup=lambda r,case,mode,cycle:next(x for x in r['rows'] if x['case']==case and x['mode']==mode and x['cycles']==cycle)
    late=lookup(cpu,'constant1p5','production32',2047)
    check('in-run late clock lies within saved acquisition',0<late['absolute_echo_s']<2072.5765)
    check('production32 known-velocity gate demonstrably FAILS late',abs(late['velocity_m_s']-1.5)>.002)
    for cycle in (0,1024,2047):
        for mode in ('fine64','production64'):
            check('constant f64 gate:'+str((cycle,mode)),abs(lookup(cpu,'constant1p5',mode,cycle)['velocity_m_s']-1.5)<.002)
    for row in gpu['rows']:
        other=lookup(cpu,row['case'],row['mode'],row['cycles'])
        close('CPU/GPU clock agreement:'+str((row['case'],row['mode'],row['cycles'])),row['velocity_m_s'],other['velocity_m_s'],tol=1e-10)
    close('Float32 ULP near late clock',float(np.spacing(np.float32(late['absolute_echo_s']))),.000244140625,tol=1e-12)
    t=json.loads((BASE/'adc_audit_v1/report.json').read_text())
    check('no erroneous extra half-dwell correction',t['apply_extra_half_dwell'] is False)
    check('all122880 saved ADC first-sample timestamps verified',sum(x['profiles'] for x in t['encodings'])==122880)
    result=dict(experiment='E-049',status='PASS_CONTROL_VALIDATION_WITH_CONFIRMED_PRODUCTION32_CLOCK_FAILURE',
        created_utc=datetime.now(timezone.utc).isoformat(),checks=len(checks),check_names=checks,
        distinct_files_hashed=len(hashes),legacy_identities_unchanged=39,
        frozen_source_bindings=frozen_bindings,
        constant_velocity_table=[dict(time_s=lookup(cpu,'constant1p5','production32',c)['absolute_echo_s'],cycle=c,
             production32_m_s=lookup(cpu,'constant1p5','production32',c)['velocity_m_s'],
             production64_m_s=lookup(cpu,'constant1p5','production64',c)['velocity_m_s'],
             fine32_m_s=lookup(cpu,'constant1p5','fine32',c)['velocity_m_s'],fine64_m_s=lookup(cpu,'constant1p5','fine64',c)['velocity_m_s']) for c in (0,1024,2047)],
        no_1m=True,full_phantom_recovery_demonstrated=False,source_sha256=digest(Path(__file__)))
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','checks','distinct_files_hashed','constant_velocity_table')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    validate(p.parse_args().output)
