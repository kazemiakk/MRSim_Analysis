"""Deterministic equal-volume known motion; no CFD source substitution."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import tomllib
import h5py
import numpy as np

def digest(path):
    with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main(run):
    output=run/'inputs_v1';stage=run/'inputs_v1.building'
    if output.exists() or stage.exists():raise FileExistsError(output)
    # Sequence preparation supplies independently auditable actual timing.
    meta=tomllib.loads((run/'sequences_v1/report.toml').read_text())
    baseline_echo=float(meta['baseline_echo_s'])
    period=float(meta['rf_period_s'])
    assert abs(period-1.012/15)<1e-9
    radius=.004579049785927147
    x=(np.arange(16)+.5-8)*.000625
    y=(np.arange(16)+.5-8)*.000625
    z=(np.arange(8)+.5-4)*.00125
    positions=np.stack(np.meshgrid(x,y,z,indexing='ij'),axis=-1).reshape(-1,3)
    positions=positions[np.sum(positions[:,1:]**2,axis=1)<radius**2]
    n=len(positions)
    omega=2*np.pi/period;phi=-np.pi/4
    u=dict(static=np.zeros(n),uniform_osc=np.full(n,2.2),
        parabolic_osc=2.2*(1-np.sum(positions[:,1:]**2,axis=1)/radius**2))
    times=np.linspace(0,period,4097)
    basis=(np.sin(omega*(times-baseline_echo)+phi)-np.sin(phi))/omega
    basis[-1]=basis[0]
    extreme=positions[:,0,None]+u['uniform_osc'][:,None]*basis
    assert np.max(abs(extreme))<.05
    stage.mkdir()
    with h5py.File(stage/'controls.h5','w') as f:
        for name,value in dict(positions=positions,rho=np.full(n,1/n),T1=np.full(n,1.3),T2=np.full(n,.1),
            omega_s_inv=omega,phi_rad=phi,echo_baseline_s=baseline_echo,motion_period_s=period,
            path_time_s=times,path_basis_s=basis,radius_m=radius,length_m=.01).items():f[name]=value
        for name,value in u.items():f['u_'+name]=value
    registration=dict(experiment='E-056',status='PASS_DETERMINISTIC_CONTROL_INPUTS',
        created_utc=datetime.now(timezone.utc).isoformat(),physical_particles=n,
        evidence_class='Analytical synthetic acquisition-physics mechanism control, not CFD validation',
        input_sha256=digest(stage/'controls.h5'),script_sha256=digest(Path(__file__)),
        protocol_sha256=digest(run/'code/PROTOCOL.md'),sequence_report_sha256=digest(run/'sequences_v1/report.toml'),
        motion_period_s=period,baseline_echo_s=baseline_echo,radius_m=radius,
        position_extent_m=np.max(abs(extreme)),models=list(u),new_1m_started=False)
    (stage/'registration.json').write_text(json.dumps(registration,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    stage.rename(output)
    print(json.dumps(registration),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    main(p.parse_args().run)
