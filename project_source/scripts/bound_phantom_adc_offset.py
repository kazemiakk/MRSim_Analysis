"""Bound E048 reference error due to the newly verified half-dwell offset."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import tomllib
import uuid

import h5py
import numpy as np
from scripts.audit_phantom_reference import catmull_weights
from scripts.build_aorta_inlet_picker import sha
from scripts.diagnose_phantom_signal import ROOT,E047
from scripts.dp150_hemodynamic_reference import load_periodic_reference
from scripts.prepare_phantom_encoding import BASE


def bound(output):
    if output.exists():raise FileExistsError(output)
    stage=output.with_name(output.name+'.building-'+uuid.uuid4().hex[:8]);stage.mkdir(parents=True)
    info=tomllib.loads((E047/'phantom_inputs_v1/case_info.toml').read_text())
    timing=E047/'phantom_inputs_v1/legacy_timing.h5'
    source=ROOT/'outputs/dp150_mri_pipeline/flow_periodic/dp150_periodic.pvd'
    registration=dict(experiment='E-049',created_utc=datetime.now(timezone.utc).isoformat(),
        status='REGISTERED_BEFORE_HALF_DWELL_BOUND',inputs={str(p):sha(p) for p in [
            Path(__file__).resolve(),timing,source,ROOT/'outputs/phantom_diagnosis_e048/reference_v1/reference_fields.h5']})
    (stage/'registration.json').write_text(json.dumps(registration,indent=2)+'\n')
    weights=[]
    with h5py.File(timing) as f:
        first=f['REF/adc_first_s'][:];dwell=f['REF/dwell_s'][:];labels=f['REF/labels'][:]
    for b in range(15):
        ids=np.flatnonzero(labels[:,2]==b)
        actual=(first[ids,None]+(np.arange(400)+.5)[None,:]*dwell[ids,None]).ravel()
        phase=np.mod(actual-info['seam_time_s'],info['period_s'])/info['period_s']
        weights.append(catmull_weights(phase))
    weights=np.array(weights)
    with h5py.File(ROOT/'outputs/phantom_diagnosis_e048/reference_v1/reference_fields.h5') as f:
        old=f['adc_weights'][:]
    if not np.allclose(weights.sum(axis=1),1,atol=1e-12):raise AssertionError('Nonunit weights')
    flow=load_periodic_reference(source)
    delta=np.zeros((15,flow.active_mesh.n_points,3),float)
    for i in range(72):
        v=np.asarray(flow._frame_vel(i)).astype(float)*(.72/info['period_s'])
        delta+=(weights[:,i]-old[:,i])[:,None,None]*v[None]
    per_bin=np.max(np.linalg.norm(delta,axis=-1),axis=1)
    # Norm is invariant to the source-to-MRI component permutation. Spatial
    # barycentric interpolation and positive voxel quadrature are convex, so
    # nodal maximum bounds every averaged voxel on fixed support. Triangle
    # inequality then bounds the change of any pooled fixed-support RMSE.
    report=dict(experiment='E-049',status='PASS_HALF_DWELL_REFERENCE_BOUND',
        actual_ADC_offset_s=float(np.median(dwell)/2),
        maximum_voxel_reference_delta_bound_m_s=float(per_bin.max()),
        pooled_RMSE_change_bound_m_s=float(np.sqrt(np.mean(per_bin**2))),
        per_bin_vector_delta_bound_m_s=per_bin.tolist(),
        assumptions='Same conditioned matched mesh, retiming, Catmull-Rom interpolation, fixed support; convex barycentric/positive geometry averaging. No image or time fitting.',
        conclusion='Bound on comparator timing correction, not a newly measured Koma error or full reference-grid rebuild.')
    (stage/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    with h5py.File(stage/'timing_fields.h5','w') as f:
        f['actual_ADC_weights']=weights;f['historical_gate_delay_weights']=old
        f['per_bin_maximum_nodal_vector_delta_m_s']=per_bin
    (stage/'MANIFEST.json').write_text(json.dumps({p.name:sha(p) for p in stage.iterdir() if p.is_file()},indent=2)+'\n')
    stage.rename(output);print(json.dumps(report))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    bound(p.parse_args().output)
