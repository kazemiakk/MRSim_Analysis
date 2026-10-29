"""Read-only phase/reference diagnosis; no empirical correction is applied."""
from pathlib import Path
import numpy as np,h5py
from analyze_physics import load_group,predict,complex_array
run=Path('D:/MRSim/outputs/acquisition_physics_e056')
with h5py.File(run/'inputs_v1/controls.h5') as f:
    positions=f['positions'][:];rho=f['rho'][:];u=f['u_parabolic_osc'][:]
    t1=float(f['T1'][0]);t2=float(f['T2'][0]);times=f['path_time_s'][:];factor=f['path_basis_s'][:];period=float(f['motion_period_s'][()])
with h5py.File(run/'sequences_v1/waveforms.h5') as f:
    g=f['baseline/REF'];m=load_group(g,80,16,32);m['gamma']=float(f['gamma_Hz_T'][()])
    print('RF phases',m['rf_phase_rad'][:25],flush=True)
    print('ADC compensation phase',np.angle(m['compensation'][:15,40]),flush=True)
    print('RF start',m['rf_start_s'][:3],flush=True)
    print('templateB1absphase',np.max(abs(m['templates'][0]['b1'])),np.angle(m['templates'][0]['b1'][np.argmax(abs(m['templates'][0]['b1']))]),flush=True)
    print('psi',g['shots/0000/rf_phase_rad_at_knots'][:].min(),g['shots/0000/rf_phase_rad_at_knots'][:].max(),flush=True)
    for name in ['labels','compensation','profile_rf_index0','adc_first_s','dwell_s']:m[name]=m[name][:15]
    last=int(m['profile_rf_index0'][-1])+1
    for name in ['rf_start_s','rf_scale','rf_phase_rad']:m[name]=m[name][:last]
    prediction,_=predict(m,positions,rho,u,t1,t2,times,factor,period,(80,16,32,15),require_complete=False)
with h5py.File(run/'prefix_v1/parabolic_osc_baseline_ref_fine.h5') as f:actual=complex_array(f['raw_signal_f64']).ravel()
for i in range(15):
    a=actual.reshape(-1,80)[i];b=prediction.reshape(-1,80)[i]
    ids=np.argsort(abs(b))[-3:]
    print(i,'actual',a[ids], 'prediction',b[ids], 'diagnostic ratio',a[ids]/b[ids],flush=True)
