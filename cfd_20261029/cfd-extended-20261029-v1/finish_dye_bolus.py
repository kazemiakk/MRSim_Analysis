"""Verify stored dye fields and document scope without claiming clinical validation."""
from pathlib import Path
import argparse,csv,json,sys
import numpy as np
sys.path.insert(0,r'D:\FluentAgent\automation\20261029-extended-hemodynamics-v1')
from extended_common import *


def main(out,fine_path):
    result=load(out/'result.json');assert result['status']=='available_first_cycle_transport_complete' and result['error'] is None
    frames=load(out/'frames.json');rows=list(csv.DictReader((out/'transport-history.csv').open()))
    mapping=np.load(PREVIOUS/'comparisons/fluid-ROI-mapping.npz');v=mapping['volume_m3'];roi=mapping['ellipsoid_mask']
    minimum=1.;maximum=0.;superposition=0.;max_storage_relative_error=0.
    for frame in frames:
        with np.load(out/frame['file']) as field:
            for side in ['right','left']:
                c=field[side];assert c.shape==v.shape and np.isfinite(c).all()
                minimum=min(minimum,float(c.min()));maximum=max(maximum,float(c.max()))
                if frame['time_s']>frames[0]['time_s']:
                    r=next(r for r in rows if r['injection']==side and abs(float(r['solver_time_s'])-frame['time_s'])<1e-10)
                    actual=float(np.dot(v,c))*1e6;expected=float(r['whole_domain_label_volume_ml'])
                    err=abs(actual-expected)/max(float(r['injected_label_volume_ml']),1e-30)
                    max_storage_relative_error=max(max_storage_relative_error,err)
                    assert err<1e-6
            superposition=max(superposition,float((field['right']+field['left']).max()))
    assert minimum>=-1e-6 and maximum<=1.000001 and superposition<=1.000001
    summary=[]
    for side in ['right','left']:
        subset=[r for r in rows if r['injection']==side];last=subset[-1]
        peak=max(subset,key=lambda r:float(r['ellipsoidal_ROI_mean_C']))
        summary.append({'injection':side,'inlet_zone':'inlet_b1' if side=='right' else 'inlet_b2',
            'injected_virtual_label_volume_ml':float(last['injected_label_volume_ml']),
            'max_raw_relative_mass_defect':max(float(r['relative_mass_balance_error']) for r in subset),
            'max_corrected_ledger_relative_error':max(float(r['corrected_ledger_relative_error']) for r in subset),
            'peak_ellipsoidal_ROI_mean_C':float(peak['ellipsoidal_ROI_mean_C']),
            'peak_time_after_injection_s':float(peak['time_since_injection_s']),
            'scope':'Qualitative first-cycle virtual dye, not validated clinical clearance or residence time'})
    with (out/'injection-summary.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    fine=fine_path/'fields/C-t0010000us.npz'
    sensitivity=[]
    with np.load(fine) as a,np.load(out/'fields/C-t0010000us.npz') as b:
        for side in ['right','left']:
            error=float(np.sqrt(np.dot(v,(a[side]-b[side])**2)/np.dot(v,a[side]**2)))
            sensitivity.append({'injection':side,'elapsed_ms':5,'coarse_step_ms':5,'fine_step_ms':2.5,'volume_weighted_relative_C_L2_difference':error})
    save(out/'initial-time-step-sensitivity.json',{'tests':sensitivity,
        'meaning':'One initial-time comparison, NOT a convergence study. Material sensitivity prohibits validated quantitative clearance claims.'})
    save(out/'stored-field-checks.json',{'status':'passed_scoped_checks','frames_checked':len(frames),'concentration_min':minimum,
        'concentration_max':maximum,'maximum_right_plus_left_C':superposition,
        'max_saved_float32_volume_error_relative_to_injected_label':max_storage_relative_error,
        'source_hashes_unchanged':result['original_sources_unchanged'],
        'constant_field_check':load(out/'actual-scalar-grid-constant-check.json'),
        'not_validated':['Flow periodicity','Scalar timestep/mesh independence','Clinical contrast physics','Clinical clearance or residence time']})
    with (out/'README.md').open('x',encoding='utf-8') as f:
        f.write('''# Right- and left-ICA virtual dye injection

Open the two videos in `visualization/`:

- `right-ICA-virtual-dye.mp4`: 100-ms injection through B1 / Right ICA only.
- `left-ICA-virtual-dye.mp4`: 100-ms injection through B2 / Left ICA only.

Each video shows 3D volumetric virtual dye, a grayscale angiography-style projection and the
ellipsoidal sac-ROI concentration curves. They use common concentration and projection scales.
Ten seconds of playback shows the 863 ms covered by existing CFD states (solver time 5–868 ms).
Frames are interpolated between saved scalar fields for smooth playback, not additional CFD solutions.

## What was simulated

A dimensionless passive numerical tracer has C=0 initially, C=1 at the selected ICA for 100 ms,
then C=0. All other incoming boundaries have C=0. Right and left are independent experiments,
using the same unmodified patient-specific CFD. No new Fluent/Navier–Stokes session, blood material,
inflow waveform, outlet fraction, vessel geometry or wall condition was changed. The pulse is a
controlled virtual experiment, not a measured catheter injection, iodine dose or MRI acquisition.

The original flow has 1,169,992 native fluid cells. This QUALITATIVE dye preview uses 181,612
connected scalar agglomerates in 0.25-mm centroid bins, not just the vessel wall. Every group is a
union of face-connected native fluid cells. Separate wall-separated passages are never merged;
the exact original fluid domain, total volume and CFD mesh are preserved. Directed native fluxes
across group boundaries are retained, including opposing flows rather than canceling them first.
Group concentrations are mapped back to native cells for storage. This is NOT a claim that one
independent scalar equation was solved for every original tetrahedron. Coarsening adds numerical
diffusion. The movie uses a separate 0.4-mm display grid; no anatomical surface is replaced by voxels.
The grayscale view integrates concentration along model-coordinate Y (units C·mm), not calibrated x-ray
attenuation. Dye colors and opacity are fixed across both injections. No frame-wise rescaling.
The projection uses fixed display gamma=0.5 to make dilute dye visible, and the 3D palette starts
at a visible light orange rather than white. These are shared display settings, not altered data,
an iodine attenuation model or independent normalization of each injection.

## Numerical limitations that affect interpretation

The initial strictly conservative dye operator failed its concentration bound at 60 ms. A uniform
C=1 test on the problematic native flux interval changed locally by up to about 7.7%, exposing
local discrete carrier-continuity defects. Failed tests remain in separate folders; no clipping
or changes to the original CFD were used to hide this.

The final scheme solves bounded concentration advection with the native continuity term explicitly
accounted: m dC/dt + sum(F C_upwind) – C sum(F)=0. It preserves a constant concentration and leaves
the original flow untouched. It is NOT claimed to be the original strictly conservative scalar solve.
Both the raw label-mass defect and the inherited C div(F) ledger are reported at every scalar step.
The raw cumulative defect must stay below 0.1% of injected label and the corrected ledger error
below 1e-5 relative. The observed maxima are in `injection-summary.csv`; no concentration clipping.

Backward Euler and first-order upwind introduce numerical diffusion. A half-time-step check at
the first saved interval shows material concentration-field sensitivity (see the JSON result).
This is a QUALITATIVE visualization, not transport-converged quantitative washout. Molecular
diffusion, contrast-agent density/viscosity, gravity, injection-induced momentum and scanner response
were not calibrated or modeled. Existing blood-flow periodicity also remains unvalidated.

The sac selection is the previously approved exploratory ellipsoid, not a validated anatomical neck.
RRT/WSS are not dye concentration, and these videos are not clinical angiograms or rupture-risk tools.

## Inspectable records

`protocol.json`: injection and numerical settings.
`transport-history.csv`: concentration bounds, label volumes, inherited defect and solver residuals.
`frames.json` and `fields/`: original-volume-cell scalar fields at saved times.
`source-manifest.json`: original case and 95 data-file hashes; originals were preserved.
`stored-field-checks.json`: stored-field, superposition and constant-preservation checks.
`initial-time-step-sensitivity.json`: limited refinement comparison, not convergence validation.
`visualization/video-manifest.json`: encoded video details.

No simulation or patient data were uploaded. New outputs are on D:.
''')
    print('DYE_HANDOFF_CHECKS_PASSED '+str(out),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--fine-pilot',type=Path,required=True)
    a=p.parse_args();main(a.out,a.fine_pilot)
