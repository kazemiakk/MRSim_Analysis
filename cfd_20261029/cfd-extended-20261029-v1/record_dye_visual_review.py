"""Record the completed encoded-frame review and concise viewing qualifications."""
from pathlib import Path
import json,sys
sys.path.insert(0,r'D:\FluentAgent\automation\20261029-extended-hemodynamics-v1')
from extended_common import PROJECT,load,save
out=PROJECT/'results/20261029-virtual-angiography-connected-250um-v1';folder=out/'visualization'
save(folder/'visual-review-completed.json',{'status':'share_with_caveats_qualitative_preview',
    'reviewed':'Four frames extracted from each actual MP4, plus full-size source frame',
    'sampled_frames':[28,55,110,239],
    'observed':['Dye starts in the selected right or left ICA, not the opposite inlet',
        'Volumetric plume changes across time; it is not a static wall contour',
        'The original gray vessel boundary remains unchanged in both videos',
        'Projection, time label and curve cursor progress together',
        'Labels, colorbars and curve legend are visible without clipping'],
    'full_decoding':'Both 240-frame MP4s decoded successfully',
    'not_established':['Clinical imaging agreement','Full-flow periodicity','Transport mesh/time convergence','Validated clearance time']})
with (out/'VIEWING-NOTES.md').open('x',encoding='utf-8') as f:
    f.write('''# Virtual dye videos

- [Right ICA / B1 injection](visualization/right-ICA-virtual-dye.mp4)
- [Left ICA / B2 injection](visualization/left-ICA-virtual-dye.mp4)

These are two independent 100-ms injections, not a simultaneous bilateral injection.
Both concentration curves are shown for comparison; the title identifies the injection displayed.
Each video shows 3D volumetric dye, a grayscale model-Y projection and the approved exploratory
ellipsoidal sac ROI's mean concentration. C is normalized to the injected inlet concentration.
Ten seconds of playback covers 863 ms of stored flow (solver time 5–868 ms).

The original geometry and blood-flow results are unchanged. For speed, the dye preview uses
181,612 connected scalar groups in 0.25-mm bins, preserving the original fluid domain and never
joining wall-separated passages. The original blood-flow mesh still has 1,169,992 tetrahedra.
The display grid is 0.4 mm; both movies use identical display scales and fixed projection gamma=0.5.

QUALITATIVE PREVIEW: early native-cell comparisons found approximately 13–20% concentration-field
differences at 180 ms after injection. The initial half-time-step check found about 12% differences.
These are not converged clinical contrast, clearance-time or residence-time measurements.
The underlying flow is one first cycle with periodicity unvalidated. The ROI is not a validated neck.

Faint dye can disappear visually below the rendering scale while nonzero concentration remains.
Do not interpret an apparently empty final image as complete clearance.

Both encoded MP4s were fully decoded and representative frames visually inspected.
Detailed protocol, numerical accounting and limitations: [README](README.md).
No original CFD inputs were changed and no patient data were uploaded.
''')
print('VISUAL_REVIEW_RECORDED',flush=True)
