# E-054: accelerating/sheared multi-spin encoding controls

User authorization: small acceleration/shear controls, followed by a corrected
300k confirmation, followed by 1M only if coverage and local errors improve.
This registration reserves E-054 and D-057 outside the frozen migration tree.
Canonical governance amendments are queued until E-053 retirement, so the
every-file migration manifest is not invalidated. No prior artifact is edited.

Question: does the pinned Float64 implementation reproduce the complex signal
and encoded velocity of prescribed accelerating/sheared spin populations?
Evidence class: controlled numerical verification, not independent physical
validation, full-volume stenosis accuracy, inlet/wall validation or convergence.

Use the original phantom REF/X/Y/Z Pulseq files, 3T, VENC 5 m/s. Extract the first
physical RF-to-ADC shot, with its actual 400 dwell-centered samples. Each model
has 32 equal-volume spins on a predefined 4 x 4 x 2 subvoxel lattice centered at
the origin. Weights sum to one. T1=1.3s, T2=0.1s, off-resonance=0.

Prescribe positions relative to the actual echo with
x=x0+(v0+s*y0)*tau+0.5*a*tau^2; y=y0; z=z0.
The shear mapping preserves volume exactly and does not need particle deletion,
inlet recycling or trajectory wrapping during the shot. All eight predefined
models remain in the result: static, +1.5 and -0.75 m/s uniform flow,
+/-200 m/s^2 axial acceleration at 1.5 m/s, +/-1000 s^-1 transverse shear at
1.5 m/s, and +200 m/s^2 with +1000 s^-1 shear at 1.5 m/s.

Sample paths every <=0.25us. Koma FlowPath uses those same piecewise-linear
positions. Run CPU Float64 with fine gradient/RF stepping (1us/0.5us), and with
the production gradient/RF stepping (Inf/50us). Also run production stepping at
the in-run late absolute clock 2047*saved_RR. Late coordinates use the same
relative trajectories at the shifted absolute time, not a retimed CFD curve.

Independent expectation: original E-049 Python finite-RF RK4, then independent
free-precession integration, on every individual spin. Sum with the physical
weights in complex signal space before deriving phase. Confirm physical
gradients against independently parsed Pulseq trapezoids and report that the
RF waveform is shared input, not an independently predicted excitation.

Gates fixed before any candidate signal:

- All 400 samples finite, all eight models and REF/X/Y/Z present.
- At the predefined central ADC sample, all encoded and reference signals
  exceed 5% of their respective readout maxima. A failure is retained, not
  repaired by choosing another sample or lowering the threshold.
- Fine Koma versus independent complex signal relative L2 <=0.003 per model
  and encoding; central phase-derived component error <=0.002 m/s.
- Production-step central component error versus independent <=0.005 m/s.
- Late versus early production central component change <=0.002 m/s.
- Weight sum within 1e-12 of one; prescribed transverse coordinates invariant
  within 1e-15m; no resets, wraps or hidden density renormalization.

Report all errors versus the independent encoded expectation AND the separate
instantaneous arithmetic echo mean. A real encoding-response discrepancy is
not relabeled a numerical failure. Preserve unfavorable models and failed runs.
No gain/time/phase fitting, threshold relaxation or phase filling is permitted.

Passing permits the separately preregistered 300k confirmation to proceed after
resource/identity/preflight checks. It does not certify repeated-RF magnitude,
full 3D image formation, sparse sampling, inlet supply or throat velocity.
An additional image-formation control remains necessary if these diagnostics
identify unresolved signal or encoding behavior. A failed gate stops scale-up.

1M remains conditional: use the same grid/sequence/timing/mask/CFD source and
compare local errors on identical common support and flow on identical areas,
with each arm's own coverage reported separately. Improvement must exceed
small-pilot sampling variability and not worsen relevant flow agreement.
