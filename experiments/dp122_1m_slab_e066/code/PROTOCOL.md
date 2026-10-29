# E-066 / D-069 — dp122 1M slab-restricted acquisition (spin-density test of the peak-velocity overshoot)

Registered 2026-10-27 by user decision ("Slab-restricted 1M (Recommended)"); this is the user's explicit decision on
the held 1M question. Root `D:/MRSim/outputs/dp122_1m_slab_e066`.

**Hypothesis (from E-065 extended analysis).** The dp122 peak-velocity overshoot (Koma 2.87 vs CFD 1.91 m/s) arises
in shear-layer voxels holding ~1 spin. With ~3.3x the spin density the overshoot (T2 max error, p99 error) shrinks.

**Spins.** All 1M dp122 trajectories (E-065 tracking) whose |S| (scanner frame, E-060 rotation) is <= 60 mm at any
of the 1440 steps (~41%). Same layout as E-055/E-065 (sentinel, rho 1, T1 1.3, T2 0.1, native period Float32(0.72));
cycle map recomputed within the subset with the same upstream mapping module (as E-055 did for its subset).

**Gate G1 (restriction identity, before the 1M run).** The E-065 300k phantom versus the same 300k restricted by the
same rule: REF and Z over a 600-profile prefix on GPU1 (h* 16 us, E-065 numerics). Pass: relative L2 <= 1e-3 for both.

**Then.** E-065 runner copy with the particle count generalized; prefix (step ladder, cross-GPU identity, estimate;
stop if > 12 GPU-hours); acquisition REF+X GPU1, Y+Z GPU0; E-065 analysis and extended analysis on the new images
against the same E-065 references. Primary readouts: T2 max and p99 error, error-voxel spin counts, flow T1, centreline.
