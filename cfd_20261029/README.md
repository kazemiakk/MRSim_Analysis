# CFD & Virtual Contrast Angiography Analysis Scripts (2026-10-29)

This directory contains the offline post-processing, hemodynamic analysis, and passive-scalar virtual contrast angiography scripts for the three cerebral aneurysm inflow conditions (`patient_specific`, `individualized`, and `generic`):

## Directories

- `cfd-followup-20261029-v1/`:
  - Cycle-cohort particle tracking recoloring (`render_cycle_cohorts.py`, `render_presentation_copies.py`)
  - Exploratory sac residence and regional velocity/WSS RMSE comparisons (`analyze_latest.py`, `summarize_and_verify.py`)
- `cfd-extended-20261029-v1/`:
  - Extended wall and volumetric hemodynamics: Normalized WSS, Transverse WSS, RRT, Vorticity, Q-criterion, Helicity, Kinetic Energy, Viscous Dissipation, and ROI exchange (`analyze_extended.py`, `weighted_residence.py`)
  - Single-artery (Right ICA / Left ICA) Virtual Contrast Angiography (`dye_bolus.py`, `render_dye_bolus.py`, `render_dye_comparison.py`)
  - Simultaneous Bilateral (Right + Left ICA) 10-Cycle Repeated Virtual Contrast Angiography (`bilateral_10cycle_angiography.py`)
  - Pre-filled Aneurysm Sac Virtual-Dye Washout (`sac_washout_3cases.py`)
