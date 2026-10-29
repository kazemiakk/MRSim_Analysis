# MRSim_Analysis project boundary

This repository is only the stenotic-phantom and aortic-arch CFD -> particle tracking -> KomaMRI phase-contrast MRI project.

- Never add the separate TB CFD, cerebral aneurysm/ICA, aneurysm video, angiography, bolus, or dye-washout projects.
- Do not bulk-copy the desktop workspace. Use reviewed source-only paths and inspect staged changes before publishing.
- Never commit private DICOM/scan arrays, CFD/geometry/trajectories, credentials, chat histories, or installed environments. Do not override `.gitignore` for data.
- Preserve existing upstream licenses. Do not attribute files to a particular agent unless supported by an actual record.
- Completed source versions and scientific artifacts are immutable. Preserve failed attempts locally; use a new version/output for changes.
- Keep CFD save cadence distinct from interpolated integration timestep, separate REF/X/Y/Z distinct from Hadamard encoding, and display derivatives distinct from quantitative data.
- Compare matched times, grids, and retained regions; report error with coverage. Same-source CFD comparisons are internal consistency, not independent validation.
- Do not imply measured aortic MRI exists or that the future digital-twin calibration loop is implemented.
- A source backup does not certify a scientific gate, experimental completion, or authorization to rerun GPU jobs. Obtain project-owner direction before failed scientific gates, new registered metrics, or expensive runs.
- Check `SOURCE_MANIFEST.json` and the per-run code first. Do not use the old root examples as the latest protocol by default.
