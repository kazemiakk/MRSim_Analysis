# MRSim Analysis: CFD-driven virtual phase-contrast MRI

This repository is for the MRSimTracks CFD -> particle tracking -> KomaMRI project: the stenotic vessel phantom and aortic-arch case, their synthetic phase-contrast MRI acquisitions, reconstruction, comparison, and related figure tools.

**Out of scope:** TB CFD, cerebral-aneurysm/ICA simulations, aneurysm videos, virtual contrast angiography, dye-bolus studies, and their post-processing. Those are separate projects and must not be added here.

## Source layout

| Location | Contents |
| --- | --- |
| `project_source/` | Source snapshot of the locally developed MRSimTracks package, scripts, tests, Julia/Koma workflows, and pinned dependency definitions. Upstream MIT license retained. |
| `experiments/<run>/code/` | Versioned phantom/aorta controls, acquisition, reconstruction, analysis, and presentation code from E054-E058 and E060-E066, plus related presentation workflows. |
| `figure_tools/` | Related magnitude/phase and Bland-Altman figure-layout source versions. |
| `SOURCE_MANIFEST.json` | Relative source locations, exact SHA-256 identities, exclusions, and source-copy checks. |
| `koma/`, `from charles/` | Preserved earlier Koma examples and original supplied sequence helpers, with their existing licenses. These are historical, not automatically the current acquisition. |
| Root phantom scripts | Preserved earlier tracking/visualization entry points. |

The source snapshot is dated **2026-10-29**. Experiment code includes historical versions, controls, sensitivities, and proposed stages. Its presence does not mean an experiment completed or passed, or that a larger run is authorized. Five known syntax-invalid failed plotting attempts remain preserved in the local research workspace and are listed as excluded in the source manifest; they are not presented as working code.

## Workflow and scientific scope

1. Read matched CFD fields, geometry, units, axes, and frame times.
2. Generate particle trajectories and moving-spin phantoms.
3. Simulate separate REF/X/Y/Z encodings in KomaMRI and reconstruct k-space.
4. Compare reconstructed velocity and section flow with CFD using the documented grid, simulated ADC times, and retained support.
5. Report errors together with coverage and limitations.

CFD drives both the simulated motion and its comparator: this is same-source internal consistency, not independent physical validation. Aortic point-cloud tracking remains approximate. The controlled phantom input includes a synthetic final-80-ms transition; interpolated tracking timesteps are not native CFD save intervals. Measured 4D-flow MRI exists for the stenotic phantom, not for the aortic geometry. Any measured-phantom comparison must disclose inflow/protocol differences. Future physical-data calibration is not an implemented vascular digital twin.

## Running the code

This is a **source-code backup**, not a standalone data/environment release. No scientific simulation was rerun for this synchronization. Case scripts retain their original local paths, hash contracts, and environment assumptions; configure the corresponding inputs locally before use. Package definitions are in `project_source/pyproject.toml` and `project_source/uv.lock`; Julia project/manifest files accompany the relevant workflows. Do not upgrade pinned environments merely to make an old script launch.

Copied Python files passed syntax parsing and all published source copies were checked against their original hashes. Scientific, GPU, full-data, and Julia execution tests were not rerun. Consult a run's bound local reports before interpreting results; code alone is not a scientific acceptance certificate.

## Privacy, licensing, and history

Raw DICOM, measured scan arrays, private CFD/geometry/trajectory files, credentials, environments, caches, and generated experiment outputs are excluded. Private scanner-ingestion scripts from E059 and private journals/chat histories are also not part of this source release. Never override data exclusions or bulk-copy the desktop workspace into this repository.

MRSimTracks source originates from [mcgrathcm/MRSimTracks](https://github.com/mcgrathcm/MRSimTracks), base commit `e3c0cae`, plus local research developments. Preserve `project_source/LICENSE` and the MRSimTracks/JuliaHealth license notices with redistributed helpers. No new blanket license is asserted over material without an existing license.

Unrelated aneurysm/angiography code was removed from the current branch in this cleanup. Earlier Git commits still contain that previous material; history was not rewritten, and local originals were not deleted.
