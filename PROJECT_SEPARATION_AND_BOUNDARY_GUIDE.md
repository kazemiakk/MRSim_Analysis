# Master Project Separation & Boundary Architecture Guide

**Generated Date:** October 30, 2026  
**Workspace:** `C:\Users\lambda-3\Desktop\MRSim\`

---

## 1. Executive Forensic Summary: How the Two Projects Intersected

A forensic review of the workstation directory structure, Git commit histories, and LLM chat archives revealed that two fundamentally distinct biomedical research projects were co-located in the same root workspace (`C:\Users\lambda-3\Desktop\MRSim\`):

1. **Project 1: Cerebral Aneurysm CFD & Virtual Angiography (Patient TB)**
2. **Project 2: KomaMRI / MRSimTracks (Stenotic Phantom & Aortic Arch PC-MRI)**

### Why They Became Conflated in the Chat History:
* **Shared Physical Directory:** The workspace folder is named `MRSim`. However, inside `MRSim`, Ansys Fluent CFD meshes, scripts, and videos for the cerebral aneurysm were saved alongside the `MRSIM` subfolder containing the KomaMRI repositories.
* **Chronological Chat Evolution in Codex:**
  * **April – August 2026:** Codex was used extensively for **KomaMRI / MRSimTracks** (214 sessions archived in `MRSIM/codex_history/`, summarized in `MRSimTracks_CHAT_HISTORY_TECHNICAL_SUMMARY.md`).
  * **September – October 2026:** Codex was used in the same workspace to build the **Cerebral Aneurysm CFD** models, inflow waveforms, particle tracking, and virtual angiography.
* **Antigravity / Gemini Chat Handoff:**
  * When this Antigravity session began with the prompt: *"can you see my chat with codex and continue from there? what is the status?"*, the agent loaded recent session context in `MRSim`.
  * Later, when the user asked to push changes to GitHub (*"Push the updated code/scripts in MRSim_Analysis (kazemiakk/MRSim_Analysis)..."*), the agent almost pushed Aneurysm CFD scripts to the `MRSim_Analysis` repo, triggering an intentional repository boundary conflict because `MRSim_Analysis` had a commit (`12e169a`) explicitly barring aneurysm data.

---

## 2. Definitive Separation Matrix

| Feature | Project 1: Cerebral Aneurysm CFD | Project 2: KomaMRI / MRSimTracks |
| :--- | :--- | :--- |
| **Primary Clinical Focus** | Patient TB Anterior Communicating Artery (ACom) aneurysm | 3D printed stenotic flow phantom (`dp150`/`dp101`) & human aortic arch (`dp2`) |
| **Research Question** | How do inflow boundary waveforms impact intra-sac hemodynamics, multi-cycle particle trapping, and virtual fluoroscopy/DSA? | How do spatial/temporal resolution, VENC, and acceleration affect 4D Flow Phase-Contrast MRI velocity reconstruction? |
| **Primary Physics** | Incompressible Navier–Stokes, passive scalar advection, Beer–Lambert X-ray line-integral attenuation | Lagrangian particle trajectories, Bloch equations, RF excitation, phase-contrast gradient encoding (REF/X/Y/Z) |
| **Software Stack** | ANSYS Fluent (v261), Python (`dye_bolus.py`, `bilateral_10cycle_angiography.py`), PyVista, Matplotlib, FFmpeg | Julia (`KomaMRI.jl`), Python (`MRSimTracks` package, RK4 particle tracking, HDF5, phase unwrapping) |
| **Primary Directories** | • `C:\Users\lambda-3\Desktop\MRSim\20261029-ALL-FIGURES-AND-VIDEOS\`<br>• `D:\FluentAgent\results\`<br>• `C:\Users\lambda-3\Desktop\MRSim\cfd-extended-20261029-v1\`<br>• `tb-cfd*`, `aneurysm-sac-v1`, `cleaned-rim-outlet-extension-v1` | • `C:\Users\lambda-3\Desktop\MRSim\MRSIM\MRSim_Analysis`<br>• `C:\Users\lambda-3\Desktop\MRSim\MRSIM\MRSimTracks`<br>• `C:\Users\lambda-3\Desktop\MRSim\MRSIM\codex_history\`<br>• `C:\Users\lambda-3\Desktop\MRSim\github_sync_mrsim_20261029_v1\` |
| **Git Repositories** | **None (Local only).** Do NOT push to GitHub. | • `https://github.com/kazemiakk/MRSim_Analysis`<br>• `https://github.com/mcgrathcm/MRSimTracks` |
| **Key Master Documents** | • `00_FINAL_PRESENTATION_DECK.html` (16 slides)<br>• `00_FINAL_PRESENTATION_SPEAKER_GUIDE.md`<br>• `20261029-FINAL-CFD-AND-ANGIOGRAPHY-SUMMARY.md` | • `MRSimTracks_CHAT_HISTORY_TECHNICAL_SUMMARY.md`<br>• `DP150_MRI_PIPELINE.md`<br>• `AGENTS.md` (Repository Boundary) |

---

## 3. Physical Clean-up & Replacement Actions Taken

To completely eliminate cross-contamination between the two projects on disk:

1. **Relocated Loose Legacy Aneurysm Files from Koma Repositories:**
   * Found two untracked exploratory files from July 9, 2026 residing in the Koma folders:
     * `C:\Users\lambda-3\Desktop\MRSim\MRSIM\MRSim_Analysis\aneurysm_tracks_high.h5`
     * `C:\Users\lambda-3\Desktop\MRSim\MRSIM\MRSimTracks\aneurysm_caps.vtp`
   * **Action Taken:** Both files were cleanly moved to:
     `C:\Users\lambda-3\Desktop\MRSim\20261029-ALL-FIGURES-AND-VIDEOS\previous-two-weeks-archive\legacy-july-aneurysm-exploratory\`
   * **Verification:** Re-scanned both `MRSIM/MRSim_Analysis` and `MRSIM/MRSimTracks`. Both repositories now contain **0 aneurysm files**. `MRSim_Analysis` working tree is clean.

2. **Verified Aneurysm Presentation & Media Archives:**
   * Scanned `C:\Users\lambda-3\Desktop\MRSim\20261029-ALL-FIGURES-AND-VIDEOS\` and `D:\FluentAgent\results\`.
   * **Verification:** Confirmed **0 Koma / phantom / aorta files** exist inside the Aneurysm archive. All 345 files and 16 presentation slides strictly belong to the cerebral aneurysm project.

---

## 4. Operational Protocol for Future Chat Sessions

To ensure future AI coding assistants (Antigravity, Codex, or Claude) never mix the two projects again:

### When Working on Project 1 (Aneurysm CFD & Angiography):
* **Workspace / Subdirectory:** Direct the agent to `C:\Users\lambda-3\Desktop\MRSim\20261029-ALL-FIGURES-AND-VIDEOS\` or `D:\FluentAgent\`.
* **Prompt Anchor:** *"I am working on the Cerebral Aneurysm CFD project (Patient TB, ANSYS Fluent, Virtual Angiography, presentation deck). Do not touch or commit to MRSim_Analysis or MRSimTracks."*
* **Master Output:** Refer to `00_FINAL_PRESENTATION_DECK.html` and `20261029-FINAL-CFD-AND-ANGIOGRAPHY-SUMMARY.md`.

### When Working on Project 2 (KomaMRI / Aorta & Phantom PC-MRI):
* **Workspace / Subdirectory:** Direct the agent explicitly into `C:\Users\lambda-3\Desktop\MRSim\MRSIM\MRSim_Analysis\` or `C:\Users\lambda-3\Desktop\MRSim\MRSIM\MRSimTracks\`.
* **Prompt Anchor:** *"I am working on the KomaMRI / MRSimTracks project (Stenotic Phantom dp150 and Aorta dp2, Julia PC-MRI). Follow the rules in AGENTS.md. Exclude all cerebral aneurysm data."*
* **Master Output:** Refer to `MRSimTracks_CHAT_HISTORY_TECHNICAL_SUMMARY.md` and `DP150_MRI_PIPELINE.md`.

