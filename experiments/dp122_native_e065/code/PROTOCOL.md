# E-065 / D-068 — dp122 CFD (peak inflow matched to the 4D-flow scan)

Registered 2026-10-26 by user decision ("go ahead") after the CFD inventory. Root `D:/MRSim/outputs/dp122_native_e065`.

**Input.** Fluent dp122 (`peak_velocity` 3.9048: inflow peak 120 mL/s, mean 54.5 mL/s; scan ~118 / ~50 mL/s), copied
read-only from `E:/CFD Phantom Data/New DataSets/phantom/0.8 M/complete/dp122` to
`D:/MRSim/external_inputs_e065/dp122_fluent` (223 files, all SHA-256 identical; list `dp122_source_sha256.txt`).

**Stage S0 (CPU only, this registration).** Convert run FFF.1-28 frames 0–72 with an E-065 copy of the frozen dp150
converter (only file names changed); compute section flow waveforms (inlet, 30 mm upstream, throat, outlet); report
peak, mean, timing and cycle closure (first vs last frame). No tracking or Koma.

**Known limits.** Same inflow shape and 0.72 s (effective 0.7225 s) cycle as dp150, not the scan's 1.012 s; likely a
single saved cycle. Any retiming must be stated. GPU stages (tracking, Koma) need a cost estimate and user approval.
