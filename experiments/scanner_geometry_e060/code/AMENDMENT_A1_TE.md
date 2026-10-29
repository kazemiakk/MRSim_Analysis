# E-060 amendment A1 — TE and hardware (registered before any sequence build)

The protocol's stop rule triggered: TE 2.74 ms is infeasible with the extended
builder at in-plane VENC 1.5 m/s. Probes (`probe_te_e060.jl`,
`probe_prephaser_e060.jl`; no signal, no files kept):

| Hardware | RF | Shortest passing prephaser | Minimum TE |
|---|---|---:|---:|
| 40 mT/m, 180 T/m/s | TBP 4, a 0.46 | 1.00 ms | 3.2029 ms |
| 40 mT/m, 180 T/m/s | TBP 1, a 0 | 1.00 ms | 3.1179 ms |
| 45 mT/m, 200 T/m/s | TBP 4, a 0.46 | 0.95 ms | 3.1129 ms |
| 45 mT/m, 200 T/m/s | TBP 1, a 0 | 0.95 ms | 3.0379 ms |

At 0.8 ms prephaser the RO/PE encodings exceed Gmax (merged trailing velocity
lobe plus readout prephaser). The scanner reached 2.74 ms with features this
builder lacks (its protocol allows asymmetric echo).

User decision (2026-10-24): **MAGNETOM Skyra gradient specification, 45 mT/m and
200 T/m/s** (published spec; user may correct), prephaser 0.95 ms, TBP 4 /
Hanning a = 0.46, **minimum TE (≈3.1129 ms)**, reported as the actual TE and never
relabelled 2.74 ms. B1 limit 10 µT unchanged. Everything else per PROTOCOL.md.
The 40 mT/m regression gate still uses the original real-sequence hardware.
This TE is ~0.37 ms longer than the scan's; it is a declared protocol difference.
