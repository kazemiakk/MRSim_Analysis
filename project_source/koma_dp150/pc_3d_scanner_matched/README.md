# Scanner-matched dp150 3D PC-GRE protocol

This protocol preserves the original `koma_dp150/pc_3d` files and matches the
requested values extracted from `phantom_8_28_2025_metadeta.rtf`:

| Parameter | Scanner metadata | Serialized simulator | Status |
|---|---:|---:|---|
| TR | 46.88 ms | 46.88 ms | Exact |
| TE | 2.74 ms | 2.7399995 ms | Within 1 ns |
| Flip angle | 8 degrees | 8 degrees | Exact |
| Cardiac phases | 15 | 15 | Exact |
| R-R interval | 1012 ms | 1012.0000000 ms | Within numerical precision |
| Nominal phase spacing | 67.4667 ms | 67.466666-67.466667 ms | 1 ns raster split |
| Pixel spacing | 1.25 x 1.25 mm | 1.25 x 1.25 mm | Exact |
| Slice thickness | 2.5 mm | 2.5 mm | Exact |
| Voxel size | 1.25 x 1.25 x 2.5 mm | 1.25 x 1.25 x 2.5 mm | Exact |
| Pixel bandwidth | 350 Hz/pixel | 349.993 Hz/pixel | 0.002% raster error |

The physical dp150 model is 466.2 mm long, so the sequence uses a
500 x 80 x 80 mm FOV and 400 x 64 x 32 matrix. This retains the requested
voxel size while fully containing the model. VENC remains 4.5 m/s because it
was not included in the requested parameter changes. A compact, unwindowed
TBP=1 slice-selective excitation is normalized from its finite sampled
waveform to realize both the 8 degree flip angle and 2.74 ms TE. The second velocity-encoding lobe is
overlapped with the Cartesian prephaser; this preserves both zero velocity-
gradient area and the requested VENC first moment while reducing minimum TE.
The common prephaser is retimed to 0.80 ms to keep every merged waveform within
the configured 40 mT/m and 180 T/m/s hardware limits.
The protocol uses a 1 ns Pulseq ADC/block timing raster so the reported
bandwidth, TE, TR, and R-R interval can be represented without microsecond
rounding bias.

The scanner-matched simulation wrapper sets `B0 = 3 T` and maps the normalized
periodic `FlowPath` in the existing 18.7 GB phantom from its stored 716.9 ms
period to the requested 1012 ms R-R interval in memory. The source `.phantom`
is not copied or modified. The simulation report records the original period,
effective period, and scale factor. Because Koma's stored flow path uses
`Float32` time, its effective period is 1011.999965 ms (a 0.000035 ms numerical
difference from the sequence R-R interval).

Generate and validate the four isolated Pulseq files with:

```powershell
$julia = if ($env:MRSIM_JULIA_EXE) {
  $env:MRSIM_JULIA_EXE
} else {
  (Get-Command julia -ErrorAction Stop).Source
}
& $julia --project=koma_dp150 koma_dp150\pc_3d_scanner_matched\generate.jl
```

Run the scanner-matched simulation with:

```powershell
& $julia --project=koma_dp150 koma_dp150\pc_3d_scanner_matched\simulate.jl
```

The full 1-million-spin, four-encoding simulation is substantially larger than
the previous two-bin protocol. Use `DP150_BATCH_SIZE` for resumable batching.

The requested image and cardiac parameters are matched, but this is not a full
replica of the proprietary scanner acquisition. The serialized four-encoding
protocol lasts 8292.1792 s (2073.0448 s per REF/X/Y/Z encoding), whereas the
metadata reports an acquisition duration of 1553.44 s. Scanner acceleration,
coil sensitivity, reconstruction, noise, and the vendor's velocity-encoding
scheme are not specified by the supplied metadata and are therefore not
claimed as matched. The DICOM also identifies retrospective synchronization;
this deterministic simulation samples the same 15 nominal phase times but does
not reproduce the scanner's retrospective gating implementation.
