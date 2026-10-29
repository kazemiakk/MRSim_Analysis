# 3D PC-GRE example

This bundle generates separate `REF`, `X`, `Y`, and `Z` Pulseq files for a
retrospective 3D PC-GRE acquisition with:

- 280×128×128 mm FOV;
- 140×64×64 matrix (2 mm isotropic);
- 250 cm/s Venc;
- two cardiac frames;
- 720 ms requested RR (718 ms realized);
- 10-degree flip angle;
- zero added gradient spoiler;
- 1.112 ms raster-compatible ADC duration (8 µs dwell).

The scripts require Julia with the repository's pinned Koma environment. From
the repository root, set `MRSIM_JULIA_EXE` when Julia is not on `PATH`, then
run:

```powershell
$julia = if ($env:MRSIM_JULIA_EXE) {
  (Resolve-Path -LiteralPath $env:MRSIM_JULIA_EXE).Path
} else {
  $command = Get-Command julia -ErrorAction SilentlyContinue
  if ($command) {
    $command.Source
  } else {
    $installed = @(Get-ChildItem -Path `
      (Join-Path $env:LOCALAPPDATA 'Programs\Julia-*\bin\julia.exe') `
      -File -ErrorAction SilentlyContinue | Sort-Object FullName -Descending)
    if (-not $installed.Count) {
      throw 'Julia was not found. Install Julia or set MRSIM_JULIA_EXE.'
    }
    $installed[0].FullName
  }
}
& $julia --startup-file=no --project=koma_aorta koma_aorta\pc_3d\generate.jl
```

The generator validates timing and hardware limits before replacing
`pc_3d/ref.seq`, `x.seq`, `y.seq`, and `z.seq`.
