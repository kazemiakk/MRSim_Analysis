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

The scripts require Julia with KomaMRI and Unitful available. From the bundle
root, run:

```sh
julia --project=/path/to/compatible/koma/environment pc_3d/generate.jl
```

The generator validates timing and hardware limits before replacing
`pc_3d/ref.seq`, `x.seq`, `y.seq`, and `z.seq`.
