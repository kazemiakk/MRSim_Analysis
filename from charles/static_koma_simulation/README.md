# Static phantom KomaMRI simulation

This bundle contains the scripts and Julia environment needed to:

1. Convert a particle-position HDF5 file into a centered static KomaMRI phantom.
2. Convert the same data into a periodic KomaMRI motion phantom with a user-defined RR interval.
3. Simulate `ref.seq`, sort the labeled acquisitions, and reconstruct both temporal bins.

The input HDF5, Pulseq sequence, generated phantom, and reconstruction are not included.

## Setup

Use Julia 1.12, start Julia in this directory, and instantiate the environment:

```julia
import Pkg
Pkg.activate(".")
Pkg.instantiate()
```

In `static_phantom.jl`, replace `"PATH TO INPUT H5 FILE"` with the HDF5 path, then run:

```julia
include("static_phantom.jl")
```

This writes `aorta_static.phantom` and `aorta_static_projections.png` beside the script.

To build a motion phantom, set `MOTION_H5_FILE` and `RR_INTERVAL` in
`motion_phantom.jl`, then run:

```julia
include("motion_phantom.jl")
```

The script uses the HDF5 `position` and `reset` arrays to create a periodic `FlowPath`.
It centers and reorders the complete motion envelope in the same way as the static phantom,
then maps one complete source cycle onto the requested RR interval. It writes
`aorta_motion.phantom` beside the script.

In `simulate_ref.jl`, replace `"PATH TO ref.seq"` with the Pulseq path, then run:

```julia
include("simulate_ref.jl")
```

This writes `ref_static.jld2` containing `rawsort` and the reconstructed `result`.

## GPU backend

The bundled environment and script use Metal on Apple Silicon. For an NVIDIA GPU, replace
`using Metal` with `using CUDA` in `simulate_ref.jl` and add CUDA.jl to the environment:

```julia
import Pkg
Pkg.add("CUDA")
```

Load only one functional GPU backend. If Metal and CUDA are both loaded, KomaMRI falls back
to CPU execution.
