# The frozen direct_sort helper expects FFTW exports in its calling module.
# Import them explicitly without changing its implementation or simulator.
using FFTW
include(joinpath(@__DIR__, "run_acquisition.jl"))
