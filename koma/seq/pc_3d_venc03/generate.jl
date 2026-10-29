using KomaMRI

# Venc-0.3 m/s variant of pc_3d/generate.jl.
#
# Rationale: the aortic dp2 CFD field peaks at 0.241 m/s (measured over all
# 2,340,480 nodes x 72 frames). Against the bundle's Venc of 2.5 m/s that is only
# ~17 deg of peak phase; at 0.3 m/s it is ~145 deg, safely inside +/-180 and worth
# roughly 8x in velocity SNR.
#
# Consequence to respect downstream: the larger first moment needs a longer
# bipolar, which lengthens TR, which changes the RR that `build_pc` realizes.
# Nothing may hard-code 0.718 s any more -- read DEF["RR"] from the written file.
# Hand-derived prediction, to be checked against the @info line below:
#   module 0.960 -> 2.240 ms, TR ~4.87 ms, 74 lines/bin, RR ~0.7208 s.
#
# Everything else is identical to the bundled generator. Charles's originals stay
# untouched in ../pc_3d/ and in from charles/.

root = dirname(@__DIR__)
include(joinpath(root, "gradient_design.jl"))
include(joinpath(root, "utils.jl"))
include(joinpath(root, "cartesian.jl"))
include(joinpath(root, "excitation.jl"))
include(joinpath(root, "gre.jl"))
include(joinpath(root, "PC.jl"))

sys = Scanner(Gmax=40e-3, Smax=180, B1=10e-6)
FOV = (0.280, 0.128, 0.128)
matrix = (140, 64, 64)
requested_adc_duration = 1e-3
adc_dwell = max(
    round(Int, requested_adc_duration / (matrix[1] - 1) / sys.ADC_Δt),
    1,
) * sys.ADC_Δt
BWpp = 1 / (matrix[1] * adc_dwell)

const VENC = parse(Float64, get(ENV, "KOMA_VENC", "0.3"))

seqs = build_pc_encoding_scans(
    FOV,
    matrix,
    sys;
    venc=VENC,
    cardiac_bins=2,
    RR=0.720,
    BWpp,
    flip_angle=deg2rad(10),
    spoil_phase=0.0,
    n_ramp_shots=10,
    trigger_channel=nothing,
    rf_spoil_increment=deg2rad(117),
)

@info "ADC timing" requested_adc_duration realized_adc_duration=adc_dwell * (matrix[1] - 1) adc_dwell BWpp
@info "Venc" VENC

for (encoding, seq) in pairs(seqs)
    check_timing(seq, sys)
    check_hw_limits(seq, sys)
    @info "Realized timing" encoding TR=seq.DEF["TR"] RR=seq.DEF["RR"] lines_per_bin=seq.DEF["LinesPerCardiacBin"]
    write_seq(seq, joinpath(@__DIR__, "$(lowercase(String(encoding))).seq"))
end
