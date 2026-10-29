using KomaMRI

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

seqs = build_pc_encoding_scans(
    FOV,
    matrix,
    sys;
    venc=2.5,
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

for (encoding, seq) in pairs(seqs)
    check_timing(seq, sys)
    check_hw_limits(seq, sys)
    write_seq(seq, joinpath(@__DIR__, "$(lowercase(String(encoding))).seq"))
end
