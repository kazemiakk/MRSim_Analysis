using KomaMRI
using Test

include(joinpath(dirname(@__DIR__), "motion_timing.jl"))

path = FlowPath(
    dx=Float32[0 1],
    dy=Float32[0 0],
    dz=Float32[0 0],
    spin_reset=falses(1, 2),
)
motion = Motion(
    action=path,
    time=Periodic(period=0.7169f0, asymmetry=1.0f0),
    spins=AllSpins(),
)
scaled, original_period, effective_period = rescale_periodic_motion(motion, 1.012)

@test original_period ≈ 0.7169 atol=1e-7
@test effective_period ≈ 1.012 atol=1e-7
@test scaled.time.periodic
@test effective_period == Float64(Float32(1.012))
@test scaled.time.t_unit == motion.time.t_unit
@test scaled.action === motion.action
@test motion.time.t_end ≈ 0.7169f0
@test scaled.time.t_end ≈ 1.012f0

@info "Motion timing rescale test passed" original_period effective_period

root = dirname(@__DIR__)
include(joinpath(root, "gradient_design.jl"))
include(joinpath(root, "utils.jl"))
include(joinpath(root, "cartesian.jl"))
include(joinpath(root, "excitation.jl"))
include(joinpath(root, "gre.jl"))
include(joinpath(root, "PC.jl"))

sys = Scanner(Gmax=40e-3, Smax=180, B1=10e-6)
matrix = (16, 4, 4)
adc_dwell = max(
    round(Int, 1e-3 / (matrix[1] - 1) / sys.ADC_Δt),
    1,
) * sys.ADC_Δt
sequences = build_pc_encoding_scans(
    (0.064, 0.016, 0.016),
    matrix,
    sys;
    venc=2.0,
    cardiac_bins=1,
    RR=0.05,
    BWpp=1 / (matrix[1] * adc_dwell),
    flip_angle=deg2rad(10),
    spoil_phase=0.0,
    n_ramp_shots=0,
    trigger_channel=nothing,
    rf_spoil_increment=0.0,
)
for sequence in values(sequences)
    check_timing(sequence, sys)
    check_hw_limits(sequence, sys)
    @test sequence.DEF["ExactRRTiming"] == 0
    @test sequence.DEF["EchoDelay"] == 0.0
end

@info "Legacy default sequence-build path passed"
