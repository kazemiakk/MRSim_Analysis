using KomaMRI
using KomaMRICore
using Statistics
using TOML
using Unitful

include(joinpath(@__DIR__, "gradient_design.jl"))
include(joinpath(@__DIR__, "utils.jl"))
include(joinpath(@__DIR__, "cartesian.jl"))
include(joinpath(@__DIR__, "excitation.jl"))
include(joinpath(@__DIR__, "gre.jl"))
include(joinpath(@__DIR__, "PC.jl"))
include(joinpath(@__DIR__, "perfect_spoiling.jl"))

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
const OUTPUT_DIR = joinpath(
    REPO_ROOT,
    "outputs",
    "dp150_mri_pipeline",
    "koma",
)
mkpath(OUTPUT_DIR)

scanner = Scanner(Gmax=40e-3, Smax=180, B1=10e-6)
fov = (0.064, 0.016, 0.016)
matrix = (16, 4, 4)
venc = 2.0
test_velocity = 0.4
requested_adc_duration = 1e-3
adc_dwell = max(
    round(Int, requested_adc_duration / (matrix[1] - 1) / scanner.ADC_Δt),
    1,
) * scanner.ADC_Δt
bwpp = 1 / (matrix[1] * adc_dwell)

sequences = build_pc_encoding_scans(
    fov,
    matrix,
    scanner;
    venc,
    cardiac_bins=1,
    RR=0.05,
    BWpp=bwpp,
    flip_angle=deg2rad(10),
    spoil_phase=0.0,
    n_ramp_shots=0,
    trigger_channel=nothing,
    rf_spoil_increment=0.0,
)

duration = dur(sequences.REF)
@assert all(isapprox(dur(sequence), duration; rtol=0, atol=1e-12) for sequence in values(sequences))

function moving_phantom(axis::Int)
    displacement = zeros(Float32, 3)
    displacement[axis] = Float32(test_velocity * duration)
    motion = translate(
        displacement[1],
        displacement[2],
        displacement[3],
        TimeRange(0.0f0, Float32(duration)),
    )
    return Phantom(
        name="known_constant_velocity",
        x=Float32[0],
        y=Float32[0],
        z=Float32[0],
        ρ=Float32[1],
        T1=Float32[100],
        T2=Float32[100],
        motion=motion,
    )
end

sim_params = KomaMRICore.default_sim_params()
sim_params["gpu"] = false
sim_params["precision"] = "f32"
sim_params["sim_method"] = BlochWithPerfectSpoiling()
sim_params["Δt"] = Inf

function profile_samples(raw)
    return reduce(vcat, vec(profile.data) for profile in raw.profiles)
end

axis_encodings = ((1, :X), (2, :Y), (3, :Z))
measured_phase = Float64[]
measured_velocity = Float64[]
phase_error = Float64[]
expected_phase = pi * test_velocity / venc

for (axis, encoding) in axis_encodings
    phantom = moving_phantom(axis)
    reference = simulate(phantom, sequences.REF, scanner; sim_params)
    encoded = simulate(phantom, getproperty(sequences, encoding), scanner; sim_params)
    reference_signal = profile_samples(reference)
    encoded_signal = profile_samples(encoded)
    @assert length(reference_signal) == length(encoded_signal)
    valid = abs.(reference_signal) .> 100eps(Float32)
    unit_ratio = encoded_signal[valid] .* conj.(reference_signal[valid])
    unit_ratio ./= abs.(unit_ratio)
    phase = angle(sum(unit_ratio))
    velocity = phase * venc / pi
    error = angle(exp(im * (phase - expected_phase)))
    push!(measured_phase, phase)
    push!(measured_velocity, velocity)
    push!(phase_error, error)
    @info "Velocity sign test" encoding phase velocity error
end

tolerance = 0.02
passed = all(abs.(phase_error) .<= tolerance)
report = Dict(
    "status" => passed ? "PASS" : "FAIL",
    "test_velocity_m_per_s" => test_velocity,
    "venc_m_per_s" => venc,
    "expected_phase_rad" => expected_phase,
    "measured_phase_rad_xyz" => measured_phase,
    "reconstructed_velocity_m_per_s_xyz" => measured_velocity,
    "wrapped_phase_error_rad_xyz" => phase_error,
    "phase_error_tolerance_rad" => tolerance,
    "validated_formula" => "velocity = angle(encoded * conj(reference)) * Venc / pi",
    "simulation_method" => "Bloch with perfect transverse spoiling before RF",
)
report_file = joinpath(OUTPUT_DIR, "pc_velocity_sign_report.toml")
open(report_file, "w") do io
    TOML.print(io, report; sorted=true)
end
passed || error("PC velocity sign validation failed; see $report_file")
@info "PC velocity sign validation passed" report=report_file
