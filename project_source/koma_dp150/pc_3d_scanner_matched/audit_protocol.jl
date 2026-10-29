using KomaMRI
using TOML

const ROOT = dirname(@__DIR__)
include(joinpath(ROOT, "provenance.jl"))
using .Dp150Provenance

const EXPECTED = Dict(
    "TE" => 2.74e-3,
    "TR" => 46.88e-3,
    "RR" => 1.012,
    "CardiacBins" => 15,
    "FlipAngleDegrees" => 8.0,
    "BandwidthPerPixel" => 350.0,
    "Venc" => parse(Float64, get(ENV, "DP150_VENC_M_PER_S", "4.5")),
)
const ENCODINGS = (REF="ref.seq", X="x.seq", Y="y.seq", Z="z.seq")
const RESTRICTED_SOURCE_SHA256 =
    "fa27bb8ecaacdabc91c56f8be934401c5f91f864863bd11e035a2b30d0669a52"
const DEIDENTIFIED_COMPANION_SHA256 =
    "a83fe4aa5b7514158814f6b614f0917514c8e59e3809f3f7f41190c88371b219"
const SEQUENCE_DIR = normpath(get(
    ENV,
    "DP150_AUDIT_SEQUENCE_DIR",
    @__DIR__,
))
const AUDIT_WRAPPER_FILE = normpath(get(
    ENV,
    "DP150_AUDIT_WRAPPER_FILE",
    @__FILE__,
))
const GENERATOR_WRAPPER_FILE = normpath(get(
    ENV,
    "DP150_GENERATOR_WRAPPER_FILE",
    joinpath(SEQUENCE_DIR, "generate.jl"),
))

EXPECTED["Venc"] > 0 || error("DP150_VENC_M_PER_S must be positive.")

function measured_te(sequence)
    starts = get_block_start_times(sequence)
    adc_block = findfirst(adc -> adc.N > 0, sequence.ADC)
    isnothing(adc_block) && error("Sequence has no acquired ADC event.")
    rf_candidates = Int[]
    for block in 1:adc_block
        any(rf -> dur(rf) > 0, view(sequence.RF, :, block)) &&
            push!(rf_candidates, block)
    end
    isempty(rf_candidates) && error("No RF event precedes the first ADC event.")
    rf_block = last(rf_candidates)
    rf = only(filter(event -> dur(event) > 0, collect(view(sequence.RF, :, rf_block))))
    adc = sequence.ADC[adc_block]
    n_precenter = Int(sequence.DEF["Nx"]) ÷ 2
    rf_time = starts[rf_block] + rf.delay + rf_center(rf)
    adc_time = starts[adc_block] + adc.delay + n_precenter * dwell(adc)
    return adc_time - rf_time
end

function measured_first_acquired_flip_angle(sequence)
    adc_block = findfirst(adc -> adc.N > 0, sequence.ADC)
    isnothing(adc_block) && error("Sequence has no acquired ADC event.")
    rf_blocks = [
        block for block in 1:adc_block
        if any(rf -> dur(rf) > 0, view(sequence.RF, :, block))
    ]
    isempty(rf_blocks) && error("No RF event precedes the first ADC event.")
    return maximum(get_flip_angles(sequence[last(rf_blocks)]))
end

function measured_phase_intervals(sequence)
    starts = get_block_start_times(sequence)
    centers = Float64[]
    n_precenter = Int(sequence.DEF["Nx"]) ÷ 2
    for block in eachindex(sequence.ADC)
        adc = sequence.ADC[block]
        adc.N > 0 || continue
        push!(centers, starts[block] + adc.delay + n_precenter * dwell(adc))
        length(centers) == Int(sequence.DEF["CardiacBins"]) + 1 && break
    end
    length(centers) == Int(sequence.DEF["CardiacBins"]) + 1 || error(
        "Not enough acquired profiles to audit a complete cardiac cycle.")
    return diff(centers)
end

function first_profile_m1_at_kspace_center(sequence)
    adc_block = findfirst(adc -> adc.N > 0, sequence.ADC)
    isnothing(adc_block) && error("Sequence has no acquired ADC event.")
    rf_blocks = [
        block for block in 1:adc_block
        if any(rf -> dur(rf) > 0, view(sequence.RF, :, block))
    ]
    isempty(rf_blocks) && error("No RF event precedes the first ADC event.")
    shot = sequence[last(rf_blocks):adc_block]
    _, m1_adc = get_M1(
        shot;
        sampling_rule=MaxStepSizeRule(1, 1e-6),
    )
    center_sample = Int(sequence.DEF["Nx"]) ÷ 2 + 1
    size(m1_adc, 1) == Int(sequence.DEF["Nx"]) || error(
        "Unexpected first-shot ADC sample count.")
    return vec(m1_adc[center_sample, :])
end

results = Dict{String,Any}()
m1_at_center = Dict{String,Vector{Float64}}()
for (encoding, filename) in pairs(ENCODINGS)
    path = joinpath(SEQUENCE_DIR, filename)
    isfile(path) || error("Missing sequence file: $path")
    sequence = read_seq(path; verify_signature=true, verbose=false)
    sys = Scanner(
        B0=3.0,
        Gmax=40e-3,
        Smax=180,
        B1=10e-6,
        ADC_Δt=1e-9,
        DUR_Δt=1e-9,
    )
    check_timing(sequence, sys)
    check_hw_limits(sequence, sys)

    te = measured_te(sequence)
    intervals = measured_phase_intervals(sequence)
    matrix = Int[sequence.DEF["Nx"], sequence.DEF["Ny"], sequence.DEF["Nz"]]
    fov = Float64.(sequence.DEF["FOV"])
    voxel = fov ./ matrix
    adc = first(filter(event -> event.N > 0, sequence.ADC))
    bandwidth = 1 / (matrix[1] * dwell(adc))
    flip_angle = measured_first_acquired_flip_angle(sequence)
    m1_at_center[String(encoding)] = first_profile_m1_at_kspace_center(sequence)

    abs(te - EXPECTED["TE"]) ≤ 1e-9 || error("$encoding TE mismatch: $te")
    abs(Float64(sequence.DEF["TR"]) - EXPECTED["TR"]) ≤ 1e-12 ||
        error("$encoding TR mismatch")
    abs(Float64(sequence.DEF["RR"]) - EXPECTED["RR"]) ≤ 1e-12 ||
        error("$encoding RR mismatch")
    Int(sequence.DEF["CardiacBins"]) == EXPECTED["CardiacBins"] ||
        error("$encoding cardiac-bin mismatch")
    abs(Float64(sequence.DEF["FlipAngleDegrees"]) - EXPECTED["FlipAngleDegrees"]) ≤ 1e-12 ||
        error("$encoding flip-angle mismatch")
    abs(Float64(sequence.DEF["Venc"]) - EXPECTED["Venc"]) ≤ 1e-12 ||
        error("$encoding serialized VENC mismatch")
    abs(flip_angle - EXPECTED["FlipAngleDegrees"]) ≤ 1e-3 ||
        error("$encoding RF-waveform flip-angle mismatch: $flip_angle degrees")
    maximum(abs.(voxel .- [1.25e-3, 1.25e-3, 2.5e-3])) ≤ 1e-12 ||
        error("$encoding voxel-size mismatch")
    abs(bandwidth - EXPECTED["BandwidthPerPixel"]) < 0.01 ||
        error("$encoding bandwidth mismatch: $bandwidth")
    abs(sum(intervals) - EXPECTED["RR"]) ≤ 2e-9 ||
        error("$encoding cardiac-cycle timing mismatch")

    results[String(encoding)] = Dict(
        "file" => filename,
        "bytes" => filesize(path),
        "sha256" => file_sha256(path),
        "te_ms_readback" => 1e3 * te,
        "tr_ms" => 1e3 * Float64(sequence.DEF["TR"]),
        "rr_ms" => 1e3 * sum(intervals),
        "minimum_phase_interval_ms" => 1e3 * minimum(intervals),
        "maximum_phase_interval_ms" => 1e3 * maximum(intervals),
        "bandwidth_hz_per_pixel_readback" => bandwidth,
        "matrix" => matrix,
        "fov_mm" => 1e3 .* fov,
        "voxel_size_mm" => 1e3 .* voxel,
        "flip_angle_degrees_definition" => Float64(sequence.DEF["FlipAngleDegrees"]),
        "flip_angle_degrees_waveform" => flip_angle,
        "venc_m_per_s_definition" => Float64(sequence.DEF["Venc"]),
        "timing_check" => "passed",
        "hardware_check" => "passed",
        "sequence_duration_s" => dur(sequence),
    )
end

venc_target = EXPECTED["Venc"]
venc_results = Dict{String,Any}()
reference_m1 = m1_at_center["REF"]
for (encoding, axis_index) in (("X", 1), ("Y", 2), ("Z", 3))
    delta = m1_at_center[encoding] .- reference_m1
    encoded_m1 = abs(delta[axis_index])
    off_axis_m1 = maximum(abs.(delta[setdiff(1:3, [axis_index])]))
    derived_venc = 1 / (2encoded_m1)
    abs(derived_venc - venc_target) < 0.01 || error(
        "$encoding waveform-derived VENC mismatch: $derived_venc m/s")
    off_axis_m1 < 1e-7 || error(
        "$encoding contains an unexpected off-axis differential first moment.")
    venc_results[encoding] = Dict(
        "differential_m1_hz_s_per_m" => delta,
        "encoded_axis_absolute_m1_hz_s_per_m" => encoded_m1,
        "maximum_off_axis_absolute_m1_hz_s_per_m" => off_axis_m1,
        "waveform_derived_venc_m_per_s" => derived_venc,
    )
end

protocol_report_path = joinpath(SEQUENCE_DIR, "protocol_report.toml")
isfile(protocol_report_path) || error(
    "Missing source-adjacent protocol report: $protocol_report_path",
)
repository_root = dirname(ROOT)
reference_directory = normpath(joinpath(
    dirname(repository_root),
    "reference_inputs",
    "scanner_metadata",
))
source_metadata_path = normpath(get(
    ENV,
    "DP150_SOURCE_METADATA",
    joinpath(reference_directory, "phantom_8_28_2025_metadeta.rtf"),
))
source_metadata_companion_path = normpath(get(
    ENV,
    "DP150_SOURCE_METADATA_COMPANION",
    joinpath(reference_directory, "README.md"),
))

function optional_hash_status(path::AbstractString, expected::AbstractString, label::AbstractString)
    if !isfile(path)
        return "optional external $label not present; expected hash retained"
    end
    actual = file_sha256(path)
    actual == expected || error(
        "$label hash mismatch: expected $expected, found $actual",
    )
    return "optional external $label present and hash verified"
end

source_metadata_status = optional_hash_status(
    source_metadata_path,
    RESTRICTED_SOURCE_SHA256,
    "restricted scanner metadata",
)
source_metadata_companion_status = optional_hash_status(
    source_metadata_companion_path,
    DEIDENTIFIED_COMPANION_SHA256,
    "de-identified metadata companion",
)
report = Dict(
    "status" => "PASS",
    "provenance_schema_version" => 1,
    "julia_version" => string(VERSION),
    "koma_mri_version" => string(Base.pkgversion(KomaMRI)),
    "audit_script_sha256" => file_sha256(@__FILE__),
    "audit_wrapper_sha256" => file_sha256(AUDIT_WRAPPER_FILE),
    "generator_script_sha256" => file_sha256(joinpath(@__DIR__, "generate.jl")),
    "generator_wrapper_sha256" => file_sha256(GENERATOR_WRAPPER_FILE),
    "protocol_report_file" => "protocol_report.toml",
    "protocol_report_sha256" => file_sha256(protocol_report_path),
    "source_metadata" => "phantom_8_28_2025_metadeta.rtf",
    "source_metadata_sha256" => RESTRICTED_SOURCE_SHA256,
    "source_metadata_status" => source_metadata_status,
    "source_metadata_companion" => "README.md",
    "source_metadata_companion_sha256" => DEIDENTIFIED_COMPANION_SHA256,
    "source_metadata_companion_status" => source_metadata_companion_status,
    "source_metadata_binding_scope" =>
        "the audit binds protocol_report.toml and serialized .seq files; " *
        "restricted metadata is optional external provenance, is not copied, " *
        "and is not required to execute the simulation",
    "scanner_metadata_matrix" => [64, 64],
    "simulation_geometry" =>
        "geometry-adapted 400x64x32 matrix and 500x80x80 mm FOV; this does " *
        "not reproduce the scanner matrix/FOV",
    "protocol_match_scope" =>
        "matches 1.25x1.25x2.5 mm voxel size, 15 phase labels, TR, TE, " *
        "flip angle, and bandwidth; VENC and model-containing matrix/FOV " *
        "are explicitly adapted",
    "bandwidth_tolerance_hz_per_pixel" => 0.01,
    "te_readback_tolerance_ns" => 1.0,
    "encodings" => results,
    "velocity_encoding" => venc_results,
    "venc_target_m_per_s" => venc_target,
    "total_four_encoding_duration_s" => sum(
        result["sequence_duration_s"] for result in values(results)),
)
audit_file = joinpath(SEQUENCE_DIR, "protocol_audit.toml")
temporary_audit_file = audit_file * ".partial"
open(temporary_audit_file, "w") do io
    TOML.print(io, report; sorted=true)
end
mv(temporary_audit_file, audit_file; force=true)
@info "Scanner-matched protocol audit passed" report_file=audit_file
