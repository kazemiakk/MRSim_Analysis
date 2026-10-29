using CUDA
using FFTW
using HDF5
using JLD2
using KomaMRI
using KomaMRICore
using Plots
using Statistics
using TOML

include(joinpath(@__DIR__, "perfect_spoiling.jl"))
include(joinpath(@__DIR__, "motion_timing.jl"))
include(joinpath(@__DIR__, "provenance.jl"))

using .Dp150Provenance

const PROVENANCE_SCHEMA_VERSION = 1

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
const DEFAULT_OUTPUT_ROOT = joinpath(
    REPO_ROOT,
    "outputs",
    "dp150_mri_pipeline",
    "koma",
)
const DEFAULT_PHANTOM = joinpath(DEFAULT_OUTPUT_ROOT, "dp150_motion_1m.phantom")
const DEFAULT_SEQUENCE_DIR = joinpath(@__DIR__, "pc_3d")

"""Return a stable, platform-normalized absolute artifact path."""
function canonical_artifact_path(path::AbstractString)
    value = replace(normpath(abspath(path)), '\\' => '/')
    return Sys.iswindows() ? lowercase(value) : value
end

"""Accept an artifact path moved with this repository, without rewriting it."""
function same_repository_artifact(stored_path::AbstractString, current_path::AbstractString)
    stored = canonical_artifact_path(stored_path)
    current = canonical_artifact_path(current_path)
    stored == current && return true

    root_prefix = rstrip(canonical_artifact_path(REPO_ROOT), '/') * "/"
    if !startswith(current, root_prefix)
        @warn "Current artifact is outside repository root" current root_prefix
        return false
    end
    relative = chop(current; head=length(root_prefix), tail=0)
    matches = endswith(stored, "/" * relative)
    matches || @warn "Rejected relocated artifact path" stored current relative
    return matches
end

phantom_file = normpath(get(ENV, "DP150_PHANTOM", DEFAULT_PHANTOM))
sequence_dir = normpath(get(ENV, "DP150_SEQUENCE_DIR", DEFAULT_SEQUENCE_DIR))
output_root = normpath(get(ENV, "DP150_SIM_OUTPUT_DIR", DEFAULT_OUTPUT_ROOT))
run_name = get(ENV, "DP150_RUN_NAME", "motion_1m")
report_name = get(ENV, "DP150_REPORT_NAME", "simulation_report.toml")
requested_encodings = uppercase.(strip.(split(
    get(ENV, "DP150_ENCODINGS", "REF,X,Y,Z"),
    ",",
)))
gpu_device = parse(Int, get(ENV, "DP150_GPU_DEVICE", "0"))
physical_gpu_index = get(ENV, "DP150_PHYSICAL_GPU_INDEX", "not constrained")
expected_particle_count = parse(Int, get(
    ENV,
    "DP150_EXPECTED_PARTICLE_COUNT",
    "0",
))
reuse_results = lowercase(get(ENV, "DP150_REUSE_RESULTS", "true")) in ("1", "true", "yes")
preflight_only = lowercase(get(
    ENV,
    "DP150_PREFLIGHT_ONLY",
    "false",
)) in ("1", "true", "yes")
batch_size = parse(Int, get(ENV, "DP150_BATCH_SIZE", "0"))
max_block_length = parse(Int, get(ENV, "DP150_MAX_BLOCK_LENGTH", "512"))
allow_unvalidated_block_length = lowercase(get(
    ENV,
    "DP150_ALLOW_UNVALIDATED_BLOCK_LENGTH",
    "false",
)) in ("1", "true", "yes")
scanner_b0 = parse(Float64, get(ENV, "DP150_B0_T", "1.5"))
rescale_motion_to_sequence_rr = lowercase(get(
    ENV,
    "DP150_RESCALE_MOTION_TO_SEQUENCE_RR",
    "false",
)) in ("1", "true", "yes")
seam_safe_motion = lowercase(get(
    ENV,
    "DP150_SEAM_SAFE_MOTION",
    "false",
)) in ("1", "true", "yes")
minimum_adc_seam_margin_s = parse(Float64, get(
    ENV,
    "DP150_MINIMUM_ADC_SEAM_MARGIN_S",
    "0.030",
))
seam_reset_marker_lead_s = parse(Float64, get(
    ENV,
    "DP150_SEAM_RESET_MARKER_LEAD_S",
    string(DEFAULT_SEAM_RESET_MARKER_LEAD_S),
))
seam_reset_marker_duration_s = parse(Float64, get(
    ENV,
    "DP150_SEAM_RESET_MARKER_DURATION_S",
    string(DEFAULT_SEAM_RESET_MARKER_DURATION_S),
))
run_dir = joinpath(output_root, "simulation_$(run_name)")
mkpath(run_dir)

valid_encodings = ["REF", "X", "Y", "Z"]
@assert !isempty(requested_encodings)
@assert all(encoding in valid_encodings for encoding in requested_encodings)
@assert length(unique(requested_encodings)) == length(requested_encodings)
@assert CUDA.functional()
@assert batch_size >= 0
@assert expected_particle_count >= 0
@assert max_block_length >= 1
@assert scanner_b0 > 0
@assert minimum_adc_seam_margin_s >= 0
@assert 0 < seam_reset_marker_duration_s < seam_reset_marker_lead_s
@assert basename(report_name) == report_name "DP150_REPORT_NAME must be a file name"
@assert endswith(lowercase(report_name), ".toml") "DP150_REPORT_NAME must end in .toml"
@assert max_block_length == 512 || allow_unvalidated_block_length "dp150 production is validated only with max_block_length=512"
seam_safe_motion && !rescale_motion_to_sequence_rr && error(
    "DP150_SEAM_SAFE_MOTION requires DP150_RESCALE_MOTION_TO_SEQUENCE_RR=true.")

sequence_files = Dict(
    encoding => normpath(joinpath(sequence_dir, "$(lowercase(encoding)).seq"))
    for encoding in requested_encodings
)
simulation_wrapper_file = normpath(get(
    ENV,
    "DP150_SIMULATION_WRAPPER_FILE",
    @__FILE__,
))
protocol_audit_file = normpath(joinpath(sequence_dir, "protocol_audit.toml"))
project_file = normpath(joinpath(@__DIR__, "Project.toml"))
manifest_file = normpath(joinpath(@__DIR__, "Manifest.toml"))
phantom_sha256 = file_sha256(phantom_file)
sequence_sha256 = Dict(
    encoding => file_sha256(sequence_files[encoding])
    for encoding in requested_encodings
)
simulation_script_sha256 = file_sha256(@__FILE__)
simulation_wrapper_sha256 = file_sha256(simulation_wrapper_file)
perfect_spoiling_sha256 = file_sha256(joinpath(@__DIR__, "perfect_spoiling.jl"))
motion_timing_script_sha256 = file_sha256(joinpath(@__DIR__, "motion_timing.jl"))
project_sha256 = file_sha256(project_file)
manifest_sha256 = file_sha256(manifest_file)
protocol_audit_sha256 = isfile(protocol_audit_file) ?
    file_sha256(protocol_audit_file) : "not_present"
@info "Bound simulation inputs by SHA-256" phantom_sha256 sequence_sha256

@info "Loading Koma phantom" file=phantom_file
phantom = read_phantom(phantom_file)
if expected_particle_count > 0
    @assert length(phantom) == expected_particle_count "Expected $(expected_particle_count) particles, loaded $(length(phantom))."
end
@assert all(isfinite, phantom.x)
@assert all(isfinite, phantom.y)
@assert all(isfinite, phantom.z)

sequences = Dict(
    encoding => read_seq(
        sequence_files[encoding];
        verify_signature=true,
        verbose=false,
    )
    for encoding in requested_encodings
)

sequence_rr_values = Float64[sequence.DEF["RR"] for sequence in values(sequences)]
@assert all(isapprox(rr, sequence_rr_values[1]; rtol=0, atol=1e-12) for rr in sequence_rr_values)
sequence_rr = sequence_rr_values[1]

namedtuple_dict(value::NamedTuple) = Dict(
    String(key) => getfield(value, key) for key in keys(value)
)

reference_encoding = haskey(sequences, "REF") ? "REF" : first(requested_encodings)
reference_sequence = sequences[reference_encoding]
reference_cardiac_bins = Int(reference_sequence.DEF["CardiacBins"])
reference_adc_window_count = count(adc -> adc.N > 0, reference_sequence.ADC)
reference_adc_window_count % reference_cardiac_bins == 0 || error(
    "Reference ADC count is not divisible by the cardiac-bin count.")
acquisition_group_count = div(
    reference_adc_window_count,
    reference_cardiac_bins,
)
seam_marker_results = Dict{String,Any}()
seam_marker_times_s = Float64[]
effective_sequence_transform_signature = "none"

motion_timing = Dict{String,Any}(
    "rescale_requested" => rescale_motion_to_sequence_rr,
    "seam_safe_requested" => seam_safe_motion,
    "sequence_rr_s" => sequence_rr,
)
if phantom.motion isa Motion
    original_period = Float64(phantom.motion.time.t_end - phantom.motion.time.t_start)
    effective_period = original_period
    if rescale_motion_to_sequence_rr
        phantom.motion, original_period, effective_period = rescale_periodic_motion(
            phantom.motion,
            sequence_rr,
        )
    end
    if seam_safe_motion
        phantom.motion.action isa FlowPath || error(
            "Seam-safe motion requires a FlowPath action.")
        phantom.motion.time.periodic || error(
            "Seam-safe motion requires a periodic TimeCurve.")
        alignment = align_periodic_motion_seam_to_first_adc_midbin(
            phantom.motion,
            first_adc_kspace_center(reference_sequence),
            sequence_rr,
            reference_cardiac_bins,
        )
        phantom.motion = alignment.motion
        effective_period = Float64(
            phantom.motion.time.t_end - phantom.motion.time.t_start,
        )

        adc_seam_audits = Dict{String,Any}()
        marker_summaries = Dict{String,Any}()
        for encoding in requested_encodings
            source_sequence = sequences[encoding]
            cardiac_bins = Int(source_sequence.DEF["CardiacBins"])
            adc_window_count = count(adc -> adc.N > 0, source_sequence.ADC)
            adc_window_count % cardiac_bins == 0 || error(
                "$encoding ADC count is not divisible by its cardiac-bin count.")
            group_count = div(adc_window_count, cardiac_bins)
            group_count == acquisition_group_count || error(
                "$encoding has a different acquisition-group count.")
            isapprox(
                first_adc_kspace_center(source_sequence),
                first_adc_kspace_center(reference_sequence);
                rtol=0,
                atol=1e-12,
            ) || error("$encoding has a different first ADC-center time.")

            inserted = insert_periodic_seam_reset_markers(
                source_sequence,
                alignment.realized_seam_time_s,
                effective_period,
                acquisition_group_count;
                lead_s=seam_reset_marker_lead_s,
                duration_s=seam_reset_marker_duration_s,
            )
            sequences[encoding] = inserted.sequence
            seam_marker_results[encoding] = inserted
            isempty(seam_marker_times_s) ?
                append!(seam_marker_times_s, inserted.marker_times_s) :
                inserted.marker_times_s == seam_marker_times_s || error(
                    "$encoding reset-marker times differ from the reference.")
            adc_audit = audit_adc_sample_windows_against_periodic_seam(
                inserted.sequence,
                alignment.realized_seam_time_s,
                effective_period;
                minimum_margin_s=minimum_adc_seam_margin_s,
            )
            adc_seam_audits[encoding] = namedtuple_dict(adc_audit)
            records = inserted.marker_records
            marker_summaries[encoding] = Dict(
                "marker_count" => inserted.marker_count,
                "first_marker_start_s" => first(inserted.marker_times_s),
                "last_marker_start_s" => last(inserted.marker_times_s),
                "minimum_clearance_before_s" => minimum(
                    record.clearance_before_s for record in records
                ),
                "minimum_clearance_after_s" => minimum(
                    record.clearance_after_s for record in records
                ),
                "marker_lead_s" => inserted.marker_lead_s,
                "marker_duration_s" => inserted.marker_duration_s,
                "marker_amplitude_float64" => inserted.marker_amplitude_float64,
                "marker_amplitude_float32" => inserted.marker_amplitude_float32,
                "marker_underflows_to_exact_zero_float32" =>
                    inserted.marker_underflows_to_exact_zero_float32,
                "adc_timing_and_labels_exactly_unchanged" =>
                    inserted.adc_timing_and_labels_exactly_unchanged,
                "block_durations_exactly_unchanged" =>
                    inserted.block_durations_exactly_unchanged,
                "gradients_exactly_unchanged" =>
                    inserted.gradients_exactly_unchanged,
                "adc_events_exactly_unchanged" =>
                    inserted.adc_events_exactly_unchanged,
                "source_sequence_file_modified" => false,
            )
        end

        transform_configuration = Dict{String,Any}(
            "transform" => "periodic_motion_seam_safe_v1",
            "all_encoding_protocol_audit_sha256" => protocol_audit_sha256,
            "realized_seam_time_s" => alignment.realized_seam_time_s,
            "realized_motion_period_s" => effective_period,
            "cardiac_bins" => reference_cardiac_bins,
            "acquisition_group_count" => acquisition_group_count,
            "marker_lead_s" => seam_reset_marker_lead_s,
            "marker_duration_s" => seam_reset_marker_duration_s,
            "minimum_adc_seam_margin_s" => minimum_adc_seam_margin_s,
            "motion_timing_script_sha256" => motion_timing_script_sha256,
            "subset_invariant_signature" => true,
        )
        effective_sequence_transform_signature = provenance_signature(
            transform_configuration,
        )
        motion_timing["seam_alignment"] = namedtuple_dict(alignment)
        delete!(motion_timing["seam_alignment"], "motion")
        motion_timing["adc_seam_audits"] = adc_seam_audits
        motion_timing["reset_marker_summaries"] = marker_summaries
        motion_timing["effective_sequence_transform_signature"] =
            effective_sequence_transform_signature
        motion_timing["acquisition_group_count"] = acquisition_group_count
        motion_timing["minimum_adc_seam_margin_s"] = minimum_adc_seam_margin_s
        motion_timing["controlled_approximation"] =
            "endpoint magnetization reset is deliberately applied before the " *
            "periodic coordinate seam; this prevents stale state transfer but " *
            "does not create physical particle identity continuity"
    end
    motion_timing["motion_type"] = string(typeof(phantom.motion.action))
    motion_timing["periodic"] = phantom.motion.time.periodic
    motion_timing["original_period_s"] = original_period
    motion_timing["effective_period_s"] = effective_period
    motion_timing["time_scale_factor"] = effective_period / original_period
    motion_timing["mapping"] = seam_safe_motion ?
        "FlowPath rescaled to RR and periodic seam placed at the half-bin gap" :
        rescale_motion_to_sequence_rr ?
        "stored normalized FlowPath mapped linearly to the sequence RR" :
        "stored motion timing used unchanged"
elseif phantom.motion isa NoMotion
    motion_timing["motion_type"] = "NoMotion"
    motion_timing["mapping"] = "not applicable"
else
    error("Unsupported phantom motion type $(typeof(phantom.motion)).")
end
motion_timing_signature = if phantom.motion isa Motion
    seam_safe_motion ?
        "periodic_motion_s=$(motion_timing["effective_period_s"]);" *
        "seam_safe_transform=$effective_sequence_transform_signature" :
        "periodic_motion_s=$(motion_timing["effective_period_s"])"
else
    "no_motion"
end

function validate_sequence(sequence, encoding)
    nx = Int(sequence.DEF["Nx"])
    ny = Int(sequence.DEF["Ny"])
    nz = Int(sequence.DEF["Nz"])
    nphases = Int(sequence.DEF["CardiacBins"])
    @assert String(sequence.DEF["Encoding"]) == encoding
    labels = get_labels(sequence)
    adc_blocks = [block for (block, event) in enumerate(sequence) if is_ADC_on(event)]
    adc_labels = labels[adc_blocks]
    label_tuples = [(label.LIN, label.PAR, label.PHS) for label in adc_labels]
    @assert length(adc_blocks) == ny * nz * nphases
    @assert length(unique(label_tuples)) == length(label_tuples)
    @assert extrema(label.LIN for label in adc_labels) == (0, ny - 1)
    @assert extrema(label.PAR for label in adc_labels) == (0, nz - 1)
    @assert extrema(label.PHS for label in adc_labels) == (0, nphases - 1)
    return (nx, ny, nz, nphases, length(adc_blocks))
end

function sort_raw(raw, shape)
    nx, ny, nz, nphases, _ = shape
    sorted = zeros(ComplexF32, nx, ny, nz, nphases)
    filled = falses(ny, nz, nphases)
    for profile in raw.profiles
        line = Int(profile.head.idx.kspace_encode_step_1)
        partition = Int(profile.head.idx.kspace_encode_step_2)
        phase = Int(profile.head.idx.phase)
        @assert 0 <= line < ny
        @assert 0 <= partition < nz
        @assert 0 <= phase < nphases
        @assert size(profile.data) == (nx, 1)
        @assert !filled[line + 1, partition + 1, phase + 1]
        sorted[:, line + 1, partition + 1, phase + 1] .= profile.data[:, 1]
        filled[line + 1, partition + 1, phase + 1] = true
    end
    @assert all(filled)
    return sorted
end

function local_motion_batch(
    phantom,
    first_spin,
    last_spin,
    global_reset_schedule,
)
    spin_range = first_spin:last_spin
    batch_phantom = phantom[spin_range]
    if phantom.motion isa Motion
        batch_action = phantom.motion.action[spin_range]
        if global_reset_schedule !== nothing
            sample_count = size(batch_action.dx, 2)
            @assert length(global_reset_schedule) == sample_count
            T = eltype(batch_action.dx)
            zero_path = zeros(T, 1, sample_count)
            batch_action = FlowPath(
                dx=vcat(batch_action.dx, zero_path),
                dy=vcat(batch_action.dy, zero_path),
                dz=vcat(batch_action.dz, zero_path),
                spin_reset=vcat(
                    batch_action.spin_reset,
                    reshape(global_reset_schedule, 1, :),
                ),
            )
            sentinel = Phantom(
                name="zero_density_reset_timing_sentinel",
                x=T[0],
                y=T[0],
                z=T[0],
                ρ=T[0],
                T1=T[1.3],
                T2=T[0.1],
            )
            batch_phantom = batch_phantom + sentinel
        end
        batch_phantom.motion = Motion(
            action=batch_action,
            time=phantom.motion.time,
            spins=AllSpins(),
        )
    elseif !(phantom.motion isa NoMotion)
        error("Resumable batching currently supports NoMotion or one Motion")
    end
    return batch_phantom
end

"""Load mandatory checkpoint provenance with a fail-closed legacy error."""
function load_required_provenance(path::AbstractString)
    try
        return JLD2.load(path, "provenance")
    catch exception
        message = sprint(showerror, exception)
        throw(ArgumentError(
            "Refusing reuse of $path: required SHA provenance is missing or " *
            "unreadable; the existing artifact was left unchanged. Choose a " *
            "fresh DP150_RUN_NAME to recompute. Cause: $message",
        ))
    end
end

function simulate_batched(
    phantom,
    sequence,
    scanner,
    sim_params,
    encoding,
    run_dir,
    batch_size,
    base_provenance,
)
    particle_count = length(phantom)
    batch_count = cld(particle_count, batch_size)
    accumulator = nothing
    batch_files = String[]
    batch_runtimes = Float64[]
    batch_dir = joinpath(run_dir, "$(lowercase(encoding))_batches")
    mkpath(batch_dir)
    global_reset_schedule = if phantom.motion isa Motion &&
        phantom.motion.action isa FlowPath
        vec(any(phantom.motion.action.spin_reset; dims=1))
    else
        nothing
    end

    batch_params = copy(sim_params)
    batch_params["return_type"] = "mat"
    for batch_index in 1:batch_count
        first_spin = (batch_index - 1) * batch_size + 1
        last_spin = min(batch_index * batch_size, particle_count)
        batch_file = joinpath(
            batch_dir,
            "batch_$(lpad(batch_index, 4, '0'))_of_$(lpad(batch_count, 4, '0')).jld2",
        )
        batch_signal = nothing
        elapsed = 0.0
        batch_provenance = merge(
            copy(base_provenance),
            Dict(
                "artifact_kind" => "signal_batch",
                "spin_range" => [first_spin, last_spin],
                "batch_count" => batch_count,
                "batching_contract" => "flowpath_local_rows_with_global_reset_sentinel_v1",
            ),
        )
        if reuse_results && isfile(batch_file)
            stored_provenance = load_required_provenance(batch_file)
            require_matching_provenance(
                stored_provenance,
                batch_provenance;
                artifact=batch_file,
            )
            stored_encoding = String(JLD2.load(batch_file, "encoding"))
            stored_range = Int.(JLD2.load(batch_file, "spin_range"))
            stored_phantom = normpath(String(JLD2.load(batch_file, "phantom_file")))
            stored_sequence = normpath(String(JLD2.load(batch_file, "sequence_file")))
            @assert stored_encoding == encoding
            @assert stored_range == [first_spin, last_spin]
            @assert same_repository_artifact(stored_phantom, phantom_file)
            @assert same_repository_artifact(stored_sequence, sequence_files[encoding])
            if rescale_motion_to_sequence_rr
                stored_motion_timing = String(JLD2.load(
                    batch_file,
                    "motion_timing_signature",
                ))
                @assert stored_motion_timing == motion_timing_signature
            end
            batch_signal = JLD2.load(batch_file, "signal")
            @assert all(isfinite, batch_signal)
            @info "Reusing validated signal batch" encoding batch_index batch_count file=batch_file
        else
            @info "Starting CUDA signal batch" encoding batch_index batch_count spins=(last_spin - first_spin + 1)
            batch_phantom = local_motion_batch(
                phantom,
                first_spin,
                last_spin,
                global_reset_schedule,
            )
            start = time()
            batch_signal = simulate(
                batch_phantom,
                sequence,
                scanner;
                sim_params=batch_params,
            )
            elapsed = time() - start
            batch_signal = ComplexF32.(batch_signal)
            temporary = batch_file * ".partial"
            JLD2.jldsave(
                temporary;
                signal=batch_signal,
                encoding,
                spin_range=[first_spin, last_spin],
                phantom_file,
                sequence_file=sequence_files[encoding],
                provenance=batch_provenance,
                provenance_schema_version=PROVENANCE_SCHEMA_VERSION,
                phantom_sha256,
                sequence_sha256=sequence_sha256[encoding],
                run_configuration_sha256=batch_provenance["run_configuration_sha256"],
                elapsed_seconds=elapsed,
                motion_timing_signature,
            )
            mv(temporary, batch_file; force=true)
            batch_phantom = nothing
            GC.gc()
            CUDA.reclaim()
        end
        if accumulator === nothing
            accumulator = zeros(ComplexF32, size(batch_signal))
        else
            @assert size(batch_signal) == size(accumulator)
        end
        accumulator .+= batch_signal
        push!(batch_files, batch_file)
        push!(batch_runtimes, elapsed)
        batch_signal = nothing
        GC.gc()
    end
    raw = KomaMRICore.signal_to_raw_data(accumulator, sequence)
    return raw, Dict(
        "batch_count" => batch_count,
        "batch_size_requested" => batch_size,
        "batch_files" => batch_files,
        "batch_runtime_seconds" => batch_runtimes,
        "provenance_schema_version" => PROVENANCE_SCHEMA_VERSION,
        "phantom_sha256" => phantom_sha256,
        "sequence_sha256" => sequence_sha256[encoding],
        "run_configuration_sha256" => base_provenance["run_configuration_sha256"],
        "signal_combination" => "complex ADC signals summed across disjoint spin batches",
        "linearity_basis" => "independent-spin MR signal is additive; KomaMRICore tests partitioned raw signals",
        "motion_batch_indexing" => "FlowPath rows sliced globally, then reindexed to local AllSpins",
        "global_reset_timing" => "zero-density sentinel preserves the unsplit reset-time union in every batch",
    )
end

sim_params = KomaMRICore.default_sim_params()
sim_params["gpu"] = true
sim_params["gpu_device"] = gpu_device
sim_params["precision"] = "f32"
sim_params["sim_method"] = BlochWithPerfectSpoiling()
sim_params["Δt"] = Inf
sim_params["max_block_length"] = max_block_length
scanner = Scanner(B0=scanner_b0)

if seam_safe_motion
    reset_application_audit = audit_periodic_seam_reset_markers(
        sequences[reference_encoding],
        phantom.motion,
        seam_marker_times_s;
        marker_duration_s=seam_reset_marker_duration_s,
        sim_method=sim_params["sim_method"],
        gradient_max_step_s=sim_params[KOMA_DELTA_T_KEY],
        rf_max_step_s=sim_params[KOMA_DELTA_T_RF_KEY],
        max_block_length=sim_params["max_block_length"],
        max_rf_block_length=sim_params["max_rf_block_length"],
    )
    motion_timing["reset_application_audit"] = namedtuple_dict(
        reset_application_audit,
    )
    motion_timing["precision_audit"] = namedtuple_dict(
        periodic_motion_precision_audit(
            phantom.motion,
            sequence_rr,
            acquisition_group_count,
        ),
    )
    @info "Seam-safe acquisition preflight passed" marker_count=reset_application_audit.marker_count endpoint_reset_spin_count=reset_application_audit.endpoint_reset_spin_count effective_sequence_transform_signature
end

if preflight_only
    seam_safe_motion || error(
        "DP150_PREFLIGHT_ONLY is supported only for seam-safe motion runs.")
    preflight_report = Dict{String,Any}(
        "status" => "PASS",
        "evidence_class" =>
            "deterministic source-path and post-discretization preflight",
        "simulation_started" => false,
        "run_name" => run_name,
        "phantom_file" => phantom_file,
        "phantom_sha256" => phantom_sha256,
        "particle_count" => length(phantom),
        "sequence_source_sha256" => Dict(
            encoding => sequence_sha256[encoding]
            for encoding in sort(requested_encodings)
        ),
        "effective_sequence_transform_signature" =>
            effective_sequence_transform_signature,
        "motion_timing_signature" => motion_timing_signature,
        "motion_timing" => motion_timing,
        "simulation_script_sha256" => simulation_script_sha256,
        "simulation_wrapper_sha256" => simulation_wrapper_sha256,
        "motion_timing_script_sha256" => motion_timing_script_sha256,
        "project_sha256" => project_sha256,
        "manifest_sha256" => manifest_sha256,
        "julia_version" => string(VERSION),
    )
    preflight_file = joinpath(run_dir, "seam_safe_preflight.toml")
    temporary_preflight_file = preflight_file * ".partial"
    open(temporary_preflight_file, "w") do io
        TOML.print(io, preflight_report; sorted=true)
    end
    mv(temporary_preflight_file, preflight_file; force=true)
    @info "Seam-safe preflight-only workflow complete" report=preflight_file
    exit(0)
end

function encoding_provenance(
    encoding::AbstractString,
    artifact_kind::AbstractString,
    shape;
    extra::AbstractDict=Dict{String,Any}(),
)
    nx, ny, nz, nphases, _ = shape
    sequence = sequences[encoding]
    configuration = Dict{String,Any}(
        "provenance_schema_version" => PROVENANCE_SCHEMA_VERSION,
        "encoding" => String(encoding),
        "phantom_sha256" => phantom_sha256,
        "sequence_sha256" => sequence_sha256[encoding],
        "protocol_audit_sha256" => protocol_audit_sha256,
        "particle_count" => length(phantom),
        "expected_particle_count" => expected_particle_count,
        "simulation_script_sha256" => simulation_script_sha256,
        "simulation_wrapper_sha256" => simulation_wrapper_sha256,
        "perfect_spoiling_sha256" => perfect_spoiling_sha256,
        "motion_timing_script_sha256" => motion_timing_script_sha256,
        "project_sha256" => project_sha256,
        "manifest_sha256" => manifest_sha256,
        "julia_version" => string(VERSION),
        "scanner_B0_T" => scanner_b0,
        "rescale_motion_to_sequence_rr" => rescale_motion_to_sequence_rr,
        "seam_safe_motion" => seam_safe_motion,
        "effective_sequence_transform_signature" =>
            effective_sequence_transform_signature,
        "minimum_adc_seam_margin_s" => minimum_adc_seam_margin_s,
        "seam_reset_marker_lead_s" => seam_reset_marker_lead_s,
        "seam_reset_marker_duration_s" => seam_reset_marker_duration_s,
        "motion_timing_signature" => motion_timing_signature,
        "precision" => "f32",
        "simulation_method" => "BlochWithPerfectSpoiling",
        "gradient_max_step_s" => "Inf",
        "maximum_simulation_block_length" => max_block_length,
        "unvalidated_block_length_override" => allow_unvalidated_block_length,
        "spin_batch_size_requested" => batch_size,
        "matrix" => [nx, ny, nz],
        "cardiac_bins" => nphases,
        "FOV_m" => Float64.(sequence.DEF["FOV"]),
        "Venc_m_per_s" => Float64(sequence.DEF["Venc"]),
        "RR_s" => Float64(sequence.DEF["RR"]),
    )
    provenance = merge(
        configuration,
        Dict(
            "artifact_kind" => String(artifact_kind),
            "run_configuration_sha256" => provenance_signature(configuration),
        ),
    )
    merge!(provenance, extra)
    return provenance
end

sequence_meta = Dict{String,Any}()
encoding_runtime = Dict{String,Float64}()
result_files = Dict{String,String}()
batch_metadata = Dict{String,Any}()
run_configuration_sha256_by_encoding = Dict{String,String}()
reused_encodings = String[]
total_start = time()

for encoding in requested_encodings
    sequence = sequences[encoding]
    shape = validate_sequence(sequence, encoding)
    nx, ny, nz, nphases, profiles = shape
    fov = Float64.(sequence.DEF["FOV"])
    voxel_size = fov ./ [nx, ny, nz]
    sequence_meta[encoding] = Dict(
        "matrix" => [nx, ny, nz],
        "cardiac_bins" => nphases,
        "profiles" => profiles,
        "RR_s" => Float64(sequence.DEF["RR"]),
        "TR_s" => Float64(sequence.DEF["TR"]),
        "TE_s" => haskey(sequence.DEF, "TE") ?
            Float64(sequence.DEF["TE"]) : "not serialized in this sequence",
        "nominal_phase_spacing_s" => haskey(sequence.DEF, "NominalPhaseInterval") ?
            Float64(sequence.DEF["NominalPhaseInterval"]) :
            Float64(sequence.DEF["RR"]) / nphases,
        "FOV_m" => fov,
        "voxel_size_m" => voxel_size,
        "pixel_spacing_m" => voxel_size[1:2],
        "slice_thickness_m" => voxel_size[3],
        "flip_angle_degrees" => haskey(sequence.DEF, "FlipAngleDegrees") ?
            Float64(sequence.DEF["FlipAngleDegrees"]) :
            "not serialized in this sequence",
        "bandwidth_hz_per_pixel" => 1 / (nx * dwell(first(filter(
            event -> event.N > 0,
            sequence.ADC,
        )))),
        "Venc_m_per_s" => Float64(sequence.DEF["Venc"]),
        "velocity_axis" => String(sequence.DEF["VelocityEncodingAxis"]),
        "file" => sequence_files[encoding],
        "sha256" => sequence_sha256[encoding],
    )
    result_file = joinpath(run_dir, "$(lowercase(encoding))_result.jld2")
    result_provenance = encoding_provenance(
        encoding,
        "encoding_result",
        shape;
        extra=Dict("image_shape" => [nx, ny, nz, nphases]),
    )
    run_configuration_sha256_by_encoding[encoding] =
        result_provenance["run_configuration_sha256"]
    if reuse_results && isfile(result_file)
        stored_provenance = load_required_provenance(result_file)
        require_matching_provenance(
            stored_provenance,
            result_provenance;
            artifact=result_file,
        )
        stored_encoding = String(JLD2.load(result_file, "encoding"))
        stored_phantom = normpath(String(JLD2.load(result_file, "phantom_file")))
        stored_sequence = normpath(String(JLD2.load(result_file, "sequence_file")))
        stored_image = JLD2.load(result_file, "image")
        @assert stored_encoding == encoding
        @assert same_repository_artifact(stored_phantom, phantom_file)
        @assert same_repository_artifact(
            stored_sequence,
            sequence_files[encoding],
        )
        @assert size(stored_image) == (nx, ny, nz, nphases)
        @assert all(isfinite, stored_image)
        if rescale_motion_to_sequence_rr
            stored_motion_timing = String(JLD2.load(
                result_file,
                "motion_timing_signature",
            ))
            @assert stored_motion_timing == motion_timing_signature
        end
        batch_metadata[encoding] = JLD2.load(
            result_file,
            "batch_metadata_for_encoding",
        )
        result_files[encoding] = result_file
        encoding_runtime[encoding] = Float64(JLD2.load(result_file, "elapsed_seconds"))
        push!(reused_encodings, encoding)
        @info "Reusing validated encoding result" encoding file=result_file
        continue
    end
    @info "Starting CUDA simulation" encoding spins=length(phantom) profiles matrix=(nx, ny, nz, nphases) gpu_device
    start = time()
    if batch_size > 0 && batch_size < length(phantom)
        raw, batch_metadata[encoding] = simulate_batched(
            phantom,
            sequence,
            scanner,
            sim_params,
            encoding,
            run_dir,
            batch_size,
            encoding_provenance(
                encoding,
                "encoding_simulation",
                shape,
            ),
        )
    else
        raw = simulate(phantom, sequence, scanner; sim_params)
        batch_metadata[encoding] = Dict(
            "batch_count" => 1,
            "batch_size_requested" => batch_size,
            "provenance_schema_version" => PROVENANCE_SCHEMA_VERSION,
            "phantom_sha256" => phantom_sha256,
            "sequence_sha256" => sequence_sha256[encoding],
            "run_configuration_sha256" =>
                result_provenance["run_configuration_sha256"],
            "signal_combination" => "none; direct full-phantom simulation",
        )
    end
    rawsort = sort_raw(raw, shape)
    image = fftshift(
        ifft(ifftshift(rawsort, (1, 2, 3)), (1, 2, 3)),
        (1, 2, 3),
    )
    elapsed = time() - start
    encoding_runtime[encoding] = elapsed
    temporary = result_file * ".partial"
    JLD2.jldsave(
        temporary;
        rawsort,
        image,
        encoding,
        sequence_file=sequence_files[encoding],
        phantom_file,
        provenance=result_provenance,
        provenance_schema_version=PROVENANCE_SCHEMA_VERSION,
        phantom_sha256,
        sequence_sha256=sequence_sha256[encoding],
        run_configuration_sha256=result_provenance["run_configuration_sha256"],
        batch_metadata_for_encoding=batch_metadata[encoding],
        elapsed_seconds=elapsed,
        motion_timing_signature,
        effective_sequence_transform_signature,
        seam_safe_motion,
        gpu_name=string(CUDA.name(CUDA.device())),
        gpu_device_zero_based=gpu_device,
        physical_gpu_index,
    )
    mv(temporary, result_file; force=true)
    result_files[encoding] = result_file
    @info "Encoding complete" encoding elapsed_seconds=elapsed file=result_file
    raw = nothing
    rawsort = nothing
    image = nothing
    GC.gc()
    CUDA.reclaim()
end

reconstruction_file = ""
figure_file = ""
velocity_qc = Dict{String,Any}()

if all(encoding in requested_encodings for encoding in valid_encodings)
    images = Dict(
        encoding => JLD2.load(result_files[encoding], "image")
        for encoding in valid_encodings
    )
    reference = images["REF"]
    magnitude = Float32.(abs.(reference))
    nphases = size(reference, 4)
    mask = falses(size(magnitude))
    for phase in 1:nphases
        phase_magnitude = @view magnitude[:, :, :, phase]
        threshold = 0.05f0 * maximum(phase_magnitude)
        @views mask[:, :, :, phase] .= phase_magnitude .>= threshold
    end

    venc = Float32(sequences["REF"].DEF["Venc"])
    velocity = zeros(Float32, size(reference)..., 3)
    for (component, encoding) in enumerate(("X", "Y", "Z"))
        phase_difference = angle.(images[encoding] .* conj.(reference))
        @views velocity[:, :, :, :, component] .= phase_difference .* (venc / Float32(pi))
        @views velocity[:, :, :, :, component][.!mask] .= 0
    end
    speed = sqrt.(sum(abs2, velocity; dims=5))[:, :, :, :, 1]
    @assert all(isfinite, magnitude)
    @assert all(isfinite, velocity)
    @assert maximum(abs, velocity) <= venc * (1 + 10eps(Float32))

    component_stats = Dict{String,Any}()
    for (component, name) in enumerate(("X", "Y", "Z"))
        values = abs.(velocity[:, :, :, :, component][mask])
        component_stats[name] = Dict(
            "median_absolute_m_per_s" => Float64(median(values)),
            "p95_absolute_m_per_s" => Float64(quantile(values, 0.95)),
            "maximum_absolute_m_per_s" => Float64(maximum(values)),
        )
    end
    velocity_qc = Dict(
        "mask_threshold_fraction_of_phase_max" => 0.05,
        "masked_voxel_count" => count(mask),
        "venc_m_per_s" => Float64(venc),
        "phase_formula" => "velocity = angle(encoded * conj(reference)) * Venc / pi",
        "component_statistics" => component_stats,
        "maximum_speed_m_per_s" => Float64(maximum(speed)),
    )

    reconstruction_file = joinpath(run_dir, "pc_reconstruction.h5")
    temporary = reconstruction_file * ".partial"
    h5open(temporary, "w") do file
        write(file, "reference_magnitude", magnitude)
        write(file, "velocity", velocity)
        write(file, "speed", speed)
        write(file, "mask", UInt8.(mask))
        HDF5.attributes(file)["dimension_order_julia"] = "readout,phase,slice,cardiac_bin[,component]"
        HDF5.attributes(file)["velocity_components"] = "X,Y,Z"
        HDF5.attributes(file)["Venc_m_per_s"] = venc
        HDF5.attributes(file)["RR_s"] = Float32(sequences["REF"].DEF["RR"])
        HDF5.attributes(file)["FOV_m"] = Float32.(sequences["REF"].DEF["FOV"])
        HDF5.attributes(file)["matrix"] = Int32.([
            sequences["REF"].DEF["Nx"],
            sequences["REF"].DEF["Ny"],
            sequences["REF"].DEF["Nz"],
        ])
        HDF5.attributes(file)["cardiac_bins"] =
            Int32(sequences["REF"].DEF["CardiacBins"])
        HDF5.attributes(file)["voxel_size_m"] = Float32.(
            sequences["REF"].DEF["FOV"] ./ [
                sequences["REF"].DEF["Nx"],
                sequences["REF"].DEF["Ny"],
                sequences["REF"].DEF["Nz"],
            ],
        )
        HDF5.attributes(file)["phase_formula"] = velocity_qc["phase_formula"]
        HDF5.attributes(file)["provenance_schema_version"] =
            PROVENANCE_SCHEMA_VERSION
        HDF5.attributes(file)["source_phantom_sha256"] = phantom_sha256
        for encoding in valid_encodings
            HDF5.attributes(file)["source_sequence_sha256_$encoding"] =
                sequence_sha256[encoding]
            HDF5.attributes(file)["run_configuration_sha256_$encoding"] =
                run_configuration_sha256_by_encoding[encoding]
        end
        HDF5.attributes(file)["simulation_script_sha256"] =
            simulation_script_sha256
        HDF5.attributes(file)["simulation_wrapper_sha256"] =
            simulation_wrapper_sha256
        HDF5.attributes(file)["perfect_spoiling_sha256"] =
            perfect_spoiling_sha256
        HDF5.attributes(file)["motion_timing_script_sha256"] =
            motion_timing_script_sha256
        HDF5.attributes(file)["seam_safe_motion"] = seam_safe_motion
        HDF5.attributes(file)["effective_sequence_transform_signature"] =
            effective_sequence_transform_signature
        HDF5.attributes(file)["motion_timing_signature"] =
            motion_timing_signature
        HDF5.attributes(file)["project_sha256"] = project_sha256
        HDF5.attributes(file)["manifest_sha256"] = manifest_sha256
        HDF5.attributes(file)["protocol_audit_sha256"] = protocol_audit_sha256
        HDF5.attributes(file)["julia_version"] = string(VERSION)
    end
    mv(temporary, reconstruction_file; force=true)

    projection = dropdims(maximum(magnitude[:, :, :, 1]; dims=3); dims=3)
    speed_projection = dropdims(maximum(speed[:, :, :, 1]; dims=3); dims=3)
    p1 = Plots.heatmap(
        projection',
        color=:grays,
        axis=nothing,
        title="Reference magnitude | bin 1",
        colorbar=false,
    )
    p2 = Plots.heatmap(
        speed_projection',
        color=:turbo,
        axis=nothing,
        title="Speed (m/s) | bin 1",
        colorbar=true,
        clim=(0, Float64(venc)),
    )
    figure = Plots.plot(p1, p2; layout=(2, 1), size=(1500, 700), dpi=160)
    figure_file = joinpath(run_dir, "pc_reconstruction_qc.png")
    temporary_figure_file = joinpath(run_dir, "pc_reconstruction_qc.partial.png")
    Plots.savefig(figure, temporary_figure_file)
    mv(temporary_figure_file, figure_file; force=true)

    if phantom.motion isa NoMotion
        static_p95 = maximum(
            stats["p95_absolute_m_per_s"] for stats in values(component_stats)
        )
        static_limit = 0.05
        velocity_qc["static_zero_velocity_acceptance"] = Dict(
            "passed" => static_p95 <= static_limit,
            "maximum_component_p95_absolute_m_per_s" => static_p95,
            "limit_m_per_s" => static_limit,
        )
    end
end

total_elapsed = time() - total_start
passed = !haskey(velocity_qc, "static_zero_velocity_acceptance") ||
    velocity_qc["static_zero_velocity_acceptance"]["passed"]
result_artifacts = Dict(
    encoding => Dict(
        "file" => result_files[encoding],
        "bytes" => filesize(result_files[encoding]),
        "sha256" => file_sha256(result_files[encoding]),
    )
    for encoding in requested_encodings
)
run_configuration_sha256 = provenance_signature(
    Dict(
        encoding => run_configuration_sha256_by_encoding[encoding]
        for encoding in sort(requested_encodings)
    ),
)
reconstruction_bytes = isfile(reconstruction_file) ? filesize(reconstruction_file) : 0
reconstruction_sha256 = isfile(reconstruction_file) ?
    file_sha256(reconstruction_file) : ""
qc_figure_bytes = isfile(figure_file) ? filesize(figure_file) : 0
qc_figure_sha256 = isfile(figure_file) ? file_sha256(figure_file) : ""
report = Dict(
    "status" => passed ? "PASS" : "FAIL",
    "provenance_schema_version" => PROVENANCE_SCHEMA_VERSION,
    "run_configuration_sha256" => run_configuration_sha256,
    "run_name" => run_name,
    "phantom_file" => phantom_file,
    "phantom_sha256" => phantom_sha256,
    "particle_count" => length(phantom),
    "expected_particle_count" => expected_particle_count,
    "sequence_dir" => sequence_dir,
    "encodings" => requested_encodings,
    "gpu_device_zero_based" => gpu_device,
    "physical_gpu_index" => physical_gpu_index,
    "gpu_name" => CUDA.name(CUDA.device()),
    "scanner_B0_T" => scanner_b0,
    "seam_safe_motion" => seam_safe_motion,
    "effective_sequence_transform_signature" =>
        effective_sequence_transform_signature,
    "motion_timing" => motion_timing,
    "simulation_method" => "Bloch with perfect transverse spoiling before RF",
    "simulation_script_sha256" => simulation_script_sha256,
    "simulation_wrapper_sha256" => simulation_wrapper_sha256,
    "perfect_spoiling_sha256" => perfect_spoiling_sha256,
    "motion_timing_script_sha256" => motion_timing_script_sha256,
    "project_sha256" => project_sha256,
    "manifest_sha256" => manifest_sha256,
    "protocol_audit_sha256" => protocol_audit_sha256,
    "julia_version" => string(VERSION),
    "gradient_spoiling_simulated" => false,
    "precision" => "f32",
    "gradient_max_step_s" => "Inf (event boundaries only, matching supplied script)",
    "maximum_simulation_block_length" => max_block_length,
    "validated_simulation_block_length" => 512,
    "unvalidated_block_length_override" => allow_unvalidated_block_length,
    "spin_batch_size" => batch_size,
    "batch_metadata" => batch_metadata,
    "sequence" => sequence_meta,
    "encoding_runtime_seconds" => encoding_runtime,
    "summed_encoding_runtime_seconds" => sum(values(encoding_runtime)),
    "total_runtime_seconds" => total_elapsed,
    "result_files" => result_files,
    "result_artifacts" => result_artifacts,
    "reconstruction_file" => reconstruction_file,
    "reconstruction_bytes" => reconstruction_bytes,
    "reconstruction_sha256" => reconstruction_sha256,
    "qc_figure" => figure_file,
    "qc_figure_bytes" => qc_figure_bytes,
    "qc_figure_sha256" => qc_figure_sha256,
    "velocity_qc" => velocity_qc,
    "result_reuse_enabled" => reuse_results,
    "reused_encodings" => reused_encodings,
)
report_file = joinpath(run_dir, report_name)
temporary_report_file = report_file * ".partial"
open(temporary_report_file, "w") do io
    TOML.print(io, report; sorted=true)
end
mv(temporary_report_file, report_file; force=true)
passed || error("Simulation QC failed; see $report_file")
@info "Simulation workflow complete" report=report_file total_elapsed
