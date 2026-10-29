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
include(joinpath(@__DIR__, "provenance.jl"))

using .AortaProvenance

const PROVENANCE_SCHEMA_VERSION = 1

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
const DEFAULT_OUTPUT_ROOT = joinpath(
    REPO_ROOT,
    "outputs",
    "aorta_dp2_mri_pipeline",
    "koma",
)
const DEFAULT_PHANTOM = joinpath(DEFAULT_OUTPUT_ROOT, "aorta_motion_1m.phantom")
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

phantom_file = normpath(get(ENV, "AORTA_PHANTOM", DEFAULT_PHANTOM))
sequence_dir = normpath(get(ENV, "AORTA_SEQUENCE_DIR", DEFAULT_SEQUENCE_DIR))
output_root = normpath(get(ENV, "AORTA_SIM_OUTPUT_DIR", DEFAULT_OUTPUT_ROOT))
run_name = get(ENV, "AORTA_RUN_NAME", "motion_1m")
requested_encodings = uppercase.(strip.(split(
    get(ENV, "AORTA_ENCODINGS", "REF,X,Y,Z"),
    ",",
)))
gpu_device = parse(Int, get(ENV, "AORTA_GPU_DEVICE", "0"))
physical_gpu_index = parse(Int, get(
    ENV,
    "AORTA_PHYSICAL_GPU_INDEX",
    string(gpu_device),
))
reuse_results = lowercase(get(ENV, "AORTA_REUSE_RESULTS", "true")) in ("1", "true", "yes")
batch_size = parse(Int, get(ENV, "AORTA_BATCH_SIZE", "0"))
max_block_length = parse(Int, get(ENV, "AORTA_MAX_BLOCK_LENGTH", "512"))
allow_unvalidated_block_length = lowercase(get(
    ENV,
    "AORTA_ALLOW_UNVALIDATED_BLOCK_LENGTH",
    "false",
)) in ("1", "true", "yes")
run_dir = joinpath(output_root, "simulation_$(run_name)")
mkpath(run_dir)

valid_encodings = ["REF", "X", "Y", "Z"]
@assert !isempty(requested_encodings)
@assert all(encoding in valid_encodings for encoding in requested_encodings)
@assert length(unique(requested_encodings)) == length(requested_encodings)
@assert CUDA.functional()
@assert physical_gpu_index >= 0
@assert batch_size >= 0
@assert max_block_length >= 1
@assert max_block_length == 512 || allow_unvalidated_block_length "aorta production is validated only with max_block_length=512"

sequence_files = Dict(
    encoding => normpath(joinpath(sequence_dir, "$(lowercase(encoding)).seq"))
    for encoding in requested_encodings
)
phantom_sha256 = file_sha256(phantom_file)
sequence_sha256 = Dict(
    encoding => file_sha256(sequence_files[encoding])
    for encoding in requested_encodings
)
simulation_script_sha256 = file_sha256(@__FILE__)
perfect_spoiling_sha256 = file_sha256(joinpath(@__DIR__, "perfect_spoiling.jl"))
@info "Bound simulation inputs by SHA-256" phantom_sha256 sequence_sha256

@info "Loading Koma phantom" file=phantom_file
phantom = read_phantom(phantom_file)
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
            "fresh AORTA_RUN_NAME to recompute. Cause: $message",
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
                "batch_size_requested" => batch_size,
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
                elapsed_seconds=elapsed,
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
scanner = Scanner()

function encoding_provenance(
    encoding::AbstractString,
    artifact_kind::AbstractString;
    extra::AbstractDict=Dict{String,Any}(),
)
    provenance = Dict{String,Any}(
        "provenance_schema_version" => PROVENANCE_SCHEMA_VERSION,
        "artifact_kind" => String(artifact_kind),
        "encoding" => String(encoding),
        "run_name" => run_name,
        "phantom_sha256" => phantom_sha256,
        "sequence_sha256" => sequence_sha256[encoding],
        "particle_count" => length(phantom),
        "simulation_script_sha256" => simulation_script_sha256,
        "perfect_spoiling_sha256" => perfect_spoiling_sha256,
        "julia_version" => string(VERSION),
        "precision" => "f32",
        "simulation_method" => "BlochWithPerfectSpoiling",
        "gradient_max_step_s" => "Inf",
        "maximum_simulation_block_length" => max_block_length,
    )
    merge!(provenance, extra)
    return provenance
end

sequence_meta = Dict{String,Any}()
encoding_runtime = Dict{String,Float64}()
result_files = Dict{String,String}()
batch_metadata = Dict{String,Any}()
reused_encodings = String[]
total_start = time()

for encoding in requested_encodings
    sequence = sequences[encoding]
    shape = validate_sequence(sequence, encoding)
    nx, ny, nz, nphases, profiles = shape
    sequence_meta[encoding] = Dict(
        "matrix" => [nx, ny, nz],
        "cardiac_bins" => nphases,
        "profiles" => profiles,
        "RR_s" => Float64(sequence.DEF["RR"]),
        "TR_s" => Float64(sequence.DEF["TR"]),
        "FOV_m" => Float64.(sequence.DEF["FOV"]),
        "Venc_m_per_s" => Float64(sequence.DEF["Venc"]),
        "velocity_axis" => String(sequence.DEF["VelocityEncodingAxis"]),
        "file" => sequence_files[encoding],
        "sha256" => sequence_sha256[encoding],
    )
    result_file = joinpath(run_dir, "$(lowercase(encoding))_result.jld2")
    result_provenance = encoding_provenance(
        encoding,
        "encoding_result";
        extra=Dict(
            "image_shape" => [nx, ny, nz, nphases],
            "spin_batch_size_requested" => batch_size,
        ),
    )
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
        batch_metadata[encoding] = JLD2.load(
            result_file,
            "batch_metadata_for_encoding",
        )
        result_files[encoding] = result_file
        encoding_runtime[encoding] = 0.0
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
            encoding_provenance(encoding, "encoding_simulation"),
        )
    else
        raw = simulate(phantom, sequence, scanner; sim_params)
        batch_metadata[encoding] = Dict(
            "batch_count" => 1,
            "batch_size_requested" => batch_size,
            "provenance_schema_version" => PROVENANCE_SCHEMA_VERSION,
            "phantom_sha256" => phantom_sha256,
            "sequence_sha256" => sequence_sha256[encoding],
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
        batch_metadata_for_encoding=batch_metadata[encoding],
        elapsed_seconds=elapsed,
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

reference_reconstruction_file = ""
reference_figure_file = ""
reference_qc = Dict{String,Any}()

if "REF" in requested_encodings
    reference = JLD2.load(result_files["REF"], "image")
    magnitude = Float32.(abs.(reference))
    @assert all(isfinite, magnitude)
    @assert maximum(magnitude) > 0
    reference_sequence = sequences["REF"]
    reference_reconstruction_file = joinpath(
        run_dir,
        "reference_reconstruction.h5",
    )
    temporary = reference_reconstruction_file * ".partial"
    h5open(temporary, "w") do file
        write(file, "reference_magnitude", magnitude)
        HDF5.attributes(file)["dimension_order_julia"] = "readout,phase,slice,cardiac_bin"
        HDF5.attributes(file)["FOV_m"] = Float32.(reference_sequence.DEF["FOV"])
        HDF5.attributes(file)["matrix"] = Int32.([
            reference_sequence.DEF["Nx"],
            reference_sequence.DEF["Ny"],
            reference_sequence.DEF["Nz"],
        ])
        HDF5.attributes(file)["Venc_m_per_s"] = Float32(reference_sequence.DEF["Venc"])
        HDF5.attributes(file)["RR_s"] = Float32(reference_sequence.DEF["RR"])
        HDF5.attributes(file)["provenance_schema_version"] = PROVENANCE_SCHEMA_VERSION
        HDF5.attributes(file)["run_name"] = run_name
        HDF5.attributes(file)["physical_gpu_index"] = physical_gpu_index
        HDF5.attributes(file)["source_phantom_sha256"] = phantom_sha256
        HDF5.attributes(file)["source_sequence_sha256_REF"] = sequence_sha256["REF"]
        HDF5.attributes(file)["simulation_script_sha256"] = simulation_script_sha256
        HDF5.attributes(file)["perfect_spoiling_sha256"] = perfect_spoiling_sha256
    end
    mv(temporary, reference_reconstruction_file; force=true)

    phase_plots = Any[]
    for phase in 1:size(magnitude, 4)
        projection = dropdims(
            maximum(magnitude[:, :, :, phase]; dims=3);
            dims=3,
        )
        push!(phase_plots, Plots.heatmap(
            projection',
            color=:grays,
            axis=nothing,
            title="Reference magnitude | bin $phase",
            colorbar=false,
        ))
    end
    figure = Plots.plot(
        phase_plots...;
        layout=(length(phase_plots), 1),
        size=(1500, 350 * length(phase_plots)),
        dpi=160,
    )
    reference_figure_file = joinpath(run_dir, "reference_reconstruction_qc.png")
    Plots.savefig(figure, reference_figure_file)

    if phantom.motion isa NoMotion && size(magnitude, 4) >= 2
        first_bin = vec(magnitude[:, :, :, 1])
        second_bin = vec(magnitude[:, :, :, 2])
        threshold = 0.05f0 * max(maximum(first_bin), maximum(second_bin))
        support = (first_bin .>= threshold) .| (second_bin .>= threshold)
        first_values = first_bin[support]
        second_values = second_bin[support]
        first_normalized = first_values ./ mean(first_values)
        second_normalized = second_values ./ mean(second_values)
        normalized_rmse = sqrt(mean(abs2, first_normalized .- second_normalized))
        correlation = cor(first_values, second_values)
        static_passed = correlation >= 0.99 && normalized_rmse <= 0.10
        reference_qc["static_bin_consistency"] = Dict(
            "passed" => static_passed,
            "support_voxel_count" => count(support),
            "magnitude_correlation" => Float64(correlation),
            "mean_normalized_rmse" => Float64(normalized_rmse),
            "correlation_limit_min" => 0.99,
            "normalized_rmse_limit_max" => 0.10,
            "threshold_fraction_of_global_max" => 0.05,
        )
    end
    reference = nothing
    magnitude = nothing
    GC.gc()
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
        mask_threshold = 0.05f0 * maximum(phase_magnitude)
        @views mask[:, :, :, phase] .= phase_magnitude .>= mask_threshold
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
        HDF5.attributes(file)["phase_formula"] = velocity_qc["phase_formula"]
        HDF5.attributes(file)["provenance_schema_version"] = PROVENANCE_SCHEMA_VERSION
        HDF5.attributes(file)["run_name"] = run_name
        HDF5.attributes(file)["physical_gpu_index"] = physical_gpu_index
        HDF5.attributes(file)["source_phantom_sha256"] = phantom_sha256
        for encoding in valid_encodings
            HDF5.attributes(file)["source_sequence_sha256_$encoding"] =
                sequence_sha256[encoding]
        end
        HDF5.attributes(file)["simulation_script_sha256"] = simulation_script_sha256
        HDF5.attributes(file)["perfect_spoiling_sha256"] = perfect_spoiling_sha256
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
    Plots.savefig(figure, figure_file)

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
reference_passed = !haskey(reference_qc, "static_bin_consistency") ||
    reference_qc["static_bin_consistency"]["passed"]
velocity_passed = !haskey(velocity_qc, "static_zero_velocity_acceptance") ||
    velocity_qc["static_zero_velocity_acceptance"]["passed"]
passed = reference_passed && velocity_passed
report = Dict(
    "status" => passed ? "PASS" : "FAIL",
    "provenance_schema_version" => PROVENANCE_SCHEMA_VERSION,
    "run_name" => run_name,
    "phantom_file" => phantom_file,
    "phantom_sha256" => phantom_sha256,
    "particle_count" => length(phantom),
    "sequence_dir" => sequence_dir,
    "encodings" => requested_encodings,
    "gpu_device_zero_based" => gpu_device,
    "physical_gpu_index" => physical_gpu_index,
    "gpu_name" => CUDA.name(CUDA.device()),
    "simulation_method" => "Bloch with perfect transverse spoiling before RF",
    "simulation_script_sha256" => simulation_script_sha256,
    "perfect_spoiling_sha256" => perfect_spoiling_sha256,
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
    "total_runtime_seconds" => total_elapsed,
    "result_files" => result_files,
    "reference_reconstruction_file" => reference_reconstruction_file,
    "reference_qc_figure" => reference_figure_file,
    "reference_qc" => reference_qc,
    "reconstruction_file" => reconstruction_file,
    "qc_figure" => figure_file,
    "velocity_qc" => velocity_qc,
    "result_reuse_enabled" => reuse_results,
    "reused_encodings" => reused_encodings,
)
report_file = joinpath(run_dir, "simulation_report.toml")
temporary_report = report_file * ".partial"
open(temporary_report, "w") do io
    TOML.print(io, report; sorted=true)
end
mv(temporary_report, report_file; force=true)
passed || error("Simulation QC failed; see $report_file")
@info "Simulation workflow complete" report=report_file total_elapsed
