using Dates
using HDF5
using KomaMRI
using KomaMRICore
using Printf
using SHA
using Statistics
using TOML

const TIMING_AUDIT_ENCODINGS = ("REF", "X", "Y", "Z")
const TIMING_AUDIT_FAILURE = "FAIL_GLOBAL_MOTION_SEAM_OVERLAPS_ADC"
const TIMING_AUDIT_INCOMPLETE = "FAIL_TIMING_AUDIT_INCOMPLETE"
const TIMING_AUDIT_SCHEMA_VERSION = 1

struct SequenceTiming
    encoding::String
    path::String
    rr_s::Float64
    cardiac_bins::Int
    matrix::NTuple{3,Int}
    centers_s::Vector{Float64}
    first_samples_s::Vector{Float64}
    last_samples_s::Vector{Float64}
    dwells_s::Vector{Float64}
    sample_counts::Vector{Int}
    phs::Vector{Int}
    lin::Vector{Int}
    par::Vector{Int}
end

wrapped_delta(value, reference, period) =
    mod(value - reference + period / 2, period) - period / 2

function file_binding(path::AbstractString, repo_root::AbstractString)
    isfile(path) || throw(ArgumentError("Missing audit input: $path"))
    return Dict{String,Any}(
        "path" => replace(relpath(path, repo_root), '\\' => '/'),
        "bytes" => filesize(path),
        "sha256" => open(path, "r") do io
            bytes2hex(SHA.sha256(io))
        end,
    )
end

function read_sequence_timing(path::AbstractString, encoding::AbstractString)
    sequence = read_seq(path; verify_signature=true, verbose=false)
    starts = get_block_start_times(sequence)
    labels = get_labels(sequence)
    adc_blocks = findall(adc -> adc.N > 0, sequence.ADC)

    centers_s = Float64[]
    first_samples_s = Float64[]
    last_samples_s = Float64[]
    dwells_s = Float64[]
    sample_counts = Int[]
    phs = Int[]
    lin = Int[]
    par = Int[]
    sizehint!(centers_s, length(adc_blocks))
    sizehint!(first_samples_s, length(adc_blocks))
    sizehint!(last_samples_s, length(adc_blocks))
    sizehint!(dwells_s, length(adc_blocks))
    sizehint!(sample_counts, length(adc_blocks))
    sizehint!(phs, length(adc_blocks))
    sizehint!(lin, length(adc_blocks))
    sizehint!(par, length(adc_blocks))

    for block in adc_blocks
        adc = sequence.ADC[block]
        dwell_s = Float64(dwell(adc))
        first_sample_s = Float64(starts[block] + adc.delay)
        center_s = first_sample_s + (adc.N ÷ 2) * dwell_s
        last_sample_s = first_sample_s + (adc.N - 1) * dwell_s
        push!(centers_s, center_s)
        push!(first_samples_s, first_sample_s)
        push!(last_samples_s, last_sample_s)
        push!(dwells_s, dwell_s)
        push!(sample_counts, adc.N)
        push!(phs, labels[block].PHS)
        push!(lin, labels[block].LIN)
        push!(par, labels[block].PAR)
    end

    matrix = (
        Int(sequence.DEF["Nx"]),
        Int(sequence.DEF["Ny"]),
        Int(sequence.DEF["Nz"]),
    )
    return SequenceTiming(
        String(encoding),
        normpath(path),
        Float64(sequence.DEF["RR"]),
        Int(sequence.DEF["CardiacBins"]),
        matrix,
        centers_s,
        first_samples_s,
        last_samples_s,
        dwells_s,
        sample_counts,
        phs,
        lin,
        par,
    )
end

function samples_around_boundary(
    first_sample_s::Real,
    dwell_s::Real,
    sample_count::Integer,
    boundary_s::Real,
)
    before = count(
        sample -> first_sample_s + sample * dwell_s < boundary_s,
        0:(sample_count - 1),
    )
    at = count(
        sample -> first_sample_s + sample * dwell_s == boundary_s,
        0:(sample_count - 1),
    )
    return (before=before, at=at, after=sample_count - before - at)
end

function signed_adc_seam_margin_s(
    first_sample_s,
    last_sample_s,
    period_s;
    seam_offset_s=0.0,
)
    first_cycle = floor(Int, (first_sample_s - seam_offset_s) / period_s)
    last_cycle = floor(Int, (last_sample_s - seam_offset_s) / period_s)
    candidate_boundaries = (
        seam_offset_s .+
        ((first_cycle - 1):(last_cycle + 1)) .* period_s
    )
    margins = Float64[]
    for boundary_s in candidate_boundaries
        if first_sample_s <= boundary_s <= last_sample_s
            push!(margins, -min(
                boundary_s - first_sample_s,
                last_sample_s - boundary_s,
            ))
        else
            push!(margins, min(
                abs(boundary_s - first_sample_s),
                abs(boundary_s - last_sample_s),
            ))
        end
    end
    return minimum(margins)
end

function nearest_phase_mapping(center_mod_period_s, period_s, cardiac_bins)
    mapping = Int[]
    errors_s = Float64[]
    for true_bin in 0:(cardiac_bins - 1)
        target = true_bin * period_s / cardiac_bins
        distances = abs.(wrapped_delta.(center_mod_period_s, target, period_s))
        saved_phs = argmin(distances) - 1
        push!(mapping, saved_phs)
        push!(errors_s, wrapped_delta(
            center_mod_period_s[saved_phs + 1],
            target,
            period_s,
        ))
    end
    length(unique(mapping)) == cardiac_bins || error(
        "Nearest-grid phase mapping is not one-to-one.",
    )
    return mapping, errors_s
end

function interval_phase_mapping(center_mod_period_s, period_s, cardiac_bins)
    mapping = fill(-1, cardiac_bins)
    for (saved_index, center_s) in enumerate(center_mod_period_s)
        true_bin = mod(floor(Int, center_s / (period_s / cardiac_bins)), cardiac_bins)
        mapping[true_bin + 1] == -1 || error(
            "Interval phase mapping assigns multiple saved PHS values to bin $true_bin.",
        )
        mapping[true_bin + 1] = saved_index - 1
    end
    all(>=(0), mapping) || error("Interval phase mapping is incomplete.")
    return mapping
end

function summarize_sequence_timing(timing::SequenceTiming, motion_period_s::Real)
    bins = timing.cardiac_bins
    profile_count = length(timing.centers_s)
    profile_count % bins == 0 || error(
        "$(timing.encoding) ADC profile count is not divisible by cardiac bins.",
    )
    groups = profile_count ÷ bins
    expected_groups = timing.matrix[2] * timing.matrix[3]
    centers = reshape(timing.centers_s, bins, groups)
    first_samples = reshape(timing.first_samples_s, bins, groups)
    last_samples = reshape(timing.last_samples_s, bins, groups)
    dwells = reshape(timing.dwells_s, bins, groups)
    sample_counts = reshape(timing.sample_counts, bins, groups)
    phs = reshape(timing.phs, bins, groups)
    lin = reshape(timing.lin, bins, groups)
    par = reshape(timing.par, bins, groups)

    expected_phs = reshape(collect(0:(bins - 1)), bins, 1)
    phs_repeats = all(phs .== expected_phs)
    spatial_labels_constant_within_group = all(
        all(lin[:, group] .== lin[1, group]) &&
        all(par[:, group] .== par[1, group])
        for group in 1:groups
    )
    spatial_group_count = length(Set(
        (lin[1, group], par[1, group]) for group in 1:groups
    ))

    first_mod_rr = mod.(centers[:, 1], timing.rr_s)
    last_mod_rr = mod.(centers[:, end], timing.rr_s)
    max_group_rr_drift_s = maximum(
        abs(wrapped_delta(
            mod(centers[phase, group], timing.rr_s),
            first_mod_rr[phase],
            timing.rr_s,
        ))
        for phase in 1:bins, group in 1:groups
    )

    first_mod_motion = mod.(centers[:, 1], motion_period_s)
    last_mod_motion = mod.(centers[:, end], motion_period_s)
    last_group_motion_drift_s = [
        wrapped_delta(last_mod_motion[phase], first_mod_motion[phase], motion_period_s)
        for phase in 1:bins
    ]

    crossings_by_phs = zeros(Int, bins)
    signed_seam_margins_s = Float64[]
    samples_before = [Int[] for _ in 1:bins]
    samples_after = [Int[] for _ in 1:bins]
    crossing_examples = [Dict{String,Any}[] for _ in 1:bins]
    for group in 1:groups, phase in 1:bins
        first_sample_s = first_samples[phase, group]
        last_sample_s = last_samples[phase, group]
        push!(signed_seam_margins_s, signed_adc_seam_margin_s(
            first_sample_s,
            last_sample_s,
            motion_period_s,
        ))
        first_cycle = floor(Int, first_sample_s / motion_period_s)
        last_cycle = floor(Int, last_sample_s / motion_period_s)
        first_cycle == last_cycle && continue
        last_cycle == first_cycle + 1 || error(
            "An ADC readout crosses more than one motion boundary.",
        )
        boundary_s = (first_cycle + 1) * motion_period_s
        counts = samples_around_boundary(
            first_sample_s,
            dwells[phase, group],
            sample_counts[phase, group],
            boundary_s,
        )
        crossings_by_phs[phase] += 1
        push!(samples_before[phase], counts.before)
        push!(samples_after[phase], counts.after)
        if length(crossing_examples[phase]) < 2
            push!(crossing_examples[phase], Dict{String,Any}(
                "spatial_group_one_based" => group,
                "saved_phs_zero_based" => phase - 1,
                "first_sample_s" => first_sample_s,
                "center_sample_s" => centers[phase, group],
                "last_sample_s" => last_sample_s,
                "motion_boundary_s" => boundary_s,
                "samples_before_boundary" => counts.before,
                "samples_exactly_at_boundary" => counts.at,
                "samples_after_boundary" => counts.after,
            ))
        end
    end

    crossing_sample_ranges = Dict{String,Any}()
    for phase in 1:bins
        isempty(samples_before[phase]) && continue
        crossing_sample_ranges[string(phase - 1)] = Dict{String,Any}(
            "before_min" => minimum(samples_before[phase]),
            "before_max" => maximum(samples_before[phase]),
            "after_min" => minimum(samples_after[phase]),
            "after_max" => maximum(samples_after[phase]),
            "examples" => crossing_examples[phase],
        )
    end

    nearest_mapping, _ = nearest_phase_mapping(
        first_mod_motion,
        motion_period_s,
        bins,
    )
    saved_phs_to_true_bin = fill(-1, bins)
    for (true_index, saved_phs) in enumerate(nearest_mapping)
        saved_phs_to_true_bin[saved_phs + 1] = true_index - 1
    end
    all(>=(0), saved_phs_to_true_bin) || error(
        "Nearest-grid inverse phase mapping is incomplete.",
    )
    motion_period_f32 = Float32(motion_period_s)
    motion_period_from_f32_s = Float64(motion_period_f32)
    f32_phase_residual_s = Matrix{Float64}(undef, bins, groups)
    for group in 1:groups, phase in 1:bins
        simulated_time_s = Float32(centers[phase, group])
        simulated_motion_phase_s = Float64(mod(
            simulated_time_s,
            motion_period_f32,
        ))
        target_s = saved_phs_to_true_bin[phase] *
            motion_period_from_f32_s / bins
        f32_phase_residual_s[phase, group] = wrapped_delta(
            simulated_motion_phase_s,
            target_s,
            motion_period_from_f32_s,
        )
    end
    minimum_residual_index = argmin(f32_phase_residual_s)
    maximum_residual_index = argmax(f32_phase_residual_s)
    minimum_residual_s = f32_phase_residual_s[minimum_residual_index]
    maximum_residual_s = f32_phase_residual_s[maximum_residual_index]
    per_phs_residual_span_s = [
        maximum(f32_phase_residual_s[phase, :]) -
        minimum(f32_phase_residual_s[phase, :])
        for phase in 1:bins
    ]

    return Dict{String,Any}(
        "encoding" => timing.encoding,
        "matrix_readout_phase_slice" => collect(timing.matrix),
        "cardiac_bins" => bins,
        "adc_profiles" => profile_count,
        "spatial_groups" => groups,
        "expected_spatial_groups" => expected_groups,
        "spatial_group_count_from_labels" => spatial_group_count,
        "phs_zero_through_fourteen_repeats_for_every_group" => phs_repeats,
        "spatial_labels_constant_within_each_group" =>
            spatial_labels_constant_within_group,
        "first_group_center_modulo_sequence_rr_s" => first_mod_rr,
        "last_group_center_modulo_sequence_rr_s" => last_mod_rr,
        "maximum_absolute_group_modulo_rr_drift_s" => max_group_rr_drift_s,
        "first_group_center_modulo_motion_period_s" => first_mod_motion,
        "last_group_center_modulo_motion_period_s" => last_mod_motion,
        "last_group_motion_phase_drift_from_first_group_s" =>
            last_group_motion_drift_s,
        "maximum_absolute_last_group_motion_phase_drift_s" =>
            maximum(abs, last_group_motion_drift_s),
        "minimum_signed_adc_seam_margin_s" => minimum(signed_seam_margins_s),
        "minimum_signed_adc_seam_margin_ms" =>
            1e3 * minimum(signed_seam_margins_s),
        "signed_adc_seam_margin_definition" =>
            "Positive is the minimum separation between an ADC window and a " *
            "periodic seam; negative means the seam lies inside an ADC window, " *
            "with magnitude equal to the nearer endpoint distance.",
        "float32_absolute_time_phase_residual" => Dict{String,Any}(
            "definition" =>
                "wrapped_delta(Float64(mod(Float32(adc_center_absolute_s), " *
                "Float32(T))), nearest_true_bin*T/15, Float64(Float32(T)))",
            "minimum_s" => minimum_residual_s,
            "maximum_s" => maximum_residual_s,
            "global_span_s" => maximum_residual_s - minimum_residual_s,
            "global_span_ms" => 1e3 * (maximum_residual_s - minimum_residual_s),
            "minimum_location" => Dict{String,Any}(
                "saved_phs_zero_based" => minimum_residual_index[1] - 1,
                "spatial_group_one_based" => minimum_residual_index[2],
                "adc_center_float64_s" => centers[minimum_residual_index],
                "adc_center_float32_as_float64_s" =>
                    Float64(Float32(centers[minimum_residual_index])),
            ),
            "maximum_location" => Dict{String,Any}(
                "saved_phs_zero_based" => maximum_residual_index[1] - 1,
                "spatial_group_one_based" => maximum_residual_index[2],
                "adc_center_float64_s" => centers[maximum_residual_index],
                "adc_center_float32_as_float64_s" =>
                    Float64(Float32(centers[maximum_residual_index])),
            ),
            "per_saved_phs_residual_span_s" => per_phs_residual_span_s,
            "interpretation" =>
                "Casting long absolute sequence times to Float32 adds " *
                "line-dependent cardiac-phase residual beyond the fixed " *
                "Float32-period mismatch.",
        ),
        "motion_boundary_crossings_by_saved_phs_zero_based" =>
            crossings_by_phs,
        "crossing_sample_count_ranges_by_saved_phs" => crossing_sample_ranges,
        "checks" => Dict{String,Any}(
            "profile_count_matches_matrix_times_bins" =>
                profile_count == expected_groups * bins,
            "spatial_group_count_matches_matrix" => groups == expected_groups,
            "spatial_labels_cover_every_group_once" =>
                spatial_group_count == expected_groups,
            "phs_order_is_uniform" => phs_repeats,
            "spatial_labels_are_group_constant" =>
                spatial_labels_constant_within_group,
            "no_adc_overlaps_motion_boundary" => sum(crossings_by_phs) == 0,
        ),
    )
end

function cross_encoding_timing(timings)
    reference = first(timings)
    max_center_difference_s = 0.0
    max_first_difference_s = 0.0
    max_last_difference_s = 0.0
    labels_identical = true
    definitions_identical = true
    for timing in Iterators.drop(timings, 1)
        length(timing.centers_s) == length(reference.centers_s) ||
            return Dict{String,Any}(
                "identical" => false,
                "reason" => "ADC profile counts differ across encodings",
            )
        max_center_difference_s = max(
            max_center_difference_s,
            maximum(abs.(timing.centers_s .- reference.centers_s)),
        )
        max_first_difference_s = max(
            max_first_difference_s,
            maximum(abs.(timing.first_samples_s .- reference.first_samples_s)),
        )
        max_last_difference_s = max(
            max_last_difference_s,
            maximum(abs.(timing.last_samples_s .- reference.last_samples_s)),
        )
        labels_identical &= timing.phs == reference.phs
        labels_identical &= timing.lin == reference.lin
        labels_identical &= timing.par == reference.par
        definitions_identical &= timing.rr_s == reference.rr_s
        definitions_identical &= timing.cardiac_bins == reference.cardiac_bins
        definitions_identical &= timing.matrix == reference.matrix
    end
    return Dict{String,Any}(
        "identical" =>
            max_center_difference_s == 0 &&
            max_first_difference_s == 0 &&
            max_last_difference_s == 0 &&
            labels_identical &&
            definitions_identical,
        "maximum_absolute_adc_center_difference_s" => max_center_difference_s,
        "maximum_absolute_first_sample_difference_s" => max_first_difference_s,
        "maximum_absolute_last_sample_difference_s" => max_last_difference_s,
        "labels_identical" => labels_identical,
        "definitions_identical" => definitions_identical,
    )
end

function voxel_occupancy(positions, fov, matrix)
    counts = zeros(Int32, matrix)
    for particle in axes(positions, 2)
        indices = ntuple(3) do axis
            coordinate = positions[axis, particle]
            index = floor(
                Int,
                (coordinate + fov[axis] / 2) / fov[axis] * matrix[axis],
            ) + 1
            clamp(index, 1, matrix[axis])
        end
        counts[indices...] += 1
    end
    return counts
end

function endpoint_discontinuity(motion_h5, fov, matrix, motion_report)
    initial, endpoint, last_reset_count, nsteps, source_dt_s = h5open(
        motion_h5,
        "r",
    ) do file
        initial = read(file, "initial_position")
        position = file["position"]
        endpoint = position[:, :, size(position, 3)]
        reset = file["reset"]
        last_reset_count = sum(reset[:, size(reset, 2)])
        nsteps = Int(read_attribute(file, "n_steps"))
        source_dt_s = Float64(read_attribute(file, "dt"))
        initial, endpoint, last_reset_count, nsteps, source_dt_s
    end

    displacement_m = vec(sqrt.(sum((endpoint .- initial) .^ 2; dims=1)))
    initial_occupancy = voxel_occupancy(initial, fov, matrix)
    endpoint_occupancy = voxel_occupancy(endpoint, fov, matrix)
    union_mask = (initial_occupancy .> 0) .| (endpoint_occupancy .> 0)
    occupancy_correlation = cor(
        Float64.(initial_occupancy[union_mask]),
        Float64.(endpoint_occupancy[union_mask]),
    )
    total_particles = sum(initial_occupancy)
    total_variation = sum(abs.(endpoint_occupancy .- initial_occupancy)) /
        (2 * total_particles)

    return Dict{String,Any}(
        "source_poststep_count" => nsteps,
        "source_dt_s" => source_dt_s,
        "particle_count" => length(displacement_m),
        "mean_endpoint_to_initial_displacement_m" => mean(displacement_m),
        "median_endpoint_to_initial_displacement_m" => median(displacement_m),
        "minimum_endpoint_to_initial_displacement_m" => minimum(displacement_m),
        "maximum_endpoint_to_initial_displacement_m" => maximum(displacement_m),
        "quantiles_endpoint_to_initial_displacement_m" => Dict{String,Any}(
            "p01" => quantile(displacement_m, 0.01),
            "p05" => quantile(displacement_m, 0.05),
            "p50" => quantile(displacement_m, 0.50),
            "p95" => quantile(displacement_m, 0.95),
            "p99" => quantile(displacement_m, 0.99),
        ),
        "fraction_displacement_gt_25_4_mm" =>
            count(>(0.0254), displacement_m) / length(displacement_m),
        "fraction_displacement_gt_1_25_mm" =>
            count(>(0.00125), displacement_m) / length(displacement_m),
        "tracker_reset_count_at_final_source_sample" => Int(last_reset_count),
        "tracker_reset_events_reported" =>
            Int(motion_report["tracker_reset_events"]),
        "forced_cycle_boundary_reset_count_reported" =>
            Int(motion_report["cycle_boundary_reset_count"]),
        "total_flowpath_reset_flags_reported" =>
            Int(motion_report["total_flowpath_reset_flags"]),
        "initial_occupied_voxels" => count(>(0), initial_occupancy),
        "endpoint_occupied_voxels" => count(>(0), endpoint_occupancy),
        "union_occupied_voxels" => count(union_mask),
        "initial_endpoint_occupancy_correlation_on_union" =>
            occupancy_correlation,
        "occupancy_total_variation_fraction" => total_variation,
        "interpretation" =>
            "The stored particle identities and voxel occupancy do not close at the " *
            "periodic seam; forcing every endpoint reset changes magnetization but " *
            "does not make coordinates continuous.",
    )
end

function koma_reset_semantics_probe(reference_sequence, motion_period_s, sample_count)
    sequence = read_seq(reference_sequence; verify_signature=true, verbose=false)
    sequence = sequence[1:min(200, length(sequence))]
    endpoint_reset = reshape([falses(sample_count - 1); true], 1, :)
    flow = FlowPath(
        dx=zeros(Float32, 1, sample_count),
        dy=zeros(Float32, 1, sample_count),
        dz=zeros(Float32, 1, sample_count),
        spin_reset=endpoint_reset,
    )
    motion = Motion(
        action=flow,
        time=Periodic(period=Float32(motion_period_s), asymmetry=1.0f0),
        spins=SpinRange(1:1),
    )
    discrete = discretize(sequence; motion)
    parts, _ = KomaMRICore.get_sim_ranges(
        discrete;
        max_block_length=512,
        max_rf_block_length=Inf,
        eval_intervals_per_step=1,
    )

    candidates = Dict{String,Any}[]
    for (part_index, part) in enumerate(parts)
        first(discrete.t[part]) <= motion_period_s <= last(discrete.t[part]) ||
            continue
        unit_t = KomaMRICore.KomaMRIBase.unit_time(discrete.t[part], motion.time)
        reset_mask = vec(KomaMRICore.get_mask(endpoint_reset, unit_t))
        reset_indices = findall(!iszero, reset_mask)
        push!(candidates, Dict{String,Any}(
            "simulation_part_one_based" => part_index,
            "discrete_range_first" => first(part),
            "discrete_range_last" => last(part),
            "part_start_s" => first(discrete.t[part]),
            "part_end_s" => last(discrete.t[part]),
            "part_points" => length(part),
            "adc_samples_in_part" => sum(discrete.ADC[part]),
            "reset_mask_true_points" => length(reset_indices),
            "first_reset_mask_true_s" => isempty(reset_indices) ? nothing :
                first(discrete.t[part][reset_indices]),
            "last_reset_mask_true_s" => isempty(reset_indices) ? nothing :
                last(discrete.t[part][reset_indices]),
            "reset_mask_at_part_end" => !iszero(last(reset_mask)),
        ))
    end
    isempty(candidates) && error("Koma reset semantics probe did not span the seam.")

    core_source = dirname(pathof(KomaMRICore))
    base_source = dirname(pathof(KomaMRICore.KomaMRIBase))
    return Dict{String,Any}(
        "koma_mri_version" => string(Base.pkgversion(KomaMRI)),
        "koma_mri_core_version" => string(Base.pkgversion(KomaMRICore)),
        "simulation_max_block_length" => 512,
        "probe_parts_spanning_first_motion_boundary" => candidates,
        "magnetization_reset_rule" =>
            "Flow.jl samples the reset mask but applies only mask[:, end] to " *
            "the carried magnetization after the GPU signal kernel.",
        "coordinate_rule" =>
            "ArbitraryAction.jl linearly interpolates dx/dy/dz and does not " *
            "consult spin_reset.",
        "source_files" => Dict{String,Any}(
            "flow_reset" => normpath(joinpath(core_source, "simulation", "Flow.jl")),
            "bloch_gpu" => normpath(joinpath(
                core_source,
                "simulation",
                "SimMethods",
                "Bloch",
                "gpu",
                "BlochGPU.jl",
            )),
            "arbitrary_action" => normpath(joinpath(
                base_source,
                "motion",
                "actions",
                "ArbitraryAction.jl",
            )),
        ),
        "observed_current_segmentation_consequence" =>
            "The endpoint reset mask is true only at internal points of the " *
            "seam-spanning precession part and false at that part's end, so the " *
            "intended global magnetization reset is not applied there; coordinates " *
            "still wrap inside the ADC readout.",
    )
end

function expected_raw_hashes(production_audit_json)
    text = read(production_audit_json, String)
    hashes = Dict{String,String}()
    for encoding in TIMING_AUDIT_ENCODINGS
        filename = lowercase(encoding) * "_result.jld2"
        pattern = Regex(
            "\\\"" * replace(filename, "." => "\\.") *
            "\\\"\\s*:\\s*\\{[^}]*\\\"digest\\\"\\s*:\\s*\\\"([0-9a-f]{64})\\\"",
        )
        match_result = match(pattern, text)
        isnothing(match_result) && error(
            "Could not read expected hash for $filename from production audit.",
        )
        hashes[encoding] = match_result.captures[1]
    end
    return hashes
end

function default_audit_paths()
    tracks_root = normpath(joinpath(@__DIR__, ".."))
    repo_root = dirname(tracks_root)
    run_dir = normpath(joinpath(
        tracks_root,
        "outputs",
        "dp150_mri_pipeline",
        "koma",
        "simulation_motion_1m_scanner_matched_venc5",
    ))
    sequence_dir = normpath(joinpath(
        @__DIR__,
        "pc_3d_scanner_matched_venc5",
    ))
    output_dir = normpath(joinpath(
        run_dir,
        "final_image_analysis",
        "motion_timing_diagnostic",
    ))
    return Dict{String,Any}(
        "repo_root" => repo_root,
        "tracks_root" => tracks_root,
        "run_dir" => run_dir,
        "sequence_dir" => sequence_dir,
        "output_dir" => output_dir,
        "output_json" => joinpath(output_dir, "motion_timing_audit.json"),
        "simulation_report" => joinpath(run_dir, "simulation_report.toml"),
        "motion_report" => joinpath(
            tracks_root,
            "outputs",
            "dp150_mri_pipeline",
            "koma",
            "motion_phantom_report.toml",
        ),
        "motion_h5" => joinpath(
            tracks_root,
            "outputs",
            "dp150_mri_pipeline",
            "dp150_motion_1m_mri.h5",
        ),
        "motion_phantom" => joinpath(
            tracks_root,
            "outputs",
            "dp150_mri_pipeline",
            "koma",
            "dp150_motion_1m.phantom",
        ),
        "reconstruction" => joinpath(run_dir, "pc_reconstruction.h5"),
        "production_audit" => joinpath(
            run_dir,
            "final_image_analysis",
            "production_data_audit.json",
        ),
    )
end

function build_timing_audit(paths=default_audit_paths())
    repo_root = paths["repo_root"]
    sequence_dir = paths["sequence_dir"]
    run_dir = paths["run_dir"]
    simulation_report = TOML.parsefile(paths["simulation_report"])
    motion_report = TOML.parsefile(paths["motion_report"])
    motion_period_s = Float64(simulation_report["motion_timing"]["effective_period_s"])

    sequence_paths = Dict(
        encoding => joinpath(sequence_dir, lowercase(encoding) * ".seq")
        for encoding in TIMING_AUDIT_ENCODINGS
    )
    result_paths = Dict(
        encoding => joinpath(run_dir, lowercase(encoding) * "_result.jld2")
        for encoding in TIMING_AUDIT_ENCODINGS
    )
    timings = [
        read_sequence_timing(sequence_paths[encoding], encoding)
        for encoding in TIMING_AUDIT_ENCODINGS
    ]
    summaries = Dict(
        timing.encoding => summarize_sequence_timing(timing, motion_period_s)
        for timing in timings
    )
    cross_encoding = cross_encoding_timing(timings)

    reference = first(timings)
    reference_mod_motion = mod.(
        reference.centers_s[1:reference.cardiac_bins],
        motion_period_s,
    )
    nearest_mapping, nearest_errors_s = nearest_phase_mapping(
        reference_mod_motion,
        motion_period_s,
        reference.cardiac_bins,
    )
    interval_mapping = interval_phase_mapping(
        reference_mod_motion,
        motion_period_s,
        reference.cardiac_bins,
    )
    source_period_s = Float64(motion_report["source_cycle_s"])
    source_phase_s = reference_mod_motion ./ motion_period_s .* source_period_s

    protocol_audit = TOML.parsefile(joinpath(sequence_dir, "protocol_audit.toml"))
    expected_result_hash = expected_raw_hashes(paths["production_audit"])
    sequence_bindings = Dict{String,Any}()
    result_bindings = Dict{String,Any}()
    for encoding in TIMING_AUDIT_ENCODINGS
        sequence_binding = file_binding(sequence_paths[encoding], repo_root)
        sequence_binding["expected_sha256_from_protocol_audit"] =
            protocol_audit["encodings"][encoding]["sha256"]
        sequence_binding["matches_expected_sha256"] =
            sequence_binding["sha256"] ==
            sequence_binding["expected_sha256_from_protocol_audit"]
        sequence_bindings[encoding] = sequence_binding

        result_binding = file_binding(result_paths[encoding], repo_root)
        result_binding["expected_sha256_from_production_data_audit"] =
            expected_result_hash[encoding]
        result_binding["matches_expected_sha256"] =
            result_binding["sha256"] ==
            result_binding["expected_sha256_from_production_data_audit"]
        result_bindings[encoding] = result_binding
    end

    fov = Tuple(Float64.(simulation_report["sequence"]["REF"]["FOV_m"]))
    matrix = Tuple(Int.(simulation_report["sequence"]["REF"]["matrix"]))
    endpoint = endpoint_discontinuity(
        paths["motion_h5"],
        fov,
        matrix,
        motion_report,
    )
    semantics_probe = koma_reset_semantics_probe(
        sequence_paths["REF"],
        motion_period_s,
        Int(endpoint["source_poststep_count"]) + 1,
    )

    total_crossings = sum(
        sum(summary["motion_boundary_crossings_by_saved_phs_zero_based"])
        for summary in values(summaries)
    )
    phs8_crossings_each = Dict(
        encoding => summaries[encoding][
            "motion_boundary_crossings_by_saved_phs_zero_based"
        ][9]
        for encoding in TIMING_AUDIT_ENCODINGS
    )
    expected_groups = reference.matrix[2] * reference.matrix[3]
    uniform_current_fault = all(
        phs8_crossings_each[encoding] == expected_groups &&
        sum(summaries[encoding][
            "motion_boundary_crossings_by_saved_phs_zero_based"
        ]) == expected_groups
        for encoding in TIMING_AUDIT_ENCODINGS
    )

    rr_s = reference.rr_s
    period_difference_s = motion_period_s - rr_s
    last_group_drift_s = (expected_groups - 1) * (rr_s - motion_period_s)
    minimum_adc_seam_margin_ms = minimum(
        summaries[encoding]["minimum_signed_adc_seam_margin_ms"]
        for encoding in TIMING_AUDIT_ENCODINGS
    )
    reset_probe_parts = semantics_probe[
        "probe_parts_spanning_first_motion_boundary"
    ]
    global_reset_applied_proof = !isempty(reset_probe_parts) && all(
        part["reset_mask_true_points"] > 0 && part["reset_mask_at_part_end"]
        for part in reset_probe_parts
    )
    bindings_pass = all(
        sequence_bindings[encoding]["matches_expected_sha256"] &&
        result_bindings[encoding]["matches_expected_sha256"]
        for encoding in TIMING_AUDIT_ENCODINGS
    )
    structural_checks_pass = all(
        all(
            value
            for (key, value) in summaries[encoding]["checks"]
            if key != "no_adc_overlaps_motion_boundary"
        )
        for encoding in TIMING_AUDIT_ENCODINGS
    )

    status = total_crossings > 0 ? TIMING_AUDIT_FAILURE : "PASS"
    status == "PASS" && !bindings_pass && (status = TIMING_AUDIT_INCOMPLETE)
    status == "PASS" && !cross_encoding["identical"] &&
        (status = TIMING_AUDIT_INCOMPLETE)

    return Dict{String,Any}(
        "schema_version" => TIMING_AUDIT_SCHEMA_VERSION,
        "audit_created_at" => string(now()),
        "status" => status,
        "scope" =>
            "Fail-closed timing and periodic-motion seam audit of the completed " *
            "DP150 1M scanner-matched VENC5 run.",
        "production_artifacts_modified" => false,
        "diagnostic_output_only" => true,
        "run_directory" => replace(relpath(run_dir, repo_root), '\\' => '/'),
        "verdict" => Dict{String,Any}(
            "scientific_use_of_current_phs8" => "REJECT",
            "global_motion_seam_overlaps_adc" => total_crossings > 0,
            "total_seam_crossing_readouts_all_encodings" => total_crossings,
            "phs8_crossings_each_encoding" => phs8_crossings_each,
            "fault_is_uniform_across_all_2048_groups_and_four_encodings" =>
                uniform_current_fault,
            "relabeling_or_masking_can_repair_crossed_raw_readout" => false,
            "new_simulation_required_for_corrected_phs8" => true,
        ),
        "pilot_acceptance_gates" => Dict{String,Any}(
            "status" => status,
            "observed" => Dict{String,Any}(
                "seam_crossing_readout_count" => total_crossings,
                "minimum_adc_seam_margin_ms" => minimum_adc_seam_margin_ms,
                "global_reset_applied_proof" => global_reset_applied_proof,
                "forced_global_reset_present" =>
                    endpoint["forced_cycle_boundary_reset_count_reported"] > 0,
            ),
            "requirements" => Dict{String,Any}(
                "seam_crossing_readout_count_equal" => 0,
                "minimum_adc_seam_margin_ms_greater_than" => 0.0,
                "global_reset_rule" =>
                    "If a controlled global reset remains in the immediate pilot, " *
                    "prove it is applied at an explicit simulation-part boundary " *
                    "outside every ADC window. The durable production fix is a " *
                    "genuinely identity-continuous periodic representation.",
            ),
            "checks" => Dict{String,Any}(
                "zero_seam_crossings" => total_crossings == 0,
                "positive_adc_seam_margin" => minimum_adc_seam_margin_ms > 0,
                "global_reset_applied_if_present" =>
                    endpoint["forced_cycle_boundary_reset_count_reported"] == 0 ||
                    global_reset_applied_proof,
            ),
        ),
        "artifact_bindings" => Dict{String,Any}(
            "all_expected_sequence_and_raw_hashes_match" => bindings_pass,
            "sequences" => sequence_bindings,
            "raw_results" => result_bindings,
            "reconstruction" => file_binding(paths["reconstruction"], repo_root),
            "simulation_report" => file_binding(
                paths["simulation_report"],
                repo_root,
            ),
            "motion_report" => file_binding(paths["motion_report"], repo_root),
            "source_motion_h5" => file_binding(paths["motion_h5"], repo_root),
            "motion_phantom" => file_binding(paths["motion_phantom"], repo_root),
            "production_data_audit" => file_binding(
                paths["production_audit"],
                repo_root,
            ),
            "build_motion_phantom_script" => file_binding(
                joinpath(@__DIR__, "build_motion_phantom.jl"),
                repo_root,
            ),
            "simulation_script" => file_binding(
                joinpath(@__DIR__, "simulate_pc.jl"),
                repo_root,
            ),
            "sequence_generator" => file_binding(
                joinpath(dirname(sequence_dir), "pc_3d_scanner_matched", "generate.jl"),
                repo_root,
            ),
            "sequence_builder" => file_binding(
                joinpath(@__DIR__, "PC.jl"),
                repo_root,
            ),
        ),
        "protocol_timing" => Dict{String,Any}(
            "sequence_rr_s" => rr_s,
            "effective_motion_period_s" => motion_period_s,
            "motion_period_minus_sequence_rr_s" => period_difference_s,
            "motion_period_minus_sequence_rr_ns" => 1e9 * period_difference_s,
            "spatial_groups" => expected_groups,
            "last_group_motion_phase_drift_s" => last_group_drift_s,
            "last_group_motion_phase_drift_us" => 1e6 * last_group_drift_s,
            "initial_ramp_shots" => 10,
            "ramp_duration_s" => 10 * Float64(
                simulation_report["sequence"]["REF"]["TR_s"],
            ),
            "trigger_channel" => nothing,
            "motion_time_origin" => "absolute sequence t=0",
        ),
        "sequence_uniformity" => Dict{String,Any}(
            "across_encodings" => cross_encoding,
            "encodings" => summaries,
            "non_seam_structural_checks_pass" => structural_checks_pass,
        ),
        "phase_mapping" => Dict{String,Any}(
            "saved_phs_zero_based" => collect(0:(reference.cardiac_bins - 1)),
            "first_group_center_modulo_motion_period_s" => reference_mod_motion,
            "first_group_center_mapped_to_source_cycle_s" => source_phase_s,
            "source_saved_phs_for_true_bin" => nearest_mapping,
            "source_saved_phs_for_true_bin_convention" =>
                "Nearest circular ADC-center sample to the true phase grid " *
                "q*T/15, with true bin 0 anchored at the motion seam.",
            "nearest_grid_signed_timing_errors_s" => nearest_errors_s,
            "chronological_interval_saved_phs_for_true_bin" => interval_mapping,
            "chronological_interval_convention" =>
                "floor(mod(center,T)/(T/15)); unlike nearest-grid mapping, this " *
                "starts with saved PHS9 because PHS8 center lies just before the seam.",
            "warning" =>
                "Saved PHS8 is the nearest sample to true phase zero, but its " *
                "400-sample ADC window crosses the seam and is not scientifically " *
                "recoverable by relabeling.",
        ),
        "global_motion_seam" => Dict{String,Any}(
            "crossings_each_encoding_by_saved_phs_zero_based" => Dict(
                encoding => summaries[encoding][
                    "motion_boundary_crossings_by_saved_phs_zero_based"
                ]
                for encoding in TIMING_AUDIT_ENCODINGS
            ),
            "representative_ref_crossing" => summaries["REF"][
                "crossing_sample_count_ranges_by_saved_phs"
            ]["8"],
            "cause" =>
                "The one-time 10-shot preparation advances absolute Koma motion " *
                "phase so saved PHS8 is centered 0.596 ms before the periodic seam; " *
                "its 2.850 ms sampled ADC window continues beyond the seam.",
        ),
        "endpoint_discontinuity" => endpoint,
        "koma_runtime_semantics_probe" => semantics_probe,
        "code_evidence" => Dict{String,Any}(
            "forced_global_reset" =>
                "koma_dp150/build_motion_phantom.jl:111-120",
            "flowpath_periodic_construction" =>
                "koma_dp150/build_motion_phantom.jl:134-138",
            "single_initial_ramp_without_trigger" => "koma_dp150/PC.jl:535-555",
            "saved_phs_sequence_order" => "koma_dp150/PC.jl:558-596",
            "koma_coordinate_linear_interpolation" =>
                "KomaMRIBase/src/motion/actions/ArbitraryAction.jl:10-24",
            "koma_reset_uses_block_end_mask" =>
                "KomaMRICore/src/simulation/Flow.jl:58-71",
            "koma_gpu_signal_before_carried_state_reset" =>
                "KomaMRICore/src/simulation/SimMethods/Bloch/gpu/BlochGPU.jl:95-116",
        ),
        "required_correction" => [
            "Treat current saved PHS8 as invalid raw acquisition, not a display hole.",
            "Align motion phase so no artificial periodic seam overlaps any ADC window.",
            "For the immediate 100k diagnostic pilot only, the global reset may be " *
            "retained and relocated as an explicitly controlled approximation, but " *
            "the jump gate must prove it occurs outside every ADC window and at an " *
            "applied simulation-part boundary.",
            "For durable production, replace the forced all-spin endpoint reset with " *
            "a genuinely identity-continuous periodic ensemble or a validated " *
            "reset-aware piecewise through-flow representation. Deleting the reset " *
            "alone while retaining a coordinate jump is not acceptable.",
            "Run this audit as a preflight gate; require zero seam-crossing ADCs.",
            "Validate a corrected 100k four-encoding pilot before a corrected 1M run.",
        ],
        "limitations" => [
            "This audit diagnoses timing, periodic representation, and provenance; " *
            "it does not claim that this seam is the sole cause of every black voxel.",
            "Other exclusions remain attributable to the hard 5% REF mask, finite " *
            "particle occupancy near boundaries, and REF-specific cancellation or " *
            "flow dephasing in the post-stenotic jet.",
        ],
    )
end

function json_string(io, value::AbstractString)
    print(io, '"')
    for character in value
        if character == '"'
            print(io, "\\\"")
        elseif character == '\\'
            print(io, "\\\\")
        elseif character == '\b'
            print(io, "\\b")
        elseif character == '\f'
            print(io, "\\f")
        elseif character == '\n'
            print(io, "\\n")
        elseif character == '\r'
            print(io, "\\r")
        elseif character == '\t'
            print(io, "\\t")
        elseif Int(character) < 0x20
            @printf(io, "\\u%04x", Int(character))
        else
            print(io, character)
        end
    end
    print(io, '"')
end

function write_json(io, value, indent=0)
    if value === nothing
        print(io, "null")
    elseif value isa Bool
        print(io, value ? "true" : "false")
    elseif value isa Integer
        print(io, value)
    elseif value isa AbstractFloat
        isfinite(value) || error("JSON audit refuses non-finite numbers.")
        @printf(io, "%.17g", Float64(value))
    elseif value isa AbstractString || value isa Symbol
        json_string(io, string(value))
    elseif value isa AbstractDict
        keys_sorted = sort!(string.(collect(keys(value))))
        normalized = Dict(string(key) => item for (key, item) in pairs(value))
        print(io, '{')
        if !isempty(keys_sorted)
            print(io, '\n')
            for (index, key) in enumerate(keys_sorted)
                print(io, repeat(" ", indent + 2))
                json_string(io, key)
                print(io, ": ")
                write_json(io, normalized[key], indent + 2)
                index == length(keys_sorted) || print(io, ',')
                print(io, '\n')
            end
            print(io, repeat(" ", indent))
        end
        print(io, '}')
    elseif value isa AbstractArray || value isa Tuple
        print(io, '[')
        if !isempty(value)
            print(io, '\n')
            for (index, item) in enumerate(value)
                print(io, repeat(" ", indent + 2))
                write_json(io, item, indent + 2)
                index == length(value) || print(io, ',')
                print(io, '\n')
            end
            print(io, repeat(" ", indent))
        end
        print(io, ']')
    else
        error("Unsupported JSON audit value type: $(typeof(value))")
    end
end

function atomic_write_json(path::AbstractString, value)
    output_dir = dirname(path)
    mkpath(output_dir)
    temporary, io = mktemp(output_dir; cleanup=false)
    try
        write_json(io, value)
        print(io, '\n')
        flush(io)
        close(io)
        mv(temporary, path; force=true)
    catch
        isopen(io) && close(io)
        isfile(temporary) && rm(temporary; force=true)
        rethrow()
    end
    return path
end

function run_timing_audit(paths=default_audit_paths())
    output_json = paths["output_json"]
    audit = try
        build_timing_audit(paths)
    catch error_value
        Dict{String,Any}(
            "schema_version" => TIMING_AUDIT_SCHEMA_VERSION,
            "audit_created_at" => string(now()),
            "status" => TIMING_AUDIT_INCOMPLETE,
            "production_artifacts_modified" => false,
            "diagnostic_output_only" => true,
            "error" => sprint(showerror, error_value),
        )
    end
    atomic_write_json(output_json, audit)
    status = audit["status"]
    println("Timing audit status: $status")
    println("Timing audit JSON: $output_json")
    return audit
end

function main()
    audit = run_timing_audit()
    audit["status"] == "PASS" && return 0
    return audit["status"] == TIMING_AUDIT_FAILURE ? 2 : 3
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
