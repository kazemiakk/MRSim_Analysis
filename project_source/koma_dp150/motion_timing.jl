using KomaMRI
using KomaMRICore

const DEFAULT_SEAM_RESET_MARKER_LEAD_S = 350e-6
const DEFAULT_SEAM_RESET_MARKER_DURATION_S = 2e-6
const KOMA_DELTA_T_KEY = string(Char(0x0394), "t")
const KOMA_DELTA_T_RF_KEY = KOMA_DELTA_T_KEY * "_rf"

function _best_typed_endpoint(start::T, requested_period_s::Real) where {T<:AbstractFloat}
    requested_period = Float64(requested_period_s)
    requested_period > 0 || error("Requested period must be positive.")
    nominal = T(Float64(start) + requested_period)
    candidates = unique(T[prevfloat(nominal), nominal, nextfloat(nominal)])
    filter!(candidate -> candidate > start, candidates)
    isempty(candidates) && error("No representable endpoint follows the start time.")
    errors = [
        abs((Float64(candidate) - Float64(start)) - requested_period)
        for candidate in candidates
    ]
    return candidates[argmin(errors)]
end

function _retime_curve(curve, new_start::T, requested_period_s::Real) where {T<:AbstractFloat}
    source_period = curve.t_end - curve.t_start
    source_period > 0 || error("Motion TimeCurve must have positive duration.")
    new_end = _best_typed_endpoint(new_start, requested_period_s)
    realized_period = new_end - new_start
    normalized = (curve.t .- curve.t_start) ./ source_period
    retimed = new_start .+ normalized .* realized_period
    retimed[firstindex(retimed)] = new_start
    retimed[lastindex(retimed)] = new_end
    return retimed
end

"""
    rescale_periodic_motion(motion, target_period)

Rescale a one-period Koma motion curve without changing its scalar type.  Koma
requires the `Motion`, action, and `TimeCurve` to share one scalar type, so a
`Float32` `FlowPath` necessarily receives the nearest representable `Float32`
period.  Callers that require scanner-timing alignment must audit that rounding
explicitly rather than silently promoting the very large action arrays.
"""
function rescale_periodic_motion(motion::Motion, target_period::Real)
    curve = motion.time
    curve.periodic || error("Motion time rescaling requires a periodic TimeCurve.")
    curve.periods isa Real || error(
        "Motion time rescaling does not support a vector of unequal periods.")
    isapprox(Float64(curve.periods), 1.0; rtol=0, atol=1e-7) || error(
        "Motion time rescaling expects one baseline period.")
    source_period = Float64(curve.t_end - curve.t_start)
    source_period > 0 || error("Motion TimeCurve must have positive duration.")

    T = eltype(curve.t)
    scaled_t = _retime_curve(curve, T(curve.t_start), target_period)
    scaled_curve = TimeCurve(
        t=scaled_t,
        t_unit=T.(curve.t_unit),
        periodic=true,
        periods=one(T),
        t_start=first(scaled_t),
        t_end=last(scaled_t),
    )
    scaled_motion = Motion(
        action=motion.action,
        time=scaled_curve,
        spins=motion.spins,
    )
    return scaled_motion, source_period, Float64(last(scaled_t) - first(scaled_t))
end

"""
    shift_periodic_motion_seam(motion, requested_seam_time)

Shift the start/end of a periodic `TimeCurve` while preserving its duration,
action arrays, and scalar type.  The returned realized seam is rounded to the
motion scalar type; the source `Motion` is not mutated.
"""
function shift_periodic_motion_seam(
    motion::Motion,
    requested_seam_time::Real,
    ;
    requested_period_s::Union{Nothing,Real}=nothing,
)
    curve = motion.time
    curve.periodic || error("Motion seam alignment requires a periodic TimeCurve.")
    curve.periods isa Real || error(
        "Motion seam alignment does not support a vector of unequal periods.")

    T = eltype(curve.t)
    realized_seam_time = T(requested_seam_time)
    target_period = isnothing(requested_period_s) ?
        Float64(curve.t_end - curve.t_start) : Float64(requested_period_s)
    shifted_t = _retime_curve(curve, realized_seam_time, target_period)
    shifted_curve = TimeCurve(
        t=shifted_t,
        t_unit=T.(curve.t_unit),
        periodic=true,
        periods=T(curve.periods),
        t_start=first(shifted_t),
        t_end=last(shifted_t),
    )
    shifted_motion = Motion(
        action=motion.action,
        time=shifted_curve,
        spins=motion.spins,
    )
    return (
        motion=shifted_motion,
        requested_seam_time_s=Float64(requested_seam_time),
        realized_seam_time_s=Float64(realized_seam_time),
        rounding_error_s=Float64(realized_seam_time) - Float64(requested_seam_time),
        requested_period_s=target_period,
        realized_period_s=Float64(last(shifted_t) - first(shifted_t)),
        realized_period_error_s=
            Float64(last(shifted_t) - first(shifted_t)) - target_period,
    )
end

"""Return the first ADC k-space-center time in seconds."""
function first_adc_kspace_center(sequence)
    starts = get_block_start_times(sequence)
    block = findfirst(adc -> adc.N > 0, sequence.ADC)
    isnothing(block) && error("Sequence has no ADC samples.")
    adc = sequence.ADC[block]
    center_sample = Int(adc.N) ÷ 2
    return Float64(starts[block] + adc.delay + center_sample * dwell(adc))
end

"""
    align_periodic_motion_seam_to_first_adc_midbin(
        motion, first_adc_center, sequence_rr, cardiac_bins)

Place the periodic seam one half cardiac-bin spacing before the first ADC
k-space center.  The requested position is calculated in `Float64`, then
rounded once to the motion scalar type by `shift_periodic_motion_seam`.
"""
function align_periodic_motion_seam_to_first_adc_midbin(
    motion::Motion,
    first_adc_center::Real,
    sequence_rr::Real,
    cardiac_bins::Integer,
)
    sequence_rr > 0 || error("Sequence RR must be positive.")
    cardiac_bins > 0 || error("Cardiac-bin count must be positive.")
    target_center_offset = Float64(sequence_rr) / (2 * Int(cardiac_bins))
    requested_seam = Float64(first_adc_center) - target_center_offset
    shifted = shift_periodic_motion_seam(
        motion,
        requested_seam;
        requested_period_s=sequence_rr,
    )
    realized_period = Float64(
        shifted.motion.time.t_end - shifted.motion.time.t_start,
    )
    realized_center_offset = mod(
        Float64(first_adc_center) - shifted.realized_seam_time_s,
        realized_period,
    )
    return merge(
        shifted,
        (
            target_first_center_offset_s=target_center_offset,
            realized_first_center_offset_s=realized_center_offset,
            first_center_offset_error_s=
                realized_center_offset - target_center_offset,
        ),
    )
end

"""Return immutable ADC timing/label records for an exact before/after check."""
function adc_timing_records(sequence)
    starts = get_block_start_times(sequence)
    labels = get_labels(sequence)
    return [
        (
            block=block,
            first_sample_s=Float64(starts[block] + adc.delay),
            dwell_s=Float64(dwell(adc)),
            sample_count=Int(adc.N),
            line_zero_based=Int(labels[block].LIN),
            partition_zero_based=Int(labels[block].PAR),
            phase_zero_based=Int(labels[block].PHS),
        )
        for (block, adc) in enumerate(sequence.ADC) if adc.N > 0
    ]
end

function _event_arrays_exactly_equal(left, right)
    size(left) == size(right) || return false
    return all(eachindex(left)) do index
        isapprox(left[index], right[index]; rtol=0, atol=0)
    end
end

function _periodic_seam_marker_times(
    sequence_duration_s::Real,
    first_seam_s::Real,
    period_s::Real,
    marker_count::Integer;
    lead_s::Real,
    duration_s::Real,
)
    sequence_duration = Float64(sequence_duration_s)
    first_seam = Float64(first_seam_s)
    period = Float64(period_s)
    lead = Float64(lead_s)
    marker_duration = Float64(duration_s)
    sequence_duration > 0 || error("Sequence duration must be positive.")
    period > 0 || error("Motion period must be positive.")
    marker_count > 0 || error("At least one reset marker is required.")
    0 < marker_duration < lead < period || error(
        "Require 0 < marker duration < marker lead < motion period.")
    marker_times = [
        first_seam + (index - 1) * period - lead
        for index in 1:Int(marker_count)
    ]
    first(marker_times) >= 0 || error("First reset marker occurs before the sequence.")
    last(marker_times) + marker_duration <= sequence_duration || error(
        "Last reset marker extends beyond the sequence duration.")
    return marker_times
end

"""
    insert_periodic_seam_reset_markers(sequence, first_seam, period, marker_count)

Return a copy of `sequence` with one simulation-only RF segmentation marker
inside the field-free interval immediately before each periodic motion seam.
The marker amplitude is `nextfloat(0.0)` in Float64, which makes the RF event
visible to Koma's block splitter but becomes exactly zero in the required
Float32 simulation.  No block duration, ADC event, label, gradient event, or
source sequence file is changed.

This is a controlled workaround for KomaMRICore 0.12.3 applying `FlowPath`
resets only at simulation-part boundaries.  It does not make discontinuous
particle identities physically continuous.
"""
function insert_periodic_seam_reset_markers(
    sequence,
    first_seam_s::Real,
    period_s::Real,
    marker_count::Integer;
    lead_s::Real=DEFAULT_SEAM_RESET_MARKER_LEAD_S,
    duration_s::Real=DEFAULT_SEAM_RESET_MARKER_DURATION_S,
)
    size(sequence.RF, 1) == 1 || error(
        "Reset-marker insertion currently requires a one-coil sequence.")
    marker_amplitude_t = nextfloat(0.0)
    marker_amplitude_t > 0 || error("Reset marker must be nonzero before discretization.")
    Float32(marker_amplitude_t) == 0 || error(
        "Reset marker must underflow to exact zero in Float32.")

    starts = get_block_start_times(sequence)
    original_adc = adc_timing_records(sequence)
    original_duration = Float64(dur(sequence))
    marker_times = _periodic_seam_marker_times(
        original_duration,
        first_seam_s,
        period_s,
        marker_count;
        lead_s,
        duration_s,
    )
    corrected = copy(sequence)
    used_blocks = Set{Int}()
    records = NamedTuple[]

    for (marker_index, marker_time) in enumerate(marker_times)
        block = searchsortedlast(starts, marker_time)
        1 <= block <= length(sequence) || error(
            "Reset marker $marker_index is outside the sequence blocks.")
        block in used_blocks && error(
            "More than one reset marker maps to sequence block $block.")
        block_start = Float64(starts[block])
        block_end = Float64(starts[block + 1])
        local_delay = marker_time - block_start
        clearance_before = local_delay
        clearance_after = block_end - (marker_time + Float64(duration_s))
        clearance_before > 0 && clearance_after > 0 || error(
            "Reset marker $marker_index does not fit strictly inside block $block.")
        KomaMRI.KomaMRIBase.is_Delay(sequence[block]) || error(
            "Reset marker $marker_index maps to non-field-free block $block.")
        isempty(sequence.EXT[block]) || error(
            "Reset marker $marker_index maps to a delay block with extensions.")

        marker = RF(
            marker_amplitude_t,
            Float64(duration_s),
            0.0,
            local_delay,
        )
        corrected.RF[1, block] = marker
        push!(used_blocks, block)
        push!(records, (
            marker_index=marker_index,
            block=block,
            marker_start_s=marker_time,
            marker_end_s=marker_time + Float64(duration_s),
            local_delay_s=local_delay,
            clearance_before_s=clearance_before,
            clearance_after_s=clearance_after,
        ))
    end

    corrected_adc = adc_timing_records(corrected)
    corrected_adc == original_adc || error(
        "Reset-marker insertion changed ADC timing or labels.")
    corrected.DUR == sequence.DUR || error(
        "Reset-marker insertion changed block durations.")
    gradients_unchanged = _event_arrays_exactly_equal(
        corrected.GR,
        sequence.GR,
    )
    adc_events_unchanged = _event_arrays_exactly_equal(
        corrected.ADC,
        sequence.ADC,
    )
    gradients_unchanged || error(
        "Reset-marker insertion changed gradient events.")
    adc_events_unchanged || error(
        "Reset-marker insertion changed ADC events.")
    corrected.EXT == sequence.EXT || error(
        "Reset-marker insertion changed sequence extensions.")
    corrected.DEF == sequence.DEF || error(
        "Reset-marker insertion changed sequence definitions.")
    Float64(dur(corrected)) == original_duration || error(
        "Reset-marker insertion changed total sequence duration.")

    return (
        sequence=corrected,
        marker_records=records,
        marker_times_s=marker_times,
        marker_count=length(records),
        marker_lead_s=Float64(lead_s),
        marker_duration_s=Float64(duration_s),
        marker_amplitude_float64=marker_amplitude_t,
        marker_amplitude_float32=Float32(marker_amplitude_t),
        marker_underflows_to_exact_zero_float32=
            Float32(marker_amplitude_t) == 0,
        adc_timing_and_labels_exactly_unchanged=corrected_adc == original_adc,
        block_durations_exactly_unchanged=corrected.DUR == sequence.DUR,
        gradients_exactly_unchanged=gradients_unchanged,
        adc_events_exactly_unchanged=adc_events_unchanged,
        source_sequence_file_modified=false,
        limitation=
            "numerical reset-boundary workaround; particle identity remains " *
            "discontinuous at the periodic coordinate seam",
    )
end

"""
    audit_periodic_seam_reset_markers(sequence, motion, marker_times; ...)

Discretize the effective sequence exactly as the simulator does and prove, for
every marker, that (1) a precession part ends at the marker, (2) Koma's actual
constant-next `FlowPath` mask selects the all-spin endpoint reset at that
Float32 time, and (3) the marker has zero B1/gradients/ADC after the production
precision conversion.  Throws on the first failed invariant.
"""
function audit_periodic_seam_reset_markers(
    sequence,
    motion::Motion,
    marker_times_s;
    marker_duration_s::Real=DEFAULT_SEAM_RESET_MARKER_DURATION_S,
    sim_method=Bloch(),
    gradient_max_step_s::Real=Inf,
    rf_max_step_s::Real=5e-5,
    max_block_length::Integer=512,
    max_rf_block_length::Real=Inf,
    atol_s::Real=1e-9,
)
    motion.action isa FlowPath || error(
        "Reset-marker audit requires a FlowPath motion action.")
    motion.time.periodic || error(
        "Reset-marker audit requires periodic motion.")
    max_block_length > 0 || error("Maximum simulation block length must be positive.")
    reset = motion.action.spin_reset
    spin_count, motion_sample_count = size(reset)
    spin_count > 0 && motion_sample_count > 1 || error(
        "FlowPath reset array has invalid dimensions.")
    endpoint_reset_count = count(@view reset[:, end])
    endpoint_reset_count == spin_count || error(
        "Endpoint reset is not asserted for every spin " *
        "($endpoint_reset_count/$spin_count).")

    # Discretization depends only on the union of reset times, not on which
    # spin owns each flag.  A one-row sentinel preserves that union exactly
    # while avoiding a redundant scan of the full Nspin x Ntime reset matrix.
    # This is the same timing contract used by `local_motion_batch` in the
    # production simulator.
    action_type = eltype(motion.action.dx)
    global_reset_schedule = vec(any(reset; dims=1))
    zero_path = zeros(action_type, 1, motion_sample_count)
    discretization_motion = Motion(
        action=FlowPath(
            dx=zero_path,
            dy=copy(zero_path),
            dz=copy(zero_path),
            spin_reset=reshape(global_reset_schedule, 1, :),
        ),
        time=motion.time,
        spins=AllSpins(),
    )

    params = KomaMRICore.default_sim_params(Dict{String,Any}(
        "sim_method" => sim_method,
        "precision" => "f32",
        KOMA_DELTA_T_KEY => Float64(gradient_max_step_s),
        KOMA_DELTA_T_RF_KEY => Float64(rf_max_step_s),
        "max_block_length" => Int(max_block_length),
        "max_rf_block_length" => Float64(max_rf_block_length),
    ))
    sampling_rule = KomaMRICore.simulation_sampling_rule(sim_method, params)
    discrete = discretize(
        sequence;
        sampling_rule,
        motion=discretization_motion,
        freq_in_phase=params["freq_in_phase"],
    )
    parts, part_is_excitation = KomaMRICore.get_sim_ranges(
        discrete;
        max_block_length=Int(max_block_length),
        max_rf_block_length=Float64(max_rf_block_length),
        eval_intervals_per_step=KomaMRICore.eval_intervals_per_step(sim_method),
    )

    marker_duration = Float64(marker_duration_s)
    tolerance = Float64(atol_s)
    probe_reset = @view reset[1:1, :]
    reset_part_count = 0
    marker_excitation_part_count = 0
    minimum_endpoint_unit_time = Inf
    maximum_endpoint_unit_time = -Inf
    maximum_float32_marker_b1_t = 0.0f0
    first_part = 0
    last_part = 0

    precession_part_indices = findall(!, part_is_excitation)
    precession_end_times = Float64[
        discrete.t[last(parts[part_index])]
        for part_index in precession_part_indices
    ]
    issorted(precession_end_times) || error(
        "Precession-part end times are not sorted.")
    excitation_part_indices = findall(identity, part_is_excitation)
    excitation_start_times = Float64[
        discrete.t[first(parts[part_index])]
        for part_index in excitation_part_indices
    ]
    excitation_end_times = Float64[
        discrete.t[last(parts[part_index])]
        for part_index in excitation_part_indices
    ]
    issorted(excitation_start_times) || error(
        "Excitation-part start times are not sorted.")

    for (marker_index, marker_time_value) in enumerate(marker_times_s)
        marker_time = Float64(marker_time_value)
        precession_left = searchsortedfirst(
            precession_end_times,
            marker_time - tolerance,
        )
        precession_right = searchsortedlast(
            precession_end_times,
            marker_time + tolerance,
        )
        precession_candidates = precession_left <= precession_right ?
            precession_part_indices[precession_left:precession_right] : Int[]
        length(precession_candidates) == 1 || error(
            "Marker $marker_index has $(length(precession_candidates)) " *
            "precession parts ending at its start; expected exactly one.")
        precession_part = only(precession_candidates)
        reset_time_f32 = Float32(discrete.t[last(parts[precession_part])])
        query_time = reshape(Float32[reset_time_f32], 1, 1)
        unit = KomaMRI.KomaMRIBase.unit_time(query_time, motion.time)
        mask = KomaMRICore.get_mask(probe_reset, unit)
        only(mask) == 1 || error(
            "Marker $marker_index does not select the endpoint FlowPath reset " *
            "after Float32 timing conversion.")
        unit_value = Float64(only(unit))
        minimum_endpoint_unit_time = min(minimum_endpoint_unit_time, unit_value)
        maximum_endpoint_unit_time = max(maximum_endpoint_unit_time, unit_value)
        reset_part_count += 1

        excitation_left = searchsortedfirst(
            excitation_start_times,
            marker_time - tolerance,
        )
        excitation_right = searchsortedlast(
            excitation_start_times,
            marker_time + marker_duration + tolerance,
        )
        excitation_candidates = Int[]
        if excitation_left <= excitation_right
            for offset in excitation_left:excitation_right
                part_start = excitation_start_times[offset]
                part_end = excitation_end_times[offset]
                if part_end > part_start &&
                    marker_time - tolerance <= part_start &&
                    part_end <= marker_time + marker_duration + tolerance
                    push!(excitation_candidates, excitation_part_indices[offset])
                end
            end
        end
        isempty(excitation_candidates) && error(
            "Marker $marker_index produced no excitation simulation part.")
        marker_excitation_part_count += length(excitation_candidates)
        first_part == 0 && (first_part = first(excitation_candidates))
        last_part = last(excitation_candidates)
        for part_index in excitation_candidates
            indices = parts[part_index]
            b1_f32 = ComplexF32.(discrete.B1[indices])
            all(iszero, b1_f32) || error(
                "Marker $marker_index has nonzero Float32 B1.")
            maximum_float32_marker_b1_t = max(
                maximum_float32_marker_b1_t,
                maximum(abs, b1_f32; init=0.0f0),
            )
            all(iszero, Float32.(discrete.Gx[indices])) || error(
                "Marker $marker_index has nonzero Gx.")
            all(iszero, Float32.(discrete.Gy[indices])) || error(
                "Marker $marker_index has nonzero Gy.")
            all(iszero, Float32.(discrete.Gz[indices])) || error(
                "Marker $marker_index has nonzero Gz.")
            any(discrete.ADC[indices]) && error(
                "Marker $marker_index overlaps ADC sampling.")
        end
    end

    marker_count = length(marker_times_s)
    reset_part_count == marker_count || error(
        "Not every marker created a verified reset boundary.")
    return (
        passed=true,
        evidence_class="deterministic source-path and post-discretization audit",
        marker_count=marker_count,
        verified_precession_reset_part_count=reset_part_count,
        marker_excitation_part_count=marker_excitation_part_count,
        endpoint_reset_spin_count=endpoint_reset_count,
        total_spin_count=spin_count,
        motion_sample_count=motion_sample_count,
        reset_time_union_flag_count=count(global_reset_schedule),
        reset_time_union_proxy_exact=true,
        all_endpoint_resets_selected_after_float32_conversion=true,
        minimum_endpoint_unit_time=minimum_endpoint_unit_time,
        maximum_endpoint_unit_time=maximum_endpoint_unit_time,
        maximum_float32_marker_b1_t=Float64(maximum_float32_marker_b1_t),
        marker_gradients_exactly_zero_float32=true,
        marker_adc_off=true,
        maximum_simulation_block_length=Int(max_block_length),
        first_marker_excitation_part=first_part,
        last_marker_excitation_part=last_part,
        discrete_time_point_count=length(discrete.t),
        simulation_part_count=length(parts),
        reset_application_basis=
            "each verified non-excitation part invokes outflow_spin_reset! " *
            "at its final time in KomaMRICore 0.12.3",
        limitation=
            "proves reset execution for the forced endpoint flag; it does " *
            "not establish physical identity continuity across the seam",
    )
end

"""
    periodic_motion_precision_audit(motion, sequence_rr, acquisition_groups)

Quantify the type-preserving period rounding and its worst accumulated phase
drift over the acquisition groups.  A positive acquisition-relative drift
means sequence windows occur progressively later relative to the motion seam.
"""
function periodic_motion_precision_audit(
    motion::Motion,
    sequence_rr::Real,
    acquisition_groups::Integer,
)
    motion.time.periodic || error("Period precision audit requires periodic motion.")
    acquisition_groups > 0 || error("Acquisition-group count must be positive.")
    motion_period = Float64(motion.time.t_end - motion.time.t_start)
    sequence_period = Float64(sequence_rr)
    stored_minus_sequence = motion_period - sequence_period
    acquisition_relative_drift = -stored_minus_sequence
    maximum_accumulated_drift =
        abs(acquisition_relative_drift) * max(Int(acquisition_groups) - 1, 0)
    return (
        motion_time_scalar_type=string(eltype(motion.time.t)),
        float64_flowpath_promotion_applied=false,
        float64_flowpath_promotion_reason=
            "Koma Motion, FlowPath, and TimeCurve require one scalar type; " *
            "the existing Float32 action arrays were not copied or doubled",
        exact_sequence_rr_s=sequence_period,
        stored_motion_period_s=motion_period,
        stored_motion_period_minus_sequence_rr_s=stored_minus_sequence,
        acquisition_relative_drift_per_group_s=acquisition_relative_drift,
        acquisition_group_count=Int(acquisition_groups),
        maximum_accumulated_drift_s=maximum_accumulated_drift,
    )
end

function _nearest_periodic_seam_distance(
    time_s::Float64,
    seam_time_s::Float64,
    period_s::Float64,
)
    seam_index = round((time_s - seam_time_s) / period_s)
    return abs(time_s - (seam_time_s + seam_index * period_s))
end

"""
    audit_adc_sample_windows_against_periodic_seam(
        sequence, seam_time, motion_period; minimum_margin_s, atol_s)

Audit every closed ADC sample window, from its first through last discrete
sample, against all periodic seams.  Treating the complete interval as unsafe
when it contains a seam is stronger than checking only the discrete sample
times.  Throws before simulation if any window intersects a seam or if the
minimum clearance is below `minimum_margin_s`.
"""
function audit_adc_sample_windows_against_periodic_seam(
    sequence,
    seam_time::Real,
    motion_period::Real;
    minimum_margin_s::Real=0.0,
    atol_s::Real=1e-12,
)
    seam = Float64(seam_time)
    period = Float64(motion_period)
    required_margin = Float64(minimum_margin_s)
    tolerance = Float64(atol_s)
    isfinite(seam) || error("Periodic seam time must be finite.")
    period > 0 && isfinite(period) || error("Motion period must be finite and positive.")
    required_margin >= 0 && isfinite(required_margin) || error(
        "Required ADC seam margin must be finite and non-negative.")
    tolerance >= 0 && isfinite(tolerance) || error(
        "ADC seam audit tolerance must be finite and non-negative.")

    starts = get_block_start_times(sequence)
    labels = get_labels(sequence)
    adc_blocks = findall(adc -> adc.N > 0, sequence.ADC)
    isempty(adc_blocks) && error("Sequence has no ADC sample windows.")
    minimum_margin = Inf
    minimum_margin_block = 0
    minimum_margin_label = (LIN=-1, PAR=-1, PHS=-1)
    crossing_count = 0
    maximum_window_duration = 0.0

    for block in adc_blocks
        adc = sequence.ADC[block]
        sample_spacing = Float64(dwell(adc))
        sample_spacing > 0 || error("ADC sample spacing must be positive.")
        sample_start = Float64(starts[block] + adc.delay)
        sample_end = sample_start + (Int(adc.N) - 1) * sample_spacing
        sample_end >= sample_start || error("ADC sample window has negative duration.")
        maximum_window_duration = max(
            maximum_window_duration,
            sample_end - sample_start,
        )

        next_seam_index = ceil(Int, (sample_start - seam) / period)
        next_seam = seam + next_seam_index * period
        crosses = sample_start - tolerance <= next_seam <= sample_end + tolerance
        crossing_count += crosses
        margin = crosses ? 0.0 : min(
            _nearest_periodic_seam_distance(sample_start, seam, period),
            _nearest_periodic_seam_distance(sample_end, seam, period),
        )
        if margin < minimum_margin
            minimum_margin = margin
            minimum_margin_block = block
            label = labels[block]
            minimum_margin_label = (
                LIN=Int(label.LIN),
                PAR=Int(label.PAR),
                PHS=Int(label.PHS),
            )
        end
    end

    cardiac_bins = Int(sequence.DEF["CardiacBins"])
    length(adc_blocks) % cardiac_bins == 0 || error(
        "ADC window count is not divisible by the cardiac-bin count.")
    acquisition_groups = length(adc_blocks) ÷ cardiac_bins
    phase_spacing = Float64(sequence.DEF["RR"]) / cardiac_bins
    passed = crossing_count == 0 && minimum_margin + tolerance >= required_margin
    passed || throw(ArgumentError(
        "ADC seam audit failed: crossings=$crossing_count, minimum margin " *
        "$(minimum_margin) s, required $(required_margin) s.",
    ))
    return (
        passed=true,
        audit_scope=
            "every closed interval from first through last ADC sample; " *
            "stronger than discrete-sample-only checking",
        adc_window_count=length(adc_blocks),
        acquisition_group_count=acquisition_groups,
        seam_crossing_window_count=crossing_count,
        minimum_seam_margin_s=minimum_margin,
        required_minimum_seam_margin_s=required_margin,
        minimum_seam_margin_fraction_of_bin=minimum_margin / phase_spacing,
        minimum_margin_block=minimum_margin_block,
        minimum_margin_line_zero_based=minimum_margin_label.LIN,
        minimum_margin_partition_zero_based=minimum_margin_label.PAR,
        minimum_margin_saved_phs_zero_based=minimum_margin_label.PHS,
        maximum_adc_sample_window_duration_s=maximum_window_duration,
        seam_time_s=seam,
        motion_period_s=period,
        sequence_rr_s=Float64(sequence.DEF["RR"]),
        cardiac_bins=cardiac_bins,
        nominal_phase_spacing_s=phase_spacing,
        tolerance_s=tolerance,
    )
end
