using KomaMRI
using KomaMRI.PulseDesigner:
    build_trigger, make_delay, make_label, make_trapezoid
using Unitful


_pc_first_moment(venc) = 1 / (2γ * venc)

function _pc_bipolar_timing(first_moment, sys)
    ramp = ceil(cbrt(first_moment / (2sys.limits.Smax)) / sys.limits.GR_Δt) * sys.limits.GR_Δt
    gradient = first_moment / (2ramp^2)
    gradient ≤ sys.limits.Gmax && return (0.0, ramp)

    ramp = ceil(sys.limits.Gmax / sys.limits.Smax / sys.limits.GR_Δt) * sys.limits.GR_Δt
    flat = ceil(
        max((-3ramp + sqrt(ramp^2 + 4first_moment / sys.limits.Gmax)) / 2, 0.0) /
        sys.limits.GR_Δt,
    ) * sys.limits.GR_Δt
    return (flat, ramp)
end

function _pc_velocity_module(axis, first_moment, flat, ramp, sys)
    duration = 2 * (flat + 2ramp)
    if iszero(first_moment)
        seq = Sequence(sys)
        @addblock seq += make_delay(duration * u"s")
        return seq
    end

    gradient = first_moment / ((flat + ramp) * (flat + 2ramp))
    gradient ≤ sys.limits.Gmax || error(
        "Velocity-encoding gradient $(round(gradient * 1e3, digits=2)) mT/m " *
        "exceeds Gmax=$(round(sys.limits.Gmax * 1e3, digits=2)) mT/m.")

    function gradient_event(amplitude)
        return make_trapezoid(;
            amplitude=amplitude * u"T/m",
            flat_time=flat * u"s",
            rise_time=ramp * u"s",
            fall_time=ramp * u"s",
            sys,
        )
    end

    seq = Sequence(sys)
    positive = gradient_event(gradient)
    negative = gradient_event(-gradient)
    if axis === :RO
        @addblock seq += (; x=positive) + (; x=negative)
    elseif axis === :PE
        @addblock seq += (; y=positive) + (; y=negative)
    elseif axis === :SS
        @addblock seq += (; z=positive) + (; z=negative)
    else
        error("velocity_axis must be :RO, :PE, or :SS.")
    end
    return seq
end

_pc_axis_index(axis) = axis === :RO ? 1 : axis === :PE ? 2 : axis === :SS ? 3 :
    error("velocity_axis must be :RO, :PE, or :SS.")

function _pc_merged_velocity_events(axis, first_moment, flat, ramp, pre_duration, sys)
    first_duration = flat + 2ramp
    first_duration > 0 || error("Velocity-encoding lobe duration must be positive.")
    pre_duration ≥ 2sys.limits.GR_Δt || error(
        "Readout prephaser is too short for merged velocity encoding.")

    second_ramp = floor(pre_duration / (2sys.limits.GR_Δt)) * sys.limits.GR_Δt
    second_flat = pre_duration - 2second_ramp
    first_center = first_duration / 2
    second_center = first_duration + pre_duration / 2
    velocity_area = first_moment / (second_center - first_center)
    first_amplitude = velocity_area / (flat + ramp)
    second_amplitude = -velocity_area / (second_flat + second_ramp)

    abs(first_amplitude) ≤ sys.limits.Gmax || error(
        "Merged leading velocity lobe exceeds Gmax.")
    abs(second_amplitude) ≤ sys.limits.Gmax || error(
        "Merged trailing velocity lobe exceeds Gmax.")
    abs(first_amplitude) / ramp ≤ sys.limits.Smax || error(
        "Merged leading velocity lobe exceeds Smax.")
    abs(second_amplitude) / second_ramp ≤ sys.limits.Smax || error(
        "Merged trailing velocity lobe exceeds Smax.")

    leading_gradient = make_trapezoid(;
        amplitude=first_amplitude * u"T/m",
        flat_time=flat * u"s",
        rise_time=ramp * u"s",
        fall_time=ramp * u"s",
        sys,
    )
    trailing_gradient = make_trapezoid(;
        amplitude=second_amplitude * u"T/m",
        flat_time=second_flat * u"s",
        rise_time=second_ramp * u"s",
        fall_time=second_ramp * u"s",
        sys,
    )

    leading = Sequence(sys)
    if axis === :RO
        @addblock leading += (; x=leading_gradient)
    elseif axis === :PE
        @addblock leading += (; y=leading_gradient)
    else
        @addblock leading += (; z=leading_gradient)
    end
    return leading, trailing_gradient, velocity_area
end

function _pc_merge_prephaser(readout, axis, trailing_gradient)
    merged = deepcopy(readout)
    axis_index = _pc_axis_index(axis)
    merged.GR[axis_index, 1] = merged.GR[axis_index, 1] + trailing_gradient
    return merged
end

function _pc_retime_prephaser(readout, requested_duration, sys)
    requested_duration ≥ 2sys.limits.GR_Δt || error(
        "prephaser_duration must span at least two gradient rasters.")
    duration = round(Int, requested_duration / sys.limits.GR_Δt) * sys.limits.GR_Δt
    abs(duration - requested_duration) ≤ sys.limits.GR_Δt / 2 + 1e-12 || error(
        "prephaser_duration cannot be represented on the gradient raster.")
    ramp = floor(duration / (2sys.limits.GR_Δt)) * sys.limits.GR_Δt
    flat = duration - 2ramp

    retimed = deepcopy(readout)
    for axis in 1:3
        moment = area(retimed.GR[axis, 1])
        amplitude = moment / (flat + ramp)
        retimed.GR[axis, 1] = make_trapezoid(;
            amplitude=amplitude * u"T/m",
            flat_time=flat * u"s",
            rise_time=ramp * u"s",
            fall_time=ramp * u"s",
            sys,
        )
    end
    retimed.DUR[1] = duration
    return retimed
end

_pc_gradient_area(gradient) = gradient.A * (gradient.T + gradient.rise)
_pc_rf_phase(index, increment) = increment * (index - 1) * index / 2

function _pc_echo_time(shot, readout_samples)
    starts = get_block_start_times(shot)
    rf_centers = Float64[]
    adc_centers = Float64[]
    for block in eachindex(shot.DUR)
        for rf in view(shot.RF, :, block)
            dur(rf) > 0 || continue
            push!(rf_centers, starts[block] + rf.delay + rf_center(rf))
        end
        adc = shot.ADC[block]
        adc.N > 0 || continue
        n_precenter = readout_samples ÷ 2
        push!(
            adc_centers,
            starts[block] + adc.delay + n_precenter * dwell(adc),
        )
    end
    length(rf_centers) == 1 || error(
        "A PC-GRE shot must contain exactly one RF event to determine TE.")
    length(adc_centers) == 1 || error(
        "A PC-GRE shot must contain exactly one ADC event to determine TE.")
    return only(adc_centers) - only(rf_centers)
end

function _pc_raster_delay(requested, raster, name)
    requested >= -1e-12 || error(
        "$name cannot be reached: the requested timing is " *
        "$(round(-requested * 1e3, digits=6)) ms shorter than the minimum.")
    ticks = max(round(Int, requested / raster), 0)
    return ticks * raster
end

function _pc_disable_adc!(seq)
    for block in eachindex(seq.ADC)
        seq.ADC[block].N > 0 || continue
        seq.ADC[block] = ADC(0, 0.0)
        empty!(seq.EXT[block])
    end
    return seq
end

function _pc_shot(
    excitation,
    readout,
    velocity_module,
    encoding,
    cardiac_bin,
    rf_index,
    rf_spoil_increment;
    flip_scale=1.0,
    write_set_label=true,
    write_phs_label=true,
    echo_delay=0.0,
    echo_delay_after_prephaser=false,
    tr_delay=0.0,
)
    phase = cis(_pc_rf_phase(rf_index, rf_spoil_increment))
    shot = rf_excitation(excitation, flip_scale * phase) + velocity_module
    if echo_delay > 0 && echo_delay_after_prephaser
        length(readout) >= 2 || error(
            "A delayed merged prephaser requires at least two readout blocks.")
        shot += phase * readout[1]
        shot += make_delay(echo_delay * u"s")
        shot += phase * readout[2:length(readout)]
    else
        echo_delay > 0 && (shot += make_delay(echo_delay * u"s"))
        shot += phase * readout
    end
    tr_delay > 0 && (shot += make_delay(tr_delay * u"s"))
    labels = Extension[]
    write_set_label &&
        pushfirst!(labels, make_label(:SET, :SET, encoding - 1))
    write_phs_label &&
        push!(labels, make_label(:SET, :PHS, cardiac_bin - 1))
    for block in eachindex(shot.ADC)
        shot.ADC[block].N > 0 || continue
        append!(shot.EXT[block], labels)
    end
    return shot
end


"""
    build_pc(FOV, matrix, sys; venc, cardiac_bins, RR, BWpp, kwargs...)

Build a beat-interleaved two- or three-dimensional cine PC-GRE sequence.
`FOV=(readout, phase, slice_or_slab_thickness)` uses metres and `matrix` is
`(Nx, Ny)` or `(Nx, Ny, Nz)`. In 3D, `FOV[3]` defines both the slab-select
thickness and the partition FOV. Each Cartesian encoding is acquired in every
cardiac bin with a reference (`SET=0`) and one velocity-encoded (`SET=1`)
acquisition on successive acquisition windows. `PHS` identifies the zero-based
cardiac bin, `LIN` the phase-encoding line, and `PAR` the 3D partition.

By default, the requested `RR` is divided into `cardiac_bins`, then each bin is
rounded to the nearest whole number of complete TRs. With `exact_rr=true`, a
raster-compatible delay is distributed across the cardiac bins so the complete
cycle matches `RR`; individual phase intervals differ by at most one block
raster. A final incomplete line group is filled with ADC-disabled dummy shots.
RF spoiling remains continuous across triggers, encoding beats, line groups,
and dummy shots.

# Required keywords
- `venc`: Velocity aliasing limit. [`m/s`]
- `cardiac_bins`: Number of cardiac phases per trigger.
- `RR`: Approximate R-R interval. [`s`]
- `BWpp`: Requested ADC bandwidth per acquired readout pixel. [`Hz/pixel`]

# Optional keywords
- `velocity_axis=:SS`: Velocity-encoding direction: `:RO`, `:PE`, or `:SS`.
- `flip_angle=π/8`: Excitation flip angle. [`rad`]
- `rf_bandwidth=nothing`: Slice-selective RF bandwidth. [`Hz`]
- `rf_time_bandwidth_product=4`: Slice-selective RF time-bandwidth product.
- `rf_window=0.46`: Raised-cosine RF window coefficient.
- `rf_frequency_offset=0.0`: RF frequency offset. [`Hz`]
- `spoil_phase=4π`: Slice-direction spoiler phase across one slice or slab. [`rad`]
- `n_ramp_shots=10`: ADC-disabled linear flip-angle ramp after each trigger,
  or once at sequence start when `trigger_channel=nothing`.
- `trigger_channel=:physio1`: Pulseq physiological trigger channel. Use
  `nothing` for a continuous retrospective acquisition.
- `post_trigger_delay=0.0`: Delay included in the trigger block. [`s`]
- `view_order=:linear`: Phase-encoding order, `:linear` or `:center_out` in 2D.
  Three-dimensional filling is linear with `ky` varying fastest.
- `rf_spoil_increment=deg2rad(117)`: RF-spoiling phase increment. [`rad`]
- `target_TE=nothing`: Requested RF-center-to-k-space-center echo time. [`s`]
- `target_TR=nothing`: Requested complete shot repetition time. [`s`]
- `exact_rr=false`: Add phase-padding delays so the complete cardiac cycle
  matches the requested `RR` instead of rounding it to complete TRs.
- `merge_velocity_prephaser=false`: For separate REF/X/Y/Z scans, overlap the
  second velocity-encoding lobe with the Cartesian prephaser. This preserves
  zero velocity-gradient area and VENC first moment while reducing minimum TE.
- `prephaser_duration=nothing`: Optional common Cartesian prephaser duration.
  A longer duration can keep a merged velocity/prephaser waveform within the
  gradient and slew limits. [`s`]
- `exact_rf_flip=false`: Normalize the finite sampled sinc waveform to the
  requested flip angle instead of using the legacy ideal-sinc approximation.
"""
function build_pc(
    FOV,
    matrix,
    sys;
    venc,
    cardiac_bins,
    RR,
    BWpp,
    velocity_axis=:SS,
    flip_angle=π / 8,
    rf_bandwidth=nothing,
    rf_time_bandwidth_product=4,
    rf_window=0.46,
    rf_frequency_offset=0.0,
    spoil_phase=4π,
    n_ramp_shots=10,
    trigger_channel=:physio1,
    post_trigger_delay=0.0,
    view_order=:linear,
    rf_spoil_increment=deg2rad(117),
    target_TE=nothing,
    target_TR=nothing,
    exact_rr=false,
    merge_velocity_prephaser=false,
    prephaser_duration=nothing,
    exact_rf_flip=false,
    slab_thickness=nothing,
    _encoding_axis=nothing,
)
    length(FOV) == 3 ||
        error("FOV must contain (readout, phase, slice_or_slab_thickness).")
    length(matrix) in (2, 3) ||
        error("matrix must contain (Nx, Ny) or (Nx, Ny, Nz).")
    all(value -> value > 0, FOV) || error("FOV entries must be positive.")
    all(value -> value isa Integer && value > 0, matrix) ||
        error("matrix entries must be positive integers.")
    vencs = venc isa Number ? (Float64(venc), Float64(venc), Float64(venc)) : Tuple(Float64.(venc))
    length(vencs) == 3 && all(>(0), vencs) || error("venc must be positive: scalar or (RO, PE, SS).")
    isnothing(slab_thickness) || slab_thickness > 0 || error("slab_thickness must be positive.")
    cardiac_bins isa Integer && cardiac_bins > 0 ||
        error("cardiac_bins must be a positive integer.")
    RR > 0 || error("RR must be positive.")
    BWpp > 0 || error("BWpp must be positive.")
    velocity_axis in (:RO, :PE, :SS) ||
        error("velocity_axis must be :RO, :PE, or :SS.")
    isnothing(_encoding_axis) || _encoding_axis in (:REF, :RO, :PE, :SS) ||
        error("encoding axis must be :REF, :RO, :PE, or :SS.")
    flip_angle > 0 || error("flip_angle must be positive.")
    spoil_phase ≥ 0 || error("spoil_phase must be non-negative.")
    n_ramp_shots isa Integer && n_ramp_shots ≥ 0 ||
        error("n_ramp_shots must be a non-negative integer.")
    post_trigger_delay ≥ 0 ||
        error("post_trigger_delay must be non-negative.")
    isnothing(target_TE) || target_TE > 0 || error("target_TE must be positive.")
    isnothing(target_TR) || target_TR > 0 || error("target_TR must be positive.")
    rf_time_bandwidth_product > 0 ||
        error("rf_time_bandwidth_product must be positive.")
    0 ≤ rf_window ≤ 1 || error("rf_window must be between zero and one.")
    exact_rr isa Bool || error("exact_rr must be true or false.")
    merge_velocity_prephaser isa Bool ||
        error("merge_velocity_prephaser must be true or false.")
    exact_rf_flip isa Bool || error("exact_rf_flip must be true or false.")
    isnothing(prephaser_duration) || prephaser_duration > 0 ||
        error("prephaser_duration must be positive.")
    merge_velocity_prephaser && isnothing(_encoding_axis) && error(
        "merge_velocity_prephaser requires a separate REF/X/Y/Z scan.")
    view_order in (:linear, :center_out) ||
        error("view_order must be :linear or :center_out.")
    length(matrix) == 3 && view_order != :linear &&
        error("three-dimensional PC currently supports only linear view ordering.")

    FOV = Tuple(Float64.(FOV))
    matrix = Tuple(Int.(matrix))
    excitation = slice_selective_sinc(
        flip_angle,
        isnothing(slab_thickness) ? FOV[3] : Float64(slab_thickness),
        sys;
        BW=rf_bandwidth,
        TBP=rf_time_bandwidth_product,
        a=rf_window,
        Δf=rf_frequency_offset,
        exact_flip=exact_rf_flip,
    )
    fixed_area = ntuple(
        axis -> _pc_gradient_area(excitation.GR[axis, 2]),
        3,
    )
    spoiler_area = spoil_phase / (2π * γ * FOV[3])
    readout_kernel = sgre_base(
        length(matrix) == 2 ? FOV[1:2] : FOV,
        matrix,
        sys,
        BWpp;
        fixed_area,
        spoil_area=(0.0, 0.0, spoiler_area),
    )

    # Shared lobe timing from the largest first moment; each axis keeps pi/VENC_axis.
    timing_moment = _pc_first_moment(minimum(vencs))
    flat, ramp = _pc_bipolar_timing(timing_moment, sys)
    axis_moment(axis) = _pc_first_moment(vencs[_pc_axis_index(axis)])
    first_moment = isnothing(_encoding_axis) || _encoding_axis === :REF ?
        axis_moment(velocity_axis) : axis_moment(_encoding_axis)
    reference_module = _pc_velocity_module(:SS, 0.0, flat, ramp, sys)
    write_set_label = isnothing(_encoding_axis)
    write_phs_label = cardiac_bins > 1
    velocity_modules = if write_set_label
        (
            reference_module,
            _pc_velocity_module(velocity_axis, first_moment, flat, ramp, sys),
        )
    elseif _encoding_axis === :REF
        (reference_module,)
    else
        (_pc_velocity_module(_encoding_axis, first_moment, flat, ramp, sys),)
    end

    ky = collect(-(matrix[2] ÷ 2):(matrix[2] - 1 - matrix[2] ÷ 2))
    spatial_encodings = if length(matrix) == 2
        ky
    else
        kz = collect(-(matrix[3] ÷ 2):(matrix[3] - 1 - matrix[3] ÷ 2))
        [(line, partition) for partition in kz for line in ky]
    end
    readouts = Dict(
        encoding => length(matrix) == 2 ?
            readout_kernel(encoding) : readout_kernel(encoding...)
        for encoding in spatial_encodings
    )
    if !isnothing(prephaser_duration)
        readouts = Dict(
            encoding => _pc_retime_prephaser(readout, prephaser_duration, sys)
            for (encoding, readout) in readouts
        )
    end
    merged_velocity_area = 0.0
    if merge_velocity_prephaser
        leading_duration = flat + 2ramp
        reference_module = Sequence(sys)
        @addblock reference_module += make_delay(leading_duration * u"s")
        if _encoding_axis === :REF
            velocity_modules = (reference_module,)
        else
            leading, trailing, merged_velocity_area = _pc_merged_velocity_events(
                _encoding_axis,
                first_moment,
                flat,
                ramp,
                first(readouts).second.DUR[1],
                sys,
            )
            readouts = Dict(
                encoding => _pc_merge_prephaser(readout, _encoding_axis, trailing)
                for (encoding, readout) in readouts
            )
            velocity_modules = (leading,)
        end
    end
    base_representative = _pc_shot(
        excitation,
        readouts[first(spatial_encodings)],
        velocity_modules[1],
        1,
        1,
        1,
        rf_spoil_increment,
        write_set_label=write_set_label,
        write_phs_label=write_phs_label,
    )
    base_TE = _pc_echo_time(base_representative, matrix[1])
    echo_delay = isnothing(target_TE) ? 0.0 : _pc_raster_delay(
        target_TE - base_TE,
        sys.limits.DUR_Δt,
        "target_TE",
    )
    echo_representative = _pc_shot(
        excitation,
        readouts[first(spatial_encodings)],
        velocity_modules[1],
        1,
        1,
        1,
        rf_spoil_increment,
        write_set_label=write_set_label,
        write_phs_label=write_phs_label,
        echo_delay=echo_delay,
        echo_delay_after_prephaser=merge_velocity_prephaser,
    )
    actual_TE = _pc_echo_time(echo_representative, matrix[1])
    isnothing(target_TE) || abs(actual_TE - target_TE) ≤ sys.limits.DUR_Δt / 2 + 1e-12 ||
        error("target_TE cannot be represented on the block-duration raster.")

    minimum_TR = dur(echo_representative)
    tr_delay = isnothing(target_TR) ? 0.0 : _pc_raster_delay(
        target_TR - minimum_TR,
        sys.limits.DUR_Δt,
        "target_TR",
    )
    representative = _pc_shot(
        excitation,
        readouts[first(spatial_encodings)],
        velocity_modules[1],
        1,
        1,
        1,
        rf_spoil_increment,
        write_set_label=write_set_label,
        write_phs_label=write_phs_label,
        echo_delay=echo_delay,
        echo_delay_after_prephaser=merge_velocity_prephaser,
        tr_delay=tr_delay,
    )
    TR = dur(representative)
    isnothing(target_TR) || abs(TR - target_TR) ≤ sys.limits.DUR_Δt / 2 + 1e-12 ||
        error("target_TR cannot be represented on the block-duration raster.")

    phase_padding = zeros(Float64, cardiac_bins)
    if exact_rr
        requested_phase_interval = RR / cardiac_bins
        requested_phase_interval + 1e-12 ≥ TR || error(
            "RR/cardiac_bins must be at least one complete TR when exact_rr=true.")
        lines_per_bin = max(floor(Int, requested_phase_interval / TR + 1e-12), 1)
        padding_duration = RR - cardiac_bins * lines_per_bin * TR
        padding_ticks = round(Int, padding_duration / sys.limits.DUR_Δt)
        padding_ticks ≥ 0 || error("Exact RR padding cannot be negative.")
        base_ticks, extra_ticks = divrem(padding_ticks, cardiac_bins)
        for cardiac_bin in eachindex(phase_padding)
            phase_padding[cardiac_bin] =
                (base_ticks + (cardiac_bin ≤ extra_ticks ? 1 : 0)) * sys.limits.DUR_Δt
        end
        actual_phase_intervals = lines_per_bin * TR .+ phase_padding
        actual_RR = sum(actual_phase_intervals)
        abs(actual_RR - RR) ≤ sys.limits.DUR_Δt / 2 + 1e-12 || error(
            "RR cannot be represented on the block-duration raster.")
    else
        lines_per_bin = max(round(Int, RR / (cardiac_bins * TR)), 1)
        actual_phase_intervals = fill(lines_per_bin * TR, cardiac_bins)
        actual_RR = sum(actual_phase_intervals)
    end
    encoding_groups = if length(matrix) == 2
        cartesian_line_order(spatial_encodings, lines_per_bin; view_order)
    else
        [spatial_encodings[first:min(first + lines_per_bin - 1, end)]
         for first in 1:lines_per_bin:length(spatial_encodings)]
    end

    trigger_duration = ceil_to_raster(post_trigger_delay, sys.limits.DUR_Δt)
    trigger = isnothing(trigger_channel) ? nothing : build_trigger(
        trigger_channel;
        duration=trigger_duration * u"s",
        sys,
    )

    sequence = Sequence(sys)
    rf_index = 1
    @addblock begin
        for encodings in encoding_groups, encoding in eachindex(velocity_modules)
            isnothing(trigger) || (sequence += trigger)

            if n_ramp_shots > 0 && (!isnothing(trigger) || rf_index == 1)
                ramp_start = min(1.0, (3π / 180) / flip_angle)
                for scale in range(ramp_start, 1.0; length=n_ramp_shots)
                    shot = _pc_shot(
                        excitation,
                        readouts[first(encodings)],
                        merge_velocity_prephaser ? velocity_modules[encoding] : reference_module,
                        1,
                        1,
                        rf_index,
                        rf_spoil_increment;
                        flip_scale=scale,
                        write_set_label=write_set_label,
                        write_phs_label=write_phs_label,
                        echo_delay=echo_delay,
                        echo_delay_after_prephaser=merge_velocity_prephaser,
                        tr_delay=tr_delay,
                    )
                    sequence += _pc_disable_adc!(shot)
                    rf_index = rf_index + 1
                end
            end

            for cardiac_bin in 1:cardiac_bins
                for spatial_encoding in encodings
                    sequence += _pc_shot(
                        excitation,
                        readouts[spatial_encoding],
                        velocity_modules[encoding],
                        encoding,
                        cardiac_bin,
                        rf_index,
                        rf_spoil_increment,
                        write_set_label=write_set_label,
                        write_phs_label=write_phs_label,
                        echo_delay=echo_delay,
                        echo_delay_after_prephaser=merge_velocity_prephaser,
                        tr_delay=tr_delay,
                    )
                    rf_index = rf_index + 1
                end
                for _ in (length(encodings) + 1):lines_per_bin
                    shot = _pc_shot(
                        excitation,
                        readouts[last(encodings)],
                        velocity_modules[encoding],
                        encoding,
                        cardiac_bin,
                        rf_index,
                        rf_spoil_increment,
                        write_set_label=write_set_label,
                        write_phs_label=write_phs_label,
                        echo_delay=echo_delay,
                        echo_delay_after_prephaser=merge_velocity_prephaser,
                        tr_delay=tr_delay,
                    )
                    sequence += _pc_disable_adc!(shot)
                    rf_index = rf_index + 1
                end
                phase_padding[cardiac_bin] > 0 &&
                    (sequence += make_delay(phase_padding[cardiac_bin] * u"s"))
            end
        end
    end

    actual_BWpp = 1 / (matrix[1] * _design_cartesian_readout(
        FOV[1], matrix[1], sys, BWpp).dwell)
    sequence.DEF["FOV"] = collect(FOV)
    sequence.DEF["Nx"] = matrix[1]
    sequence.DEF["Ny"] = matrix[2]
    sequence.DEF["Nz"] = length(matrix) == 3 ? matrix[3] : 1
    sequence.DEF["VoxelSize"] = collect(FOV ./ matrix)
    sequence.DEF["TE"] = actual_TE
    sequence.DEF["TR"] = TR
    sequence.DEF["RequestedTE"] = isnothing(target_TE) ? actual_TE : Float64(target_TE)
    sequence.DEF["RequestedTR"] = isnothing(target_TR) ? TR : Float64(target_TR)
    sequence.DEF["RequestedRR"] = Float64(RR)
    sequence.DEF["RR"] = actual_RR
    sequence.DEF["CardiacBins"] = cardiac_bins
    sequence.DEF["LinesPerCardiacBin"] = lines_per_bin
    sequence.DEF["NominalPhaseInterval"] = Float64(RR / cardiac_bins)
    sequence.DEF["MinimumPhaseInterval"] = minimum(actual_phase_intervals)
    sequence.DEF["MaximumPhaseInterval"] = maximum(actual_phase_intervals)
    sequence.DEF["ExactRRTiming"] = exact_rr ? 1 : 0
    sequence.DEF["EchoDelay"] = echo_delay
    sequence.DEF["TRFillDelay"] = tr_delay
    sequence.DEF["BandwidthPerPixel"] = actual_BWpp
    sequence.DEF["Venc"] = isnothing(_encoding_axis) || _encoding_axis === :REF ?
        minimum(vencs) : vencs[_pc_axis_index(_encoding_axis)]
    sequence.DEF["VencRO"] = vencs[1]
    sequence.DEF["VencPE"] = vencs[2]
    sequence.DEF["VencSS"] = vencs[3]
    sequence.DEF["SlabThickness"] = isnothing(slab_thickness) ? FOV[3] : Float64(slab_thickness)
    sequence.DEF["VelocityEncodingAxis"] = String(
        isnothing(_encoding_axis) ? velocity_axis : _encoding_axis)
    sequence.DEF["RfSpoilIncrement"] = Float64(rf_spoil_increment)
    sequence.DEF["FlipAngle"] = Float64(flip_angle)
    sequence.DEF["FlipAngleDegrees"] = Float64(rad2deg(flip_angle))
    sequence.DEF["RfTimeBandwidthProduct"] = Float64(rf_time_bandwidth_product)
    sequence.DEF["RfWindowCoefficient"] = Float64(rf_window)
    sequence.DEF["RfFrequencyOffset"] = Float64(rf_frequency_offset)
    sequence.DEF["ExactRfFlipNormalization"] = exact_rf_flip ? 1 : 0
    sequence.DEF["VelocityPrephaserMerged"] = merge_velocity_prephaser ? 1 : 0
    sequence.DEF["MergedVelocityLobeArea"] = merged_velocity_area
    sequence.DEF["PrephaserDuration"] = first(readouts).second.DUR[1]
    isnothing(trigger_channel) ||
        (sequence.DEF["TriggerChannel"] = String(trigger_channel))

    @info "PC timing" requested_TE=target_TE actual_TE requested_TR=target_TR TR requested_RR=RR actual_RR cardiac_bins lines_per_bin phase_interval_range=extrema(actual_phase_intervals) requested_BWpp=BWpp actual_BWpp
    return sequence
end


"""
    build_pc_encoding_scans(FOV, matrix, sys; kwargs...)

Build four full-matrix two- or three-dimensional PC-GRE scans with separate
`REF`, `X`, `Y`, and `Z` velocity encodings. The returned named tuple contains
four sequence objects. Because each file identifies its encoding, ADC events
never carry `SET` labels. They carry `PHS` only when more than one cardiac phase
is requested.

All keywords are forwarded to [`build_pc`](@ref). Reference and velocity-
encoded scans use matched module timing. Each retrospective scan receives its
own initial ramp; triggered scans ramp after each trigger as usual.
"""
function build_pc_encoding_scans(
    FOV,
    matrix,
    sys;
    kwargs...,
)
    names = (:REF, :X, :Y, :Z)
    axes = (:REF, :RO, :PE, :SS)
    sequences = ntuple(length(axes)) do index
        build_pc(
            FOV,
            matrix,
            sys;
            kwargs...,
            _encoding_axis=axes[index],
        )
    end
    result = NamedTuple{names}(sequences)
    for (name, sequence) in pairs(result)
        sequence.DEF["Encoding"] = String(name)
    end
    return result
end
