using KomaMRI
using KomaMRICore
using Test

include(joinpath(@__DIR__, "motion_timing.jl"))

@testset "Seam-safe numerical reset markers" begin
    sequence = Sequence()
    for _ in 1:3
        addblock!(sequence, Duration(1.0))
    end
    original = copy(sequence)

    action = FlowPath(
        dx=zeros(Float32, 1, 2),
        dy=zeros(Float32, 1, 2),
        dz=zeros(Float32, 1, 2),
        spin_reset=reshape(Bool[false, true], 1, 2),
    )
    motion = Motion(
        action=action,
        time=TimeCurve(
            t=Float32[0.5, 1.5],
            t_unit=Float32[0, 1],
            periodic=true,
            periods=1.0f0,
            t_start=0.5f0,
            t_end=1.5f0,
        ),
        spins=AllSpins(),
    )

    inserted = insert_periodic_seam_reset_markers(
        sequence,
        0.5,
        1.0,
        3,
    )
    @test inserted.marker_count == 3
    @test inserted.marker_times_s == [0.49965, 1.49965, 2.49965]
    @test inserted.marker_underflows_to_exact_zero_float32
    @test inserted.adc_timing_and_labels_exactly_unchanged
    @test inserted.block_durations_exactly_unchanged
    @test inserted.gradients_exactly_unchanged
    @test count(is_RF_on, inserted.sequence.RF) == 3
    @test count(is_RF_on, original.RF) == 0
    @test _event_arrays_exactly_equal(sequence.RF, original.RF)
    @test inserted.sequence.DUR == original.DUR
    @test _event_arrays_exactly_equal(inserted.sequence.GR, original.GR)
    @test _event_arrays_exactly_equal(inserted.sequence.ADC, original.ADC)

    audited = audit_periodic_seam_reset_markers(
        inserted.sequence,
        motion,
        inserted.marker_times_s;
        marker_duration_s=inserted.marker_duration_s,
        max_block_length=512,
    )
    @test audited.passed
    @test audited.marker_count == 3
    @test audited.verified_precession_reset_part_count == 3
    @test audited.endpoint_reset_spin_count == 1
    @test audited.maximum_float32_marker_b1_t == 0
    @test audited.marker_gradients_exactly_zero_float32
    @test audited.marker_adc_off
end

@testset "Marker insertion fails closed outside field-free blocks" begin
    sequence = Sequence()
    addblock!(sequence, RF(1e-6, 1e-3), Duration(1.0))
    @test_throws ErrorException insert_periodic_seam_reset_markers(
        sequence,
        0.5,
        1.0,
        1,
    )
end
