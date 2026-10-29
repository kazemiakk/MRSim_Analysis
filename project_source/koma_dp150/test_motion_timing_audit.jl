using Test

include(joinpath(@__DIR__, "audit_motion_timing.jl"))

@testset "DP150 timing-audit primitives" begin
    motion_period_s = Float64(Float32(1.012))
    centers_mod_rr_s = [
        0.47167099950000013,
        0.5391376665000003,
        0.6066043335000004,
        0.6740710005000006,
        0.7415376675000007,
        0.8090043345000009,
        0.876471001500001,
        0.9439376685000009,
        1.0114043355000009,
        0.06687100250000078,
        0.13433766950000092,
        0.201804335500001,
        0.26927100150000105,
        0.3367376675000009,
        0.4042043335000012,
    ]
    centers_mod_motion_s = mod.(centers_mod_rr_s, motion_period_s)
    nearest_mapping, nearest_errors = nearest_phase_mapping(
        centers_mod_motion_s,
        motion_period_s,
        15,
    )
    interval_mapping = interval_phase_mapping(
        centers_mod_motion_s,
        motion_period_s,
        15,
    )
    @test nearest_mapping == [8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7]
    @test interval_mapping == [9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8]
    @test maximum(abs, nearest_errors) < 0.0007

    counts = samples_around_boundary(
        1.009975735500001,
        7.143e-6,
        400,
        motion_period_s,
    )
    @test counts.before == 284
    @test counts.at == 0
    @test counts.after == 116
    @test signed_adc_seam_margin_s(
        1.009975735500001,
        1.012825792500001,
        motion_period_s,
    ) < 0
    @test isapprox(
        signed_adc_seam_margin_s(
            0.47024239950000013,
            0.4730924565000001,
            1.012;
            seam_offset_s=0.4379376661666668,
        ),
        0.032304733333333335;
        atol=1e-12,
        rtol=0,
    )
end

@testset "Atomic JSON output" begin
    mktempdir() do directory
        output = joinpath(directory, "audit.json")
        atomic_write_json(output, Dict(
            "status" => TIMING_AUDIT_FAILURE,
            "mapping" => [8, 9, 10],
            "safe" => false,
        ))
        text = read(output, String)
        @test occursin("\"status\": \"$TIMING_AUDIT_FAILURE\"", text)
        @test occursin("\"safe\": false", text)
        @test isempty(filter(name -> occursin(".tmp", name), readdir(directory)))
    end
end

@testset "Completed DP150 run fails closed on a uniform PHS8 seam" begin
    paths = default_audit_paths()
    sequence_paths = Dict(
        encoding => joinpath(
            paths["sequence_dir"],
            lowercase(encoding) * ".seq",
        )
        for encoding in TIMING_AUDIT_ENCODINGS
    )
    simulation_report = TOML.parsefile(paths["simulation_report"])
    motion_period_s = Float64(
        simulation_report["motion_timing"]["effective_period_s"],
    )
    timings = [
        read_sequence_timing(sequence_paths[encoding], encoding)
        for encoding in TIMING_AUDIT_ENCODINGS
    ]
    summaries = Dict(
        timing.encoding => summarize_sequence_timing(timing, motion_period_s)
        for timing in timings
    )

    @test cross_encoding_timing(timings)["identical"]
    for encoding in TIMING_AUDIT_ENCODINGS
        summary = summaries[encoding]
        @test summary["adc_profiles"] == 30_720
        @test summary["spatial_groups"] == 2_048
        @test summary["phs_zero_through_fourteen_repeats_for_every_group"]
        @test summary["maximum_absolute_group_modulo_rr_drift_s"] < 4e-12
        f32_residual = summary["float32_absolute_time_phase_residual"]
        @test isapprox(
            f32_residual["minimum_s"],
            -0.0006451845169067605;
            atol=2e-12,
            rtol=0,
        )
        @test isapprox(
            f32_residual["maximum_s"],
            -0.00040159225463864967;
            atol=2e-12,
            rtol=0,
        )
        @test isapprox(
            f32_residual["global_span_ms"],
            0.24359226226811082;
            atol=2e-9,
            rtol=0,
        )
        crossings = summary[
            "motion_boundary_crossings_by_saved_phs_zero_based"
        ]
        @test crossings == [0, 0, 0, 0, 0, 0, 0, 0, 2_048, 0, 0, 0, 0, 0, 0]
        @test !summary["checks"]["no_adc_overlaps_motion_boundary"]
    end
end
