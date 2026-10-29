ENV["DP150_SEQUENCE_DIR"] = @__DIR__
ENV["DP150_SIMULATION_WRAPPER_FILE"] = @__FILE__
ENV["DP150_RUN_NAME"] = get(ENV, "DP150_RUN_NAME", "motion_1m_scanner_matched")
ENV["DP150_B0_T"] = get(ENV, "DP150_B0_T", "3.0")
ENV["DP150_RESCALE_MOTION_TO_SEQUENCE_RR"] = get(
    ENV,
    "DP150_RESCALE_MOTION_TO_SEQUENCE_RR",
    "true",
)
include(joinpath(dirname(@__DIR__), "simulate_pc.jl"))
