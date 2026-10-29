ENV["AORTA_MOTION_BATCH_SIZE"] = get(
    ENV,
    "AORTA_MOTION_BATCH_SIZE",
    "20000",
)

include(joinpath(@__DIR__, "run_motion_all.jl"))
