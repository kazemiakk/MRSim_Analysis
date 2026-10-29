ENV["DP150_SEQUENCE_OUTPUT_DIR"] = @__DIR__
ENV["DP150_VENC_M_PER_S"] = "5.0"
ENV["DP150_GENERATOR_WRAPPER_FILE"] = @__FILE__

include(joinpath(dirname(@__DIR__), "pc_3d_scanner_matched", "generate.jl"))
