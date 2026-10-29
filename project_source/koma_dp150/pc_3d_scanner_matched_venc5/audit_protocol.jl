ENV["DP150_AUDIT_SEQUENCE_DIR"] = @__DIR__
ENV["DP150_VENC_M_PER_S"] = "5.0"
ENV["DP150_AUDIT_WRAPPER_FILE"] = @__FILE__
ENV["DP150_GENERATOR_WRAPPER_FILE"] = joinpath(@__DIR__, "generate.jl")

include(joinpath(dirname(@__DIR__), "pc_3d_scanner_matched", "audit_protocol.jl"))
