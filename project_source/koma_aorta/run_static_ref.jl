ENV["CUDA_VISIBLE_DEVICES"] = get(ENV, "AORTA_PHYSICAL_GPU", "1")
default_phantom = normpath(joinpath(
    @__DIR__,
    "..",
    "outputs",
    "aorta_dp2_mri_pipeline",
    "koma",
    "aorta_static_2m.phantom",
))
ENV["AORTA_PHANTOM"] = get(ENV, "AORTA_PHANTOM", default_phantom)
ENV["AORTA_ENCODINGS"] = get(ENV, "AORTA_ENCODINGS", "REF")
ENV["AORTA_RUN_NAME"] = get(ENV, "AORTA_RUN_NAME", "static_2m_ref")
ENV["AORTA_BATCH_SIZE"] = get(ENV, "AORTA_STATIC_BATCH_SIZE", "0")
ENV["AORTA_GPU_DEVICE"] = get(ENV, "AORTA_GPU_DEVICE", "0")

include(joinpath(@__DIR__, "simulate_pc.jl"))
