using JLD2
using LinearAlgebra
using TOML

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
const DEFAULT_KOMA_ROOT = joinpath(
    REPO_ROOT,
    "outputs",
    "dp150_mri_pipeline",
    "koma",
)

koma_root = normpath(get(ENV, "DP150_KOMA_OUTPUT_DIR", DEFAULT_KOMA_ROOT))
direct_file = normpath(get(
    ENV,
    "DP150_DIRECT_RESULT",
    joinpath(koma_root, "simulation_motion_smoke20k", "ref_result.jld2"),
))
batched_file = normpath(get(
    ENV,
    "DP150_BATCHED_RESULT",
    joinpath(koma_root, "simulation_motion_batchcheck20k_v3", "ref_result.jld2"),
))
report_file = normpath(get(
    ENV,
    "DP150_BATCH_VALIDATION_REPORT",
    joinpath(koma_root, "batched_signal_validation.toml"),
))

@assert isfile(direct_file)
@assert isfile(batched_file)

direct_raw = JLD2.load(direct_file, "rawsort")
batched_raw = JLD2.load(batched_file, "rawsort")
direct_image = JLD2.load(direct_file, "image")
batched_image = JLD2.load(batched_file, "image")
@assert size(direct_raw) == size(batched_raw)
@assert size(direct_image) == size(batched_image)

function comparison(reference, candidate)
    delta = candidate .- reference
    relative_l2 = norm(delta) / max(norm(reference), eps(Float64))
    maximum_relative_to_peak = maximum(abs, delta) /
        max(maximum(abs, reference), eps(Float64))
    return Dict(
        "relative_l2_error" => Float64(relative_l2),
        "maximum_absolute_error_relative_to_reference_peak" => Float64(maximum_relative_to_peak),
        "isapprox_rtol_1e_3" => isapprox(reference, candidate; rtol=1e-3, atol=0),
    )
end

raw_comparison = comparison(direct_raw, batched_raw)
image_comparison = comparison(direct_image, batched_image)
passed = raw_comparison["isapprox_rtol_1e_3"] &&
    image_comparison["isapprox_rtol_1e_3"]

report = Dict(
    "status" => passed ? "PASS" : "FAIL",
    "direct_result" => direct_file,
    "batched_result" => batched_file,
    "acceptance_basis" => "KomaMRICore partition test uses isapprox with rtol=0.001",
    "raw_kspace" => raw_comparison,
    "reconstructed_image" => image_comparison,
)
open(report_file, "w") do io
    TOML.print(io, report; sorted=true)
end

passed || error("Batched signal validation failed; see $report_file")
@info "Batched signal validation passed" report=report_file raw_relative_l2=raw_comparison["relative_l2_error"] image_relative_l2=image_comparison["relative_l2_error"]
