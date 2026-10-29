using HDF5
using JLD2
using TOML

length(ARGS) in (1, 2) || error(
    "Usage: julia audit_completed_run.jl RUN_DIR [EXPECTED_PARTICLE_COUNT]",
)

run_dir = normpath(abspath(ARGS[1]))
expected_particle_count = length(ARGS) == 2 ? parse(Int, ARGS[2]) : 1_000_000
report_file = joinpath(run_dir, "simulation_report.toml")
isfile(report_file) || error("Missing final simulation report: $report_file")
report = TOML.parsefile(report_file)

get(report, "status", "") == "PASS" || error("Final report status is not PASS.")
Int(get(report, "particle_count", -1)) == expected_particle_count || error(
    "Final report particle count does not match $expected_particle_count.",
)
Set(String.(get(report, "encodings", String[]))) == Set(("REF", "X", "Y", "Z")) ||
    error("Final report does not contain exactly REF/X/Y/Z.")
Int(get(report, "spin_batch_size", -1)) == 20_000 || error(
    "Final report does not record the production 20,000-spin batch size.",
)
Int(get(report, "maximum_simulation_block_length", -1)) == 512 || error(
    "Final report does not record the validated block length 512.",
)
get(report, "precision", "") == "f32" || error(
    "Final report precision is not f32.",
)
get(report, "unvalidated_block_length_override", true) == false || error(
    "Final report enables an unvalidated block-length override.",
)

sequence_report = get(report, "sequence", Dict{String,Any}())
reference_sequence = get(sequence_report, "REF", Dict{String,Any}())
matrix = Int.(get(reference_sequence, "matrix", Int[]))
length(matrix) == 3 || error("Final report is missing the reference matrix.")
cardiac_bins = Int(get(reference_sequence, "cardiac_bins", 0))
cardiac_bins > 0 || error("Final report has no cardiac-bin count.")
expected_image_shape = (matrix..., cardiac_bins)

function require_nonempty(path::AbstractString, label::AbstractString)
    isfile(path) || error("Missing $label: $path")
    filesize(path) > 0 || error("Empty $label: $path")
    return path
end

function same_repository_artifact(stored::AbstractString, current::AbstractString)
    stored_normalized = lowercase(replace(normpath(stored), '\\' => '/'))
    current_normalized = lowercase(replace(normpath(abspath(current)), '\\' => '/'))
    stored_normalized == current_normalized && return true
    repository_root = lowercase(replace(
        normpath(joinpath(@__DIR__, "..")),
        '\\' => '/',
    ))
    root_prefix = rstrip(repository_root, '/') * "/"
    startswith(current_normalized, root_prefix) || return false
    relative = chop(current_normalized; head=length(root_prefix), tail=0)
    return endswith(stored_normalized, "/" * relative)
end

expected_phantom = expected_particle_count == 100_000 ? joinpath(
    dirname(dirname(@__FILE__)),
    "outputs",
    "dp150_mri_pipeline",
    "koma",
    "pilot_100k",
    "dp150_motion_100k.phantom",
) : joinpath(
    dirname(dirname(@__FILE__)),
    "outputs",
    "dp150_mri_pipeline",
    "koma",
    "dp150_motion_1m.phantom",
)
reported_results = get(report, "result_files", Dict{String,Any}())

for encoding in ("REF", "X", "Y", "Z")
    result_file = require_nonempty(
        joinpath(run_dir, "$(lowercase(encoding))_result.jld2"),
        "$encoding result",
    )
    same_repository_artifact(
        String(get(reported_results, encoding, "")),
        result_file,
    ) || error("Final report result path is wrong for $encoding.")
    JLD2.jldopen(result_file, "r") do file
        required = (
            "rawsort",
            "image",
            "encoding",
            "sequence_file",
            "phantom_file",
            "elapsed_seconds",
            "motion_timing_signature",
        )
        all(haskey(file, key) for key in required) || error(
            "$encoding result is missing one or more legacy schema keys.",
        )
        String(file["encoding"]) == encoding || error(
            "$encoding result stores the wrong encoding.",
        )
        image = file["image"]
        size(image) == expected_image_shape || error(
            "$encoding image shape $(size(image)) does not match $expected_image_shape.",
        )
        all(isfinite, image) || error("$encoding image contains non-finite values.")
        rawsort = file["rawsort"]
        size(rawsort) == expected_image_shape || error(
            "$encoding raw k-space shape $(size(rawsort)) does not match " *
            "$expected_image_shape.",
        )
        all(isfinite, rawsort) || error(
            "$encoding raw k-space contains non-finite values.",
        )
        expected_sequence = joinpath(
            dirname(dirname(@__FILE__)),
            "koma_dp150",
            "pc_3d_scanner_matched_venc5",
            "$(lowercase(encoding)).seq",
        )
        same_repository_artifact(String(file["sequence_file"]), expected_sequence) ||
            error("$encoding result sequence path is not repository-relocatable.")
        same_repository_artifact(String(file["phantom_file"]), expected_phantom) ||
            error("$encoding result phantom path is not repository-relocatable.")
    end
    GC.gc()
end

batch_size = Int(report["spin_batch_size"])
batch_count = cld(expected_particle_count, batch_size)
expected_signal_shape = (prod(matrix) * cardiac_bins, 1, 1)
for encoding in ("REF", "X", "Y", "Z")
    batch_directory = joinpath(run_dir, "$(lowercase(encoding))_batches")
    isdir(batch_directory) || error("Missing $encoding batch directory.")
    expected_names = [
        "batch_$(lpad(index, 4, '0'))_of_$(lpad(batch_count, 4, '0')).jld2"
        for index in 1:batch_count
    ]
    actual_names = sort(filter(
        name -> endswith(lowercase(name), ".jld2"),
        readdir(batch_directory),
    ))
    actual_names == expected_names || error(
        "$encoding batch file set does not match the expected $batch_count checkpoints.",
    )
    for (batch_index, name) in enumerate(expected_names)
        batch_file = require_nonempty(joinpath(batch_directory, name), "$encoding batch")
        first_spin = (batch_index - 1) * batch_size + 1
        last_spin = min(batch_index * batch_size, expected_particle_count)
        JLD2.jldopen(batch_file, "r") do file
            required = (
                "signal",
                "encoding",
                "spin_range",
                "phantom_file",
                "elapsed_seconds",
                "motion_timing_signature",
            )
            all(haskey(file, key) for key in required) || error(
                "$encoding batch $batch_index is missing a legacy schema key.",
            )
            String(file["encoding"]) == encoding || error(
                "$encoding batch $batch_index stores the wrong encoding.",
            )
            Int.(file["spin_range"]) == [first_spin, last_spin] || error(
                "$encoding batch $batch_index stores the wrong spin range.",
            )
            same_repository_artifact(String(file["phantom_file"]), expected_phantom) ||
                error("$encoding batch $batch_index stores the wrong phantom path.")
            elapsed = Float64(file["elapsed_seconds"])
            isfinite(elapsed) && elapsed >= 0 || error(
                "$encoding batch $batch_index stores an invalid runtime.",
            )
            !isempty(String(file["motion_timing_signature"])) || error(
                "$encoding batch $batch_index has no motion timing signature.",
            )
            signal = file["signal"]
            size(signal) == expected_signal_shape || error(
                "$encoding batch $batch_index signal shape $(size(signal)) does not " *
                "match $expected_signal_shape.",
            )
            all(isfinite, signal) || error(
                "$encoding batch $batch_index signal contains non-finite values.",
            )
        end
        GC.gc()
    end
end

reconstruction = require_nonempty(
    joinpath(run_dir, "pc_reconstruction.h5"),
    "PC reconstruction",
)
same_repository_artifact(String(get(report, "reconstruction_file", "")), reconstruction) ||
    error("Final report reconstruction path is not repository-relocatable.")
h5open(reconstruction, "r") do file
    expected_shapes = Dict(
        "reference_magnitude" => expected_image_shape,
        "speed" => expected_image_shape,
        "mask" => expected_image_shape,
        "velocity" => (expected_image_shape..., 3),
    )
    for (name, expected_shape) in expected_shapes
        haskey(file, name) || error("Reconstruction is missing dataset '$name'.")
        dataset = file[name]
        size(dataset) == expected_shape || error(
            "Dataset '$name' shape $(size(dataset)) does not match $expected_shape.",
        )
        name == "mask" && continue
        last_dimension = ndims(dataset)
        for index in axes(dataset, last_dimension)
            selectors = ntuple(
                dimension -> dimension == last_dimension ? index : Colon(),
                last_dimension,
            )
            all(isfinite, dataset[selectors...]) || error(
                "Dataset '$name' contains non-finite values.",
            )
        end
    end
end

qc_figure = require_nonempty(joinpath(run_dir, "pc_reconstruction_qc.png"), "QC figure")
same_repository_artifact(String(get(report, "qc_figure", "")), qc_figure) ||
    error("Final report QC path is not repository-relocatable.")
temporary_files = String[]
for (directory, _, files) in walkdir(run_dir)
    for file in files
        lowered = lowercase(file)
        if endswith(lowered, ".partial") || occursin(".partial.", lowered) ||
            endswith(lowered, ".tmp") || occursin(".tmp.", lowered)
            push!(temporary_files, joinpath(directory, file))
        end
    end
end
isempty(temporary_files) || error(
    "Completed run contains temporary artifacts: $(join(temporary_files, ", "))",
)

legacy = !haskey(report, "provenance_schema_version")
label = legacy ? "PASS_DP150_LEGACY_COMPLETED_RUN_READ_ONLY" :
    "PASS_DP150_PROVENANCE_COMPLETED_RUN_READ_ONLY"
println(
    "$label particles=$expected_particle_count matrix=$(join(matrix, 'x')) " *
    "cardiac_bins=$cardiac_bins",
)
