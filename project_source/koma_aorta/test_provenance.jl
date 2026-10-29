include(joinpath(@__DIR__, "provenance.jl"))

using .AortaProvenance
using JLD2

mktempdir() do directory
    source = joinpath(directory, "source.bin")
    write(source, UInt8[0x00, 0x01, 0x02, 0xff])
    @assert file_sha256(source) ==
        "3d1f57c984978ef98a18378c8166c1cb8ede02c03eeb6aee7e2f121dfeee3e56"
end

expected = Dict(
    "provenance_schema_version" => 1,
    "phantom_sha256" => "phantom-digest",
    "sequence_sha256" => "sequence-digest",
    "particle_count" => 1_000_000,
)
@assert require_matching_provenance(
    copy(expected),
    expected;
    artifact="matching-test.jld2",
)

mktempdir() do directory
    checkpoint = joinpath(directory, "checkpoint.jld2")
    JLD2.jldsave(checkpoint; provenance=expected, payload=ComplexF32[1 + 2im])
    stored = JLD2.load(checkpoint, "provenance")
    @assert require_matching_provenance(
        stored,
        expected;
        artifact=checkpoint,
    )
end

for invalid in (
    Dict(key => value for (key, value) in expected if key != "phantom_sha256"),
    merge(copy(expected), Dict("particle_count" => 100_000)),
)
    refused = false
    try
        require_matching_provenance(
            invalid,
            expected;
            artifact="refusal-test.jld2",
        )
    catch exception
        refused = exception isa ArgumentError &&
            occursin("left unchanged", sprint(showerror, exception))
    end
    @assert refused
end

println("PASS_AORTA_PROVENANCE_NONCUDA")
