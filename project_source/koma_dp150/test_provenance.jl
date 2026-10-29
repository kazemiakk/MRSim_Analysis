using JLD2
using Test

include(joinpath(@__DIR__, "provenance.jl"))
using .Dp150Provenance

mktempdir() do directory
    source = joinpath(directory, "known.bin")
    write(source, UInt8[0x00, 0x01, 0x02, 0x03])
    @test file_sha256(source) ==
        "054edec1d0211f624fed0cbca9d4f9400b0e491c43742af2c5b0abebf0c990d8"

    expected = Dict{String,Any}(
        "schema" => 1,
        "encoding" => "REF",
        "matrix" => [4, 3, 2],
        "B0_T" => 3.0,
    )
    signature = provenance_signature(expected)
    @test length(signature) == 64
    @test signature == provenance_signature(copy(expected))
    @test require_matching_provenance(expected, expected; artifact="fixture")
    @test_throws ArgumentError require_matching_provenance(
        Dict("schema" => 1),
        expected;
        artifact="fixture",
    )
    @test_throws ArgumentError require_matching_provenance(
        merge(copy(expected), Dict("B0_T" => 1.5)),
        expected;
        artifact="fixture",
    )

    checkpoint = joinpath(directory, "checkpoint.jld2")
    JLD2.jldsave(checkpoint; provenance=expected)
    stored = JLD2.load(checkpoint, "provenance")
    @test require_matching_provenance(stored, expected; artifact=checkpoint)
end

println("PASS_DP150_PROVENANCE_NONCUDA")
