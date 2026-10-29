using Test
using TOML

include(joinpath(@__DIR__, "provenance.jl"))
using .Dp150Provenance

base_protocol_dir = joinpath(@__DIR__, "pc_3d_scanner_matched")
protocol_dirs = (
    base_protocol_dir,
    joinpath(@__DIR__, "pc_3d_scanner_matched_venc5"),
)
const SOURCE_METADATA_SHA256 = "fa27bb8ecaacdabc91c56f8be934401c5f91f864863bd11e035a2b30d0669a52"
const SOURCE_METADATA_COMPANION_SHA256 = "a83fe4aa5b7514158814f6b614f0917514c8e59e3809f3f7f41190c88371b219"

for protocol_dir in protocol_dirs
    audit_path = joinpath(protocol_dir, "protocol_audit.toml")
    report = TOML.parsefile(audit_path)
    @test report["status"] == "PASS"
    @test report["provenance_schema_version"] == 1
    @test report["source_metadata_sha256"] == SOURCE_METADATA_SHA256
    @test report["source_metadata_companion_sha256"] == SOURCE_METADATA_COMPANION_SHA256
    @test occursin("optional external provenance", report["source_metadata_binding_scope"])
    @test occursin("not required", report["source_metadata_binding_scope"])
    @test report["scanner_metadata_matrix"] == [64, 64]
    @test occursin("does not reproduce", report["simulation_geometry"])
    @test occursin("matrix/FOV are explicitly adapted", report["protocol_match_scope"])
    @test report["protocol_report_file"] == "protocol_report.toml"
    @test report["protocol_report_sha256"] == file_sha256(joinpath(
        protocol_dir,
        report["protocol_report_file"],
    ))
    @test report["audit_script_sha256"] == file_sha256(joinpath(
        base_protocol_dir,
        "audit_protocol.jl",
    ))
    audit_wrapper = protocol_dir == base_protocol_dir ?
        joinpath(base_protocol_dir, "audit_protocol.jl") :
        joinpath(protocol_dir, "audit_protocol.jl")
    generator_wrapper = protocol_dir == base_protocol_dir ?
        joinpath(base_protocol_dir, "generate.jl") :
        joinpath(protocol_dir, "generate.jl")
    @test report["audit_wrapper_sha256"] == file_sha256(audit_wrapper)
    @test report["generator_script_sha256"] == file_sha256(joinpath(
        base_protocol_dir,
        "generate.jl",
    ))
    @test report["generator_wrapper_sha256"] == file_sha256(generator_wrapper)

    for encoding in ("REF", "X", "Y", "Z")
        artifact = report["encodings"][encoding]
        expected_file = "$(lowercase(encoding)).seq"
        @test artifact["file"] == expected_file
        @test !isabspath(artifact["file"])
        sequence_path = joinpath(protocol_dir, artifact["file"])
        @test artifact["bytes"] == filesize(sequence_path)
        @test artifact["sha256"] == file_sha256(sequence_path)
    end
end

println("PASS_DP150_PROTOCOL_AUDIT_BINDING_NONCUDA")
