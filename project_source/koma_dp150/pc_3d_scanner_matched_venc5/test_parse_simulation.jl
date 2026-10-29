paths = (
    joinpath(dirname(@__DIR__), "simulate_pc.jl"),
    joinpath(dirname(@__DIR__), "provenance.jl"),
    joinpath(dirname(@__DIR__), "audit_completed_run.jl"),
    joinpath(@__DIR__, "simulate.jl"),
    joinpath(@__DIR__, "simulate_100k.jl"),
    joinpath(@__DIR__, "simulate_100k_seam_safe.jl"),
    joinpath(@__DIR__, "simulate_1m_seam_safe.jl"),
    joinpath(@__DIR__, "generate.jl"),
    joinpath(@__DIR__, "audit_protocol.jl"),
    joinpath(@__DIR__, "benchmark_cpu.jl"),
)
function contains_parse_error(value)
    value isa Expr || return false
    value.head in (:error, :incomplete) && return true
    return any(contains_parse_error, value.args)
end

for path in paths
    parsed = Meta.parseall(read(path, String))
    contains_parse_error(parsed) && error("Parse error in $path")
end
@info "VENC 5.0 simulation scripts parse successfully" paths
