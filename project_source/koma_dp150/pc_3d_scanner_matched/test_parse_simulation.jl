paths = (
    joinpath(dirname(@__DIR__), "simulate_pc.jl"),
    joinpath(@__DIR__, "simulate.jl"),
)
for path in paths
    Meta.parseall(read(path, String))
end
@info "Simulation scripts parse successfully" paths
