using HDF5
using KomaMRI
using Plots

# Set this to the particle HDF5 input before running the script.
const H5_FILE = "PATH TO INPUT H5 FILE"
const PHANTOM_FILE = joinpath(@__DIR__, "aorta_static.phantom")
const PROJECTION_FILE = joinpath(@__DIR__, "aorta_static_projections.png")

position, nsteps, nparticles = h5open(H5_FILE, "r") do file
    position = read(file, "position")
    nsteps = read_attribute(file, "n_steps")
    nparticles = read_attribute(file, "n_particles")

    @assert size(position) == (3, nparticles, nsteps)
    @assert all(isfinite, position)

    position, nsteps, nparticles
end

points = reshape(position, 3, :)
mins = vec(minimum(points; dims=2))
maxs = vec(maximum(points; dims=2))
extents = maxs - mins
centers = (maxs + mins) / 2
axis_order = sortperm(extents; rev=true)

coords = map(axis_order) do axis
    vec(points[axis, :]) .- centers[axis]
end
x, y, z = coords

nspins = nsteps * nparticles
phantom = Phantom(
    name="aorta_static",
    x=x,
    y=y,
    z=z,
    ρ=ones(Float32, nspins),
    T1=fill(Float32(1.3), nspins),
    T2=fill(Float32(0.1), nspins),
)

write_phantom(phantom, PHANTOM_FILE)

saved = read_phantom(PHANTOM_FILE)
saved_coords = (saved.x, saved.y, saved.z)
saved_extents = [maximum(coord) - minimum(coord) for coord in saved_coords]

@assert length(saved) == nspins
@assert saved.motion isa NoMotion
@assert issorted(saved_extents; rev=true)
@assert all(coord -> isapprox(minimum(coord), -maximum(coord); rtol=1e-6), saved_coords)

projection_kwargs = (
    bins=250,
    aspect_ratio=:equal,
    color=:viridis,
)

xy = Plots.histogram2d(saved.x, saved.y; projection_kwargs..., title="XY", xlabel="X (m)", ylabel="Y (m)")
xz = Plots.histogram2d(saved.x, saved.z; projection_kwargs..., title="XZ", xlabel="X (m)", ylabel="Z (m)")
yz = Plots.histogram2d(saved.y, saved.z; projection_kwargs..., title="YZ", xlabel="Y (m)", ylabel="Z (m)")
projections = Plots.plot(xy, xz, yz; layout=(1, 3), size=(2100, 650), dpi=150)
Plots.savefig(projections, PROJECTION_FILE)

@info "Wrote static phantom" file=PHANTOM_FILE spins=nspins axis_order extents=saved_extents
@info "Wrote particle projections" file=PROJECTION_FILE
