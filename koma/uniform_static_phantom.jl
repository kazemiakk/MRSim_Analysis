using HDF5
using KomaMRI
using Plots

# Build a static KomaMRI phantom from a uniform-density lumen point cloud
# (scripts/seed_uniform_lumen.py), rather than from a flattened particle
# trajectory. The trajectory-union phantom has spin density proportional to
# residence time (~1/speed), measured at a 1665x range over 2 mm voxels on aorta
# dp2; the uniform cloud measures 1.7x, which is Poisson.
#
# Centering, largest-to-smallest axis ordering, and the read-back assertions are
# kept identical to static_phantom.jl so the two phantoms are interchangeable
# downstream.

const POINTS_FILE = joinpath(@__DIR__, "..", "aorta_dp2_uniform_2M.h5")
const PHANTOM_FILE = joinpath(@__DIR__, "aorta_uniform_static.phantom")
const PROJECTION_FILE = joinpath(@__DIR__, "aorta_uniform_static_projections.png")

const T1_VALUE = 1.3f0
const T2_VALUE = 0.1f0

points, npoints = h5open(POINTS_FILE, "r") do file
    points = read(file, "points")
    npoints = read_attribute(file, "n_points")

    @assert size(points) == (3, npoints)
    @assert all(isfinite, points)

    points, npoints
end

mins = vec(minimum(points; dims=2))
maxs = vec(maximum(points; dims=2))
extents = maxs - mins
centers = (maxs + mins) / 2
axis_order = sortperm(extents; rev=true)

coords = map(axis_order) do axis
    vec(points[axis, :]) .- centers[axis]
end
x, y, z = coords

phantom = Phantom(
    name="aorta_uniform_static",
    x=x,
    y=y,
    z=z,
    ρ=ones(Float32, npoints),
    T1=fill(T1_VALUE, npoints),
    T2=fill(T2_VALUE, npoints),
)

write_phantom(phantom, PHANTOM_FILE)

saved = read_phantom(PHANTOM_FILE)
saved_coords = (saved.x, saved.y, saved.z)
saved_extents = [maximum(coord) - minimum(coord) for coord in saved_coords]

@assert length(saved) == npoints
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

@info "Wrote uniform static phantom" file=PHANTOM_FILE spins=npoints axis_order extents=saved_extents
@info "Wrote particle projections" file=PROJECTION_FILE
