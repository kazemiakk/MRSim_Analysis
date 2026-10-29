using HDF5
using KomaMRI

# Set this to the particle HDF5 input before running the script.
const MOTION_H5_FILE = "PATH TO INPUT H5 FILE"
const MOTION_PHANTOM_FILE = joinpath(@__DIR__, "aorta_motion.phantom")

# Duration of one periodically repeated cardiac cycle, in seconds.
const RR_INTERVAL = 0.8

position, reset_flags, source_dt, nsteps, nparticles = h5open(MOTION_H5_FILE, "r") do file
    position = read(file, "position")
    reset_flags = read(file, "reset")
    source_dt = read_attribute(file, "dt")
    nsteps = read_attribute(file, "n_steps")
    nparticles = read_attribute(file, "n_particles")

    @assert size(position) == (3, nparticles, nsteps)
    @assert size(reset_flags) == (nparticles, nsteps)
    @assert all(isfinite, position)

    position, reset_flags, source_dt, nsteps, nparticles
end

@assert RR_INTERVAL > 0

# Center the complete motion envelope, then reorder axes from largest to smallest extent.
points = reshape(position, 3, :)
mins = vec(minimum(points; dims=2))
maxs = vec(maximum(points; dims=2))
extents = maxs - mins
centers = (maxs + mins) / 2
axis_order = sortperm(extents; rev=true)

paths = map(axis_order) do axis
    position[axis, :, :] .- centers[axis]
end
dx, dy, dz = paths

reset = reset_flags .== 1
rr = Float32(RR_INTERVAL)
flow = FlowPath(dx=dx, dy=dy, dz=dz, spin_reset=reset)
motion = Motion(
    action=flow,
    time=Periodic(period=rr, asymmetry=0.999f0),
    spins=SpinRange(1:nparticles),
)

phantom = Phantom(
    name="aorta_motion",
    x=zeros(Float32, nparticles),
    y=zeros(Float32, nparticles),
    z=zeros(Float32, nparticles),
    ρ=ones(Float32, nparticles),
    T1=fill(Float32(1.3), nparticles),
    T2=fill(Float32(0.1), nparticles),
    motion=motion,
)

write_phantom(phantom, MOTION_PHANTOM_FILE)

saved = read_phantom(MOTION_PHANTOM_FILE)
saved_paths = (saved.motion.action.dx, saved.motion.action.dy, saved.motion.action.dz)
saved_extents = [maximum(path) - minimum(path) for path in saved_paths]

@assert length(saved) == nparticles
@assert saved.motion isa Motion
@assert saved.motion.action isa FlowPath
@assert saved.motion.time.periodic
@assert isapprox(saved.motion.time.t_end - saved.motion.time.t_start, rr)
@assert all(size(path) == (nparticles, nsteps) for path in saved_paths)
@assert issorted(saved_extents; rev=true)
@assert all(path -> isapprox(minimum(path), -maximum(path); rtol=1e-6), saved_paths)

@info "Wrote motion phantom" file=MOTION_PHANTOM_FILE spins=nparticles steps=nsteps rr=RR_INTERVAL source_cycle=nsteps * source_dt axis_order extents=saved_extents resets=count(reset)
