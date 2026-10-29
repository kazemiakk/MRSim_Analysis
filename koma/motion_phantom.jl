using HDF5
using KomaMRI

# The input must hold EXACTLY ONE cardiac cycle. This script maps the whole
# `position` array onto one RR, so a 10 s file would be replayed 10x too fast
# with no error raised. Use scripts/slice_one_cycle.py first.
const MOTION_H5_FILE = get(
    ENV,
    "KOMA_TRACKS",
    joinpath(@__DIR__, "..", "aorta_dp2_one_cycle.h5"),
)
const MOTION_PHANTOM_FILE = get(
    ENV,
    "KOMA_MOTION_PHANTOM",
    joinpath(@__DIR__, "aorta_motion.phantom"),
)

# The phantom period must equal the RR the sequence actually realizes, not the
# one that was requested: `build_pc` rounds each cardiac bin to a whole number of
# TRs, so the realized RR moves whenever TR moves -- and lowering Venc lengthens
# the bipolar, which lengthens TR. Reading it from the .seq keeps the two locked
# together instead of relying on a hard-coded constant.
const SEQUENCE_FILE = get(
    ENV,
    "KOMA_SEQ",
    joinpath(@__DIR__, "seq", "pc_3d_venc03", "ref.seq"),
)

"""
    seq_definition(path, key)

Read one numeric entry from a Pulseq file's `[DEFINITIONS]` block. Parsing the
header directly avoids loading a multi-megabyte sequence just to read a scalar.
"""
function seq_definition(path, key)
    in_definitions = false
    for line in eachline(path)
        stripped = strip(line)
        isempty(stripped) && continue
        startswith(stripped, "#") && continue
        if startswith(stripped, "[")
            in_definitions = stripped == "[DEFINITIONS]"
            continue
        end
        in_definitions || continue
        fields = split(stripped)
        length(fields) >= 2 && fields[1] == key && return parse(Float64, fields[2])
    end
    error("definition $key not found in $path")
end

const RR_INTERVAL = seq_definition(SEQUENCE_FILE, "RR")
@info "Realized RR read from sequence" SEQUENCE_FILE RR_INTERVAL

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
