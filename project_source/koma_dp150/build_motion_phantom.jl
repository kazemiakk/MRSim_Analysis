using HDF5
using KomaMRI
using TOML

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
const DEFAULT_INPUT = joinpath(
    REPO_ROOT,
    "outputs",
    "dp150_mri_pipeline",
    "dp150_motion_1m_mri.h5",
)
const DEFAULT_SEQUENCE_DIR = joinpath(@__DIR__, "pc_3d")
const DEFAULT_OUTPUT_DIR = joinpath(
    REPO_ROOT,
    "outputs",
    "dp150_mri_pipeline",
    "koma",
)

input_file = normpath(get(ENV, "DP150_MOTION_H5", DEFAULT_INPUT))
sequence_dir = normpath(get(ENV, "DP150_SEQUENCE_DIR", DEFAULT_SEQUENCE_DIR))
output_dir = normpath(get(ENV, "DP150_KOMA_OUTPUT_DIR", DEFAULT_OUTPUT_DIR))
phantom_file = normpath(get(
    ENV,
    "DP150_MOTION_PHANTOM",
    joinpath(output_dir, "dp150_motion_1m.phantom"),
))
phantom_name = get(
    ENV,
    "DP150_PHANTOM_NAME",
    splitext(basename(phantom_file))[1],
)
report_file = normpath(get(
    ENV,
    "DP150_MOTION_REPORT",
    joinpath(output_dir, "motion_phantom_report.toml"),
))
mkpath(output_dir)

sequences = Dict(
    encoding => read_seq(
        joinpath(sequence_dir, "$(lowercase(encoding)).seq");
        verify_signature=true,
        verbose=false,
    )
    for encoding in ("REF", "X", "Y", "Z")
)
rr_values = Float64[sequence.DEF["RR"] for sequence in values(sequences)]
fov_values = [Float64.(sequence.DEF["FOV"]) for sequence in values(sequences)]
matrix_values = [
    Int.([sequence.DEF["Nx"], sequence.DEF["Ny"], sequence.DEF["Nz"]])
    for sequence in values(sequences)
]
venc_values = Float64[sequence.DEF["Venc"] for sequence in values(sequences)]
@assert all(isapprox(value, rr_values[1]; rtol=0, atol=1e-12) for value in rr_values)
@assert all(isapprox(value, fov_values[1]; rtol=0, atol=1e-12) for value in fov_values)
@assert all(value == matrix_values[1] for value in matrix_values)
@assert all(isapprox(value, venc_values[1]; rtol=0, atol=1e-12) for value in venc_values)
rr = Float32(rr_values[1])
sequence_fov = fov_values[1]
sequence_matrix = matrix_values[1]

@info "Loading corrected motion HDF5" file=input_file rr_s=rr
position, initial, tracker_reset, source_dt, nsteps, nparticles,
coordinate_space, h5_fov, h5_matrix = h5open(input_file, "r") do file
    position = read(file, "position")
    initial = read(file, "initial_position")
    tracker_reset = Bool.(read(file, "reset"))
    source_dt = Float64(read_attribute(file, "dt"))
    nsteps = Int(read_attribute(file, "n_steps"))
    nparticles = Int(read_attribute(file, "n_particles"))
    coordinate_space = String(read_attribute(file, "coordinate_space"))
    h5_fov = Float64.(read_attribute(file, "fov_m"))
    h5_matrix = Int.(read_attribute(file, "matrix"))
    return (
        position,
        initial,
        tracker_reset,
        source_dt,
        nsteps,
        nparticles,
        coordinate_space,
        h5_fov,
        h5_matrix,
    )
end

@assert coordinate_space == "MRI_centered_largest_to_smallest"
@assert size(position) == (3, nparticles, nsteps)
@assert size(initial) == (3, nparticles)
@assert size(tracker_reset) == (nparticles, nsteps)
@assert all(isfinite, position)
@assert all(isfinite, initial)
@assert isapprox(nsteps * source_dt, 0.72; rtol=0, atol=1e-9)
@assert isapprox(sequence_fov, h5_fov; rtol=0, atol=1e-12)
@assert sequence_matrix == h5_matrix

# Koma FlowPath samples span the complete normalized interval [0, 1]. The
# MRSimTracks position dataset contains post-step samples, so prepend the true
# t=0 seed coordinates before constructing the path.
paths = ntuple(3) do axis
    path = Matrix{Float32}(undef, nparticles, nsteps + 1)
    path[:, 1] .= initial[axis, :]
    path[:, 2:end] .= position[axis, :, :]
    path
end
position = nothing
initial = nothing
GC.gc()

reset = Matrix{Bool}(undef, nparticles, nsteps + 1)
reset[:, 1] .= false
reset[:, 2:end] .= tracker_reset
tracker_reset_events = count(tracker_reset)
tracker_reset = nothing

# Through-flow pseudoparticle identities generally do not close at the cardiac
# wrap. Reset every spin immediately before the periodic coordinate wrap so
# magnetization is not carried from one identity to another.
reset[:, end] .= true
cycle_boundary_reset_count = nparticles
total_reset_flags = count(reset)
GC.gc()

dx, dy, dz = paths
max_abs = Float64[
    maximum(abs, dx),
    maximum(abs, dy),
    maximum(abs, dz),
]
half_margins = sequence_fov ./ 2 .- max_abs
@assert all(half_margins .> 0)

flow = FlowPath(dx=dx, dy=dy, dz=dz, spin_reset=reset)
motion = Motion(
    action=flow,
    time=Periodic(period=rr, asymmetry=1.0f0),
    spins=SpinRange(1:nparticles),
)
phantom = Phantom(
    name=phantom_name,
    x=zeros(Float32, nparticles),
    y=zeros(Float32, nparticles),
    z=zeros(Float32, nparticles),
    ρ=ones(Float32, nparticles),
    T1=fill(Float32(1.3), nparticles),
    T2=fill(Float32(0.1), nparticles),
    motion=motion,
)

@info "Writing periodic motion Koma phantom" file=phantom_file spins=nparticles samples=nsteps + 1
write_phantom(phantom, phantom_file)
@assert isfile(phantom_file)
@assert filesize(phantom_file) > 0

report = Dict(
    "status" => "PASS",
    "input_hdf5" => input_file,
    "sequence_dir" => sequence_dir,
    "phantom_file" => phantom_file,
    "phantom_name" => phantom_name,
    "phantom_bytes" => filesize(phantom_file),
    "particle_count" => nparticles,
    "source_poststep_count" => nsteps,
    "flowpath_sample_count" => nsteps + 1,
    "source_dt_s" => source_dt,
    "source_cycle_s" => nsteps * source_dt,
    "realized_sequence_rr_s" => Float64(rr),
    "motion_time_mapping" => "source 0.720 s mapped linearly to realized sequence RR",
    "periodic_asymmetry" => 1.0,
    "coordinate_space" => coordinate_space,
    "sequence_fov_m" => sequence_fov,
    "sequence_matrix" => sequence_matrix,
    "venc_m_per_s" => venc_values[1],
    "maximum_absolute_coordinate_m" => max_abs,
    "minimum_half_margin_m" => half_margins,
    "tracker_reset_events" => tracker_reset_events,
    "cycle_boundary_reset_count" => cycle_boundary_reset_count,
    "total_flowpath_reset_flags" => total_reset_flags,
    "cycle_boundary_reset_reason" => "prevent magnetization transfer across pseudoparticle identity wrap",
    "T1_s" => 1.3,
    "T2_s" => 0.1,
)
open(report_file, "w") do io
    TOML.print(io, report; sorted=true)
end

@info "Motion phantom validation passed" report=report_file margins_m=half_margins
