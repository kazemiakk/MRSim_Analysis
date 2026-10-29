using HDF5
using KomaMRI
using TOML

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
const DEFAULT_INPUT = joinpath(
    REPO_ROOT,
    "outputs",
    "aorta_dp2_mri_pipeline",
    "aorta_motion_1m.h5",
)
const DEFAULT_SEQUENCE_DIR = joinpath(@__DIR__, "pc_3d")
const DEFAULT_OUTPUT_DIR = joinpath(
    REPO_ROOT,
    "outputs",
    "aorta_dp2_mri_pipeline",
    "koma",
)

input_file = normpath(get(ENV, "AORTA_MOTION_H5", DEFAULT_INPUT))
sequence_dir = normpath(get(ENV, "AORTA_SEQUENCE_DIR", DEFAULT_SEQUENCE_DIR))
output_dir = normpath(get(ENV, "AORTA_KOMA_OUTPUT_DIR", DEFAULT_OUTPUT_DIR))
particle_limit = parse(Int, get(ENV, "AORTA_PARTICLE_LIMIT", "0"))
phantom_file = normpath(get(
    ENV,
    "AORTA_MOTION_PHANTOM",
    joinpath(output_dir, "aorta_motion_1m.phantom"),
))
report_file = joinpath(output_dir, "motion_phantom_report.toml")
mkpath(output_dir)
particle_limit >= 0 || error("AORTA_PARTICLE_LIMIT must be non-negative")

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

@info "Loading aorta motion HDF5" file=input_file rr_s=rr
position, initial, tracker_reset, source_dt, nsteps, nparticles = h5open(input_file, "r") do file
    source_dt = Float64(read_attribute(file, "dt"))
    nsteps = Int(read_attribute(file, "n_steps"))
    available_particles = Int(read_attribute(file, "n_particles"))
    nparticles = particle_limit == 0 ? available_particles : particle_limit
    nparticles <= available_particles || error(
        "AORTA_PARTICLE_LIMIT=$nparticles exceeds $available_particles available particles",
    )
    particle_indices = 1:nparticles
    position = Float32.(file["position"][:, particle_indices, :])
    initial = Float32.(file["initial_position"][:, particle_indices])
    tracker_reset = Bool.(file["reset"][particle_indices, :])
    return position, initial, tracker_reset, source_dt, nsteps, nparticles
end

@assert size(position) == (3, nparticles, nsteps)
@assert size(initial) == (3, nparticles)
@assert size(tracker_reset) == (nparticles, nsteps)
@assert all(isfinite, position)
@assert all(isfinite, initial)
@assert isapprox(nsteps * source_dt, 0.72; rtol=0, atol=1e-9)

all_points = reshape(position, 3, :)
source_mins = min.(vec(minimum(all_points; dims=2)), vec(minimum(initial; dims=2)))
source_maxs = max.(vec(maximum(all_points; dims=2)), vec(maximum(initial; dims=2)))
source_extents = source_maxs - source_mins
source_center = (source_maxs + source_mins) / 2
axis_order = sortperm(source_extents; rev=true)

# MRSimTracks stores post-step positions. Prepend the true t=0 seed positions
# so the Koma FlowPath spans the complete normalized source cycle.
paths = ntuple(3) do output_axis
    source_axis = axis_order[output_axis]
    path = Matrix{Float32}(undef, nparticles, nsteps + 1)
    path[:, 1] .= @view(initial[source_axis, :]) .- source_center[source_axis]
    path[:, 2:end] .= @view(position[source_axis, :, :]) .- source_center[source_axis]
    path
end
position = nothing
initial = nothing
all_points = nothing
GC.gc()

reset = Matrix{Bool}(undef, nparticles, nsteps + 1)
reset[:, 1] .= false
reset[:, 2:end] .= tracker_reset
tracker_reset_events = count(tracker_reset)
tracker_reset = nothing

# Through-flow identities generally do not close at the cardiac-cycle wrap.
# Reset every spin immediately before wrapping coordinates to avoid carrying
# magnetization between unrelated pseudoparticle identities.
reset[:, end] .= true
cycle_boundary_reset_count = nparticles
total_reset_flags = count(reset)
GC.gc()

dx, dy, dz = paths
centered_extents = Float64[maximum(path) - minimum(path) for path in paths]
max_abs = Float64[maximum(abs, path) for path in paths]
half_margins = sequence_fov ./ 2 .- max_abs
@assert issorted(centered_extents; rev=true)
@assert all(half_margins .> 0)

flow = FlowPath(dx=dx, dy=dy, dz=dz, spin_reset=reset)
motion = Motion(
    action=flow,
    time=Periodic(period=rr, asymmetry=1.0f0),
    spins=SpinRange(1:nparticles),
)
phantom = Phantom(
    name="aorta_motion_1m",
    x=zeros(Float32, nparticles),
    y=zeros(Float32, nparticles),
    z=zeros(Float32, nparticles),
    ρ=ones(Float32, nparticles),
    T1=fill(Float32(1.3), nparticles),
    T2=fill(Float32(0.1), nparticles),
    motion=motion,
)

@info "Writing periodic aorta motion phantom" file=phantom_file spins=nparticles samples=nsteps + 1
write_phantom(phantom, phantom_file)
@assert isfile(phantom_file)
@assert filesize(phantom_file) > 0

report = Dict(
    "status" => "PASS_APPROXIMATE_POINT_CLOUD",
    "input_hdf5" => input_file,
    "sequence_dir" => sequence_dir,
    "phantom_file" => phantom_file,
    "phantom_bytes" => filesize(phantom_file),
    "particle_count" => nparticles,
    "particle_limit_requested" => particle_limit,
    "particle_selection" => particle_limit == 0 ?
        "all particles" : "deterministic leading particle rows",
    "source_poststep_count" => nsteps,
    "flowpath_sample_count" => nsteps + 1,
    "source_dt_s" => source_dt,
    "source_cycle_s" => nsteps * source_dt,
    "realized_sequence_rr_s" => Float64(rr),
    "motion_time_mapping" => "source 0.720 s mapped linearly to realized sequence RR",
    "source_coordinate_center_m" => Float64.(source_center),
    "source_coordinate_extent_m" => Float64.(source_extents),
    "source_axis_order_largest_to_smallest_one_based" => axis_order,
    "centered_coordinate_extent_m" => centered_extents,
    "sequence_fov_m" => sequence_fov,
    "sequence_matrix" => sequence_matrix,
    "venc_m_per_s" => venc_values[1],
    "maximum_absolute_coordinate_m" => max_abs,
    "minimum_half_margin_m" => half_margins,
    "tracker_reset_events" => tracker_reset_events,
    "cycle_boundary_reset_count" => cycle_boundary_reset_count,
    "total_flowpath_reset_flags" => total_reset_flags,
    "cycle_boundary_reset_reason" => "prevent magnetization transfer across pseudoparticle identity wrap",
    "spatial_boundary_limitation" => "k-nearest-neighbor point-cloud support with seed-pool recycling; no explicit inlet/outlet caps or wall mesh",
    "T1_s" => 1.3,
    "T2_s" => 0.1,
)
open(report_file, "w") do io
    TOML.print(io, report; sorted=true)
end

@info "Motion aorta phantom validation passed" report=report_file margins_m=half_margins
