using HDF5
using KomaMRI
using TOML

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
const DEFAULT_INPUT = joinpath(
    REPO_ROOT,
    "outputs",
    "dp150_mri_pipeline",
    "dp150_static_2m_mri.h5",
)
const DEFAULT_SEQUENCE = joinpath(@__DIR__, "pc_3d", "ref.seq")
const DEFAULT_OUTPUT_DIR = joinpath(
    REPO_ROOT,
    "outputs",
    "dp150_mri_pipeline",
    "koma",
)

input_file = normpath(get(ENV, "DP150_STATIC_H5", DEFAULT_INPUT))
sequence_file = normpath(get(ENV, "DP150_REF_SEQUENCE", DEFAULT_SEQUENCE))
output_dir = normpath(get(ENV, "DP150_KOMA_OUTPUT_DIR", DEFAULT_OUTPUT_DIR))
phantom_file = normpath(get(
    ENV,
    "DP150_STATIC_PHANTOM",
    joinpath(output_dir, "dp150_static_2m.phantom"),
))
report_file = joinpath(output_dir, "static_phantom_report.toml")
mkpath(output_dir)

@info "Loading corrected static HDF5" file=input_file
position, nsteps, nparticles, coordinate_space, source_axis_order,
source_center, h5_fov, h5_matrix = h5open(input_file, "r") do file
    position = read(file, "position")
    nsteps = Int(read_attribute(file, "n_steps"))
    nparticles = Int(read_attribute(file, "n_particles"))
    coordinate_space = String(read_attribute(file, "coordinate_space"))
    source_axis_order = Int.(read_attribute(file, "source_axis_order_zero_based"))
    source_center = Float64.(read_attribute(file, "source_geometry_center_m"))
    h5_fov = Float64.(read_attribute(file, "fov_m"))
    h5_matrix = Int.(read_attribute(file, "matrix"))
    return (
        position,
        nsteps,
        nparticles,
        coordinate_space,
        source_axis_order,
        source_center,
        h5_fov,
        h5_matrix,
    )
end

@assert coordinate_space == "MRI_centered_largest_to_smallest"
@assert nsteps == 1
@assert size(position) == (3, nparticles, nsteps)
@assert all(isfinite, position)
@assert sort(source_axis_order) == [0, 1, 2]

sequence = read_seq(sequence_file; verify_signature=true, verbose=false)
sequence_fov = Float64.(sequence.DEF["FOV"])
sequence_matrix = Int.([
    sequence.DEF["Nx"],
    sequence.DEF["Ny"],
    sequence.DEF["Nz"],
])
@assert isapprox(sequence_fov, h5_fov; rtol=0, atol=1e-12)
@assert sequence_matrix == h5_matrix

x = vec(position[1, :, 1])
y = vec(position[2, :, 1])
z = vec(position[3, :, 1])
coords = (x, y, z)
mins = Float64[minimum(coord) for coord in coords]
maxs = Float64[maximum(coord) for coord in coords]
max_abs = Float64[maximum(abs, coord) for coord in coords]
half_margins = sequence_fov ./ 2 .- max_abs
@assert all(half_margins .> 0)

phantom = Phantom(
    name="dp150_static_2m",
    x=x,
    y=y,
    z=z,
    ρ=ones(Float32, nparticles),
    T1=fill(Float32(1.3), nparticles),
    T2=fill(Float32(0.1), nparticles),
)

@info "Writing static Koma phantom" file=phantom_file spins=nparticles
write_phantom(phantom, phantom_file)
@assert isfile(phantom_file)
@assert filesize(phantom_file) > 0

report = Dict(
    "status" => "PASS",
    "input_hdf5" => input_file,
    "sequence_file" => sequence_file,
    "phantom_file" => phantom_file,
    "phantom_bytes" => filesize(phantom_file),
    "particle_count" => nparticles,
    "coordinate_space" => coordinate_space,
    "source_axis_order_zero_based" => source_axis_order,
    "source_geometry_center_m" => source_center,
    "sequence_fov_m" => sequence_fov,
    "sequence_matrix" => sequence_matrix,
    "coordinate_min_m" => mins,
    "coordinate_max_m" => maxs,
    "maximum_absolute_coordinate_m" => max_abs,
    "minimum_half_margin_m" => half_margins,
    "motion" => "NoMotion",
    "T1_s" => 1.3,
    "T2_s" => 0.1,
)
open(report_file, "w") do io
    TOML.print(io, report; sorted=true)
end

@info "Static phantom validation passed" report=report_file margins_m=half_margins
