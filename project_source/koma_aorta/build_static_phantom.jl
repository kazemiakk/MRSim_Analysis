using HDF5
using KomaMRI
using TOML

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
const DEFAULT_INPUT = joinpath(
    REPO_ROOT,
    "outputs",
    "aorta_dp2_archive_2026-08-21",
    "recovered_analysis",
    "aorta_dp2_uniform_2M.h5",
)
const DEFAULT_SEQUENCE = joinpath(@__DIR__, "pc_3d", "ref.seq")
const DEFAULT_OUTPUT_DIR = joinpath(
    REPO_ROOT,
    "outputs",
    "aorta_dp2_mri_pipeline",
    "koma",
)

input_file = normpath(get(ENV, "AORTA_STATIC_H5", DEFAULT_INPUT))
sequence_file = normpath(get(ENV, "AORTA_REF_SEQUENCE", DEFAULT_SEQUENCE))
output_dir = normpath(get(ENV, "AORTA_KOMA_OUTPUT_DIR", DEFAULT_OUTPUT_DIR))
phantom_file = normpath(get(
    ENV,
    "AORTA_STATIC_PHANTOM",
    joinpath(output_dir, "aorta_static_2m.phantom"),
))
report_file = joinpath(output_dir, "static_phantom_report.toml")
mkpath(output_dir)

@info "Loading approximately volume-uniform aorta seed cloud" file=input_file
points, nparticles = h5open(input_file, "r") do file
    raw = Float32.(read(file, "points"))
    points = if size(raw, 1) == 3
        raw
    elseif size(raw, 2) == 3
        permutedims(raw)
    else
        error("Expected points with one dimension of length 3, got $(size(raw))")
    end
    expected = Int(read_attribute(file, "n_points"))
    @assert size(points) == (3, expected)
    return points, expected
end
@assert all(isfinite, points)

source_mins = vec(minimum(points; dims=2))
source_maxs = vec(maximum(points; dims=2))
source_extents = source_maxs - source_mins
source_center = (source_maxs + source_mins) / 2
axis_order = sortperm(source_extents; rev=true)

coords = map(axis_order) do axis
    vec(points[axis, :]) .- source_center[axis]
end
points = nothing
GC.gc()
x, y, z = coords
centered_extents = Float64[maximum(coord) - minimum(coord) for coord in coords]
centered_mins = Float64[minimum(coord) for coord in coords]
centered_maxs = Float64[maximum(coord) for coord in coords]

sequence = read_seq(sequence_file; verify_signature=true, verbose=false)
sequence_fov = Float64.(sequence.DEF["FOV"])
sequence_matrix = Int.([
    sequence.DEF["Nx"],
    sequence.DEF["Ny"],
    sequence.DEF["Nz"],
])
maximum_absolute_coordinate = Float64[maximum(abs, coord) for coord in coords]
half_margins = sequence_fov ./ 2 .- maximum_absolute_coordinate

@assert issorted(centered_extents; rev=true)
@assert all(half_margins .> 0)
@assert all(
    isapprox(centered_mins[index], -centered_maxs[index]; rtol=1e-6, atol=1e-8)
    for index in eachindex(centered_mins)
)

phantom = Phantom(
    name="aorta_static_2m",
    x=x,
    y=y,
    z=z,
    ρ=ones(Float32, nparticles),
    T1=fill(Float32(1.3), nparticles),
    T2=fill(Float32(0.1), nparticles),
)

@info "Writing static Koma phantom" file=phantom_file spins=nparticles
write_phantom(phantom, phantom_file)
saved = read_phantom(phantom_file)
@assert length(saved) == nparticles
@assert saved.motion isa NoMotion

report = Dict(
    "status" => "PASS",
    "input_hdf5" => input_file,
    "sequence_file" => sequence_file,
    "phantom_file" => phantom_file,
    "phantom_bytes" => filesize(phantom_file),
    "particle_count" => nparticles,
    "source_coordinate_min_m" => Float64.(source_mins),
    "source_coordinate_max_m" => Float64.(source_maxs),
    "source_coordinate_center_m" => Float64.(source_center),
    "source_coordinate_extent_m" => Float64.(source_extents),
    "source_axis_order_largest_to_smallest_one_based" => axis_order,
    "centered_coordinate_min_m" => centered_mins,
    "centered_coordinate_max_m" => centered_maxs,
    "centered_coordinate_extent_m" => centered_extents,
    "sequence_fov_m" => sequence_fov,
    "sequence_matrix" => sequence_matrix,
    "minimum_half_margin_m" => half_margins,
    "motion" => "NoMotion",
    "T1_s" => 1.3,
    "T2_s" => 0.1,
    "scientific_scope" => "Static sequence and reconstruction litmus test using approximate volume-weighted point-cloud seeding",
)
open(report_file, "w") do io
    TOML.print(io, report; sorted=true)
end

@info "Static aorta phantom validation passed" report=report_file margins_m=half_margins
