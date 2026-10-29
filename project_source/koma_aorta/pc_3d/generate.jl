using KomaMRI
using SHA
using TOML

root = dirname(@__DIR__)
include(joinpath(root, "gradient_design.jl"))
include(joinpath(root, "utils.jl"))
include(joinpath(root, "cartesian.jl"))
include(joinpath(root, "excitation.jl"))
include(joinpath(root, "gre.jl"))
include(joinpath(root, "PC.jl"))

function parse_csv_tuple(::Type{T}, name, default, expected_length) where {T}
    values = parse.(T, strip.(split(get(ENV, name, default), ",")))
    length(values) == expected_length ||
        error("$name must contain $expected_length comma-separated values")
    return Tuple(values)
end

output_dir = normpath(get(ENV, "AORTA_SEQUENCE_OUTPUT_DIR", @__DIR__))
FOV = parse_csv_tuple(
    Float64,
    "AORTA_SEQUENCE_FOV_M",
    "0.280,0.128,0.128",
    3,
)
matrix = parse_csv_tuple(
    Int,
    "AORTA_SEQUENCE_MATRIX",
    "140,64,64",
    3,
)
cardiac_bins = parse(Int, get(ENV, "AORTA_CARDIAC_BINS", "2"))
requested_RR = parse(Float64, get(ENV, "AORTA_SEQUENCE_RR_S", "0.720"))
venc = parse(Float64, get(ENV, "AORTA_SEQUENCE_VENC_M_PER_S", "2.5"))
cardiac_bins > 0 || error("AORTA_CARDIAC_BINS must be positive")
mkpath(output_dir)

sys = Scanner(Gmax=40e-3, Smax=180, B1=10e-6)
requested_adc_duration = 1e-3
adc_dwell = max(
    round(Int, requested_adc_duration / (matrix[1] - 1) / sys.ADC_Δt),
    1,
) * sys.ADC_Δt
BWpp = 1 / (matrix[1] * adc_dwell)

seqs = build_pc_encoding_scans(
    FOV,
    matrix,
    sys;
    venc,
    cardiac_bins,
    RR=requested_RR,
    BWpp,
    flip_angle=deg2rad(10),
    spoil_phase=0.0,
    n_ramp_shots=10,
    trigger_channel=nothing,
    rf_spoil_increment=deg2rad(117),
)

@info "ADC timing" requested_adc_duration realized_adc_duration=adc_dwell * (matrix[1] - 1) adc_dwell BWpp output_dir matrix cardiac_bins

sequence_report = Dict{String,Any}()
for (encoding, seq) in pairs(seqs)
    check_timing(seq, sys)
    check_hw_limits(seq, sys)
    labels = get_labels(seq)
    adc_blocks = [block for (block, event) in enumerate(seq) if is_ADC_on(event)]
    adc_labels = labels[adc_blocks]
    label_tuples = [
        (Int(label.LIN), Int(label.PAR), Int(label.PHS))
        for label in adc_labels
    ]
    expected_profiles = matrix[2] * matrix[3] * cardiac_bins
    @assert length(adc_blocks) == expected_profiles
    @assert length(unique(label_tuples)) == expected_profiles
    @assert extrema(label.LIN for label in adc_labels) == (0, matrix[2] - 1)
    @assert extrema(label.PAR for label in adc_labels) == (0, matrix[3] - 1)
    @assert extrema(label.PHS for label in adc_labels) == (0, cardiac_bins - 1)

    sequence_path = joinpath(output_dir, "$(lowercase(String(encoding))).seq")
    write_seq(seq, sequence_path)
    sequence_report[String(encoding)] = Dict(
        "path" => sequence_path,
        "bytes" => filesize(sequence_path),
        "sha256" => bytes2hex(sha256(read(sequence_path))),
        "profiles" => length(adc_blocks),
        "phase_labels_zero_based" => sort(unique(Int(label.PHS) for label in adc_labels)),
        "RR_s" => Float64(seq.DEF["RR"]),
        "TR_s" => Float64(seq.DEF["TR"]),
        "lines_per_cardiac_bin" => Int(seq.DEF["LinesPerCardiacBin"]),
        "velocity_axis" => String(seq.DEF["VelocityEncodingAxis"]),
    )
end

report = Dict(
    "status" => "PASS_SEQUENCE_LABEL_VALIDATION",
    "output_dir" => output_dir,
    "FOV_m" => collect(FOV),
    "matrix" => collect(matrix),
    "voxel_size_mm" => 1e3 .* collect(FOV) ./ collect(matrix),
    "cardiac_bins" => cardiac_bins,
    "requested_RR_s" => requested_RR,
    "realized_RR_s" => Float64(seqs.REF.DEF["RR"]),
    "phase_interval_s" => Float64(seqs.REF.DEF["RR"]) / cardiac_bins,
    "Venc_m_per_s" => venc,
    "profile_count_per_encoding" => matrix[2] * matrix[3] * cardiac_bins,
    "sequences" => sequence_report,
)
report_path = joinpath(output_dir, "sequence_report.toml")
open(report_path, "w") do io
    TOML.print(io, report; sorted=true)
end
@info "Sequence set validation passed" report_path realized_RR_s=report["realized_RR_s"] phase_interval_s=report["phase_interval_s"] profiles=report["profile_count_per_encoding"]
