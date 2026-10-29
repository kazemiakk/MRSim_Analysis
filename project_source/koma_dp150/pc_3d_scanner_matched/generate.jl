using KomaMRI
using TOML

root = dirname(@__DIR__)
include(joinpath(root, "gradient_design.jl"))
include(joinpath(root, "utils.jl"))
include(joinpath(root, "cartesian.jl"))
include(joinpath(root, "excitation.jl"))
include(joinpath(root, "gre.jl"))
include(joinpath(root, "PC.jl"))
include(joinpath(root, "provenance.jl"))

using .Dp150Provenance

const TARGET_TR = 46.88e-3
const TARGET_TE = 2.74e-3
const TARGET_RR = 1.012
const TARGET_CARDIAC_BINS = 15
const TARGET_BWPP = 350.0
const VOXEL_SIZE = (1.25e-3, 1.25e-3, 2.5e-3)
const TARGET_VENC = parse(Float64, get(ENV, "DP150_VENC_M_PER_S", "4.5"))
const SEQUENCE_OUTPUT_DIR = normpath(get(
    ENV,
    "DP150_SEQUENCE_OUTPUT_DIR",
    @__DIR__,
))
const GENERATOR_WRAPPER_FILE = normpath(get(
    ENV,
    "DP150_GENERATOR_WRAPPER_FILE",
    @__FILE__,
))

TARGET_VENC > 0 || error("DP150_VENC_M_PER_S must be positive.")
mkpath(SEQUENCE_OUTPUT_DIR)

# The measured DICOM frame is 80 x 80 mm in-plane. The dp150 model is
# 466.2 mm along its readout axis, so a 500 mm readout FOV is required to
# preserve the physical model while retaining the measured pixel spacing.
matrix = (400, 64, 32)
FOV = Tuple(matrix[index] * VOXEL_SIZE[index] for index in eachindex(matrix))

# A 1 ns design raster represents the scanner-reported 350 Hz/pixel bandwidth
# to within the integer-nanosecond Pulseq ADC dwell convention.
sys = Scanner(
    B0=3.0,
    Gmax=40e-3,
    Smax=180,
    B1=10e-6,
    ADC_Δt=1e-9,
    DUR_Δt=1e-9,
)

seqs = build_pc_encoding_scans(
    FOV,
    matrix,
    sys;
    venc=TARGET_VENC,
    cardiac_bins=TARGET_CARDIAC_BINS,
    RR=TARGET_RR,
    BWpp=TARGET_BWPP,
    flip_angle=deg2rad(8),
    rf_time_bandwidth_product=1,
    rf_window=0.0,
    exact_rf_flip=true,
    spoil_phase=0.0,
    n_ramp_shots=10,
    trigger_channel=nothing,
    rf_spoil_increment=deg2rad(117),
    target_TE=TARGET_TE,
    target_TR=TARGET_TR,
    exact_rr=true,
    merge_velocity_prephaser=true,
    prephaser_duration=0.8e-3,
)

for (encoding, seq) in pairs(seqs)
    seq.DEF["ProtocolName"] = "4D Qp150 scanner-matched simulation, VENC $(TARGET_VENC) m/s"
    seq.DEF["SourceMetadataDate"] = "2025-08-28"
    seq.DEF["PixelSpacing"] = [VOXEL_SIZE[1], VOXEL_SIZE[2]]
    seq.DEF["SliceThickness"] = VOXEL_SIZE[3]
    seq.DEF["MagneticFieldStrength"] = 3.0
    check_timing(seq, sys)
    check_hw_limits(seq, sys)
    output = joinpath(SEQUENCE_OUTPUT_DIR, "$(lowercase(String(encoding))).seq")
    temporary = output * ".partial"
    write_seq(seq, temporary)
    mv(temporary, output; force=true)
end

sequence_artifacts = Dict(
    String(encoding) => Dict(
        "file" => "$(lowercase(String(encoding))).seq",
        "bytes" => filesize(joinpath(
            SEQUENCE_OUTPUT_DIR,
            "$(lowercase(String(encoding))).seq",
        )),
        "sha256" => file_sha256(joinpath(
            SEQUENCE_OUTPUT_DIR,
            "$(lowercase(String(encoding))).seq",
        )),
    )
    for encoding in keys(seqs)
)

reference = seqs.REF
report = Dict(
    "provenance_schema_version" => 1,
    "julia_version" => string(VERSION),
    "koma_mri_version" => string(Base.pkgversion(KomaMRI)),
    "generator_script_sha256" => file_sha256(@__FILE__),
    "generator_wrapper_sha256" => file_sha256(GENERATOR_WRAPPER_FILE),
    "source_metadata" => "phantom_8_28_2025_metadeta.rtf",
    "fov_m" => Float64.(reference.DEF["FOV"]),
    "matrix" => Int[reference.DEF["Nx"], reference.DEF["Ny"], reference.DEF["Nz"]],
    "voxel_size_m" => Float64.(reference.DEF["VoxelSize"]),
    "pixel_spacing_mm" => collect(1e3 .* VOXEL_SIZE[1:2]),
    "slice_thickness_mm" => 1e3 * VOXEL_SIZE[3],
    "requested_bandwidth_hz_per_pixel" => TARGET_BWPP,
    "realized_bandwidth_hz_per_pixel" => Float64(reference.DEF["BandwidthPerPixel"]),
    "requested_te_ms" => 1e3 * TARGET_TE,
    "realized_te_ms" => 1e3 * Float64(reference.DEF["TE"]),
    "requested_tr_ms" => 1e3 * TARGET_TR,
    "realized_tr_ms" => 1e3 * Float64(reference.DEF["TR"]),
    "cardiac_bins" => Int(reference.DEF["CardiacBins"]),
    "requested_rr_ms" => 1e3 * TARGET_RR,
    "realized_rr_ms" => 1e3 * Float64(reference.DEF["RR"]),
    "nominal_phase_interval_ms" => 1e3 * Float64(reference.DEF["NominalPhaseInterval"]),
    "minimum_phase_interval_ms" => 1e3 * Float64(reference.DEF["MinimumPhaseInterval"]),
    "maximum_phase_interval_ms" => 1e3 * Float64(reference.DEF["MaximumPhaseInterval"]),
    "flip_angle_degrees" => 8.0,
    "venc_m_per_s" => TARGET_VENC,
    "encoding_files" => ["ref.seq", "x.seq", "y.seq", "z.seq"],
    "sequence_artifacts" => sequence_artifacts,
)
protocol_report_file = joinpath(SEQUENCE_OUTPUT_DIR, "protocol_report.toml")
temporary_report_file = protocol_report_file * ".partial"
open(temporary_report_file, "w") do io
    TOML.print(io, report; sorted=true)
end
mv(temporary_report_file, protocol_report_file; force=true)

@info "Scanner-matched protocol generated" output_dir=SEQUENCE_OUTPUT_DIR report
