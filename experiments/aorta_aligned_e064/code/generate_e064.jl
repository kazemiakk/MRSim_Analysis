# E-064 sequence builder: verbatim copy of E-062 code/generate_e062.jl with a second (candy-cane) FOV/matrix for the
# final set; the G0a/G0b regressions still use the production FOV/matrix.
# Frozen koma_aorta helpers are included read-only from the D copy; build_pc_e062 is a copy of
# koma_aorta/PC.jl build_pc restricted to the production path (3D, retrospective, linear order,
# one encoding per file) with two additions: an explicit lines_per_bin, a delay appended to
# every shot (shot_pad) and a delay at the start of every heartbeat group (beat_pad).
using KomaMRI
using SHA
using TOML

const KA = "D:/MRSim/MRSIM/MRSimTracks/koma_aorta"
for f in ("gradient_design.jl", "utils.jl", "cartesian.jl", "excitation.jl", "gre.jl", "PC.jl")
    include(joinpath(KA, f))
end

const ROOT = "D:/MRSim/outputs/aorta_aligned_e064"
const PRODUCTION = "D:/MRSim/MRSIM/MRSimTracks/outputs/aorta_dp2_mri_pipeline/koma/multibin_8x4mm/sequences"
const FOV = (0.280, 0.128, 0.128)
const MATRIX = (72, 32, 32)
const FOV1 = (0.288, 0.160, 0.096)      # E-064: readout = descending axis, phase = horizontal, slab = normal
const MATRIX1 = (72, 40, 24)
const VENC = 2.5
const NAMES = (:REF, :X, :Y, :Z)
const AXES = (:REF, :RO, :PE, :SS)

sha_file(path) = bytes2hex(sha256(read(path)))

# As koma_aorta/pc_3d/generate.jl.
sys = Scanner(Gmax=40e-3, Smax=180, B1=10e-6)
adc_dwell = max(round(Int, 1e-3 / (MATRIX[1] - 1) / sys.ADC_Δt), 1) * sys.ADC_Δt
BWpp = 1 / (MATRIX[1] * adc_dwell)
common = (venc=VENC, BWpp, flip_angle=deg2rad(10), spoil_phase=0.0, n_ramp_shots=10,
          rf_spoil_increment=deg2rad(117))

function build_pc_e062(FOV, matrix, sys; venc, cardiac_bins, lines_per_bin, shot_pad, beat_pad, BWpp,
                       flip_angle, spoil_phase, n_ramp_shots, rf_spoil_increment, _encoding_axis)
    FOV = Tuple(Float64.(FOV)); matrix = Tuple(Int.(matrix))
    length(matrix) == 3 || error("E-062 builder is 3D only")
    excitation = slice_selective_sinc(flip_angle, FOV[3], sys; BW=nothing)
    fixed_area = ntuple(axis -> _pc_gradient_area(excitation.GR[axis, 2]), 3)
    spoiler_area = spoil_phase / (2π * γ * FOV[3])
    readout_kernel = sgre_base(FOV, matrix, sys, BWpp; fixed_area, spoil_area=(0.0, 0.0, spoiler_area))
    first_moment = _pc_first_moment(venc)
    flat, ramp = _pc_bipolar_timing(first_moment, sys)
    reference_module = _pc_velocity_module(:SS, 0.0, flat, ramp, sys)
    write_set_label = false
    write_phs_label = cardiac_bins > 1
    velocity_modules = _encoding_axis === :REF ? (reference_module,) :
        (_pc_velocity_module(_encoding_axis, first_moment, flat, ramp, sys),)
    ky = collect(-(matrix[2] ÷ 2):(matrix[2] - 1 - matrix[2] ÷ 2))
    kz = collect(-(matrix[3] ÷ 2):(matrix[3] - 1 - matrix[3] ÷ 2))
    spatial_encodings = [(line, partition) for partition in kz for line in ky]
    readouts = Dict(encoding => readout_kernel(encoding...) for encoding in spatial_encodings)
    representative = _pc_shot(excitation, readouts[first(spatial_encodings)], velocity_modules[1], 1, 1, 1,
                              rf_spoil_increment, write_set_label=write_set_label, write_phs_label=write_phs_label)
    base_TR = dur(representative)
    TR = base_TR + shot_pad
    actual_RR = cardiac_bins * (lines_per_bin * TR) + beat_pad   # same evaluation order as build_pc
    encoding_groups = [spatial_encodings[first:min(first + lines_per_bin - 1, end)]
                       for first in 1:lines_per_bin:length(spatial_encodings)]
    pad = Sequence(sys)
    shot_pad > 0 && (@addblock pad += make_delay(shot_pad * u"s"))
    bpad = Sequence(sys)
    beat_pad > 0 && (@addblock bpad += make_delay(beat_pad * u"s"))

    sequence = Sequence(sys)
    rf_index = 1
    @addblock begin
        for encodings in encoding_groups, encoding in eachindex(velocity_modules)
            if n_ramp_shots > 0 && rf_index == 1
                ramp_start = min(1.0, (3π / 180) / flip_angle)
                for scale in range(ramp_start, 1.0; length=n_ramp_shots)
                    shot = _pc_shot(excitation, readouts[first(encodings)], reference_module, 1, 1, rf_index,
                                    rf_spoil_increment; flip_scale=scale, write_set_label, write_phs_label)
                    sequence += _pc_disable_adc!(shot)
                    shot_pad > 0 && (sequence += pad)
                    rf_index = rf_index + 1
                end
            end
            beat_pad > 0 && (sequence += bpad)
            for cardiac_bin in 1:cardiac_bins
                for spatial_encoding in encodings
                    sequence += _pc_shot(excitation, readouts[spatial_encoding], velocity_modules[encoding], encoding,
                                         cardiac_bin, rf_index, rf_spoil_increment,
                                         write_set_label=write_set_label, write_phs_label=write_phs_label)
                    shot_pad > 0 && (sequence += pad)
                    rf_index = rf_index + 1
                end
                for _ in (length(encodings) + 1):lines_per_bin
                    shot = _pc_shot(excitation, readouts[last(encodings)], velocity_modules[encoding], encoding,
                                    cardiac_bin, rf_index, rf_spoil_increment,
                                    write_set_label=write_set_label, write_phs_label=write_phs_label)
                    sequence += _pc_disable_adc!(shot)
                    shot_pad > 0 && (sequence += pad)
                    rf_index = rf_index + 1
                end
            end
        end
    end

    actual_BWpp = 1 / (matrix[1] * _design_cartesian_readout(FOV[1], matrix[1], sys, BWpp).dwell)
    sequence.DEF["FOV"] = collect(FOV)
    sequence.DEF["Nx"] = matrix[1]
    sequence.DEF["Ny"] = matrix[2]
    sequence.DEF["Nz"] = matrix[3]
    sequence.DEF["TR"] = TR
    sequence.DEF["RR"] = actual_RR
    sequence.DEF["CardiacBins"] = cardiac_bins
    sequence.DEF["LinesPerCardiacBin"] = lines_per_bin
    sequence.DEF["BandwidthPerPixel"] = actual_BWpp
    sequence.DEF["Venc"] = Float64(venc)
    sequence.DEF["VelocityEncodingAxis"] = String(_encoding_axis)
    sequence.DEF["RfSpoilIncrement"] = Float64(rf_spoil_increment)
    if shot_pad > 0 || beat_pad > 0          # absent in the zero-pad regression so G0b can be byte-exact
        sequence.DEF["ShotPad"] = shot_pad
        sequence.DEF["BeatPad"] = beat_pad
        sequence.DEF["BaseTR"] = base_TR
    end
    return sequence, base_TR
end

function e062_scans(; fov=FOV, matrix=MATRIX, kwargs...)
    out = Dict{Symbol,Any}()
    base = 0.0
    for (name, axis) in zip(NAMES, AXES)
        seq, base = build_pc_e062(fov, matrix, sys; common..., kwargs..., _encoding_axis=axis)
        seq.DEF["Encoding"] = String(name)
        out[name] = seq
    end
    return out, base
end

function write_set(seqs, dir)
    mkpath(dir)
    Dict(String(name) => begin
        path = joinpath(dir, "$(lowercase(String(name))).seq")
        write_seq(seqs[name], path)
        sha_file(path)
    end for name in NAMES)
end

production = Dict(String(n) => sha_file(joinpath(PRODUCTION, "$(lowercase(String(n))).seq")) for n in NAMES)
target = joinpath(ROOT, "sequences_v1")
stage = target * ".building"
(isdir(target) || isdir(stage)) && error("Refusing existing output $target")
mkpath(stage)
report = Dict{String,Any}("experiment" => "E-064", "stage" => "S0_sequences", "production_sha256" => production,
                          "protocol_sha256" => sha_file(joinpath(ROOT, "code", "PROTOCOL.md")),
                          "script_sha256" => sha_file(@__FILE__), "julia" => string(VERSION))

# G0a: frozen builder reproduces the production 8-bin files.
frozen = build_pc_encoding_scans(FOV, MATRIX, sys; common..., cardiac_bins=8, RR=0.720, trigger_channel=nothing)
g0a = write_set(Dict(n => frozen[n] for n in NAMES), joinpath(stage, "regression_frozen_8bin"))
report["G0a_frozen_builder_byte_identical"] = g0a == production
@info "G0a" pass = report["G0a_frozen_builder_byte_identical"]

# G0b: the E-062 builder with 8 bins, 29 lines and no pads reproduces the same files.
copy8, base8 = e062_scans(cardiac_bins=8, lines_per_bin=29, shot_pad=0.0, beat_pad=0.0)
g0b = write_set(copy8, joinpath(stage, "regression_e062_builder_8bin"))
report["G0b_e062_builder_byte_identical"] = g0b == production
report["base_TR_s"] = base8
@info "G0b" pass = report["G0b_e062_builder_byte_identical"] base8

# G0c: the E-062 set.
_, base1 = e062_scans(fov=FOV1, matrix=MATRIX1, cardiac_bins=20, lines_per_bin=11, shot_pad=0.0, beat_pad=0.0)
report["base_TR_candycane_s"] = base1
shot_pad = round(0.0033 - base1; digits=8)
shot_pad > 0 || error("base TR exceeds 3.3 ms")
beat_pad = 0.00016
seqs, base = e062_scans(fov=FOV1, matrix=MATRIX1, cardiac_bins=20, lines_per_bin=11, shot_pad=shot_pad, beat_pad=beat_pad)
report["shot_pad_s"] = shot_pad; report["beat_pad_s"] = beat_pad
checks = Dict{String,Any}()
for name in NAMES
    seq = seqs[name]
    check_timing(seq, sys)
    check_hw_limits(seq, sys)
    labels = get_labels(seq)
    adc_blocks = [block for (block, event) in enumerate(seq) if is_ADC_on(event)]
    tuples = [(Int(l.LIN), Int(l.PAR), Int(l.PHS)) for l in labels[adc_blocks]]
    checks[String(name)] = Dict("profiles" => length(adc_blocks), "unique_labels" => length(unique(tuples)),
        "phs_range" => collect(extrema(t[3] for t in tuples)), "RR_s" => Float64(seq.DEF["RR"]), "TR_s" => Float64(seq.DEF["TR"]),
        "duration_s" => dur(seq))
end
sha20 = write_set(seqs, stage)
report["G0c"] = checks
report["sequences_sha256"] = sha20
g0c = all(c["profiles"] == 19200 && c["unique_labels"] == 19200 && c["phs_range"] == [0, 19] &&
          abs(c["RR_s"] - 0.72616) < 1e-12 && abs(c["TR_s"] - 0.0033) < 1e-12 for c in values(checks))
report["G0c_pass"] = g0c
report["status"] = (report["G0a_frozen_builder_byte_identical"] && report["G0b_e062_builder_byte_identical"] && g0c) ?
    "PASS_E064_S0_JULIA_GATES" : "FAIL_E064_S0_JULIA_GATES"
open(joinpath(stage, "report_julia.toml"), "w") do io
    TOML.print(io, report; sorted=true)
end
mv(stage, target)
@info "E064 S0" status = report["status"] checks
