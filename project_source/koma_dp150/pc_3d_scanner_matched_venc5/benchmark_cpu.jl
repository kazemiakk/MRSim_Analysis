using LinearAlgebra
using Statistics
using TOML
using KomaMRI
using KomaMRICore

include(joinpath(dirname(@__DIR__), "perfect_spoiling.jl"))
include(joinpath(dirname(@__DIR__), "motion_timing.jl"))

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const PHANTOM_FILE = joinpath(
    REPO_ROOT,
    "outputs",
    "dp150_mri_pipeline",
    "koma",
    "smoke20k",
    "dp150_motion_1m.phantom",
)
const SEQUENCE_FILE = joinpath(@__DIR__, "ref.seq")
const OUTPUT_DIR = joinpath(
    REPO_ROOT,
    "outputs",
    "dp150_mri_pipeline",
    "koma",
    "simulation_motion_1m_scanner_matched_venc5",
)
const CYCLE_COUNTS = parse.(Int, split(get(
    ENV,
    "DP150_CPU_BENCH_CYCLES",
    "1,2,4",
), ","))
const PARTICLE_COUNTS = parse.(Int, split(get(
    ENV,
    "DP150_CPU_BENCH_PARTICLES",
    "250,500",
), ","))
const REPEATS = parse(Int, get(ENV, "DP150_CPU_BENCH_REPEATS", "2"))
const PRODUCTION_BATCH_SIZE = 20_000
const PRODUCTION_BATCH_COUNT = 4 * 50
const PROGRESS_FILE = joinpath(
    OUTPUT_DIR,
    "cpu_benchmark_$(Threads.nthreads())threads.progress.log",
)

all(CYCLE_COUNTS .> 0) || error("Benchmark cycle counts must be positive.")
all(PARTICLE_COUNTS .> 0) || error("Benchmark particle counts must be positive.")
REPEATS >= 2 || error("At least two benchmark repeats are required.")
mkpath(OUTPUT_DIR)

function log_progress(message)
    open(PROGRESS_FILE, "a") do io
        println(io, "$(time())\t$message")
        flush(io)
    end
end

open(PROGRESS_FILE, "w") do io
    println(io, "unix_time\tevent")
end

function rf_block_before(sequence, block)
    candidates = [
        index for index in 1:block
        if any(rf -> dur(rf) > 0, view(sequence.RF, :, index))
    ]
    isempty(candidates) && error("No RF block precedes ADC block $block.")
    return last(candidates)
end

function complete_cycles(sequence, cardiac_bins, cycle_count)
    adc_blocks = [
        block for (block, event) in enumerate(sequence)
        if is_ADC_on(event)
    ]
    required_profiles = cycle_count * cardiac_bins
    length(adc_blocks) > required_profiles || error(
        "Sequence does not contain a complete trailing boundary for $cycle_count cycles.")
    first_block = rf_block_before(sequence, adc_blocks[1])
    next_cycle_block = rf_block_before(
        sequence,
        adc_blocks[required_profiles + 1],
    )
    return sequence[first_block:(next_cycle_block - 1)]
end

@info "Loading CPU benchmark inputs" phantom=PHANTOM_FILE sequence=SEQUENCE_FILE
source_phantom = read_phantom(PHANTOM_FILE)
sequence = read_seq(SEQUENCE_FILE; verify_signature=true, verbose=false)
@assert length(source_phantom) == PRODUCTION_BATCH_SIZE
maximum(PARTICLE_COUNTS) <= length(source_phantom) || error(
    "Benchmark requests more particles than the smoke phantom contains.")
cardiac_bins = Int(sequence.DEF["CardiacBins"])
@assert cardiac_bins == 15

if source_phantom.motion isa Motion
    source_phantom.motion, _, _ = rescale_periodic_motion(
        source_phantom.motion,
        Float64(sequence.DEF["RR"]),
    )
end

sim_params = KomaMRICore.default_sim_params()
sim_params["gpu"] = false
sim_params["Nthreads"] = Threads.nthreads()
sim_params["precision"] = "f32"
sim_params["sim_method"] = BlochWithPerfectSpoiling()
sim_params["return_type"] = "mat"
sim_params[string(Char(0x0394), 't')] = Inf
sim_params["max_block_length"] = 512
scanner = Scanner(B0=3.0)

function timed_simulation(phantom, cycle_count)
    benchmark_sequence = complete_cycles(sequence, cardiac_bins, cycle_count)
    GC.gc()
    start_ns = time_ns()
    signal = simulate(
        phantom,
        benchmark_sequence,
        scanner;
        sim_params=copy(sim_params),
        verbose=false,
    )
    elapsed = (time_ns() - start_ns) / 1e9
    @assert size(signal, 1) == Int(sequence.DEF["Nx"]) * cardiac_bins * cycle_count
    @assert all(isfinite, signal)
    signal = nothing
    GC.gc()
    return elapsed
end

@info "Warming CPU simulation" threads=Threads.nthreads()
log_progress("warmup_start")
warmup_phantom = source_phantom[1:min(32, minimum(PARTICLE_COUNTS))]
warmup_seconds = timed_simulation(warmup_phantom, 1)
log_progress("warmup_complete seconds=$(warmup_seconds)")
warmup_phantom = nothing
GC.gc()

samples = Dict{String,Vector{Float64}}()
median_seconds = Dict{String,Float64}()
observation_particles = Int[]
observation_cycles = Int[]
observation_seconds = Float64[]
for particle_count in PARTICLE_COUNTS
    phantom = source_phantom[1:particle_count]
    for cycle_count in CYCLE_COUNTS
        timings = Float64[]
        for repeat in 1:REPEATS
            elapsed = timed_simulation(phantom, cycle_count)
            push!(timings, elapsed)
            push!(observation_particles, particle_count)
            push!(observation_cycles, cycle_count)
            push!(observation_seconds, elapsed)
            log_progress(
                "sample particles=$(particle_count) cycles=$(cycle_count) repeat=$(repeat) seconds=$(elapsed)")
            @info "CPU benchmark sample" particle_count cycle_count repeat elapsed_seconds=elapsed
        end
        key = "particles_$(particle_count)_cycles_$(cycle_count)"
        samples[key] = timings
        median_seconds[key] = median(timings)
    end
    phantom = nothing
    GC.gc()
end
source_phantom = nothing
GC.gc()

design = hcat(
    ones(length(observation_seconds)),
    Float64.(observation_cycles),
    Float64.(observation_particles .* observation_cycles),
)
coefficients = design \ observation_seconds
intercept_seconds = coefficients[1]
seconds_per_cycle_overhead = coefficients[2]
seconds_per_particle_cycle = coefficients[3]
seconds_per_particle_cycle > 0 || error(
    "Measured per-particle CPU slope is not positive.")
fitted = design * coefficients
residual_sum_squares = sum(abs2, observation_seconds .- fitted)
total_sum_squares = sum(abs2, observation_seconds .- mean(observation_seconds))
r_squared = total_sum_squares > 0 ? 1 - residual_sum_squares / total_sum_squares : 1.0

cycles_per_batch = Int(sequence.DEF["Ny"]) * Int(sequence.DEF["Nz"])
production_predictor = Float64[
    1,
    cycles_per_batch,
    PRODUCTION_BATCH_SIZE * cycles_per_batch,
]
estimated_seconds_per_batch = dot(production_predictor, coefficients)
estimated_seconds_per_batch > 0 || error("CPU batch-time estimate is not positive.")
estimated_total_seconds = estimated_seconds_per_batch * PRODUCTION_BATCH_COUNT
degrees_of_freedom = length(observation_seconds) - size(design, 2)
residual_variance = residual_sum_squares / degrees_of_freedom
coefficient_covariance = residual_variance * inv(design' * design)
prediction_standard_error = sqrt(dot(
    production_predictor,
    coefficient_covariance * production_predictor,
))
fit_lower_seconds = max(
    estimated_seconds_per_batch - 1.96 * prediction_standard_error,
    0.0,
) * PRODUCTION_BATCH_COUNT
fit_upper_seconds = (
    estimated_seconds_per_batch + 1.96 * prediction_standard_error
) * PRODUCTION_BATCH_COUNT

report = Dict(
    "status" => "PASS",
    "scope" => "CPU timing estimate; not a completed full CPU simulation",
    "cpu_threads" => Threads.nthreads(),
    "cpu_model" => Sys.cpu_info()[1].model,
    "benchmark_phantom" => PHANTOM_FILE,
    "benchmark_particle_counts" => PARTICLE_COUNTS,
    "production_particles" => 1_000_000,
    "production_batch_size" => PRODUCTION_BATCH_SIZE,
    "production_batch_count_all_encodings" => PRODUCTION_BATCH_COUNT,
    "cardiac_bins" => cardiac_bins,
    "cycles_per_production_batch" => cycles_per_batch,
    "profiles_per_production_batch" => cycles_per_batch * cardiac_bins,
    "cycle_counts" => CYCLE_COUNTS,
    "repeats_per_cycle_count" => REPEATS,
    "warmup_seconds" => warmup_seconds,
    "timing_samples_seconds" => samples,
    "median_seconds" => median_seconds,
    "linear_fit_intercept_seconds" => intercept_seconds,
    "linear_fit_seconds_per_cycle_overhead" => seconds_per_cycle_overhead,
    "linear_fit_seconds_per_particle_cycle" => seconds_per_particle_cycle,
    "linear_fit_r_squared" => r_squared,
    "estimated_seconds_per_20000_particle_encoding_batch" => estimated_seconds_per_batch,
    "estimated_total_seconds_four_encodings" => estimated_total_seconds,
    "estimated_total_hours_four_encodings" => estimated_total_seconds / 3600,
    "fit_noise_95_percent_hours_lower" => fit_lower_seconds / 3600,
    "fit_noise_95_percent_hours_upper" => fit_upper_seconds / 3600,
    "excludes" => [
        "full-phantom loading",
        "batch file serialization",
        "final FFT reconstruction",
    ],
    "interpretation" => "Measured under the workstation load present at benchmark time. The fit separates fixed, per-cycle, and per-particle-cycle costs. The reported 95% interval covers fit noise only, not structural uncertainty from extrapolating to 20,000 particles and 2,048 cycles.",
)

output = joinpath(
    OUTPUT_DIR,
    "cpu_timing_estimate_$(Threads.nthreads())threads.toml",
)
temporary = output * ".partial"
open(temporary, "w") do io
    TOML.print(io, report; sorted=true)
end
mv(temporary, output; force=true)
@info "CPU timing estimate complete" output estimated_hours=estimated_total_seconds / 3600 r_squared
