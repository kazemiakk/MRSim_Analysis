# Load exactly one GPU backend. Metal is for Apple Silicon.
# For an NVIDIA GPU, replace `using Metal` with `using CUDA` and add CUDA.jl:
#   import Pkg; Pkg.add("CUDA")
# Loading multiple functional GPU backends makes KomaMRI fall back to the CPU.
using Metal
using FFTW
using JLD2
using KomaMRI
using KomaMRICore

import KomaMRICore.KA

const PHANTOM_FILE = joinpath(@__DIR__, "aorta_static.phantom") # Output from static_phantom.jl
const SEQUENCE_FILE = "PATH TO ref.seq"
const RESULT_FILE = joinpath(@__DIR__, "ref_static.jld2")

struct BlochButSpoilingBeforeRF <: KomaMRICore.SimulationMethod end

function KomaMRICore.prealloc(
    ::BlochButSpoilingBeforeRF,
    backend::KA.Backend,
    phantom::Phantom{T},
    magnetization::Mag{T},
    max_block_length::Integer,
    groupsize,
) where {T<:Real}
    KomaMRICore.prealloc(
        KomaMRICore.Bloch(),
        backend,
        phantom,
        magnetization,
        max_block_length,
        groupsize,
    )
end

function KomaMRICore.run_spin_excitation!(
    phantom::Phantom{T},
    sequence::DiscreteSequence{T},
    signal::AbstractArray{Complex{T}},
    magnetization::Mag{T},
    ::BlochButSpoilingBeforeRF,
    groupsize,
    backend::KA.Backend,
    prealloc::KomaMRICore.PreallocResult{T},
) where {T<:Real}
    fill!(magnetization.xy, zero(T))
    KomaMRICore.run_spin_excitation!(
        phantom,
        sequence,
        signal,
        magnetization,
        KomaMRICore.Bloch(),
        groupsize,
        backend,
        prealloc,
    )
end

function KomaMRICore.run_spin_precession!(
    phantom::Phantom{T},
    sequence::DiscreteSequence{T},
    signal::AbstractArray{Complex{T}},
    magnetization::Mag{T},
    ::BlochButSpoilingBeforeRF,
    groupsize,
    backend::KA.Backend,
    prealloc::KomaMRICore.PreallocResult{T},
) where {T<:Real}
    KomaMRICore.run_spin_precession!(
        phantom,
        sequence,
        signal,
        magnetization,
        KomaMRICore.Bloch(),
        groupsize,
        backend,
        prealloc,
    )
end

t0 = time()

@info "Loading phantom" file=PHANTOM_FILE
phantom = read_phantom(PHANTOM_FILE)

@info "Loading sequence" file=SEQUENCE_FILE
sequence = read_seq(SEQUENCE_FILE; verify_signature=true, verbose=false)

Nx = Int(sequence.DEF["Nx"])
Ny = Int(sequence.DEF["Ny"])
Nz = Int(sequence.DEF["Nz"])
nphases = Int(sequence.DEF["CardiacBins"])

labels = get_labels(sequence)
adc_blocks = [block for (block, event) in enumerate(sequence) if is_ADC_on(event)]
adc_labels = labels[adc_blocks]
label_tuples = [(label.LIN, label.PAR, label.PHS) for label in adc_labels]

@assert length(adc_blocks) == Ny * Nz * nphases
@assert length(unique(label_tuples)) == length(label_tuples)
@assert extrema(label.LIN for label in adc_labels) == (0, Ny - 1)
@assert extrema(label.PAR for label in adc_labels) == (0, Nz - 1)
@assert extrema(label.PHS for label in adc_labels) == (0, nphases - 1)

@info "Starting simulation" spins=length(phantom) profiles=length(adc_blocks) matrix=(Nx, Ny, Nz, nphases)
scanner = Scanner()
sim_params = KomaMRICore.default_sim_params()
sim_params["sim_method"] = BlochButSpoilingBeforeRF()
sim_params["Δt"] = Inf

raw = simulate(phantom, sequence, scanner; sim_params)

@info "Sorting raw data" elapsed_time=time() - t0
rawsort = zeros(ComplexF32, Nx, Ny, Nz, nphases)
filled = falses(Ny, Nz, nphases)

for profile in raw.profiles
    line = Int(profile.head.idx.kspace_encode_step_1)
    partition = Int(profile.head.idx.kspace_encode_step_2)
    phase = Int(profile.head.idx.phase)

    @assert 0 <= line < Ny
    @assert 0 <= partition < Nz
    @assert 0 <= phase < nphases
    @assert size(profile.data) == (Nx, 1)
    @assert !filled[line + 1, partition + 1, phase + 1]

    rawsort[:, line + 1, partition + 1, phase + 1] .= profile.data[:, 1]
    filled[line + 1, partition + 1, phase + 1] = true
end

@assert all(filled)

@info "Reconstructing" elapsed_time=time() - t0
result = fftshift(ifft(ifftshift(rawsort, (1, 2, 3)), (1, 2, 3)), (1, 2, 3))

@info "Saving result" file=RESULT_FILE elapsed_time=time() - t0
JLD2.jldsave(RESULT_FILE; rawsort, result)

@info "Finished" elapsed_time=time() - t0
