using KomaMRI
using KomaMRICore

import KomaMRICore.KA

"""Bloch simulation with ideal transverse spoiling immediately before each RF pulse."""
struct BlochWithPerfectSpoiling <: KomaMRICore.SimulationMethod end

function KomaMRICore.prealloc(
    ::BlochWithPerfectSpoiling,
    backend::KA.Backend,
    phantom::Phantom{T},
    magnetization::Mag{T},
    max_block_length::Integer,
    groupsize,
) where {T<:Real}
    return KomaMRICore.prealloc(
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
    ::BlochWithPerfectSpoiling,
    groupsize,
    backend::KA.Backend,
    prealloc::KomaMRICore.PreallocResult{T},
) where {T<:Real}
    fill!(magnetization.xy, zero(T))
    return KomaMRICore.run_spin_excitation!(
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
    ::BlochWithPerfectSpoiling,
    groupsize,
    backend::KA.Backend,
    prealloc::KomaMRICore.PreallocResult{T},
) where {T<:Real}
    return KomaMRICore.run_spin_precession!(
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
