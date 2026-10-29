using KomaMRI, KomaMRICore
import KomaMRICore.KA
struct MigrationPerfectSpoiling <: KomaMRICore.SimulationMethod end
const MIGRATION_PREVIOUS_RF=Ref(false)
if isdefined(KomaMRICore,:SpinResetState)
    function KomaMRICore.prealloc(::MigrationPerfectSpoiling,b::KA.Backend,p::Phantom{T},m::KomaMRICore.Mag{T},n::Integer,nadc::Integer,g,sys) where T<:Real
        KomaMRICore.prealloc(Bloch(),b,p,m,n,nadc,g,sys)
    end
    function KomaMRICore.run_spin_excitation!(p::Phantom{T},s::DiscreteSequence{T},sig::AbstractArray{Complex{T}},m::KomaMRICore.Mag{T},sys,::MigrationPerfectSpoiling,g,b::KA.Backend,pre::KomaMRICore.PreallocResult{T}) where T<:Real
        MIGRATION_PREVIOUS_RF[]||fill!(m.xy,zero(T));MIGRATION_PREVIOUS_RF[]=true
        KomaMRICore.run_spin_excitation!(p,s,sig,m,sys,Bloch(),g,b,pre)
    end
    function KomaMRICore.run_spin_precession!(p::Phantom{T},s::DiscreteSequence{T},sig::AbstractArray{Complex{T}},m::KomaMRICore.Mag{T},sys,::MigrationPerfectSpoiling,g,b::KA.Backend,pre::KomaMRICore.PreallocResult{T}) where T<:Real
        MIGRATION_PREVIOUS_RF[]=false
        KomaMRICore.run_spin_precession!(p,s,sig,m,sys,Bloch(),g,b,pre)
    end
else
    function KomaMRICore.prealloc(::MigrationPerfectSpoiling,b::KA.Backend,p::Phantom{T},m::KomaMRICore.Mag{T},n::Integer,g) where T<:Real
        KomaMRICore.prealloc(Bloch(),b,p,m,n,g)
    end
    function KomaMRICore.run_spin_excitation!(p::Phantom{T},s::DiscreteSequence{T},sig::AbstractArray{Complex{T}},m::KomaMRICore.Mag{T},::MigrationPerfectSpoiling,g,b::KA.Backend,pre::KomaMRICore.PreallocResult{T}) where T<:Real
        MIGRATION_PREVIOUS_RF[]||fill!(m.xy,zero(T));MIGRATION_PREVIOUS_RF[]=true
        KomaMRICore.run_spin_excitation!(p,s,sig,m,Bloch(),g,b,pre)
    end
    function KomaMRICore.run_spin_precession!(p::Phantom{T},s::DiscreteSequence{T},sig::AbstractArray{Complex{T}},m::KomaMRICore.Mag{T},::MigrationPerfectSpoiling,g,b::KA.Backend,pre::KomaMRICore.PreallocResult{T}) where T<:Real
        MIGRATION_PREVIOUS_RF[]=false
        KomaMRICore.run_spin_precession!(p,s,sig,m,Bloch(),g,b,pre)
    end
end
