using Test,LinearAlgebra
include(joinpath(@__DIR__,"inlet_replenishment_pilot.jl"))

@testset "Qualified HDF5 attributes avoid CUDA export collision" begin
    @test Base.isexported(CUDA,:attributes)
    @test HDF5.attributes !== CUDA.attributes
    @test_throws UndefVarError Core.eval(Main,:attributes)
    mktempdir() do temporary
        path=joinpath(temporary,"attribute_regression.h5")
        h5open(path,"w") do f
            write(f,"test",ones(Float32,2,3))
            HDF5.attributes(f)["FOV_m"]=Float32[.28,.128,.128]
            HDF5.attributes(f)["RR_s"]=.72616f0
            HDF5.attributes(f)["Venc_m_per_s"]=2.5f0
        end
        h5open(path,"r") do f
            @test read(f,"test")==ones(Float32,2,3)
            @test read(HDF5.attributes(f)["FOV_m"])==Float32[.28,.128,.128]
            @test read(HDF5.attributes(f)["RR_s"])==.72616f0
            @test read(HDF5.attributes(f)["Venc_m_per_s"])==2.5f0
        end
    end
end

struct ReferencePerfectSpoiling <: KomaMRICore.SimulationMethod end
function KomaMRICore.prealloc(::ReferencePerfectSpoiling,b::KomaMRICore.KA.Backend,p::Phantom{T},m::KomaMRICore.Mag{T},n::Integer,g) where T<:Real
    KomaMRICore.prealloc(Bloch(),b,p,m,n,g)
end
function KomaMRICore.run_spin_excitation!(p::Phantom{T},s::DiscreteSequence{T},sig::AbstractArray{Complex{T}},m::KomaMRICore.Mag{T},::ReferencePerfectSpoiling,g,b::KomaMRICore.KA.Backend,pre::KomaMRICore.PreallocResult{T}) where T<:Real
    fill!(m.xy,zero(T));KomaMRICore.run_spin_excitation!(p,s,sig,m,Bloch(),g,b,pre)
end
function KomaMRICore.run_spin_precession!(p::Phantom{T},s::DiscreteSequence{T},sig::AbstractArray{Complex{T}},m::KomaMRICore.Mag{T},::ReferencePerfectSpoiling,g,b::KomaMRICore.KA.Backend,pre::KomaMRICore.PreallocResult{T}) where T<:Real
    KomaMRICore.run_spin_precession!(p,s,sig,m,Bloch(),g,b,pre)
end

@testset "Reset-induced RF subparts do not add perfect spoiling" begin
    sequence=read_seq(joinpath(SEQDIR,"ref.seq");verbose=false,verify_signature=true)[1:300]
    phantom=Phantom(x=collect(range(-.01,.01,length=32)),y=zeros(32),z=zeros(32),ρ=ones(32),T1=fill(1.3,32),T2=fill(.1,32))
    params=Dict{String,Any}("gpu"=>false,"Nthreads"=>1,"precision"=>"f32","return_type"=>"mat","Δt"=>Inf,"max_block_length"=>512)
    RESET_KEYTIMES[]=Float64[];params["sim_method"]=ReferencePerfectSpoiling()
    baseline=simulate(phantom,sequence,Scanner();sim_params=params,verbose=false)
    RESET_KEYTIMES[]=collect(0:.00025:dur(sequence));params["sim_method"]=BlochWithPerfectSpoiling();PREVIOUS_PART_RF[]=false
    candidate=simulate(phantom,sequence,Scanner();sim_params=params,verbose=false)
    @test size(candidate)==size(baseline)
    @test all(isfinite,candidate)
    relative_error=norm(candidate-baseline)/norm(baseline)
    println("RF_PARTITION_RELATIVE_SIGNAL_ERROR=",relative_error)
    @test relative_error<1e-4
end
