using KomaMRI, KomaMRICore, KomaMRIBase, CUDA, HDF5, JLD2, FFTW, Test, TOML, SHA
length(ARGS)==1 || error("Usage: test_environment.jl NEW_OUTPUT")
output=abspath(ARGS[1]);ispath(output)&&error("Refuse overwrite");mkpath(output)
checks=String[]
@testset "Upstream cycle map and serialization" begin
    @test :cycle_map in fieldnames(FlowPath)
    @test pkgversion(KomaMRIBase)==v"0.15.2"
    @test pkgversion(KomaMRICore)==v"0.14.0"
    sources=[3,1,4,2]
    flow=FlowPath(zeros(Float32,4,2),zeros(Float32,4,2),zeros(Float32,4,2),falses(4,2);cycle_map=sources)
    phantom=Phantom(x=zeros(Float32,4),motion=Motion(action=flow,time=Periodic(period=.72f0,asymmetry=1f0),spins=AllSpins()))
    path=joinpath(output,"map_roundtrip.phantom")
    write_phantom(phantom,path)
    @test read_phantom(path).motion.action.cycle_map==sources
    xy=ComplexF32[1+2im,3+4im,5+6im,7+8im];z=Float32[.1,.2,.3,.4]
    state=KomaMRICore.Mag(copy(xy),copy(z))
    KomaMRICore.remap_magnetization!(state,sources)
    @test state.xy==xy[sources]
    @test state.z==z[sources]
    push!(checks,"CPU destination-to-source gather and map serialization")
    @test CUDA.functional()
    gpu_state=KomaMRICore.Mag(CUDA.cu(xy),CUDA.cu(z))
    KomaMRICore.remap_magnetization!(gpu_state,CUDA.cu(sources));CUDA.synchronize()
    @test Array(gpu_state.xy)==xy[sources]
    @test Array(gpu_state.z)==z[sources]
    push!(checks,"GPU destination-to-source gather")
    @test_throws ArgumentError FlowPath(zeros(Float32,4,2),zeros(Float32,4,2),zeros(Float32,4,2),falses(4,2);cycle_map=[0,1,2,3])
end
report=Dict("status"=>"PASS_UPSTREAM_API_CPU_GPU_PREFLIGHT","checks"=>checks,
    "script_sha256"=>bytes2hex(sha256(read(@__FILE__))),"cuda_runtime"=>string(CUDA.runtime_version()),
    "physical_gpu_uuid"=>get(ENV,"CUDA_VISIBLE_DEVICES","UNSET"),"visible_gpu"=>0)
open(joinpath(output,"environment_preflight.toml"),"w") do io;TOML.print(io,report);end
println("PASS_UPSTREAM_API_CPU_GPU_PREFLIGHT")
