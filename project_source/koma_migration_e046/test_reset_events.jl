using KomaMRI, KomaMRICore, CUDA, Test, TOML
length(ARGS)==1||error("Usage: test_reset_events.jl NEW_OUTPUT")
output=ARGS[1];ispath(output)&&error("Refuse overwrite");mkpath(output)
@testset "Reset inside a part survives false final flag" begin
    action=FlowPath(zeros(Float32,1,5),zeros(Float32,1,5),zeros(Float32,1,5),reshape(Bool[false,false,true,false,false],1,:))
    motion=Motion(action=action,time=Periodic(period=.72f0,asymmetry=1f0),spins=AllSpins())
    # Probe strict window interiors, not Constant-Next's ambiguous knot at zero.
    time=Float32[.01,.3,.7]
    for gpu in (false,true)
        backend=gpu ? KomaMRICore.KA.get_backend(CUDA.zeros(Float32,1)) : KomaMRICore.KA.CPU()
        state=KomaMRICore.prealloc_spin_reset(motion,backend,1,3)
        m=gpu ? KomaMRICore.gpu(motion) : motion
        t=gpu ? CUDA.cu(time) : time
        KomaMRICore.prepare_spin_reset!(state,m,reshape(t,1,:),backend)
        @test Array(state.first_reset)==UInt32[2]
        KomaMRICore.advance_spin_reset!(state,UInt32(3))
        @test Array(state.has_reset)==Bool[true]
        xy=ComplexF32[.3+.2im];z=Float32[.4];rho=Float32[1]
        M=KomaMRICore.Mag(gpu ? CUDA.cu(xy) : xy,gpu ? CUDA.cu(z) : z)
        KomaMRICore.outflow_spin_reset!(M,state;replace_by=gpu ? CUDA.cu(rho) : rho)
        @test Array(M.xy)==ComplexF32[0]
        @test Array(M.z)==Float32[1]
    end
end
open(joinpath(output,"reset_events.toml"),"w") do io
    TOML.print(io,Dict("status"=>"PASS_CPU_GPU_INTERIOR_RESET_EVENTS","checks"=>8,
        "scope"=>"Actual earliest-event accumulation and replacement, with final sampled flag false"))
end
