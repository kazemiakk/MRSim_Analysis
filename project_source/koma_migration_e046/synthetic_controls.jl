using KomaMRI, KomaMRICore, CUDA, HDF5, TOML, SHA, Statistics, Test
include(joinpath(@__DIR__,"perfect_spoiling.jl"))
function block(t;rf=RF(0.0,0.0),gz=0.0,adc=ADC(0,0.0))
    gr=reshape([Grad(0.0,t),Grad(0.0,t),Grad(gz,t)],3,1)
    Sequence(gr,reshape([rf],1,1),[adc],[Float64(t)])
end
function probe_sequence(thickness;cycles=6)
    tr=.02;period=.72;flip=deg2rad(30);rf_dur=.001
    gz=isinf(thickness) ? 0.0 : 1000/(γ*thickness)
    b1=flip/(2*pi*γ*rf_dur)
    shot=block(rf_dur;rf=RF(b1,rf_dur),gz)+block(.001;gz=-gz/2)+block(.001)+block(.00001;adc=ADC(2,.00001))+block(tr-.00301)
    seq=Sequence()
    for c in 1:cycles
        seq+=block(.001)
        for i in 1:36
            seq+=shot
        end
        # First guard is compensated at cycle end, not added to the period.
        seq.DUR[end]-=.001;seq.GR[:,end].=[Grad(0.0,seq.DUR[end]) for _ in 1:3]
    end
    return seq
end
function synthetic_phantom(arm)
    n=1000;period=.72f0;times=Float32.(collect(0:.002:.72))
    z0=Float32.((collect(0:n-1) .+ 0.5)/n*0.1 .- 0.05)
    z=zeros(Float32,n,length(times));reset=falses(size(z))
    for j in eachindex(times)
        raw=z0.+Float32(.04/.72)*times[j]
        z[:,j].=mod.(raw .+ 0.05f0,0.1f0) .- 0.05f0
        j>1&&(reset[:,j].=z[:,j].<z[:,j-1])
    end
    real_endpoint=copy(reset[:,end]);mapping=mod1.(collect(1:n).-400,n)
    if arm=="map"
        action=FlowPath(zeros(Float32,size(z)),zeros(Float32,size(z)),z,reset;cycle_map=mapping)
    else
        reset[:,end].=true
        action=FlowPath(zeros(Float32,size(z)),zeros(Float32,size(z)),z,reset)
    end
    p=Phantom(x=zeros(Float32,n),y=zeros(Float32,n),z=zeros(Float32,n),ρ=ones(Float32,n),
        T1=fill(1.3f0,n),T2=fill(.1f0,n),motion=Motion(action=action,time=Periodic(period=period,asymmetry=1f0),spins=AllSpins()))
    @assert maximum(abs.(z[:,end][mapping].-z0))<1e-7
    return p
end
function main(output)
    ispath(output)&&error("Refuse overwrite");mkpath(output)
    params=Dict{String,Any}("gpu"=>true,"gpu_device"=>0,"Nthreads"=>1,"precision"=>"f32",
        "return_type"=>"mat","Δt"=>Inf,"max_block_length"=>512,"sim_method"=>MigrationPerfectSpoiling())
    scanner=Scanner();results=Dict{String,Any}()
    # Static analytic spoiled-GRE control: ignore startup and compare final ADC.
    seq=probe_sequence(Inf);p=Phantom(x=zeros(Float32,1),T1=Float32[1.3],T2=Float32[.1])
    MIGRATION_PREVIOUS_RF[]=false;s=vec(simulate(p,seq,scanner;sim_params=params,verbose=false))
    observed=mean(abs.(s[end-1:end]));a=deg2rad(30);e1=exp(-.02/1.3)
    expected=sin(a)*(1-e1)/(1-e1*cos(a))*exp(-.0025/.1)
    relative=abs(observed/expected-1)
    @assert relative<.02 "Static GRE analytic mismatch"
    results["static_GRE"]=Dict("observed"=>observed,"expected"=>expected,"relative_error"=>relative)
    h5open(joinpath(output,"synthetic_signals.h5"),"w") do f
        write(f,"static_signal",s)
        for (label,thickness) in [("thin_2D_slice",.005),("broad_3D_slab",.128)]
            seq=probe_sequence(thickness)
            for arm in ("nomap","map")
                p=synthetic_phantom(arm);MIGRATION_PREVIOUS_RF[]=false
                println("START_SYNTHETIC ",label," ",arm);flush(stdout)
                start=time();signal=vec(simulate(p,seq,scanner;sim_params=params,verbose=false))
                all(isfinite,signal)||error("Nonfinite synthetic signal")
                write(f,label*"_"*arm,signal)
                shot=dropdims(mean(reshape(abs.(signal),2,36,6);dims=1);dims=1)
                edge=shot[1,3:end]./shot[end,2:end-1]
                results[label*"_"*arm]=Dict("elapsed_s"=>time()-start,
                    "late_cycle_first_to_previous_last_mean_ratio"=>mean(edge),
                    "late_cycle_first_to_previous_last_ratios"=>edge)
            end
        end
    end
    results["status"]="PASS_SYNTHETIC_STATIC_AND_FLOW_CONTROLS"
    results["script_sha256"]=bytes2hex(sha256(read(@__FILE__)))
    open(joinpath(output,"synthetic_report.toml"),"w") do io;TOML.print(io,results);end
    println("PASS_SYNTHETIC_STATIC_AND_FLOW_CONTROLS")
end
length(ARGS)==1||error("Usage: synthetic_controls.jl NEW_OUTPUT")
main(ARGS[1])
