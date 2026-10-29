using KomaMRI, KomaMRICore, KomaMRIBase, HDF5, TOML, SHA

function firstshot_clock(seq)
    a=findfirst(x->x.N>0,seq.ADC)
    r=findlast(i->any(!iszero,seq.RF[1,i].A),1:a)
    copy(seq[r:a])
end

function delay_clock(t)
    Sequence(reshape([Grad(0.,t) for _ in 1:3],3,1),reshape([RF(0.,0.)],1,1),[ADC(0,0.)],[t])
end

function main_clock(root,output;gpu=false)
    ispath(output)&&error("Refuse overwrite")
    stage=output*".building";ispath(stage)&&error("Stage exists");mkpath(stage)
    input=joinpath(root,"outputs/remapped_confirmation_e047/phantom_inputs_v1/case_inputs.h5")
    info=TOML.parsefile(joinpath(dirname(input),"case_info.toml"));period=info["period_s"]
    seqs=Dict(e=>firstshot_clock(read_seq(joinpath(info["sequence_directory"],lowercase(e)*".seq");verbose=false)) for e in ("REF","X"))
    echo=get_adc_sampling_times(seqs["REF"])[201]
    cases=Dict{String,Any}[]
    for (name,acc) in (("constant1p5",0.),("accelerating200",200.))
        t=collect(range(0.,period;length=1441));tau=t.-period/2
        x=1.5tau.+.5acc.*tau.^2
        push!(cases,Dict("name"=>name,"positions"=>hcat(x,zero.(x),zero.(x)),"phase_s"=>period/2,"T1"=>1.3,"T2"=>.1))
    end
    h5open(input,"r") do f
        for (spin,name) in ((11455,"throat11455"),(31556,"throat31556"),(53157,"upstream53157"))
            p=hcat([Float64(read(f,string(c))[spin+1]).+Float64.(f[string(d)][spin+1,:]) for (c,d) in zip((:x,:y,:z),(:dx,:dy,:dz))]...)
            push!(cases,Dict("name"=>name,"positions"=>p,"phase_s"=>3.5period/15,
                            "T1"=>Float64(read(f,"T1")[spin+1]),"T2"=>Float64(read(f,"T2")[spin+1])))
        end
    end
    hashclock(p)=bytes2hex(sha256(read(p)))
    if gpu
        get(ENV,"CUDA_VISIBLE_DEVICES","")=="GPU-8be27592-e325-7b7a-f432-b84f69cb18c6"||error("Reviewed idle GPU1 required")
        filter!(c->c["name"] in ("constant1p5","throat11455"),cases)
    end
    # 2047 is within the actual 2048-cycle acquisition; earlier 2048 stress
    # controls remain preserved separately, not substituted for real end timing.
    cycle_values=gpu ? (0,2047) : (0,1024,2047)
    registration=Dict("experiment"=>"E-049","status"=>"REGISTERED_CLOCK_ORIGIN_CONTROLS",
                      "script_sha256"=>hashclock(@__FILE__),"source_sha256"=>hashclock(input),
                      "cycles"=>collect(cycle_values),"period_s"=>period,"CPU_only"=>!gpu,
                      "gpu"=>gpu,"physical_gpu_uuid"=>get(ENV,"CUDA_VISIBLE_DEVICES","NONE"),
                      "limitation"=>"Single equilibrium-RF shot after zero-field delay, not repeated RF steady state or full acquisition.")
    open(joinpath(stage,"registration.toml"),"w") do io;TOML.print(io,registration);end
    rows=Dict{String,Any}[]
    h5open(joinpath(stage,"clock_signals.h5.partial"),"w") do out
        for cycles in cycle_values,mode in ("fine64","fine32","production64","production32"),case in cases
            begin_time=cycles*period+case["phase_s"]-echo
            pos=case["positions"]
            action=FlowPath(reshape(pos[:,1].-pos[1,1],1,:),reshape(pos[:,2].-pos[1,2],1,:),reshape(pos[:,3].-pos[1,3],1,:),falses(1,1441))
            motion=Motion(action=action,time=Periodic(period=period,asymmetry=1.),spins=AllSpins())
            p=Phantom(x=[pos[1,1]],y=[pos[1,2]],z=[pos[1,3]],ρ=[1.],T1=[case["T1"]],T2=[case["T2"]],motion=motion)
            params=Dict{String,Any}("gpu"=>gpu,"gpu_device"=>0,"Nthreads"=>1,"return_type"=>"mat","max_block_length"=>512,"sim_method"=>Bloch(),
                  "precision"=>endswith(mode,"64") ? "f64" : "f32","Δt"=>startswith(mode,"fine") ? 1e-6 : Inf,
                  "Δt_rf"=>startswith(mode,"fine") ? .5e-6 : 5e-5)
            signals=Dict{String,Any}()
            for e in ("REF","X")
                q=delay_clock(begin_time)+seqs[e]
                signals[e]=vec(simulate(p,q,Scanner(limits=HardwareLimits(B0=3.));sim_params=copy(params),verbose=false))
            end
            all(isfinite,signals["REF"])&&all(isfinite,signals["X"])||error("Nonfinite clock signal")
            velocity=angle(signals["X"][201]*conj(signals["REF"][201]))*5/pi
            name=case["name"]*"_"*mode*"_"*string(cycles)
            g=HDF5.create_group(out,name);write(g,"REF",signals["REF"]);write(g,"X",signals["X"])
            push!(rows,Dict("case"=>case["name"],"mode"=>mode,"cycles"=>cycles,"absolute_echo_s"=>cycles*period+case["phase_s"],"velocity_m_s"=>Float64(velocity)))
            println("CLOCK_CONTROL ",name," velocity=",velocity);flush(stdout)
        end
    end
    mv(joinpath(stage,"clock_signals.h5.partial"),joinpath(stage,"clock_signals.h5"))
    report=merge(registration,Dict("status"=>"COMPLETE_CLOCK_ORIGIN_CONTROLS","rows"=>rows,
                 "signals_sha256"=>hashclock(joinpath(stage,"clock_signals.h5"))))
    open(joinpath(stage,"report.toml"),"w") do io;TOML.print(io,report);end
    mv(stage,output)
end

length(ARGS) in (2,3)||error("ROOT NEW_OUTPUT [GPU] required")
main_clock(ARGS[1:2]...;gpu=length(ARGS)==3&&ARGS[3]=="GPU")
