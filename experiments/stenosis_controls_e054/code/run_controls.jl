using KomaMRI, KomaMRICore, KomaMRIBase, HDF5, TOML, SHA

digest(p)=bytes2hex(sha256(read(p)))
const MODELS=("static","positive","negative","accel_positive","accel_negative","shear_positive","shear_negative","accel_shear")

function firstshot(sequence)
    adc=findfirst(a->a.N>0,sequence.ADC)
    rf=findlast(i->any(!iszero,sequence.RF[1,i].A),1:adc)
    isnothing(rf)&&error("No physical RF")
    copy(sequence[rf:adc])
end

function delay(t)
    Sequence(reshape([Grad(0.,t) for _ in 1:3],3,1),reshape([RF(0.,0.)],1,1),[ADC(0,0.)],[t])
end

function main(repo,inputs,sequence_dir,output)
    ispath(output)&&error("Refuse overwrite")
    stage=output*".building";ispath(stage)&&error("Stage exists");mkpath(stage)
    manifest=joinpath(repo,"koma_migration_e046","Manifest.toml")
    saved=TOML.parsefile(joinpath(repo,"outputs/phantom_precision_e050/mapped_f64_v1/control_report.toml"))
    digest(manifest)==saved["manifest_sha256"]||error("Pinned runtime changed")
    reg=Dict("experiment"=>"E-054","status"=>"REGISTERED_BEFORE_CPU_SIGNAL", "script_sha256"=>digest(@__FILE__),
        "manifest_sha256"=>digest(manifest),"input_sha256"=>digest(joinpath(inputs,"controls.h5")),
        "protocol_sha256"=>digest(joinpath(@__DIR__,"PROTOCOL.md")),"cpu_only"=>true,"precision"=>"f64",
        "core_version"=>string(pkgversion(KomaMRICore)),"new_300k"=>false,"new_1m"=>false)
    open(joinpath(stage,"registration.toml"),"w") do io;TOML.print(io,reg);end
    scanner=Scanner(limits=HardwareLimits(B0=3.))
    h5open(joinpath(inputs,"controls.h5"),"r") do input
        times=read(input,"time_s");adc_expected=read(input,"adc_time_s");late=read(input,"late_clock_s")
        h5open(joinpath(stage,"signals.h5.partial"),"w") do out
            for encoding in ("REF","X","Y","Z")
                shot=firstshot(read_seq(joinpath(sequence_dir,lowercase(encoding)*".seq");verbose=false,verify_signature=true))
                adc=get_adc_sampling_times(shot)
                length(adc)==400&&maximum(abs.(adc.-adc_expected))<1e-9||error("ADC contract differs")
                wave=create_group(out,encoding*"_waveform")
                gx,gy,gz=KomaMRIBase.get_grads(shot,times)
                b1,df,psi=KomaMRIBase.get_rfs(shot,times)
                for (name,value) in (("gx_T_m",gx),("gy_T_m",gy),("gz_T_m",gz),("b1_T",b1),
                    ("rf_frequency_Hz",df),("rf_phase_rad",psi),("adc_phase_compensation",KomaMRIBase.get_adc_phase_compensation(shot)))
                    write(wave,name,value)
                end
                for name in MODELS,mode in ("fine","production","late_production")
                    group=input[name];x=read(group,"x");y=read(group,"y");z=read(group,"z")
                    size(x)==(32,length(times))||error("Unexpected path layout")
                    weights=read(group,"weights");abs(sum(weights)-1)<1e-12||error("Bad volume weights")
                    offset=mode=="late_production" ? late : 0.
                    action=FlowPath(x.-x[:,1],y.-y[:,1],z.-z[:,1],falses(size(x)))
                    motion=Motion(action=action,time=TimeRange(t_start=offset,t_end=offset+times[end]),spins=AllSpins())
                    p=Phantom(name="E054_"*name,x=x[:,1],y=y[:,1],z=z[:,1],ρ=weights,
                        T1=read(group,"T1"),T2=read(group,"T2"),motion=motion)
                    fine=mode=="fine"
                    params=Dict{String,Any}("gpu"=>false,"Nthreads"=>1,"precision"=>"f64","return_type"=>"mat",
                        "Δt"=>(fine ? 1e-6 : Inf),"Δt_rf"=>(fine ? .5e-6 : 5e-5),"max_block_length"=>512,"sim_method"=>Bloch())
                    sequence=offset==0 ? shot : delay(offset)+shot
                    t=time();signal=vec(simulate(p,sequence,scanner;sim_params=params,verbose=false))
                    length(signal)==400&&all(isfinite,signal)&&eltype(signal)==ComplexF64||error("Invalid signal")
                    target_name=name*"_"*mode
                    target=haskey(out,target_name) ? out[target_name] : create_group(out,target_name)
                    write(target,encoding,signal)
                    println("E054 ",encoding," ",target_name," seconds=",time()-t);flush(stdout)
                end
            end
        end
    end
    mv(joinpath(stage,"signals.h5.partial"),joinpath(stage,"signals.h5"))
    reg["status"]="COMPLETE_CPU_MULTI_SPIN_SIGNALS";reg["signals_sha256"]=digest(joinpath(stage,"signals.h5"))
    open(joinpath(stage,"report.toml"),"w") do io;TOML.print(io,reg);end
    mv(stage,output)
end

length(ARGS)==4||error("REPO INPUTS SEQUENCE_DIR NEW_OUTPUT required")
main(ARGS...)
