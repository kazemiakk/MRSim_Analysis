using KomaMRI, KomaMRICore, KomaMRIBase, HDF5, TOML, SHA

hash_e049(path)=bytes2hex(sha256(read(path)))

function acquired_shot(seq)
    adc=findfirst(a->a.N>0,seq.ADC)
    rf=findlast(i->any(!iszero,seq.RF[1,i].A),1:adc)
    isnothing(rf)&&error("No physical RF before acquisition")
    return copy(seq[rf:adc])
end

function controls(input,sequence_dir,output;mode="fine")
    mode in ("fine","production")||error("Unknown control discretization")
    ispath(output)&&error("Refuse overwrite")
    stage=output*".building";ispath(stage)&&error("Unfinished stage already exists")
    mkpath(stage)
    registration=Dict("experiment"=>"E-049","status"=>"REGISTERED_BEFORE_CPU_SIGNAL",
        "script_sha256"=>hash_e049(@__FILE__),"input_sha256"=>hash_e049(input),
        "manifest_sha256"=>hash_e049(joinpath(@__DIR__,"Manifest.toml")),
        "sequence_sha256"=>Dict(e=>hash_e049(joinpath(sequence_dir,lowercase(e)*".seq")) for e in ("REF","X","Y","Z")),
        "core_version"=>string(pkgversion(KomaMRICore)),"julia_version"=>string(VERSION),
        "CPU_only"=>true,"new_1M"=>false,"discretization_mode"=>mode)
    open(joinpath(stage,"registration.toml"),"w") do io;TOML.print(io,registration);end
    scanner=Scanner(limits=HardwareLimits(B0=3.0))
    # Explicit high-accuracy CPU control; no production simulation setting changed.
    params=Dict{String,Any}("gpu"=>false,"Nthreads"=>1,"precision"=>"f64",
        "return_type"=>"mat","Δt"=>1e-6,"Δt_rf"=>.5e-6,"max_block_length"=>512,"sim_method"=>Bloch())
    if mode=="production"
        params["precision"]="f32";params["Δt"]=Inf;params["Δt_rf"]=5e-5
    end
    h5open(input,"r") do f
        times=read(f,"time_s");expected_adc=read(f,"adc_time_s")
        cases=sort(filter(s->startswith(s,"case_"),keys(f)))
        h5open(joinpath(stage,"signals.h5.partial"),"w") do out
            for e in ("REF","X","Y","Z")
                seq=read_seq(joinpath(sequence_dir,lowercase(e)*".seq");verbose=false,verify_signature=true)
                shot=acquired_shot(seq)
                adc=get_adc_sampling_times(shot)
                length(adc)==400||error("Expected exactly one400sample shot")
                maximum(abs.(adc.-expected_adc))<1e-9||error("ADC times differ from independent parser")
                wave=HDF5.create_group(out,e*"_waveform")
                write(wave,"time_s",times)
                gx,gy,gz=KomaMRIBase.get_grads(shot,times)
                b1,df,psi=KomaMRIBase.get_rfs(shot,times)
                for (name,value) in (("gx_T_m",gx),("gy_T_m",gy),("gz_T_m",gz),
                    ("b1_T",b1),("rf_frequency_Hz",df),("rf_phase_rad",psi),("adc_time_s",adc),
                    ("adc_phase_compensation",KomaMRIBase.get_adc_phase_compensation(shot)))
                    write(wave,name,value)
                end
                for name in cases
                    group=f[name]
                    x=read(group,"x");y=read(group,"y");z=read(group,"z")
                    action=FlowPath(reshape(x.-x[1],1,:),reshape(y.-y[1],1,:),reshape(z.-z[1],1,:),falses(1,length(times)))
                    motion=Motion(action=action,time=TimeRange(t_start=times[1],t_end=times[end]),spins=AllSpins())
                    phantom=Phantom(name="E049_"*name,x=[x[1]],y=[y[1]],z=[z[1]],ρ=[1.],
                        T1=[read(group,"T1")],T2=[read(group,"T2")],motion=motion)
                    signal=vec(simulate(phantom,shot,scanner;sim_params=copy(params),verbose=false))
                    length(signal)==400&&all(isfinite,signal)||error("Invalid signal")
                    target=haskey(out,name) ? out[name] : HDF5.create_group(out,name)
                    write(target,e,signal)
                end
                println("COMPLETE_E049_SINGLE_SHOT ",e," cases=",length(cases));flush(stdout)
            end
        end
        mv(joinpath(stage,"signals.h5.partial"),joinpath(stage,"signals.h5"))
    end
    registration["status"]="COMPLETE_SINGLE_SPIN_CPU_BLOCH_CONTROLS"
    registration["signals_sha256"]=hash_e049(joinpath(stage,"signals.h5"))
    registration["max_gradient_step_s"]=params["Δt"]
    registration["max_RF_step_s"]=params["Δt_rf"]
    open(joinpath(stage,"report.toml"),"w") do io;TOML.print(io,registration);end
    mv(stage,output)
    println("COMPLETE_SINGLE_SPIN_CPU_BLOCH_CONTROLS")
end

length(ARGS) in (3,4)||error("INPUT_H5 SEQUENCE_DIR NEW_OUTPUT [fine|production] required")
controls(ARGS[1:3]...;mode=length(ARGS)==4 ? ARGS[4] : "fine")
