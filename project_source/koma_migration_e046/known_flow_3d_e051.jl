using KomaMRI, KomaMRICore, KomaMRIBase, CUDA, HDF5, TOML, SHA
include(joinpath(@__DIR__,"export_controls.jl"))
include(joinpath(@__DIR__,"direct_sort.jl"))
include(joinpath(@__DIR__,"perfect_spoiling.jl"))
include(joinpath(@__DIR__,"sorted_event_lookup.jl"))

function e051_report(path,values)
    ispath(path)&&error("Refuse overwrite: "*path)
    open(path*".partial","w") do io;TOML.print(io,values);end
    mv(path*".partial",path)
end

function e051_phantom(input,model,duration)
    h5open(input,"r") do f
        g=f[model];p=read(g,"positions");v=vec(read(g,"velocity"));rho=vec(read(g,"rho"))
        motion=Motion(action=Translate((v.*duration)...),time=TimeRange(0.,duration),spins=AllSpins())
        return Phantom(name="E051_"*model,x=vec(p[1,:]),y=vec(p[2,:]),z=vec(p[3,:]),ρ=rho,
            T1=fill(1.3,length(rho)),T2=fill(.1,length(rho)),motion=motion)
    end
end

function e051_first_shot(seq)
    labels=get_labels(seq)
    a=findfirst(i->seq.ADC[i].N>0&&Int(labels[i].LIN)==32&&Int(labels[i].PAR)==16,eachindex(seq.ADC))
    r=findlast(i->any(a->abs(a)>1e-15,seq.RF[1,i].A),1:a)
    return copy(seq[r:a]),r,a
end

function e051_delay(t)
    Sequence(reshape([Grad(0.,t) for _ in 1:3],3,1),reshape([RF(0.,0.)],1,1),[ADC(0,0.)],[t])
end

function e051(mode,inputs,output)
    mode in ("preflight","run")||error("Unknown mode")
    ispath(output)&&error("Refuse existing output")
    gpu_uuid="GPU-8be27592-e325-7b7a-f432-b84f69cb18c6"
    get(ENV,"CUDA_VISIBLE_DEVICES","")==gpu_uuid||error("Reviewed GPU1 required")
    info=TOML.parsefile(joinpath(E046ROOT,"outputs/remapped_confirmation_e047/phantom_inputs_v1/case_info.toml"))
    frozen=TOML.parsefile(joinpath(E046ROOT,"outputs/phantom_precision_e050/mapped_f64_v1/control_report.toml"))
    frozen["status"]=="COMPLETE_E050_F64_100K_MAPPED_ACQUISITIONS"||error("E050 incomplete")
    hashfile(joinpath(@__DIR__,"Manifest.toml"))==frozen["manifest_sha256"]||error("Runtime changed")
    for (name,digest) in frozen["helper_sha256"]
        hashfile(joinpath(@__DIR__,name))==digest||error("Helper changed")
    end
    input=joinpath(inputs,"controls.h5")
    duration=h5open(f->read(f,"duration_s"),input,"r")
    reg=Dict("experiment"=>"E-051","status"=>"REGISTERED_BEFORE_SIGNAL","mode"=>mode,
        "script_sha256"=>hashfile(@__FILE__),"input_sha256"=>hashfile(input),
        "contract_sha256"=>hashfile(joinpath(inputs,"contract.json")),
        "protocol_sha256"=>hashfile(joinpath(E046ROOT,"outputs/known_flow_3d_e051/PROTOCOL.md")),
        "manifest_sha256"=>frozen["manifest_sha256"],"helper_sha256"=>frozen["helper_sha256"],
        "precision"=>"f64","physical_gpu_uuid"=>gpu_uuid,"visible_GPU"=>0,"new_1m"=>false,
        "sequence_sha256"=>info["sequence_sha256"])
    mkpath(output);e051_report(joinpath(output,"registration.toml"),reg)
    seqs=case_sequences("phantom",info)
    for e in ("REF","X","Y","Z")
        hashfile(joinpath(info["sequence_directory"],lowercase(e)*".seq"))==info["sequence_sha256"][e]||error("Sequence changed")
        starts=get_block_start_times(seqs[e]);labels=get_labels(seqs[e]);blocks=findall(a->a.N>0,seqs[e].ADC)
        h5open(joinpath(E046ROOT,"outputs/remapped_confirmation_e047/phantom_inputs_v1/legacy_timing.h5"),"r") do f
            g=f[e]
            maximum(abs.(Float64[starts[i]+seqs[e].ADC[i].delay for i in blocks].-read(g,"adc_first_s")))<1e-9||error("ADC changed")
            hcat([[Int(labels[i].LIN),Int(labels[i].PAR),Int(labels[i].PHS)] for i in blocks]...)==read(g,"labels")||error("Labels changed")
        end
    end
    params=Dict{String,Any}("gpu"=>true,"gpu_device"=>0,"Nthreads"=>1,"precision"=>"f64","return_type"=>"mat",
        "Δt"=>Inf,"Δt_rf"=>5e-5,"max_block_length"=>512,"sim_method"=>MigrationPerfectSpoiling())
    scanner=Scanner(limits=HardwareLimits(B0=3.));CUDA.device!(0)
    if mode=="preflight"
        p=e051_phantom(input,"plug_positive",duration)
        adc=findall(a->a.N>0,seqs["REF"].ADC);rows=Dict{String,Any}[]
        h5open(joinpath(output,"prefix_signals.h5.partial"),"w") do out
            for e in ("REF","X")
                q=copy(seqs[e][1:adc[15]])
                MIGRATION_PREVIOUS_RF[]=false;t=time();s=simulate(p,q,scanner;sim_params=copy(params),verbose=false)
                all(isfinite,s)&&length(s)==6000&&eltype(s)==ComplexF64||error("Invalid prefix")
                write(out,e*"_prefix",s)
                push!(rows,Dict("encoding"=>e,"kind"=>"first15profiles","seconds"=>time()-t))
            end
            for (clock_name,late) in (("early",0.),("late",2047*info["period_s"])),e in ("REF","X")
                shot,_,_=e051_first_shot(seqs[e]);q=late==0 ? shot : e051_delay(late)+shot
                MIGRATION_PREVIOUS_RF[]=false;t=time();s=simulate(p,q,scanner;sim_params=copy(params),verbose=false)
                all(isfinite,s)&&length(s)==400||error("Invalid late prefix")
                write(out,e*"_"*clock_name,s)
                push!(rows,Dict("encoding"=>e,"kind"=>clock_name*"_clock_central_line_single_shot","seconds"=>time()-t))
            end
        end
        mv(joinpath(output,"prefix_signals.h5.partial"),joinpath(output,"prefix_signals.h5"))
        e051_report(joinpath(output,"report.toml"),merge(reg,Dict("status"=>"COMPLETE_3D_CONTROL_RESOURCE_PREFIX",
            "rows"=>rows,"signal_sha256"=>hashfile(joinpath(output,"prefix_signals.h5")))))
        println("COMPLETE_E051_RESOURCE_PREFIX");return
    end
    pre=TOML.parsefile(joinpath(E046ROOT,"outputs/known_flow_3d_e051/preflight_v1/report.toml"))
    for key in ("script_sha256","input_sha256","protocol_sha256","contract_sha256","manifest_sha256","helper_sha256")
        pre[key]==reg[key]||error("Preflight identity differs: "*key)
    end
    gate=TOML.parsefile(joinpath(E046ROOT,"outputs/known_flow_3d_e051/preflight_v1/numerical_gate.toml"))
    gate["status"]=="PASS_3D_PREFIX_KNOWN_VELOCITY"||error("Numerical prefix gate failed")
    gate["prefix_report_sha256"]==hashfile(joinpath(E046ROOT,"outputs/known_flow_3d_e051/preflight_v1/report.toml"))||error("Prefix gate identity changed")
    identities=Dict{String,String}();runtimes=Dict{String,Float64}()
    for model in ("static_landmarks","plug_positive","plug_negative")
        p=e051_phantom(input,model,duration)
        for e in ("REF","X","Y","Z")
            key=model*"_"*lowercase(e);path=joinpath(output,key*".h5")
            println("START_E051 ",key);flush(stdout);MIGRATION_PREVIOUS_RF[]=false;t=time()
            signal=simulate(p,seqs[e],scanner;sim_params=copy(params),verbose=false)
            all(isfinite,signal)&&eltype(signal)==ComplexF64&&length(signal)==12288000||error("Invalid full raw signal")
            k,image=sorted_image(signal,seqs[e]);runtimes[key]=time()-t
            h5open(path*".partial","w") do f
                write(f,"raw_signal_f64",signal);write(f,"image",image)
                HDF5.attributes(f)["encoding"]=e;HDF5.attributes(f)["model"]=model
            end
            mv(path*".partial",path);identities[key]=hashfile(path)
            e051_report(joinpath(output,key*"_checkpoint.toml"),Dict("status"=>"COMPLETE_ENCODING",
                "model"=>model,"encoding"=>e,"sha256"=>identities[key],"seconds"=>runtimes[key]))
            println("FINISHED_E051 ",key," seconds=",runtimes[key]);flush(stdout)
            signal=nothing;k=nothing;image=nothing;GC.gc();CUDA.reclaim()
        end
    end
    e051_report(joinpath(output,"control_report.toml"),merge(reg,Dict("status"=>"COMPLETE_E051_3D_ACQUISITIONS",
        "result_sha256"=>identities,"runtime_s"=>runtimes,"accuracy_promoted"=>false)))
    println("COMPLETE_E051_3D_ACQUISITIONS")
end

length(ARGS)==3||error("preflight|run INPUTS OUTPUT required")
e051(ARGS...)
