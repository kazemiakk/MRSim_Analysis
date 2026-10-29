using KomaMRI, KomaMRICore, CUDA, HDF5, JLD2, FFTW, TOML, SHA
include(joinpath(@__DIR__,"export_controls.jl"))
include(joinpath(@__DIR__,"direct_sort.jl"))
include(joinpath(@__DIR__,"perfect_spoiling.jl"))
include(joinpath(@__DIR__,"sorted_event_lookup.jl"))

function require_precision_gates(root)
    for folder in ("clock_inrun_cpu_v1","clock_inrun_gpu_v1")
        report=TOML.parsefile(joinpath(root,"outputs/phantom_encoding_e049",folder,"report.toml"))
        report["status"]=="COMPLETE_CLOCK_ORIGIN_CONTROLS"||error("Clock control missing")
        values=filter(r->r["case"]=="constant1p5"&&r["mode"]=="production64",report["rows"])
        any(r->r["cycles"]==2047,values)||error("No in-run late Float64 gate")
        all(abs(r["velocity_m_s"]-1.5)<=.002 for r in values)||error("Float64 clock gate failed")
        hashfile(joinpath(root,"outputs/phantom_encoding_e049",folder,"clock_signals.h5"))==report["signals_sha256"]||error("Clock signal changed")
    end
end

function load_precision_phantom(case_dir,map_dir)
    info=TOML.parsefile(joinpath(case_dir,"case_info.toml"))
    info["case"]=="phantom"&&info["physical_particles"]==100000||error("Frozen phantom100k only")
    hashfile(joinpath(case_dir,"case_inputs.h5"))==info["neutral_input_sha256"]||error("Input changed")
    hashfile(joinpath(case_dir,"legacy_timing.h5"))==info["legacy_timing_sha256"]||error("Timing changed")
    old=TOML.parsefile(joinpath(E046ROOT,"outputs/remapped_confirmation_e047/phantom_mapped_v1/control_report.toml"))
    hashfile(joinpath(map_dir,"cycle_map.h5"))==old["map_sha256"]||error("Map differs from E047")
    hashfile(joinpath(@__DIR__,"Manifest.toml"))==old["manifest_sha256"]||error("Runtime manifest changed")
    for (name,value) in old["helper_sha256"]
        hashfile(joinpath(@__DIR__,name))==value||error("E047 helper changed: "*name)
    end
    p=h5open(joinpath(case_dir,"case_inputs.h5"),"r") do f
        reset=Bool.(read(f,"reset"))
        mapping,restored=h5open(joinpath(map_dir,"cycle_map.h5"),"r") do m
            Int.(read(m,"cycle_map")),Bool.(read(m,"restored_endpoint_reset"))
        end
        length(mapping)==100001&&mapping[end]==100001&&all(1 .<= mapping .<= 100001)||error("Bad map")
        all(mapping[1:end-1].<=100000)||error("Sentinel contamination")
        reset[:,end].=restored
        action=FlowPath(read(f,"dx"),read(f,"dy"),read(f,"dz"),reset;cycle_map=mapping)
        curve=TimeCurve(t=Float32.(read(f,"time")),t_unit=Float32.(read(f,"time_unit")),
            periodic=true,periods=1f0,t_start=Float32(info["t_start_s"]),t_end=Float32(info["t_end_s"]))
        Phantom(name="E050_precision_only_100k",x=read(f,"x"),y=read(f,"y"),z=read(f,"z"),
            ρ=read(f,"rho"),T1=read(f,"T1"),T2=read(f,"T2"),Δw=read(f,"delta_w"),
            motion=Motion(action=action,time=curve,spins=AllSpins()))
    end
    length(p)==100001&&p.ρ[end]==0||error("Wrong particle count")
    return info,p
end

function verify_precision_timing(seq,case_dir,encoding)
    starts=get_block_start_times(seq);labels=get_labels(seq);blocks=findall(a->a.N>0,seq.ADC)
    h5open(joinpath(case_dir,"legacy_timing.h5"),"r") do f
        g=f[encoding]
        maximum(abs.(Float64[starts[i]+seq.ADC[i].delay for i in blocks].-read(g,"adc_first_s")))<1e-9||error("ADC timing changed")
        maximum(abs.(Float64[dwell(seq.ADC[i]) for i in blocks].-read(g,"dwell_s")))<1e-13||error("ADC dwell changed")
        Int[seq.ADC[i].N for i in blocks]==read(g,"n")||error("ADC counts changed")
        hcat([[Int(labels[i].LIN),Int(labels[i].PAR),Int(labels[i].PHS)] for i in blocks]...)==read(g,"labels")||error("Labels changed")
    end
    return blocks
end

function precision_motion_timing(info,curve)
    # Literal promotion, not period normalization or a fitted temporal shift.
    promoted=f64(curve)
    length(curve.t)==2&&curve.periods==1f0&&curve.periodic||error("Unexpected time curve")
    Float64(curve.t_end-curve.t_start)==info["period_s"]||error("Exported Float32 period changed")
    knots=KomaMRIBase.times(promoted)
    duration=promoted.t_end-promoted.t_start
    duration==knots[end]-knots[1]||error("Reset and trajectory periods disagree")
    promoted.t_start==info["t_start_s"]&&promoted.t_end==info["t_end_s"]||error("Literal endpoints changed")
    abs(duration-info["period_s"])<=1e-7||error("Unexpected precision-promotion period difference")
    return Dict("convention"=>"literal saved endpoints promoted; no retiming",
        "t_start_s"=>promoted.t_start,"t_end_s"=>promoted.t_end,
        "exported_float32_period_s"=>info["period_s"],"promoted_motion_period_s"=>duration,
        "period_difference_s"=>duration-info["period_s"],
        "seam_difference_at_cycle2047_s"=>2047*(duration-info["period_s"]),
        "trajectory_and_reset_periods_identical"=>true)
end

function write_precision_toml(path,report)
    ispath(path)&&error("Refuse report overwrite: "*path)
    open(path*".partial","w") do io;TOML.print(io,report);end
    mv(path*".partial",path)
end

function run_precision(mode,case_dir,map_dir,output)
    mode in ("preflight","run","resume")||error("Unknown mode")
    ispath(output)&&mode!="resume"&&error("Refuse overwrite")
    mode=="resume"&&!isdir(output)&&error("No registered run to resume")
    require_precision_gates(E046ROOT)
    get(ENV,"CUDA_VISIBLE_DEVICES","")=="GPU-8be27592-e325-7b7a-f432-b84f69cb18c6"||error("Reviewed GPU1 required")
    info,phantom=load_precision_phantom(case_dir,map_dir)
    timing_contract=precision_motion_timing(info,phantom.motion.time)
    info["B0_T"]==3.0&&info["Venc_m_s"]==5.0||error("Protocol changed")
    registration=Dict("experiment"=>"E-050","status"=>"REGISTERED_BEFORE_CANDIDATE_SIGNAL",
        "mode"=>mode,"script_sha256"=>hashfile(@__FILE__),"precision"=>"f64",
        "input_sha256"=>info["neutral_input_sha256"],"timing_sha256"=>info["legacy_timing_sha256"],
        "map_sha256"=>hashfile(joinpath(map_dir,"cycle_map.h5")),
        "protocol_sha256"=>hashfile(joinpath(E046ROOT,"outputs/phantom_precision_e050/PROTOCOL.md")),
        "manifest_sha256"=>hashfile(joinpath(@__DIR__,"Manifest.toml")),
        "physical_particles"=>100000,"coupled_spin_partitioning"=>false,
        "motion_timing_contract"=>timing_contract,
        "physical_gpu_uuid"=>ENV["CUDA_VISIBLE_DEVICES"],"visible_GPU"=>0,
        "helper_sha256"=>Dict(file=>hashfile(joinpath(@__DIR__,file)) for file in
            ("export_controls.jl","direct_sort.jl","perfect_spoiling.jl","sorted_event_lookup.jl")))
    if mode=="resume"
        oldreg=TOML.parsefile(joinpath(output,"registration.toml"))
        for key in ("script_sha256","input_sha256","timing_sha256","map_sha256","protocol_sha256","manifest_sha256","precision","helper_sha256")
            oldreg[key]==registration[key]||error("Resume identity changed: "*key)
        end
    else
        mkpath(output);write_precision_toml(joinpath(output,"registration.toml"),registration)
    end
    params=Dict{String,Any}("gpu"=>true,"gpu_device"=>0,"Nthreads"=>1,"precision"=>"f64",
        "return_type"=>"mat","Δt"=>Inf,"Δt_rf"=>5e-5,"max_block_length"=>512,"sim_method"=>MigrationPerfectSpoiling())
    scanner=Scanner(limits=HardwareLimits(B0=3.0));CUDA.device!(0)
    seqs=case_sequences("phantom",info)
    for e in ("REF","X","Y","Z")
        hashfile(joinpath(info["sequence_directory"],lowercase(e)*".seq"))==info["sequence_sha256"][e]||error("Sequence changed")
    end
    refblocks=verify_precision_timing(seqs["REF"],case_dir,"REF")
    if mode=="preflight"
        # Full100k mapped set, never an independently sliced spin batch.
        runtimes=Dict{String,Float64}();signals=Dict{String,Any}()
        for profiles in (3,15)
            q=copy(seqs["REF"][1:refblocks[profiles]])
            MIGRATION_PREVIOUS_RF[]=false;start=time()
            s=simulate(phantom,q,scanner;sim_params=copy(params),verbose=false)
            CUDA.synchronize();runtimes[string(profiles)]=time()-start
            all(isfinite,s)&&length(s)==400profiles&&eltype(s)==ComplexF64||error("Invalid full100k Float64 prefix")
            signals[string(profiles)]=s
            println("E050_PREFIX profiles=",profiles," seconds=",runtimes[string(profiles)]);flush(stdout)
            GC.gc();CUDA.reclaim()
        end
        h5open(joinpath(output,"prefix_signals.h5.partial"),"w") do f
            for (name,s) in signals;write(f,"profiles_"*name,s);end
        end
        mv(joinpath(output,"prefix_signals.h5.partial"),joinpath(output,"prefix_signals.h5"))
        estimate=runtimes["15"]/15*length(refblocks)
        report=merge(registration,Dict("status"=>"PASS_COUPLED100K_F64_RESOURCE_PREFIX",
            "prefix_runtime_seconds"=>runtimes,"estimated_seconds_per_encoding"=>estimate,
            "estimated_seconds_four_encodings"=>4estimate,
            "estimate_caveat"=>"Warm prefix extrapolation, not guaranteed full-run ETA or measured peak memory",
            "float64_three_path_arrays_bytes"=>3*100001*1441*8,
            "prefix_sha256"=>hashfile(joinpath(output,"prefix_signals.h5"))))
        write_precision_toml(joinpath(output,"report.toml"),report)
        println("PASS_E050_RESOURCE_PREFIX estimated_four_encoding_seconds=",4estimate);return
    end
    preflight=TOML.parsefile(joinpath(E046ROOT,"outputs/phantom_precision_e050/preflight_v2/report.toml"))
    preflight["status"]=="PASS_COUPLED100K_F64_RESOURCE_PREFIX"||error("Resource preflight missing")
    for key in ("script_sha256","input_sha256","timing_sha256","map_sha256","manifest_sha256","protocol_sha256")
        preflight[key]==registration[key]||error("Preflight identity mismatch: "*key)
    end
    runtime=Dict{String,Float64}();identities=Dict{String,String}()
    for e in ("REF","X","Y","Z")
        seq=seqs[e];blocks=verify_precision_timing(seq,case_dir,e)
        path=joinpath(output,lowercase(e)*"_result.h5")
        checkpoint=joinpath(output,lowercase(e)*"_checkpoint.toml")
        if isfile(path)||isfile(checkpoint)
            mode=="resume"&&isfile(path)&&isfile(checkpoint)||error("Incomplete/unqualified existing encoding")
            previous=TOML.parsefile(checkpoint)
            previous["status"]=="COMPLETE_CHECKPOINT"&&previous["encoding"]==e&&previous["profiles"]==length(blocks)||error("Bad checkpoint")
            hashfile(path)==previous["sha256"]||error("Checkpoint signal changed")
            runtime[e]=previous["runtime_s"];identities[e]=previous["sha256"]
            println("REUSE_VERIFIED_E050 ",e);flush(stdout);continue
        end
        MIGRATION_PREVIOUS_RF[]=false;println("START_E050 ",e);flush(stdout);start=time()
        signal=simulate(phantom,seq,scanner;sim_params=copy(params),verbose=false)
        all(isfinite,signal)&&eltype(signal)==ComplexF64||error("Expected finite raw ComplexF64")
        k,image=sorted_image(signal,seq);runtime[e]=time()-start
        partial=tempname(output)*".partial"
        h5open(partial,"w") do f
            write(f,"raw_signal_f64",signal);write(f,"rawsort",k);write(f,"image",image)
            HDF5.attributes(f)["precision"]="f64_simulation_f32_primary_reconstruction"
            HDF5.attributes(f)["elapsed_seconds"]=runtime[e]
        end
        mv(partial,path);identities[e]=hashfile(path)
        write_precision_toml(checkpoint,
            Dict("encoding"=>e,"status"=>"COMPLETE_CHECKPOINT","sha256"=>identities[e],"profiles"=>length(blocks),"runtime_s"=>runtime[e]))
        println("FINISHED_E050 ",e," seconds=",runtime[e]);flush(stdout)
        GC.gc();CUDA.reclaim()
    end
    report=merge(registration,Dict("status"=>"COMPLETE_E050_F64_100K_MAPPED_ACQUISITIONS",
        "result_sha256"=>identities,"encoding_runtime_seconds"=>runtime,
        "actual_ADC_timing_and_labels_match_E047"=>true,"new_1m"=>false,
        "runtime_core_version"=>string(pkgversion(KomaMRICore))))
    write_precision_toml(joinpath(output,"control_report.toml"),report)
    println("COMPLETE_E050_F64_100K_MAPPED_ACQUISITIONS")
end

length(ARGS)==4||error("preflight|run|resume CASE_DIR MAP_DIR OUTPUT required")
run_precision(ARGS...)
