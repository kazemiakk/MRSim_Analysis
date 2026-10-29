# E-058 S2/S3 runner on the coupled E055 300k phantom. Loader, timing contract,
# helpers and scanner are copied from E055 code/run_300k.jl; changes are the RF
# step, the certified E-057 free-precession rule and the candidate sequences.
const ORIGINAL_HELPERS=raw"D:\MRSim\MRSIM\MRSimTracks\koma_migration_e046"
const E055=raw"D:\MRSim\outputs\phantom_density_e055"
const E057=raw"D:\MRSim\outputs\step_convergence_e057"
const ROOT=raw"D:\MRSim\outputs\paired_acquisition_e058"
const GPU1="GPU-8be27592-e325-7b7a-f432-b84f69cb18c6"
using KomaMRI, KomaMRICore, KomaMRIBase, CUDA, HDF5, JLD2, FFTW, TOML, SHA
include(joinpath(ORIGINAL_HELPERS,"export_controls.jl"))
include(joinpath(ORIGINAL_HELPERS,"direct_sort.jl"))
include(joinpath(ORIGINAL_HELPERS,"perfect_spoiling.jl"))
include(joinpath(ORIGINAL_HELPERS,"sorted_event_lookup.jl"))
include(joinpath(E057,"code","precession_step_rule.jl"))
const ENCODINGS=("REF","X","Y","Z")
const PREFIX_PROFILES=30
step_label(Δt)=isfinite(Δt) ? string(round(Int,Δt*1e6))*"us" : "inf"

# --- copied verbatim from E055 code/run_300k.jl ---------------------------------
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
    info["case"]=="phantom"&&info["physical_particles"]==300000||error("Frozen phantom300k only")
    hashfile(joinpath(case_dir,"case_inputs.h5"))==info["neutral_input_sha256"]||error("Input changed")
    hashfile(joinpath(case_dir,"legacy_timing.h5"))==info["legacy_timing_sha256"]||error("Timing changed")
    old=TOML.parsefile(joinpath(E046ROOT,"outputs/phantom_precision_e050/mapped_f64_v1/control_report.toml"))
    hashfile(joinpath(map_dir,"cycle_map.h5"))==info["map_sha256"]||error("Map differs from E047")
    hashfile(joinpath(ORIGINAL_HELPERS,"Manifest.toml"))==old["manifest_sha256"]||error("Runtime manifest changed")
    for (name,value) in old["helper_sha256"]
        hashfile(joinpath(ORIGINAL_HELPERS,name))==value||error("E047 helper changed: "*name)
    end
    p=h5open(joinpath(case_dir,"case_inputs.h5"),"r") do f
        reset=Bool.(read(f,"reset"))
        mapping,restored=h5open(joinpath(map_dir,"cycle_map.h5"),"r") do m
            Int.(read(m,"cycle_map")),Bool.(read(m,"restored_endpoint_reset"))
        end
        length(mapping)==300001&&mapping[end]==300001&&all(1 .<= mapping .<= 300001)||error("Bad map")
        all(mapping[1:end-1].<=300000)||error("Sentinel contamination")
        reset[:,end].=restored
        action=FlowPath(read(f,"dx"),read(f,"dy"),read(f,"dz"),reset;cycle_map=mapping)
        curve=TimeCurve(t=Float32.(read(f,"time")),t_unit=Float32.(read(f,"time_unit")),
            periodic=true,periods=1f0,t_start=Float32(info["t_start_s"]),t_end=Float32(info["t_end_s"]))
        Phantom(name="E055_density_300k",x=read(f,"x"),y=read(f,"y"),z=read(f,"z"),
            ρ=read(f,"rho"),T1=read(f,"T1"),T2=read(f,"T2"),Δw=read(f,"delta_w"),
            motion=Motion(action=action,time=curve,spins=AllSpins()))
    end
    length(p)==300001&&p.ρ[end]==0||error("Wrong particle count")
    return info,p
end

function precision_motion_timing(info,curve)
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
# --------------------------------------------------------------------------------

# E055 verify_precision_timing, with the timing file as an argument.
function verify_timing(seq,timing_file,encoding)
    starts=get_block_start_times(seq);labels=get_labels(seq);blocks=findall(a->a.N>0,seq.ADC)
    h5open(timing_file,"r") do f
        g=f[encoding]
        maximum(abs.(Float64[starts[i]+seq.ADC[i].delay for i in blocks].-read(g,"adc_first_s")))<1e-9||error("ADC timing changed")
        maximum(abs.(Float64[dwell(seq.ADC[i]) for i in blocks].-read(g,"dwell_s")))<1e-13||error("ADC dwell changed")
        Int[seq.ADC[i].N for i in blocks]==read(g,"n")||error("ADC counts changed")
        hcat([[Int(labels[i].LIN),Int(labels[i].PAR),Int(labels[i].PHS)] for i in blocks]...)==read(g,"labels")||error("Labels changed")
    end
    return blocks
end

function arm_sequences(arm,info)
    if arm=="baseline"
        seqs=case_sequences("phantom",info)
        for e in ENCODINGS
            hashfile(joinpath(info["sequence_directory"],lowercase(e)*".seq"))==info["sequence_sha256"][e]||error("Sequence changed")
        end
        return seqs,joinpath(E055,"inputs_v1","legacy_timing.h5")
    end
    arm=="short_echo"||error("Unknown arm")
    s0=TOML.parsefile(joinpath(ROOT,"sequences_v1/report.toml"))
    s0["status"]=="PASS_E058_SEQUENCE_CERTIFICATION_AND_CANDIDATE_AUDIT"||error("S0 certification missing")
    timing=joinpath(ROOT,"sequences_v1","short_echo_timing.h5")
    hashfile(timing)==s0["short_echo_timing_sha256"]||error("Candidate timing changed")
    candidate=merge(info,Dict("sequence_directory"=>joinpath(ROOT,"sequences_v1","short_echo")))
    for e in ENCODINGS
        hashfile(joinpath(candidate["sequence_directory"],lowercase(e)*".seq"))==s0["short_echo"][e]["sequence_sha256"]||error("Candidate sequence changed")
    end
    # Same verified zero-amplitude seam markers as the baseline (E055 case_sequences).
    return case_sequences("phantom",candidate),timing
end

function params_for(Δt_prec,Δt_rf)
    p=Dict{String,Any}("gpu"=>true,"gpu_device"=>0,"Nthreads"=>1,"precision"=>"f64",
        "return_type"=>"mat","Δt"=>Inf,"Δt_rf"=>Δt_rf,"max_block_length"=>512,"sim_method"=>MigrationPerfectSpoiling())
    isnothing(Δt_prec)||(p["sampling_rule"]=PrecessionMaxStepRule(Δt_prec,Δt_rf))
    return p
end

relative(a,b)=sqrt(sum(abs2,a.-b)/sum(abs2,b))

function identities(info)
    e057=TOML.parsefile(joinpath(E057,"grid_preflight_v1","report.toml"))
    hashfile(joinpath(E057,"code","precession_step_rule.jl"))==e057["rule_sha256"]||error("E-057 rule changed since certification")
    return Dict{String,Any}("experiment"=>"E-058","script_sha256"=>hashfile(@__FILE__),
        "protocol_sha256"=>hashfile(joinpath(ROOT,"code","PROTOCOL.md")),"rule_sha256"=>e057["rule_sha256"],
        "input_sha256"=>info["neutral_input_sha256"],"timing_sha256"=>info["legacy_timing_sha256"],
        "map_sha256"=>hashfile(joinpath(E055,"map_v1","cycle_map.h5")),
        "manifest_sha256"=>hashfile(joinpath(ORIGINAL_HELPERS,"Manifest.toml")),"precision"=>"f64",
        "physical_particles"=>300000,"coupled_spin_partitioning"=>false,"new_1m"=>false,
        "physical_gpu_uuid"=>ENV["CUDA_VISIBLE_DEVICES"],"visible_GPU"=>0,
        "helper_sha256"=>Dict(file=>hashfile(joinpath(ORIGINAL_HELPERS,file)) for file in
            ("export_controls.jl","direct_sort.jl","perfect_spoiling.jl","sorted_event_lookup.jl")))
end

function simulate_prefix(phantom,seq,blocks,profiles,scanner,params)
    q=copy(seq[1:blocks[profiles]])
    MIGRATION_PREVIOUS_RF[]=false;start=time()
    s=vec(simulate(phantom,q,scanner;sim_params=copy(params),verbose=false))
    CUDA.synchronize();elapsed=time()-start
    all(isfinite,s)&&length(s)==400profiles&&eltype(s)==ComplexF64||error("Invalid full300k Float64 prefix")
    GC.gc();CUDA.reclaim()
    return s,elapsed
end

function prefix(output)
    ispath(output)&&error("Refuse existing output")
    require_precision_gates(E046ROOT)
    get(ENV,"CUDA_VISIBLE_DEVICES","")==GPU1||error("Reviewed GPU1 required")
    info,phantom=load_precision_phantom(joinpath(E055,"inputs_v1"),joinpath(E055,"map_v1"))
    registration=merge(identities(info),Dict("status"=>"REGISTERED_BEFORE_E058_PREFIX","mode"=>"prefix",
        "motion_timing_contract"=>precision_motion_timing(info,phantom.motion.time)))
    mkpath(output);write_precision_toml(joinpath(output,"registration.toml"),registration)
    scanner=Scanner(limits=HardwareLimits(B0=3.0));CUDA.device!(0)
    signals=Dict{String,Any}();runtimes=Dict{String,Float64}();checks=Dict{String,Any}()
    base,base_timing=arm_sequences("baseline",info)
    blocks=Dict(e=>verify_timing(base[e],base_timing,e) for e in ("REF","X"))
    # R0: E055 numerics reproduce E055's saved 15-profile REF preflight prefix.
    s,t=simulate_prefix(phantom,base["REF"],blocks["REF"],15,scanner,params_for(nothing,5e-5))
    saved=h5open(joinpath(E055,"preflight_v1","prefix_signals.h5"),"r") do f;vec(read(f,"profiles_15"));end
    checks["R0_relative_L2"]=relative(s,saved);runtimes["R0_baseline_ref_15_e055"]=t
    println("E058_R0 relative_L2=",checks["R0_relative_L2"]," seconds=",t);flush(stdout)
    checks["R0_relative_L2"]<=1e-12||error("R0 failed: E055 prefix not reproduced")
    ladder=(Inf,16e-6,8e-6,4e-6)
    for Δt in ladder,e in ("REF","X")
        key="baseline_"*lowercase(e)*"_"*step_label(Δt)
        signals[key],runtimes[key]=simulate_prefix(phantom,base[e],blocks[e],PREFIX_PROFILES,scanner,params_for(Δt,.5e-6))
        println("E058_PREFIX ",key," seconds=",runtimes[key]);flush(stdout)
    end
    differences=Dict{String,Any}();chosen=nothing
    for (coarse,fine) in zip(ladder[1:end-1],ladder[2:end])
        d=Dict(e=>relative(signals["baseline_"*lowercase(e)*"_"*step_label(coarse)],signals["baseline_"*lowercase(e)*"_"*step_label(fine)]) for e in ("REF","X"))
        differences[step_label(coarse)*"_vs_"*step_label(fine)]=d
        isnothing(chosen)&&isfinite(coarse)&&all(v<=1e-4 for v in values(d))&&(chosen=coarse)
    end
    coarse_e055=Dict(e=>relative(simulate_prefix(phantom,base[e],blocks[e],PREFIX_PROFILES,scanner,params_for(nothing,5e-5))[1],
        signals["baseline_"*lowercase(e)*"_4us"]) for e in ("REF","X"))
    report=merge(registration,Dict("checks"=>checks,"self_differences"=>differences,
        "E055_numerics_vs_finest_relative_L2"=>coarse_e055,"runtime_s"=>runtimes))
    if isnothing(chosen)
        report["status"]="STOP_E058_NO_PRECESSION_STEP_MET_1E-4"
        write_precision_toml(joinpath(output,"report.toml"),report);println(report["status"]);return
    end
    report["h_star_s"]=chosen
    # Marginal per-profile cost at h*, from 15- and 30-profile warm prefixes.
    _,t15=simulate_prefix(phantom,base["REF"],blocks["REF"],15,scanner,params_for(chosen,.5e-6))
    _,t30=simulate_prefix(phantom,base["REF"],blocks["REF"],30,scanner,params_for(chosen,.5e-6))
    per_profile=max(t30-t15,0.)/15;estimate=8*(t30+per_profile*(30720-30))
    report["timing_15_30_profiles_s"]=[t15,t30];report["estimated_eight_encoding_seconds"]=estimate
    report["estimate_caveat"]="Warm prefix extrapolation from baseline REF, not a guaranteed ETA"
    report["status"]=estimate<=12*3600 ? "PASS_E058_BASELINE_PREFIX_STEP_SELECTION" : "STOP_E058_ESTIMATE_EXCEEDS_12H"
    write_precision_toml(joinpath(output,"report.toml"),report)
    println(report["status"]," h_star=",chosen," estimate_s=",estimate)
end

# Candidate check at h* and h*/2; needs the S0 candidate sequences and a passing baseline prefix.
function prefix_candidate(output)
    ispath(output)&&error("Refuse existing output")
    require_precision_gates(E046ROOT)
    get(ENV,"CUDA_VISIBLE_DEVICES","")==GPU1||error("Reviewed GPU1 required")
    base_report=TOML.parsefile(joinpath(ROOT,"prefix_v1","report.toml"))
    base_report["status"]=="PASS_E058_BASELINE_PREFIX_STEP_SELECTION"||error("Baseline prefix step selection missing")
    chosen=base_report["h_star_s"]
    info,phantom=load_precision_phantom(joinpath(E055,"inputs_v1"),joinpath(E055,"map_v1"))
    registration=merge(identities(info),Dict("status"=>"REGISTERED_BEFORE_E058_CANDIDATE_PREFIX","mode"=>"prefix_candidate",
        "h_star_s"=>chosen,"sequence_report_sha256"=>hashfile(joinpath(ROOT,"sequences_v1","report.toml"))))
    base_report["script_sha256"]==registration["script_sha256"]||error("Runner changed after baseline prefix")
    mkpath(output);write_precision_toml(joinpath(output,"registration.toml"),registration)
    scanner=Scanner(limits=HardwareLimits(B0=3.0));CUDA.device!(0)
    cand,cand_timing=arm_sequences("short_echo",info)
    cblocks=Dict(e=>verify_timing(cand[e],cand_timing,e) for e in ("REF","X"))
    cdiff=Dict{String,Any}();runtimes=Dict{String,Float64}()
    for e in ("REF","X")
        a,ta=simulate_prefix(phantom,cand[e],cblocks[e],PREFIX_PROFILES,scanner,params_for(chosen,.5e-6))
        b,tb=simulate_prefix(phantom,cand[e],cblocks[e],PREFIX_PROFILES,scanner,params_for(chosen/2,.5e-6))
        cdiff[e]=relative(a,b);runtimes["short_echo_"*lowercase(e)*"_"*step_label(chosen)]=ta
        runtimes["short_echo_"*lowercase(e)*"_"*step_label(chosen/2)]=tb
        println("E058_CANDIDATE_PREFIX ",e," self_difference=",cdiff[e]);flush(stdout)
    end
    report=merge(registration,Dict("candidate_self_difference_h_star_vs_half"=>cdiff,"runtime_s"=>runtimes,
        "status"=>all(v<=1e-4 for v in values(cdiff)) ? "PASS_E058_CANDIDATE_PREFIX" : "STOP_E058_CANDIDATE_STEP_NOT_CONVERGED"))
    write_precision_toml(joinpath(output,"report.toml"),report)
    println(report["status"])
end

function acquire(arm,output)
    arm in ("baseline","short_echo")||error("Unknown arm")
    require_precision_gates(E046ROOT)
    get(ENV,"CUDA_VISIBLE_DEVICES","")==GPU1||error("Reviewed GPU1 required")
    pre=TOML.parsefile(joinpath(ROOT,"prefix_v1","report.toml"))
    pre["status"]=="PASS_E058_BASELINE_PREFIX_STEP_SELECTION"||error("Baseline prefix step selection missing")
    cpre=TOML.parsefile(joinpath(ROOT,"prefix_candidate_v1","report.toml"))
    cpre["status"]=="PASS_E058_CANDIDATE_PREFIX"&&cpre["h_star_s"]==pre["h_star_s"]||error("Candidate prefix missing")
    info,phantom=load_precision_phantom(joinpath(E055,"inputs_v1"),joinpath(E055,"map_v1"))
    registration=merge(identities(info),Dict("status"=>"REGISTERED_BEFORE_E058_ACQUISITION","mode"=>"run","arm"=>arm,
        "RF_step_s"=>.5e-6,"free_precession_step_s"=>pre["h_star_s"],
        "sequence_report_sha256"=>hashfile(joinpath(ROOT,"sequences_v1","report.toml")),
        "motion_timing_contract"=>precision_motion_timing(info,phantom.motion.time)))
    for key in ("script_sha256","protocol_sha256","rule_sha256","input_sha256","map_sha256","manifest_sha256")
        pre[key]==registration[key]||error("Prefix identity mismatch: "*key)
    end
    cpre["sequence_report_sha256"]==registration["sequence_report_sha256"]||error("Sequences changed after candidate prefix")
    resume=isdir(output)
    if resume
        old=TOML.parsefile(joinpath(output,"registration.toml"))
        for key in ("script_sha256","protocol_sha256","rule_sha256","sequence_report_sha256","input_sha256","map_sha256","manifest_sha256","free_precession_step_s","arm")
            old[key]==registration[key]||error("Resume identity changed: "*key)
        end
    else
        mkpath(output);write_precision_toml(joinpath(output,"registration.toml"),registration)
    end
    scanner=Scanner(limits=HardwareLimits(B0=3.0));CUDA.device!(0)
    seqs,timing=arm_sequences(arm,info);params=params_for(pre["h_star_s"],.5e-6)
    runtime=Dict{String,Float64}();results=Dict{String,String}()
    for e in ENCODINGS
        seq=seqs[e];blocks=verify_timing(seq,timing,e)
        path=joinpath(output,lowercase(e)*"_result.h5");checkpoint=joinpath(output,lowercase(e)*"_checkpoint.toml")
        if isfile(path)||isfile(checkpoint)
            isfile(path)&&isfile(checkpoint)||error("Incomplete existing encoding")
            previous=TOML.parsefile(checkpoint)
            previous["status"]=="COMPLETE_CHECKPOINT"&&previous["encoding"]==e&&previous["profiles"]==length(blocks)||error("Bad checkpoint")
            hashfile(path)==previous["sha256"]||error("Checkpoint signal changed")
            runtime[e]=previous["runtime_s"];results[e]=previous["sha256"]
            println("REUSE_VERIFIED_E058 ",arm," ",e);flush(stdout);continue
        end
        MIGRATION_PREVIOUS_RF[]=false;println("START_E058 ",arm," ",e);flush(stdout);start=time()
        signal=simulate(phantom,seq,scanner;sim_params=copy(params),verbose=false)
        all(isfinite,signal)&&eltype(signal)==ComplexF64||error("Expected finite raw ComplexF64")
        k,image=sorted_image(signal,seq);runtime[e]=time()-start
        partial=tempname(output)*".partial"
        h5open(partial,"w") do f
            write(f,"raw_signal_f64",signal);write(f,"rawsort",k);write(f,"image",image)
            HDF5.attributes(f)["precision"]="f64_simulation_f32_primary_reconstruction"
            HDF5.attributes(f)["elapsed_seconds"]=runtime[e]
            HDF5.attributes(f)["free_precession_step_s"]=pre["h_star_s"]
        end
        mv(partial,path);results[e]=hashfile(path)
        write_precision_toml(checkpoint,Dict("encoding"=>e,"status"=>"COMPLETE_CHECKPOINT","sha256"=>results[e],
            "profiles"=>length(blocks),"runtime_s"=>runtime[e]))
        println("FINISHED_E058 ",arm," ",e," seconds=",runtime[e]);flush(stdout)
        GC.gc();CUDA.reclaim()
    end
    report=merge(registration,Dict("status"=>"COMPLETE_E058_F64_300K_"*uppercase(arm)*"_FINE_ACQUISITIONS",
        "result_sha256"=>results,"encoding_runtime_seconds"=>runtime,"runtime_core_version"=>string(pkgversion(KomaMRICore))))
    write_precision_toml(joinpath(output,"control_report.toml"),report)
    println(report["status"])
end

if length(ARGS)==2&&ARGS[1]=="prefix"
    prefix(ARGS[2])
elseif length(ARGS)==2&&ARGS[1]=="prefix_candidate"
    prefix_candidate(ARGS[2])
elseif length(ARGS)==3&&ARGS[1]=="run"
    acquire(ARGS[2],ARGS[3])
else
    error("prefix OUTPUT | run baseline|short_echo OUTPUT")
end
