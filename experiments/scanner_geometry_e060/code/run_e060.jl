# E-060 S2/S3 runner: the E055 coupled 300k phantom rigidly rotated into the scanner
# geometry, E-060 sequences, E-058 numerics (0.5 us RF, certified E-057 rule).
# Loader, timing contract and helpers follow E-058 code/run_real.jl. Amendment A2:
# TBP 12 sequences (sequences_v2, slab_profile_v2) and encodings split over two
# identical GPUs after a cross-GPU identity gate. Amendment A3: the periodic
# motion seam of the rotated phantom is moved to the E-060 half-bin gap (seam_v1).
const ORIGINAL_HELPERS=raw"D:\MRSim\MRSIM\MRSimTracks\koma_migration_e046"
const E055=raw"D:\MRSim\outputs\phantom_density_e055"
const E057=raw"D:\MRSim\outputs\step_convergence_e057"
const E058=raw"D:\MRSim\outputs\paired_acquisition_e058"
const ROOT=raw"D:\MRSim\outputs\scanner_geometry_e060"
const SEQUENCES="sequences_v2"
const SLAB="slab_profile_v2"
const GPU1="GPU-8be27592-e325-7b7a-f432-b84f69cb18c6"
const GPU0="GPU-3da2b149-c022-07f4-7fcb-7112684b5ac9"  # also drives the display; shared, never cleared
using KomaMRI, KomaMRICore, KomaMRIBase, CUDA, HDF5, JLD2, FFTW, TOML, SHA
include(joinpath(ORIGINAL_HELPERS,"export_controls.jl"))
include(joinpath(ORIGINAL_HELPERS,"direct_sort.jl"))
include(joinpath(ORIGINAL_HELPERS,"perfect_spoiling.jl"))
include(joinpath(ORIGINAL_HELPERS,"sorted_event_lookup.jl"))
include(joinpath(E057,"code","precession_step_rule.jl"))
const ENCODINGS=("REF","X","Y","Z")
const PREFIX_PROFILES=30
step_label(Δt)=isfinite(Δt) ? string(round(Int,Δt*1e6))*"us" : "inf"
device_uuid(i)="GPU-"*string(CUDA.uuid(CuDevice(i)))

# --- copied verbatim from E-058 code/run_real.jl ---------------------------------
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

function params_for(Δt_prec,Δt_rf;device=0)
    p=Dict{String,Any}("gpu"=>true,"gpu_device"=>device,"Nthreads"=>1,"precision"=>"f64",
        "return_type"=>"mat","Δt"=>Inf,"Δt_rf"=>Δt_rf,"max_block_length"=>512,"sim_method"=>MigrationPerfectSpoiling())
    isnothing(Δt_prec)||(p["sampling_rule"]=PrecessionMaxStepRule(Δt_prec,Δt_rf))
    return p
end

relative(a,b)=sqrt(sum(abs2,a.-b)/sum(abs2,b))

function simulate_prefix(phantom,seq,blocks,profiles,scanner,params)
    q=copy(seq[1:blocks[profiles]])
    MIGRATION_PREVIOUS_RF[]=false;start=time()
    s=vec(simulate(phantom,q,scanner;sim_params=copy(params),verbose=false))
    CUDA.synchronize();elapsed=time()-start
    all(isfinite,s)&&eltype(s)==ComplexF64||error("Invalid full300k Float64 prefix")
    GC.gc();CUDA.reclaim()
    return s,elapsed
end
# --------------------------------------------------------------------------------

"""E055 loader (run_real.jl) with an optional proper rotation (x,y,z)_new = (y, z, x + shift_z).

E055 stores absolute paths in dx/dy/dz with zero base positions, so the rotation
is a permutation of the FlowPath arrays and the translation is a constant base z.
"""
function load_phantom(case_dir,map_dir;rotate::Bool,shift_z::Float32=0f0,seam=nothing)
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
        x,y,z=read(f,"x"),read(f,"y"),read(f,"z")
        all(iszero,x)&&all(iszero,y)&&all(iszero,z)||error("Expected zero base positions")
        dx,dy,dz=read(f,"dx"),read(f,"dy"),read(f,"dz")
        action=rotate ? FlowPath(dy,dz,dx,reset;cycle_map=mapping) : FlowPath(dx,dy,dz,reset;cycle_map=mapping)
        curve=TimeCurve(t=Float32.(read(f,"time")),t_unit=Float32.(read(f,"time_unit")),
            periodic=true,periods=1f0,t_start=Float32(info["t_start_s"]),t_end=Float32(info["t_end_s"]))
        motion=Motion(action=action,time=curve,spins=AllSpins())
        if !isnothing(seam)
            # A3: frozen koma_dp150 shift; the period must stay bit-identical to E055.
            shifted=shift_periodic_motion_seam(motion,seam;requested_period_s=info["period_s"])
            shifted.realized_seam_time_s==seam&&shifted.realized_period_s==info["period_s"]||error("Seam shift not exact")
            motion=shifted.motion
        end
        Phantom(name=rotate ? "E060_rotated_300k" : "E055_density_300k",x=x,y=y,z=rotate ? fill(shift_z,length(z)) : z,
            ρ=read(f,"rho"),T1=read(f,"T1"),T2=read(f,"T2"),Δw=read(f,"delta_w"),motion=motion)
    end
    length(p)==300001&&p.ρ[end]==0||error("Wrong particle count")
    return info,p
end

# JSON.jl is not a direct dependency of the pinned project; the two Python reports
# are read for single unique scalar keys only.
function json_scalar(path,key)
    found=collect(eachmatch(Regex("\"$key\": (\"[^\"]*\"|[-+0-9.eE]+)"),read(path,String)))
    length(found)==1||error("Expected one "*key*" in "*path)
    v=found[1].captures[1]
    return startswith(v,"\"") ? v[2:end-1] : parse(Float64,v)
end

function s1_shift()
    report=joinpath(ROOT,"phantom_v1","report.json")
    json_scalar(report,"status")=="PASS_E060_PHANTOM_ROTATION"||error("S1 phantom rotation missing")
    json_scalar(report,"definition")=="(x,y,z)_new = (y, z, x - x_throat)"||error("Unexpected S1 rotation")
    translation=json_scalar(report,"translation_new_z_m")
    shift=Float32(translation)
    # Attempt 2 required exactness; the true Float32 rounding is -3.7e-9 m (S1's 0.0 used float32 arithmetic).
    abs(Float64(shift)-translation)<=1e-8||error("Float32 translation rounding exceeds 1e-8 m")
    return shift
end

marker_count(seq,info)=floor(Int,(dur(seq)-(info["seam_time_s"]-350e-6)-2e-6)/info["period_s"])+1

"""A3: E055 case info with the E-060 seam from seam_v1 (spins, map, resets and period unchanged)."""
function e060_info(info)
    seam=TOML.parsefile(joinpath(ROOT,"seam_v1","report.toml"))
    seam["status"]=="PASS_E060_SEAM_ALIGNMENT"&&seam["period_s"]==info["period_s"]||error("Seam alignment missing")
    seam["sequence_report_sha256"]==hashfile(joinpath(ROOT,SEQUENCES,"report.toml"))||error("Seam aligned to other sequences")
    return merge(info,Dict("seam_time_s"=>seam["seam_time_s"],"t_start_s"=>seam["seam_time_s"],"t_end_s"=>seam["t_end_s"],
        "expected_seam_markers"=>seam["seam_markers"]))
end

function e060_sequences(info)
    s0=TOML.parsefile(joinpath(ROOT,SEQUENCES,"report.toml"))
    s0["status"]=="PASS_E060_SEQUENCES_AND_AUDITS"&&s0["hardware"]["TBP"]==12||error("E-060 S0 v2 (TBP 12) missing")
    timing=joinpath(ROOT,SEQUENCES,"timing_e060.h5")
    hashfile(timing)==s0["timing_sha256"]||error("E-060 timing changed")
    seqs=Dict{String,Any}();counts=Dict{String,Int}()
    for e in ENCODINGS
        path=joinpath(ROOT,SEQUENCES,"scanner_geometry",lowercase(e)*".seq")
        hashfile(path)==s0["scanner_geometry"][e]["sequence_sha256"]||error("E-060 sequence changed: "*e)
        seq=read_seq(path;verbose=false,verify_signature=true)
        Int.((seq.DEF["Nx"],seq.DEF["Ny"],seq.DEF["Nz"],seq.DEF["CardiacBins"]))==(64,64,30,30)||error("Unexpected DEF shape")
        counts[e]=marker_count(seq,info)
        counts[e]==info["expected_seam_markers"]||error("Seam count differs from the seam_v1 alignment: "*string(counts[e]))
        # Same verified zero-amplitude seam markers as E055/E058, one per motion seam.
        seqs[e]=insert_verified_zero_markers!(seq,info["seam_time_s"],info["period_s"],counts[e])
    end
    return seqs,timing,counts
end

"""E055 baseline sequence with gradient axes permuted to match the rotated phantom."""
function rotated_gradients(seq)
    q=copy(seq);q.GR=seq.GR[[2,3,1],:]
    return q
end

function identities(info)
    e057=TOML.parsefile(joinpath(E057,"grid_preflight_v1","report.toml"))
    hashfile(joinpath(E057,"code","precession_step_rule.jl"))==e057["rule_sha256"]||error("E-057 rule changed since certification")
    return Dict{String,Any}("experiment"=>"E-060","script_sha256"=>hashfile(@__FILE__),
        "protocol_sha256"=>hashfile(joinpath(ROOT,"code","PROTOCOL.md")),
        "amendment_sha256"=>hashfile(joinpath(ROOT,"code","AMENDMENT_A1_TE.md")),
        "amendment_a2_sha256"=>hashfile(joinpath(ROOT,"code","AMENDMENT_A2_SLAB_GPU.md")),
        "amendment_a3_sha256"=>hashfile(joinpath(ROOT,"code","AMENDMENT_A3_SEAM.md")),
        "seam_report_sha256"=>hashfile(joinpath(ROOT,"seam_v1","report.toml")),"rule_sha256"=>e057["rule_sha256"],
        "input_sha256"=>info["neutral_input_sha256"],"map_sha256"=>hashfile(joinpath(E055,"map_v1","cycle_map.h5")),
        "manifest_sha256"=>hashfile(joinpath(ORIGINAL_HELPERS,"Manifest.toml")),"precision"=>"f64",
        "sequence_report_sha256"=>hashfile(joinpath(ROOT,SEQUENCES,"report.toml")),
        "phantom_report_sha256"=>hashfile(joinpath(ROOT,"phantom_v1","report.json")),
        "slab_report_sha256"=>hashfile(joinpath(ROOT,SLAB,"report.json")),
        "physical_particles"=>300000,"coupled_spin_partitioning"=>false,"new_1m"=>false,
        "physical_gpu_uuid"=>ENV["CUDA_VISIBLE_DEVICES"],"visible_GPU"=>0,
        "helper_sha256"=>Dict(file=>hashfile(joinpath(ORIGINAL_HELPERS,file)) for file in
            ("export_controls.jl","direct_sort.jl","perfect_spoiling.jl","sorted_event_lookup.jl")))
end

function require_slab()
    json_scalar(joinpath(ROOT,SLAB,"report.json"),"status")=="PASS_E060_SLAB_FOLD_IN_GATE"||error("Slab fold-in gate missing")
end

function prefix(output)
    ispath(output)&&error("Refuse existing output")
    require_precision_gates(E046ROOT);require_slab()
    # Both reviewed GPUs visible: device 0 = GPU1 (all prefix work), device 1 = GPU0 (identity gate only).
    get(ENV,"CUDA_VISIBLE_DEVICES","")==GPU1*","*GPU0||error("Both reviewed GPUs required, GPU1 first")
    device_uuid(0)==GPU1&&device_uuid(1)==GPU0||error("Device order differs from CUDA_VISIBLE_DEVICES")
    e058=TOML.parsefile(joinpath(E058,"prefix_v1","report.toml"))
    e058["status"]=="PASS_E058_BASELINE_PREFIX_STEP_SELECTION"||error("E-058 h* missing")
    h=e058["h_star_s"]
    info,unrotated=load_phantom(joinpath(E055,"inputs_v1"),joinpath(E055,"map_v1");rotate=false)
    registration=merge(identities(info),Dict("status"=>"REGISTERED_BEFORE_E060_PREFIX","mode"=>"prefix","h_star_s"=>h,
        "motion_timing_contract"=>precision_motion_timing(info,unrotated.motion.time)))
    mkpath(output);write_precision_toml(joinpath(output,"registration.toml"),registration)
    e060=e060_info(info)
    seqs,timing,counts=e060_sequences(e060)  # sequence, timing and seam-marker checks before any GPU work
    scanner=Scanner(limits=HardwareLimits(B0=3.0));CUDA.device!(0)
    checks=Dict{String,Any}();runtimes=Dict{String,Float64}()
    # R0: rotation invariance. Unrotated phantom + E055 sequence versus the rotated
    # phantom (no translation) + the same sequence with gradients rotated identically.
    base=case_sequences("phantom",info)
    for e in ("REF","X")
        hashfile(joinpath(info["sequence_directory"],lowercase(e)*".seq"))==info["sequence_sha256"][e]||error("Sequence changed")
    end
    bblocks=Dict(e=>verify_timing(base[e],joinpath(E055,"inputs_v1","legacy_timing.h5"),e) for e in ("REF","X"))
    original=Dict(e=>simulate_prefix(unrotated,base[e],bblocks[e],15,scanner,params_for(h,.5e-6))[1] for e in ("REF","X"))
    unrotated=nothing;GC.gc()
    _,turned=load_phantom(joinpath(E055,"inputs_v1"),joinpath(E055,"map_v1");rotate=true)
    for e in ("REF","X")
        s,_=simulate_prefix(turned,rotated_gradients(base[e]),bblocks[e],15,scanner,params_for(h,.5e-6))
        checks["R0_rotation_invariance_relative_L2_"*e]=relative(s,original[e])
        println("E060_R0 ",e," relative_L2=",checks["R0_rotation_invariance_relative_L2_"*e]);flush(stdout)
    end
    turned=nothing;GC.gc()
    all(checks["R0_rotation_invariance_relative_L2_"*e]<=1e-9 for e in ("REF","X"))||begin
        report=merge(registration,Dict("checks"=>checks,"status"=>"STOP_E060_ROTATION_NOT_INVARIANT"))
        write_precision_toml(joinpath(output,"report.toml"),report);error("Rotation invariance failed")
    end
    # Step ladder on the E-060 sequences with the translated, rotated phantom: the
    # coarsest step whose self-difference to its half is <= 1e-4 on all encodings.
    _,phantom=load_phantom(joinpath(E055,"inputs_v1"),joinpath(E055,"map_v1");rotate=true,shift_z=s1_shift(),seam=e060["seam_time_s"])
    checks["e060_motion_timing_contract"]=precision_motion_timing(e060,phantom.motion.time)
    checks["s1_translation_float32_rounding_m"]=Float64(s1_shift())-json_scalar(joinpath(ROOT,"phantom_v1","report.json"),"translation_new_z_m")
    blocks=Dict(e=>verify_timing(seqs[e],timing,e) for e in ENCODINGS)
    ladder=(h,h/2,h/4,h/8);signals=Dict{String,Any}();differences=Dict{String,Any}();chosen=nothing
    run!(e,Δt)=get!(signals,e*"_"*step_label(Δt)) do
        s,t=simulate_prefix(phantom,seqs[e],blocks[e],PREFIX_PROFILES,scanner,params_for(Δt,.5e-6))
        runtimes[e*"_"*step_label(Δt)]=t;s
    end
    for (coarse,fine) in zip(ladder[1:end-1],ladder[2:end])
        d=Dict(e=>relative(run!(e,coarse),run!(e,fine)) for e in ENCODINGS)
        differences[step_label(coarse)*"_vs_"*step_label(fine)]=d
        println("E060_PREFIX ",step_label(coarse)," vs ",step_label(fine)," ",d);flush(stdout)
        all(v<=1e-4 for v in values(d))&&(chosen=coarse;break)
    end
    for e in ENCODINGS
        checks["prefix_signal_rms_"*e]=sqrt(sum(abs2,run!(e,ladder[1]))/length(run!(e,ladder[1])))
    end
    report=merge(registration,Dict("checks"=>checks,"self_differences"=>differences,"seam_markers"=>counts,"runtime_s"=>runtimes))
    if isnothing(chosen)
        report["status"]="STOP_E060_NO_PRECESSION_STEP_MET_1E-4"
        write_precision_toml(joinpath(output,"report.toml"),report);println(report["status"]);return
    end
    report["h_star_s"]=chosen
    # A2 gate: the same REF prefix on both identical GPUs before splitting encodings.
    a,_=simulate_prefix(phantom,seqs["REF"],blocks["REF"],15,scanner,params_for(chosen,.5e-6;device=0))
    b,_=simulate_prefix(phantom,seqs["REF"],blocks["REF"],15,scanner,params_for(chosen,.5e-6;device=1))
    CUDA.device!(0)
    checks["cross_gpu_REF_15_relative_L2"]=relative(b,a)
    println("E060_CROSS_GPU relative_L2=",checks["cross_gpu_REF_15_relative_L2"]);flush(stdout)
    if checks["cross_gpu_REF_15_relative_L2"]>1e-9
        report["status"]="STOP_E060_GPUS_DISAGREE"
        write_precision_toml(joinpath(output,"report.toml"),report);println(report["status"]);return
    end
    _,t15=simulate_prefix(phantom,seqs["REF"],blocks["REF"],15,scanner,params_for(chosen,.5e-6))
    _,t30=simulate_prefix(phantom,seqs["REF"],blocks["REF"],30,scanner,params_for(chosen,.5e-6))
    per_profile=max(t30-t15,0.)/15;estimate=4*(t30+per_profile*(length(blocks["REF"])-30))
    report["timing_15_30_profiles_s"]=[t15,t30];report["estimated_four_encoding_seconds"]=estimate
    report["estimate_caveat"]="Warm prefix extrapolation from REF on one GPU (single-GPU equivalent), not a guaranteed ETA"
    report["status"]=estimate<=12*3600 ? "PASS_E060_PREFIX" : "STOP_E060_ESTIMATE_EXCEEDS_12H"
    write_precision_toml(joinpath(output,"report.toml"),report)
    println(report["status"]," h_star=",chosen," estimate_s=",estimate)
end

const IDENTITY_KEYS=("script_sha256","protocol_sha256","amendment_sha256","amendment_a2_sha256","amendment_a3_sha256","rule_sha256",
    "input_sha256","map_sha256","manifest_sha256","sequence_report_sha256","phantom_report_sha256","slab_report_sha256","seam_report_sha256")

function acquire(output,encodings)
    require_precision_gates(E046ROOT);require_slab()
    gpu=get(ENV,"CUDA_VISIBLE_DEVICES","")
    gpu in (GPU1,GPU0)||error("Exactly one reviewed GPU required")
    device_uuid(0)==gpu||error("Visible device differs from CUDA_VISIBLE_DEVICES")
    !isempty(encodings)&&allunique(encodings)&&all(e->e in ENCODINGS,encodings)||error("Bad encoding subset")
    pre=TOML.parsefile(joinpath(ROOT,"prefix_v1","report.toml"))
    pre["status"]=="PASS_E060_PREFIX"||error("E-060 prefix missing")
    e060=e060_info(TOML.parsefile(joinpath(E055,"inputs_v1","case_info.toml")))
    info,phantom=load_phantom(joinpath(E055,"inputs_v1"),joinpath(E055,"map_v1");rotate=true,shift_z=s1_shift(),seam=e060["seam_time_s"])
    registration=merge(identities(info),Dict("status"=>"REGISTERED_BEFORE_E060_ACQUISITION","mode"=>"run",
        "RF_step_s"=>.5e-6,"free_precession_step_s"=>pre["h_star_s"],
        "motion_timing_contract"=>precision_motion_timing(e060,phantom.motion.time)))
    for key in IDENTITY_KEYS
        pre[key]==registration[key]||error("Prefix identity mismatch: "*key)
    end
    # The first process registers; the other (and any resume) must match it.
    if isfile(joinpath(output,"registration.toml"))
        old=TOML.parsefile(joinpath(output,"registration.toml"))
        for key in (IDENTITY_KEYS...,"free_precession_step_s")
            old[key]==registration[key]||error("Resume identity changed: "*key)
        end
    else
        ispath(output)&&error("Output exists without registration")
        mkpath(output);write_precision_toml(joinpath(output,"registration.toml"),registration)
    end
    scanner=Scanner(limits=HardwareLimits(B0=3.0));CUDA.device!(0)
    seqs,timing,_=e060_sequences(e060);params=params_for(pre["h_star_s"],.5e-6)
    runtime=Dict{String,Float64}();results=Dict{String,String}()
    for e in encodings
        seq=seqs[e];blocks=verify_timing(seq,timing,e)
        path=joinpath(output,lowercase(e)*"_result.h5");checkpoint=joinpath(output,lowercase(e)*"_checkpoint.toml")
        if isfile(path)||isfile(checkpoint)
            isfile(path)&&isfile(checkpoint)||error("Incomplete existing encoding")
            previous=TOML.parsefile(checkpoint)
            previous["status"]=="COMPLETE_CHECKPOINT"&&previous["encoding"]==e&&previous["profiles"]==length(blocks)||error("Bad checkpoint")
            hashfile(path)==previous["sha256"]||error("Checkpoint signal changed")
            println("REUSE_VERIFIED_E060 ",e);flush(stdout);continue
        end
        MIGRATION_PREVIOUS_RF[]=false;println("START_E060 ",e);flush(stdout);start=time()
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
            "profiles"=>length(blocks),"runtime_s"=>runtime[e],"physical_gpu_uuid"=>gpu,"script_sha256"=>registration["script_sha256"]))
        println("FINISHED_E060 ",e," seconds=",runtime[e]);flush(stdout)
        GC.gc();CUDA.reclaim()
    end
    println("COMPLETE_E060_SUBSET ",join(encodings,","));flush(stdout)
end

"""After both GPU processes: verify all four checkpoints and write the control report."""
function finalize(output)
    ispath(joinpath(output,"control_report.toml"))&&error("Control report exists")
    registration=TOML.parsefile(joinpath(output,"registration.toml"))
    registration["script_sha256"]==hashfile(@__FILE__)||error("Runner changed during acquisition")
    results=Dict{String,String}();runtime=Dict{String,Float64}();gpus=Dict{String,String}()
    for e in ENCODINGS
        checkpoint=TOML.parsefile(joinpath(output,lowercase(e)*"_checkpoint.toml"))
        checkpoint["status"]=="COMPLETE_CHECKPOINT"&&checkpoint["encoding"]==e||error("Missing checkpoint "*e)
        checkpoint["script_sha256"]==registration["script_sha256"]||error("Checkpoint from another runner "*e)
        hashfile(joinpath(output,lowercase(e)*"_result.h5"))==checkpoint["sha256"]||error("Result changed "*e)
        results[e]=checkpoint["sha256"];runtime[e]=checkpoint["runtime_s"];gpus[e]=checkpoint["physical_gpu_uuid"]
    end
    report=merge(registration,Dict("status"=>"COMPLETE_E060_F64_300K_SCANNER_GEOMETRY_ACQUISITIONS",
        "result_sha256"=>results,"encoding_runtime_seconds"=>runtime,"encoding_physical_gpu_uuid"=>gpus,
        "runtime_core_version"=>string(pkgversion(KomaMRICore))))
    write_precision_toml(joinpath(output,"control_report.toml"),report)
    println(report["status"])
end

if length(ARGS)==2&&ARGS[1]=="prefix"
    prefix(ARGS[2])
elseif length(ARGS)==3&&ARGS[1]=="run"
    acquire(ARGS[2],String.(split(ARGS[3],",")))
elseif length(ARGS)==2&&ARGS[1]=="finalize"
    finalize(ARGS[2])
else
    error("prefix OUTPUT | run OUTPUT REF,X | finalize OUTPUT")
end
