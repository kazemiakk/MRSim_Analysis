using KomaMRI, KomaMRICore, CUDA, HDF5, JLD2, FFTW, TOML, SHA
include(joinpath(@__DIR__,"export_controls.jl"))
include(joinpath(@__DIR__,"direct_sort.jl"))
include(joinpath(@__DIR__,"perfect_spoiling.jl"))

function load_control(case_dir,map_dir,arm)
    info=TOML.parsefile(joinpath(case_dir,"case_info.toml"))
    file=joinpath(case_dir,"case_inputs.h5")
    hashfile(file)==info["neutral_input_sha256"]||error("Input changed")
    h5open(file,"r") do f
        reset=Bool.(read(f,"reset"));mapping=nothing
        if arm=="updated_map"
            mapping,restored=h5open(joinpath(map_dir,"cycle_map.h5"),"r") do m
                Int.(read(m,"cycle_map")),Bool.(read(m,"restored_endpoint_reset"))
            end
            length(mapping)==size(reset,1)||error("Map size mismatch")
            reset[:,end].=restored
        end
        action=arm=="updated_map" ?
            FlowPath(read(f,"dx"),read(f,"dy"),read(f,"dz"),reset;cycle_map=mapping) :
            FlowPath(read(f,"dx"),read(f,"dy"),read(f,"dz"),reset)
        curve=TimeCurve(t=Float32.(read(f,"time")),t_unit=Float32.(read(f,"time_unit")),
            periodic=true,periods=1f0,t_start=Float32(info["t_start_s"]),t_end=Float32(info["t_end_s"]))
        phantom=Phantom(name="E046_"*info["case"]*"_"*arm,x=read(f,"x"),y=read(f,"y"),z=read(f,"z"),
            ρ=read(f,"rho"),T1=read(f,"T1"),T2=read(f,"T2"),Δw=read(f,"delta_w"),
            motion=Motion(action=action,time=curve,spins=AllSpins()))
        return phantom,info
    end
end

function check_timing(sequence,e,case_dir)
    starts=get_block_start_times(sequence);labels=get_labels(sequence)
    blocks=findall(a->a.N>0,sequence.ADC)
    first=Float64[starts[i]+sequence.ADC[i].delay for i in blocks]
    ds=Float64[dwell(sequence.ADC[i]) for i in blocks]
    ns=Int[sequence.ADC[i].N for i in blocks]
    labs=hcat([[Int(labels[i].LIN),Int(labels[i].PAR),Int(labels[i].PHS)] for i in blocks]...)
    h5open(joinpath(case_dir,"legacy_timing.h5"),"r") do f
        g=f[e]
        maximum(abs.(first.-read(g,"adc_first_s")))<1e-9||error("ADC timing changed")
        maximum(abs.(ds.-read(g,"dwell_s")))<1e-13||error("ADC dwell changed")
        ns==read(g,"n")&&labs==read(g,"labels")||error("ADC labels/counts changed")
    end
    return length(blocks)
end

function main(arm,case_dir,map_dir,output)
    arm in ("legacy","updated_nomap","updated_map")||error("Unknown arm")
    ispath(output)&&error("Refuse overwrite");mkpath(output)
    modern=isdefined(KomaMRICore,:SpinResetState)
    (arm=="legacy")==!modern||error("Wrong runtime for arm")
    gates=joinpath(E046ROOT,"outputs/upstream_migration_e046")
    for (folder,file,status) in [("api_preflight_v2","environment_preflight.toml","PASS_UPSTREAM_API_CPU_GPU_PREFLIGHT"),
        ("synthetic_controls_v1","synthetic_report.toml","PASS_SYNTHETIC_STATIC_AND_FLOW_CONTROLS"),
        ("reset_events_v2","reset_events.toml","PASS_CPU_GPU_INTERIOR_RESET_EVENTS")]
        TOML.parsefile(joinpath(gates,folder,file))["status"]==status||error("Migration prerequisite failed")
    end
    get(ENV,"CUDA_VISIBLE_DEVICES","")=="GPU-8be27592-e325-7b7a-f432-b84f69cb18c6"||error("Reviewed GPU1 UUID required")
    occursin(hashfile(joinpath(map_dir,"cycle_map.h5")),read(joinpath(map_dir,"map_report.json"),String))||error("Map provenance changed")
    open(joinpath(output,"registration.toml"),"w") do io
        TOML.print(io,Dict("status"=>"REGISTERED_BEFORE_CANDIDATE_SIGNAL","arm"=>arm,
            "case_info_sha256"=>hashfile(joinpath(case_dir,"case_info.toml")),
            "input_sha256"=>hashfile(joinpath(case_dir,"case_inputs.h5")),
            "map_sha256"=>hashfile(joinpath(map_dir,"cycle_map.h5")),
            "runtime_manifest_sha256"=>hashfile(joinpath(dirname(Base.active_project()),"Manifest.toml")),
            "script_sha256"=>hashfile(@__FILE__)))
    end
    phantom,info=load_control(case_dir,map_dir,arm)
    length(phantom)==20001||error("Only20k physical spins plus timing sentinel authorized")
    seqs=case_sequences(info["case"],info)
    timings=Dict{String,Any}();runtimes=Dict{String,Float64}()
    params=Dict{String,Any}("gpu"=>true,"gpu_device"=>0,"Nthreads"=>1,"precision"=>"f32",
        "return_type"=>"mat","Δt"=>Inf,"max_block_length"=>512,
        "sim_method"=>(modern ? MigrationPerfectSpoiling() : BlochWithPerfectSpoiling()))
    scanner=modern ? Scanner(limits=HardwareLimits(B0=info["B0_T"])) : Scanner(B0=info["B0_T"])
    (modern ? scanner.limits.B0 : scanner.B0)==info["B0_T"]||error("Scanner B0 changed")
    for e in ("REF","X","Y","Z")
        seq=seqs[e];timings[e]=check_timing(seq,e,case_dir)
        if arm=="legacy"
            reset_schedule!(phantom.motion,seq);PREVIOUS_PART_RF[]=false
        end
        MIGRATION_PREVIOUS_RF[]=false
        println("START_E046 ",info["case"]," ",arm," ",e);flush(stdout)
        start=time();signal=simulate(phantom,seq,scanner;sim_params=params,verbose=false)
        all(isfinite,signal)||error("Nonfinite raw signal")
        k,image=sorted_image(signal,seq);runtimes[e]=time()-start
        target=joinpath(output,lowercase(e)*"_result.h5")
        h5open(target*".partial","w") do f
            write(f,"rawsort",k);write(f,"image",image)
            HDF5.attributes(f)["elapsed_seconds"]=runtimes[e]
        end
        mv(target*".partial",target)
        println("FINISHED_E046 ",info["case"]," ",arm," ",e," seconds=",runtimes[e]);flush(stdout)
        GC.gc();CUDA.reclaim()
    end
    report=Dict("status"=>"COMPLETE_E046_CONTROL","case"=>info["case"],"arm"=>arm,
        "physical_particles"=>20000,"total_spins"=>length(phantom),"coupled_spin_partitioning"=>false,
        "case_info_sha256"=>hashfile(joinpath(case_dir,"case_info.toml")),
        "input_sha256"=>hashfile(joinpath(case_dir,"case_inputs.h5")),
        "map_sha256"=>hashfile(joinpath(map_dir,"cycle_map.h5")),
        "runtime_core_version"=>string(pkgversion(KomaMRICore)),
        "manifest_sha256"=>hashfile(joinpath(dirname(Base.active_project()),"Manifest.toml")),
        "source_sequence_sha256"=>info["sequence_sha256"],"actual_ADC_timing_and_labels_match_legacy"=>true,
        "encoding_profile_count"=>timings,"encoding_runtime_seconds"=>runtimes,
        "script_sha256"=>hashfile(@__FILE__),"perfect_spoiling_sha256"=>hashfile(joinpath(@__DIR__,"perfect_spoiling.jl")),
        "direct_sort_sha256"=>hashfile(joinpath(@__DIR__,"direct_sort.jl")),
        "export_helpers_sha256"=>hashfile(joinpath(@__DIR__,"export_controls.jl")),
        "result_sha256"=>Dict(e=>hashfile(joinpath(output,lowercase(e)*"_result.h5")) for e in keys(seqs)),
        "physical_gpu_uuid"=>get(ENV,"CUDA_VISIBLE_DEVICES","UNSET"),"visible_device"=>0)
    open(joinpath(output,"control_report.toml.partial"),"w") do io;TOML.print(io,report);end
    mv(joinpath(output,"control_report.toml.partial"),joinpath(output,"control_report.toml"))
end

length(ARGS)==4||error("Usage: run_control.jl ARM CASE_DIR MAP_DIR NEW_OUTPUT")
if ARGS[1]=="legacy"
    # Exact preserved numerical corrections used by E045; not loaded upstream.
    include(joinpath(E046ROOT,"koma_aorta/native_zero_pilot.jl"))
end
main(ARGS...)
