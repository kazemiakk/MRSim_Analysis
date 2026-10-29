using KomaMRI, KomaMRICore, CUDA, HDF5, JLD2, FFTW, TOML, SHA
include(joinpath(@__DIR__,"export_controls.jl"))
include(joinpath(@__DIR__,"direct_sort.jl"))
include(joinpath(@__DIR__,"perfect_spoiling.jl"))
include(joinpath(@__DIR__,"sorted_event_lookup.jl"))

function run100k(case_dir,map_dir,output)
    ispath(output) && error("Refuse overwrite")
    info=TOML.parsefile(joinpath(case_dir,"case_info.toml"))
    info["status"]=="PASS_FROZEN_100K_CASE_EXPORT" && info["physical_particles"]==100000 || error("Frozen100k only")
    hashfile(joinpath(case_dir,"case_inputs.h5"))==info["neutral_input_sha256"] || error("Input changed")
    hashfile(joinpath(case_dir,"legacy_timing.h5"))==info["legacy_timing_sha256"] || error("Timing changed")
    gates=joinpath(E046ROOT,"outputs/upstream_migration_e046")
    for (folder,file,status) in [("api_preflight_v2","environment_preflight.toml","PASS_UPSTREAM_API_CPU_GPU_PREFLIGHT"),
        ("synthetic_controls_v1","synthetic_report.toml","PASS_SYNTHETIC_STATIC_AND_FLOW_CONTROLS"),
        ("reset_events_v2","reset_events.toml","PASS_CPU_GPU_INTERIOR_RESET_EVENTS"),
        ("event_lookup_v1","lookup_equivalence.toml","PASS_EXACT_SORTED_EVENT_LOOKUP")]
        TOML.parsefile(joinpath(gates,folder,file))["status"]==status || error("Prerequisite failed")
    end
    lookup=TOML.parsefile(joinpath(gates,"event_lookup_v1/lookup_equivalence.toml"))
    hashfile(joinpath(@__DIR__,"sorted_event_lookup.jl"))==lookup["helper_sha256"] || error("Lookup changed")
    get(ENV,"CUDA_VISIBLE_DEVICES","")=="GPU-8be27592-e325-7b7a-f432-b84f69cb18c6" || error("ReviewedGPU1 required")
    mapfile=joinpath(map_dir,"cycle_map.h5")
    occursin(hashfile(mapfile),read(joinpath(map_dir,"map_report.json"),String)) || error("Map changed")
    phantom=h5open(joinpath(case_dir,"case_inputs.h5"),"r") do f
        reset=Bool.(read(f,"reset"))
        mapping,restored=h5open(mapfile,"r") do m
            Int.(read(m,"cycle_map")),Bool.(read(m,"restored_endpoint_reset"))
        end
        length(mapping)==100001 && all(1 .<= mapping .<= 100001) || error("Invalid map")
        mapping[end]==100001 && all(mapping[1:end-1].<=100000) || error("Sentinel contamination")
        reset[:,end].=restored
        action=FlowPath(read(f,"dx"),read(f,"dy"),read(f,"dz"),reset;cycle_map=mapping)
        curve=TimeCurve(t=Float32.(read(f,"time")),t_unit=Float32.(read(f,"time_unit")),
            periodic=true,periods=1f0,t_start=Float32(info["t_start_s"]),t_end=Float32(info["t_end_s"]))
        Phantom(name="E047_"*info["case"]*"_100k_map",x=read(f,"x"),y=read(f,"y"),z=read(f,"z"),
            ρ=read(f,"rho"),T1=read(f,"T1"),T2=read(f,"T2"),Δw=read(f,"delta_w"),
            motion=Motion(action=action,time=curve,spins=AllSpins()))
    end
    length(phantom)==100001 && phantom.ρ[end]==0 || error("Wrong spin count")
    mkpath(output)
    registration=Dict("status"=>"REGISTERED_BEFORE_CANDIDATE_SIGNAL","experiment"=>"E-047",
        "case_info_sha256"=>hashfile(joinpath(case_dir,"case_info.toml")),
        "input_sha256"=>info["neutral_input_sha256"],"map_sha256"=>hashfile(mapfile),
        "script_sha256"=>hashfile(@__FILE__),"manifest_sha256"=>hashfile(joinpath(@__DIR__,"Manifest.toml")))
    open(joinpath(output,"registration.toml"),"w") do io;TOML.print(io,registration);end
    seqs=case_sequences(info["case"],info);runtimes=Dict{String,Float64}();counts=Dict{String,Int}()
    params=Dict{String,Any}("gpu"=>true,"gpu_device"=>0,"Nthreads"=>1,"precision"=>"f32",
        "return_type"=>"mat","Δt"=>Inf,"max_block_length"=>512,"sim_method"=>MigrationPerfectSpoiling())
    scanner=Scanner(limits=HardwareLimits(B0=info["B0_T"]))
    scanner.limits.B0==info["B0_T"] || error("B0 changed")
    for e in ("REF","X","Y","Z")
        seq=seqs[e];starts=get_block_start_times(seq);labels=get_labels(seq);blocks=findall(a->a.N>0,seq.ADC)
        h5open(joinpath(case_dir,"legacy_timing.h5"),"r") do f
            g=f[e]
            maximum(abs.(Float64[starts[i]+seq.ADC[i].delay for i in blocks].-read(g,"adc_first_s")))<1e-9 || error("ADC timing changed")
            maximum(abs.(Float64[dwell(seq.ADC[i]) for i in blocks].-read(g,"dwell_s")))<1e-13 || error("Dwell changed")
            Int[seq.ADC[i].N for i in blocks]==read(g,"n") || error("Counts changed")
            hcat([[Int(labels[i].LIN),Int(labels[i].PAR),Int(labels[i].PHS)] for i in blocks]...)==read(g,"labels") || error("Labels changed")
        end
        counts[e]=length(blocks);MIGRATION_PREVIOUS_RF[]=false
        println("START_E047 ",info["case"]," ",e);flush(stdout);start=time()
        signal=simulate(phantom,seq,scanner;sim_params=params,verbose=false)
        all(isfinite,signal) || error("Nonfinite signal")
        k,image=sorted_image(signal,seq);runtimes[e]=time()-start
        target=joinpath(output,lowercase(e)*"_result.h5")
        h5open(target*".partial","w") do f
            write(f,"rawsort",k);write(f,"image",image)
            HDF5.attributes(f)["elapsed_seconds"]=runtimes[e]
        end
        mv(target*".partial",target)
        println("FINISHED_E047 ",info["case"]," ",e," seconds=",runtimes[e]);flush(stdout)
        GC.gc();CUDA.reclaim()
    end
    report=merge(registration,Dict("status"=>"COMPLETE_E047_100K_REMAPPED","case"=>info["case"],
        "physical_particles"=>100000,"total_spins"=>100001,"coupled_spin_partitioning"=>false,
        "arm"=>"updated_map","actual_ADC_timing_and_labels_match_legacy"=>true,
        "runtime_core_version"=>string(pkgversion(KomaMRICore)),"physical_gpu_uuid"=>ENV["CUDA_VISIBLE_DEVICES"],
        "encoding_profile_count"=>counts,"encoding_runtime_seconds"=>runtimes,
        "result_sha256"=>Dict(e=>hashfile(joinpath(output,lowercase(e)*"_result.h5")) for e in keys(seqs)),
        "helper_sha256"=>Dict(file=>hashfile(joinpath(@__DIR__,file)) for file in
            ("export_controls.jl","direct_sort.jl","perfect_spoiling.jl","sorted_event_lookup.jl"))))
    open(joinpath(output,"control_report.toml.partial"),"w") do io;TOML.print(io,report);end
    mv(joinpath(output,"control_report.toml.partial"),joinpath(output,"control_report.toml"))
    println("COMPLETE_E047_100K_REMAPPED ",info["case"]);flush(stdout)
end

length(ARGS)==3 || error("CASE_DIR MAP_DIR NEW_OUTPUT required")
run100k(ARGS...)
