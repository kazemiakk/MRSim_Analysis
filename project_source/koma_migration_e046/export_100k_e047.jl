using KomaMRI, KomaMRICore, HDF5, JLD2, FFTW, TOML, SHA
include(joinpath(@__DIR__, "export_controls.jl"))

"""Export the complete frozen100k case and existing legacy complex results."""
function export100k(case, output)
    ispath(output) && error("Refuse overwrite: $output")
    old = joinpath(E046ROOT,"outputs/upstream_migration_e046",
        case=="aorta" ? "aorta_inputs_v1" : "phantom_inputs_v3")
    info = TOML.parsefile(joinpath(old,"case_info.toml"))
    info["case"]==case || error("Case mismatch")
    hashfile(info["source_phantom"])==info["source_phantom_sha256"] || error("Source changed")
    original = read_phantom(info["source_phantom"])
    length(original)==100000 || error("Complete frozen100k source required")
    if case=="phantom"
        ref=read_seq(joinpath(info["sequence_directory"],"ref.seq");verbose=false,verify_signature=true)
        original.motion,_,_=rescale_periodic_motion(original.motion,ref.DEF["RR"])
        original.motion=align_periodic_motion_seam_to_first_adc_midbin(
            original.motion,first_adc_kspace_center(ref),ref.DEF["RR"],15).motion
    end
    Float64(original.motion.time.t_start)==info["t_start_s"] || error("Time origin mismatch")
    Float64(original.motion.time.t_end)==info["t_end_s"] || error("Time end mismatch")
    a=original.motion.action; n=length(original); steps=size(a.dx,2)
    all_columns=findall(vec(all(a.spin_reset;dims=1)))
    all_columns==[steps] || error("Unexpected artificial reset columns")
    h5open(joinpath(old,"case_inputs.h5"),"r") do f
        for (key,values) in [("x",original.x),("y",original.y),("z",original.z),
            ("rho",original.ρ),("T1",original.T1),("T2",original.T2),("delta_w",original.Δw)]
            read(f,key)[1:20000]==Float32.(values[1:20000]) || error("Subset changed: $key")
        end
        for (key,values) in [("dx",a.dx),("dy",a.dy),("dz",a.dz)]
            read(f,key)[1:20000,:]==values[1:20000,:] || error("Subset paths changed")
        end
        Bool.(read(f,"reset"))[1:20000,:]==a.spin_reset[1:20000,:] || error("Subset resets changed")
    end
    mkpath(output)
    info["experiment"]="E-047"; info["physical_particles"]=n;info["total_spins_with_sentinel"]=n+1
    info["exporter_sha256"]=hashfile(@__FILE__)
    info["frozen_20k_case_info_sha256"]=hashfile(joinpath(old,"case_info.toml"))
    info["legacy_baseline_reused_after_identity_checks"]=true
    target=joinpath(output,"case_inputs.h5"); zero_path=zeros(Float32,1,steps)
    h5open(target*".partial","w") do f
        for (key,values,extra) in [("x",original.x,0f0),("y",original.y,0f0),("z",original.z,0f0),
            ("rho",original.ρ,0f0),("T1",original.T1,1.3f0),("T2",original.T2,.1f0),("delta_w",original.Δw,0f0)]
            write(f,key,vcat(Float32.(values),extra))
        end
        for (key,values) in [("dx",a.dx),("dy",a.dy),("dz",a.dz)]
            write(f,key,vcat(values,zero_path))
        end
        write(f,"reset",UInt8.(vcat(a.spin_reset,reshape(vec(any(a.spin_reset;dims=1)),1,:))))
        write(f,"time",original.motion.time.t);write(f,"time_unit",original.motion.time.t_unit)
        HDF5.attributes(f)["physical_particles"]=n
        HDF5.attributes(f)["t_start_s"]=info["t_start_s"];HDF5.attributes(f)["t_end_s"]=info["t_end_s"]
    end
    mv(target*".partial",target);info["neutral_input_sha256"]=hashfile(target)
    cp(joinpath(old,"legacy_timing.h5"),joinpath(output,"legacy_timing.h5"))
    info["legacy_timing_sha256"]=hashfile(joinpath(output,"legacy_timing.h5"))
    source=info["source_output"]; report=TOML.parsefile(joinpath(source,"simulation_report.toml"))
    identities=Dict{String,String}()
    for e in ("REF","X","Y","Z")
        sourcefile=joinpath(source,lowercase(e)*"_result.jld2")
        data=JLD2.load(sourcefile)
        data["encoding"]==e || error("Encoding mismatch")
        data["phantom_sha256"]==info["source_phantom_sha256"] || error("Legacy phantom mismatch")
        data["sequence_sha256"]==info["sequence_sha256"][e] || error("Legacy sequence mismatch")
        hashfile(joinpath(info["sequence_directory"],lowercase(e)*".seq"))==info["sequence_sha256"][e] || error("Sequence changed")
        if case=="aorta"
            hashfile(sourcefile)==report["raw_result_sha256"][e] || error("Legacy result changed")
        else
            data["effective_sequence_transform_signature"]==report["effective_sequence_transform_signature"] || error("Transform mismatch")
            expected="periodic_motion_s=$(info["period_s"]);seam_safe_transform=$(report["effective_sequence_transform_signature"])"
            data["motion_timing_signature"]==expected || error("Legacy timing mismatch")
            data["batch_metadata_for_encoding"]["batch_count"]==5 || error("Not complete100k")
        end
        all(isfinite,data["rawsort"]) && all(isfinite,data["image"]) || error("Nonfinite legacy result")
        target=joinpath(output,lowercase(e)*"_legacy_result.h5")
        h5open(target*".partial","w") do f
            write(f,"rawsort",ComplexF32.(data["rawsort"]));write(f,"image",ComplexF32.(data["image"]))
            HDF5.attributes(f)["source_result_sha256"]=hashfile(sourcefile)
            HDF5.attributes(f)["elapsed_seconds"]=data["elapsed_seconds"]
        end
        mv(target*".partial",target);identities[e]=hashfile(sourcefile)
        println("EXPORTED_E047_BASELINE ",case," ",e);flush(stdout);GC.gc()
    end
    info["legacy_result_sha256"]=identities;info["source_simulation_report_sha256"]=hashfile(joinpath(source,"simulation_report.toml"))
    info["status"]="PASS_FROZEN_100K_CASE_EXPORT"
    open(joinpath(output,"case_info.toml.partial"),"w") do io;TOML.print(io,info);end
    mv(joinpath(output,"case_info.toml.partial"),joinpath(output,"case_info.toml"))
    println("PASS_FROZEN_100K_CASE_EXPORT ",case);flush(stdout)
end

length(ARGS)==2 || error("CASE NEW_OUTPUT required")
ARGS[1] in ("aorta","phantom") || error("Unknown case")
export100k(ARGS...)
