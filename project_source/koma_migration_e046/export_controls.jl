using KomaMRI, KomaMRICore, HDF5, JLD2, FFTW, TOML, SHA
const E046ROOT=normpath(joinpath(@__DIR__,".."))
include(joinpath(E046ROOT,"koma_dp150","motion_timing.jl"))
hashfile(p)=bytes2hex(sha256(read(p)))

"""Same zero-Float32 RF markers; verify only the actually edited delay blocks.

GR/ADC/DUR/EXT arrays are never assigned or modified. This avoids the legacy
full-array approximate-event comparison, while retaining every marker check.
"""
function insert_verified_zero_markers!(seq,seam,period,count;lead=350e-6,duration=2e-6)
    starts=get_block_start_times(seq);used=Set{Int}()
    for k in 0:count-1
        time=seam+k*period-lead;b=searchsortedlast(starts,time)
        1<=b<=length(seq)||error("Marker outside sequence")
        b in used&&error("Duplicate marker block")
        seq.ADC[b].N==0&&isempty(seq.EXT[b])||error("Marker is not in empty delay")
        zero_value(v)=v isa Number ? iszero(v) : all(iszero,v)
        zero_value(seq.RF[1,b].A)||error("Marker overlaps physical RF")
        all(zero_value(seq.GR[d,b].A) for d in 1:3)||error("Marker overlaps gradient")
        local_delay=time-starts[b]
        0<local_delay&&local_delay+duration<seq.DUR[b]||error("Marker outside delay")
        seq.RF[1,b]=RF(nextfloat(0.0),duration,0.0,local_delay)
        Float32(seq.RF[1,b].A)==0||error("Nonzero physical marker")
        push!(used,b)
    end
    length(used)==count||error("Missing markers")
    return seq
end

function case_sequences(case,info)
    seqs=Dict{String,Any}()
    for e in ("REF","X","Y","Z")
        seq=read_seq(joinpath(info["sequence_directory"],lowercase(e)*".seq");verbose=false,verify_signature=true)
        if case=="phantom"
            seq=insert_verified_zero_markers!(seq,info["seam_time_s"],info["period_s"],2048)
        end
        seqs[e]=seq
    end
    return seqs
end

function sorted_image(signal,seq)
    raw=KomaMRICore.signal_to_raw_data(signal,seq)
    shape=Int.([seq.DEF["Nx"],seq.DEF["Ny"],seq.DEF["Nz"],seq.DEF["CardiacBins"]])
    k=zeros(ComplexF32,shape...);filled=falses(shape[2:end]...)
    for p in raw.profiles
        y=Int(p.head.idx.kspace_encode_step_1)+1;z=Int(p.head.idx.kspace_encode_step_2)+1;b=Int(p.head.idx.phase)+1
        filled[y,z,b]&&error("Duplicate profile");filled[y,z,b]=true;k[:,y,z,b].=p.data[:,1]
    end
    all(filled)||error("Incomplete profile set")
    image=ComplexF32.(fftshift(ifft(ifftshift(k,(1,2,3)),(1,2,3)),(1,2,3)))
    return k,image
end

function export_case(case,output)
    ispath(output)&&error("Refuse overwrite: $output");mkpath(output)
    n=20000
    if case=="aorta"
        source=joinpath(E046ROOT,"outputs/aorta_inlet_replenishment/e045_native0_100k_mri_v1")
        report=TOML.parsefile(joinpath(source,"phantom_report.toml"))
        phantomfile=joinpath(source,"inlet_pilot.phantom")
        hashfile(phantomfile)==report["phantom_sha256"]||error("Source phantom changed")
        original=read_phantom(phantomfile)
        info=Dict{String,Any}("sequence_directory"=>joinpath(E046ROOT,"outputs/aorta_inlet_replenishment/e045_native0_sequences_v4"),
            "period_s"=>Float64(original.motion.time.t_end-original.motion.time.t_start),
            "seam_time_s"=>Float64(original.motion.time.t_start),"B0_T"=>1.5,"Venc_m_s"=>2.5)
    elseif case=="phantom"
        source=joinpath(E046ROOT,"outputs/dp150_mri_pipeline/koma/simulation_motion_100k_scanner_matched_venc5_seam_safe_v3")
        report=TOML.parsefile(joinpath(source,"simulation_report.toml"))
        phantomfile=joinpath(E046ROOT,"outputs/dp150_mri_pipeline/koma/pilot_100k/dp150_motion_100k.phantom")
        hashfile(phantomfile)==report["phantom_sha256"]||error("Source phantom changed")
        hashfile(joinpath(E046ROOT,"koma_dp150/motion_timing.jl"))==report["motion_timing_script_sha256"]||error("Legacy timing implementation changed")
        original=read_phantom(phantomfile)
        seqdir=joinpath(E046ROOT,"koma_dp150/pc_3d_scanner_matched_venc5")
        ref=read_seq(joinpath(seqdir,"ref.seq");verbose=false,verify_signature=true)
        original.motion,_,_=rescale_periodic_motion(original.motion,ref.DEF["RR"])
        alignment=align_periodic_motion_seam_to_first_adc_midbin(original.motion,first_adc_kspace_center(ref),ref.DEF["RR"],15)
        original.motion=alignment.motion
        info=Dict{String,Any}("sequence_directory"=>seqdir,
            "period_s"=>Float64(original.motion.time.t_end-original.motion.time.t_start),
            "seam_time_s"=>alignment.realized_seam_time_s,"B0_T"=>3.0,"Venc_m_s"=>5.0)
    else
        error("Unknown case")
    end
    length(original)==100000||error("Expected frozen100k source")
    a=original.motion.action;global_schedule=vec(any(a.spin_reset;dims=1))
    all(a.spin_reset[:,end])||error("No explicit legacy all-spin cycle-end reset")
    all_columns=findall(vec(all(a.spin_reset;dims=1)))
    all_columns==[size(a.spin_reset,2)]||error("Unexpected additional all-spin reset columns")
    steps=size(a.dx,2);zero_path=zeros(Float32,1,steps)
    info["case"]=case;info["physical_particles"]=n;info["total_spins_with_sentinel"]=n+1
    info["source_phantom"]=phantomfile;info["source_phantom_sha256"]=hashfile(phantomfile)
    info["source_output"]=source;info["t_start_s"]=Float64(original.motion.time.t_start)
    info["t_end_s"]=Float64(original.motion.time.t_end)
    info["artificial_global_reset_columns_one_based"]=all_columns
    info["exporter_sha256"]=hashfile(@__FILE__)
    target=joinpath(output,"case_inputs.h5")
    h5open(target*".partial","w") do f
        for (name,values,extra) in [("x",original.x,0f0),("y",original.y,0f0),("z",original.z,0f0),
            ("rho",original.ρ,0f0),("T1",original.T1,1.3f0),("T2",original.T2,.1f0),("delta_w",original.Δw,0f0)]
            write(f,name,vcat(Float32.(values[1:n]),extra))
        end
        write(f,"dx",vcat(a.dx[1:n,:],zero_path));write(f,"dy",vcat(a.dy[1:n,:],zero_path));write(f,"dz",vcat(a.dz[1:n,:],zero_path))
        write(f,"reset",UInt8.(vcat(a.spin_reset[1:n,:],reshape(global_schedule,1,:))))
        write(f,"time",original.motion.time.t);write(f,"time_unit",original.motion.time.t_unit)
        HDF5.attributes(f)["physical_particles"]=n
        HDF5.attributes(f)["t_start_s"]=Float64(original.motion.time.t_start)
        HDF5.attributes(f)["t_end_s"]=Float64(original.motion.time.t_end)
    end
    mv(target*".partial",target);info["neutral_input_sha256"]=hashfile(target)
    seqs=case_sequences(case,info)
    info["sequence_sha256"]=Dict(e=>hashfile(joinpath(info["sequence_directory"],lowercase(e)*".seq")) for e in keys(seqs))
    if case=="phantom"
        for e in ("REF","X","Y","Z")
            hashfile(joinpath(info["sequence_directory"],lowercase(e)*".seq"))==report["sequence"][e]["sha256"]||error("Legacy source sequence changed")
            file=joinpath(source,lowercase(e)*"_batches","batch_0001_of_0005.jld2")
            data=JLD2.load(file)
            data["phantom_sha256"]==report["phantom_sha256"]||error("Batch phantom mismatch")
            data["sequence_sha256"]==report["sequence"][e]["sha256"]||error("Batch sequence mismatch")
            data["spin_range"]==[1,n]||error("Batch subset mismatch")
            expected="periodic_motion_s=$(info["period_s"]);seam_safe_transform=$(report["effective_sequence_transform_signature"])"
            data["motion_timing_signature"]==expected||error("Batch transform mismatch")
            k,image=sorted_image(data["signal"],seqs[e])
            target=joinpath(output,lowercase(e)*"_legacy_result.h5")
            h5open(target*".partial","w") do f
                write(f,"rawsort",k);write(f,"image",image)
                HDF5.attributes(f)["source_batch_sha256"]=hashfile(file)
                HDF5.attributes(f)["elapsed_seconds"]=data["elapsed_seconds"]
            end
            mv(target*".partial",target)
        end
        info["legacy_baseline_reused_after_identity_checks"]=true
    else
        info["legacy_baseline_reused_after_identity_checks"]=false
    end
    # Exact legacy ADC starts/dwells/labels for independent cross-version timing checks.
    h5open(joinpath(output,"legacy_timing.h5"),"w") do f
        for e in ("REF","X","Y","Z")
            seq=seqs[e];starts=get_block_start_times(seq);labels=get_labels(seq)
            blocks=findall(adc->adc.N>0,seq.ADC);g=create_group(f,e)
            write(g,"adc_first_s",Float64[starts[i]+seq.ADC[i].delay for i in blocks])
            write(g,"dwell_s",Float64[dwell(seq.ADC[i]) for i in blocks])
            write(g,"n",Int[seq.ADC[i].N for i in blocks])
            write(g,"labels",hcat([[Int(labels[i].LIN),Int(labels[i].PAR),Int(labels[i].PHS)] for i in blocks]...))
        end
    end
    info["status"]="PASS_FROZEN_CASE_EXPORT"
    open(joinpath(output,"case_info.toml"),"w") do io;TOML.print(io,info);end
    println("PASS_FROZEN_CASE_EXPORT ",case);flush(stdout)
end

if abspath(PROGRAM_FILE)==@__FILE__
    length(ARGS)==2||error("Usage: export_controls.jl CASE NEW_OUTPUT")
    export_case(ARGS...)
end
