using KomaMRI, KomaMRICore, CUDA, HDF5, JLD2, FFTW, TOML, Statistics, SHA
include(joinpath(@__DIR__, "perfect_spoiling.jl"))

const ROOT = normpath(joinpath(@__DIR__, ".."))
const SEQDIR = joinpath(ROOT, "outputs", "aorta_dp2_mri_pipeline", "koma", "multibin_8x4mm", "sequences")
const CENTER = Float32[.038389891386032104, .015447176992893219, -.023723918944597244]
const ORDER = [3,1,2]
const RESET_KEYTIMES = Ref(Float64[])
const PREVIOUS_PART_RF = Ref(false)
const RESET_MARKER_LEAD_S = 5f-5

# Koma's stock reset key offset is 1e-14 s, which disappears in Float32.
# Put a numerical key safely inside the existing Constant-Next reset window;
# this changes no RF/gradient/ADC waveform or acquisition label/time.
function KomaMRI.KomaMRIBase.add_reset_times!(t,a::FlowPath,t_start,t_end,periods)
    indices=getindex.(findall(a.spin_reset.==1),2)
    aux=t_start.+(t_end-t_start)/(size(a.spin_reset,2)-1).*(indices.-1)
    append!(t,KomaMRI.KomaMRIBase.times(aux,t_start,t_end,periods).-RESET_MARKER_LEAD_S)
end

# GPU simulation is single-threaded at this orchestration layer. Preserve RF
# coherence across reset-induced subparts; ideal spoiling belongs only at the
# start of a physical RF interval, not at an artificial numerical cut.
function KomaMRICore.run_spin_excitation!(p::Phantom{T},s::DiscreteSequence{T},sig::AbstractArray{Complex{T}},
    M::KomaMRICore.Mag{T},::BlochWithPerfectSpoiling,groupsize,backend::KomaMRICore.KA.Backend,pre::KomaMRICore.PreallocResult{T}) where T<:Real
    PREVIOUS_PART_RF[] || fill!(M.xy,zero(T))
    PREVIOUS_PART_RF[]=true
    return KomaMRICore.run_spin_excitation!(p,s,sig,M,Bloch(),groupsize,backend,pre)
end
function KomaMRICore.run_spin_precession!(p::Phantom{T},s::DiscreteSequence{T},sig::AbstractArray{Complex{T}},
    M::KomaMRICore.Mag{T},::BlochWithPerfectSpoiling,groupsize,backend::KomaMRICore.KA.Backend,pre::KomaMRICore.PreallocResult{T}) where T<:Real
    PREVIOUS_PART_RF[]=false
    return KomaMRICore.run_spin_precession!(p,s,sig,M,Bloch(),groupsize,backend,pre)
end

function reset_schedule!(motion,sequence)
    schedule=vec(any(motion.action.spin_reset;dims=1))
    z=zeros(Float32,1,length(schedule))
    probe=Motion(action=FlowPath(dx=z,dy=z,dz=z,spin_reset=reshape(schedule,1,:)),time=motion.time,spins=AllSpins())
    times=Float64[0,Float64(dur(sequence))]
    KomaMRI.KomaMRIBase.add_key_time_points!(times,probe)
    RESET_KEYTIMES[]=sort!(unique(times))
end

# Process-local, source-bound partition specialization. RF blocks are unchanged;
# precession blocks are additionally ended at actual FlowPath key times. No
# Koma installation, RF/ADC/gradient waveform, or saved sequence file is edited.
function KomaMRICore.get_sim_ranges(seqd::DiscreteSequence;max_block_length=Inf,max_rf_block_length=Inf,eval_intervals_per_step=1)
    ranges=UnitRange{Int}[];flags=Bool[]
    isempty(seqd.Δt) && return ranges,flags
    starts=[firstindex(seqd.Δt);findall(seqd.excitation_bool[2:end].!=seqd.excitation_bool[1:end-1]).+1]
    stops=[starts[2:end].-1;lastindex(seqd.Δt)]
    cuts=unique(clamp.(searchsortedfirst.(Ref(seqd.t),RESET_KEYTIMES[]),1,length(seqd.t)))
    for (start,stop) in zip(starts,stops)
        rf=seqd.excitation_bool[start]
        coarse=KomaMRICore.split_range(start:stop+1,rf ? max_rf_block_length : max_block_length,rf ? eval_intervals_per_step : 1)
        for r in coarse
            inside=cuts[searchsortedlast(cuts,first(r))+1:searchsortedfirst(cuts,last(r))-1]
            bounds=[first(r);inside;last(r)]
            for k in 1:length(bounds)-1
                bounds[k]<bounds[k+1] || continue
                push!(ranges,bounds[k]:bounds[k+1]);push!(flags,rf)
            end
        end
    end
    return ranges,flags
end
filehash(path) = open(io -> bytes2hex(sha256(io)), path)
function write_toml(path, report)
    isfile(path) && error("Refusing overwrite: $path")
    open(path * ".partial", "w") do io
        TOML.print(io, report)
    end
    mv(path * ".partial", path)
end

function build_phantom(track_dir, output)
    ispath(output) && error("Output exists: $output")
    occursin("\"status\": \"PASS_TRACKING_CONTROL\"",read(joinpath(track_dir,"tracking_report.json"),String)) || error("Tracking gates did not pass")
    mkpath(output)
    track = joinpath(track_dir, "tracks.h5")
    sequences = Dict(e => read_seq(joinpath(SEQDIR, lowercase(e)*".seq"); verbose=false, verify_signature=true) for e in ("REF","X","Y","Z"))
    rr = Float32(sequences["REF"].DEF["RR"])
    n,steps,paths,reset = h5open(track,"r") do f
        n = Int(read_attribute(f,"n_particles")); steps = Int(read_attribute(f,"n_steps"))
        n in (20000,100000) || error("Pilot only; refusing a production count")
        steps == 360 || error("Expected one recorded 720ms cycle")
        initial = read(f,"initial_position")
        position = read(f,"position")
        reset = falses(n,steps+1); reset[:,2:end] .= Bool.(read(f,"reset"))
        reset[:,end] .= true
        paths = ntuple(3) do k
            a = Matrix{Float32}(undef,n,steps+1)
            a[:,1] .= initial[ORDER[k],:] .- CENTER[ORDER[k]]
            a[:,2:end] .= position[ORDER[k],:,:] .- CENTER[ORDER[k]]
            a
        end
        return n,steps,paths,reset
    end
    fov = Float64.(sequences["REF"].DEF["FOV"])
    margins = fov ./ 2 .- [maximum(abs,p) for p in paths]
    all(margins .> 0) || error("Corrected trajectory does not fit preserved FOV")
    motion = Motion(action=FlowPath(dx=paths[1],dy=paths[2],dz=paths[3],spin_reset=reset),
                    time=Periodic(period=rr,asymmetry=1f0),spins=AllSpins())
    phantom = Phantom(name="inferred_inlet_replenishment_pilot",x=zeros(Float32,n),y=zeros(Float32,n),z=zeros(Float32,n),
                      ρ=ones(Float32,n),T1=fill(1.3f0,n),T2=fill(.1f0,n),motion=motion)
    target=joinpath(output,"inlet_pilot.phantom")
    write_phantom(phantom,target*".partial");mv(target*".partial",target)
    report=Dict("status"=>"PASS_PILOT_PHANTOM", "particle_count"=>n,"tracks_sha256"=>filehash(track),
        "phantom_sha256"=>filehash(target),"fixed_source_center_m"=>Float64.(CENTER),"axis_order_one_based"=>ORDER,
        "source_cycle_s"=>.72,"realized_rr_s"=>Float64(rr),"minimum_fov_margin_m"=>margins,
        "reset_flags"=>count(reset),"cycle_end_reset_spins"=>n,
        "scope"=>"Approximate inferred-cap point-cloud recycling; native cap connectivity unverified",
        "builder_sha256"=>filehash(@__FILE__))
    write_toml(joinpath(output,"phantom_report.toml"),report)
    return phantom,sequences
end

function reset_preflight(phantom,sequences,output)
    # End precession parts at explicit reset keys without splitting every ADC
    # interval. RF parts and ideal spoiling remain unchanged.
    params=KomaMRICore.default_sim_params(Dict{String,Any}("sim_method"=>BlochWithPerfectSpoiling(),
        "precision"=>"f32","Δt"=>Inf,"max_block_length"=>512))
    schedule=vec(any(phantom.motion.action.spin_reset;dims=1))
    ntime=length(schedule);zero_path=zeros(Float32,1,ntime)
    probe=Motion(action=FlowPath(dx=zero_path,dy=zero_path,dz=zero_path,spin_reset=reshape(schedule,1,:)),
                 time=phantom.motion.time,spins=AllSpins())
    results=Dict{String,Any}()
    for encoding in ("REF","X","Y","Z")
        reset_schedule!(phantom.motion,sequences[encoding])
        rule=KomaMRICore.simulation_sampling_rule(params["sim_method"],params)
        seqd=discretize(sequences[encoding];sampling_rule=rule,motion=probe,freq_in_phase=params["freq_in_phase"])
        parts,excitation=KomaMRICore.get_sim_ranges(seqd;max_block_length=512,
            max_rf_block_length=params["max_rf_block_length"],eval_intervals_per_step=KomaMRICore.eval_intervals_per_step(params["sim_method"]))
        ends=Float32[seqd.t[last(p)] for p in parts]
        phases=vec(KomaMRI.KomaMRIBase.unit_time(ends',phantom.motion.time))
        # Verify Constant-Next indexing against Koma's own one-hot masks.
        bins=clamp.(ceil.(Int,phases .* (ntime-1)) .+ 1,1,ntime)
        checked=0;checked_events=0
        for j in findall(schedule)
            lower=Float32((j-2)/(ntime-1));upper=Float32((j-1)/(ntime-1))
            first_hit=findfirst(x -> lower+1f-7 < x < upper-1f-7, phases)
            isnothing(first_hit) && error("Reset source sample $j has no part-end coverage in $encoding")
            single=falses(1,ntime);single[1,j]=true
            actual=KomaMRICore.get_mask(single,reshape(phases[first_hit:first_hit],1,:))
            all(actual .== 1) || error("Reset indexing did not match actual Koma mask: sample=$j phase=$(phases[first_hit]) actual=$actual")
            checked+=1
            period=Float64(phantom.motion.time.t_end)
            for cycle in 0:floor(Int,dur(sequences[encoding])/period)
                begin_time=(cycle+Float64(lower))*period+2e-7
                end_time=(cycle+Float64(upper))*period-2e-7
                end_time>dur(sequences[encoding]) && continue
                hit=searchsortedfirst(ends,begin_time)
                hit<=length(ends) && ends[hit]<end_time || error("Reset window missed in cycle=$cycle sample=$j encoding=$encoding")
                actual=KomaMRICore.get_mask(single,KomaMRI.KomaMRIBase.unit_time(reshape(ends[hit:hit],1,:),phantom.motion.time))
                all(actual .==1) || error("Actual repeated reset mask failed")
                checked_events+=1
            end
        end
        results[encoding]=Dict("part_count"=>length(parts),"discrete_time_count"=>length(seqd.t),
            "reset_sample_windows_checked"=>checked,"all_reset_windows_have_part_end"=>true,
            "repeated_reset_windows_checked"=>checked_events,
            "sequence_sha256"=>filehash(joinpath(SEQDIR,lowercase(encoding)*".seq")))
    end
    # Exercise actual magnetization replacement on CPU and GPU, not just flags.
    testtime=Float32(.5)*Float32(phantom.motion.time.t_end)
    testflow=FlowPath(dx=zeros(Float32,1,2),dy=zeros(Float32,1,2),dz=zeros(Float32,1,2),spin_reset=reshape(Bool[false,true],1,:))
    testmotion=Motion(action=testflow,time=phantom.motion.time,spins=AllSpins())
    state=KomaMRICore.Mag(ComplexF32[.3+.2im],Float32[.4])
    KomaMRICore.outflow_spin_reset!(state,reshape([testtime],1,:),testmotion;replace_by=Float32[1])
    @assert state.xy == ComplexF32[0] && state.z == Float32[1]
    gpu_state=KomaMRICore.Mag(CUDA.cu(ComplexF32[.3+.2im]),CUDA.cu(Float32[.4]))
    gpu_motion=KomaMRICore.gpu(testmotion)
    KomaMRICore.outflow_spin_reset!(gpu_state,CUDA.cu(reshape([testtime],1,:)),gpu_motion;replace_by=CUDA.cu(Float32[1]))
    @assert Array(gpu_state.xy)==ComplexF32[0] && Array(gpu_state.z)==Float32[1]
    report=Dict("status"=>"PASS_RESET_PREFLIGHT","max_block_length"=>512,
        "reason"=>"Explicit FlowPath-key splitting; RF state continuous across numerical subparts",
        "encodings"=>results,"actual_cpu_magnetization_reset_passed"=>true,"actual_gpu_magnetization_reset_passed"=>true,
        "reset_key_lead_s"=>Float64(RESET_MARKER_LEAD_S),
        "warning"=>"2ms trajectory/identity interpolation remains approximate; no boundary-resolved assertion")
    write_toml(joinpath(output,"reset_preflight.toml"),report)
    return params
end

function simulate_pilot(phantom,sequences,params,output)
    CUDA.functional() || error("CUDA not functional")
    params["gpu"]=true;params["gpu_device"]=0;params["return_type"]="raw"
    images=Dict{String,Array{ComplexF32,4}}();runtimes=Dict{String,Float64}()
    for encoding in ("REF","X","Y","Z")
        println("START_PILOT_ENCODING ",encoding);flush(stdout)
        seq=sequences[encoding];start=time()
        reset_schedule!(phantom.motion,seq)
        PREVIOUS_PART_RF[]=false
        raw=simulate(phantom,seq,Scanner();sim_params=params,verbose=false)
        matrix=Int.([seq.DEF["Nx"],seq.DEF["Ny"],seq.DEF["Nz"],seq.DEF["CardiacBins"]])
        rawsort=zeros(ComplexF32,matrix...);filled=falses(matrix[2:end]...)
        for profile in raw.profiles
            y=Int(profile.head.idx.kspace_encode_step_1)+1;z=Int(profile.head.idx.kspace_encode_step_2)+1;b=Int(profile.head.idx.phase)+1
            !filled[y,z,b] || error("Duplicate kspace profile");filled[y,z,b]=true
            rawsort[:,y,z,b].=profile.data[:,1]
        end
        all(filled) || error("Incomplete kspace")
        image=ComplexF32.(fftshift(ifft(ifftshift(rawsort,(1,2,3)),(1,2,3)),(1,2,3)))
        images[encoding]=image;runtimes[encoding]=time()-start
        target=joinpath(output,lowercase(encoding)*"_result.jld2")
        JLD2.jldsave(target*".partial";rawsort,image,encoding,elapsed_seconds=runtimes[encoding],
            phantom_sha256=filehash(joinpath(output,"inlet_pilot.phantom")),
            sequence_sha256=filehash(joinpath(SEQDIR,lowercase(encoding)*".seq")))
        mv(target*".partial",target);println("FINISHED_PILOT_ENCODING ",encoding," seconds=",runtimes[encoding]);flush(stdout)
        if encoding=="REF"
            reference_viability(image,sequences["REF"],output)
        end
        GC.gc();CUDA.reclaim()
    end
    reference=images["REF"];magnitude=abs.(reference);mask=falses(size(magnitude))
    for b in axes(magnitude,4)
        mask[:,:,:,b].=magnitude[:,:,:,b].>=.05f0*maximum(magnitude[:,:,:,b])
    end
    velocity=zeros(Float32,size(reference)...,3);venc=Float32(sequences["REF"].DEF["Venc"])
    for (k,e) in enumerate(("X","Y","Z"))
        velocity[:,:,:,:,k].=angle.(images[e].*conj.(reference)).*(venc/Float32(pi))
        component=@view velocity[:,:,:,:,k];component[.!mask].=0
    end
    target=joinpath(output,"pc_reconstruction.h5")
    h5open(target*".partial","w") do f
        write(f,"reference_magnitude",magnitude);write(f,"velocity",velocity);write(f,"mask",UInt8.(mask))
        write(f,"speed",dropdims(sqrt.(sum(abs2,velocity;dims=5));dims=5))
        HDF5.attributes(f)["FOV_m"]=Float32.(collect(sequences["REF"].DEF["FOV"]))
        HDF5.attributes(f)["RR_s"]=Float32(sequences["REF"].DEF["RR"]);HDF5.attributes(f)["Venc_m_per_s"]=venc
    end
    mv(target*".partial",target)
    write_toml(joinpath(output,"simulation_report.toml"),Dict("status"=>"COMPLETE_EXPERIMENTAL_INLET_PILOT",
        "particle_count"=>length(phantom),"physical_gpu_index"=>1,"cuda_visible_device"=>0,
        "max_block_length"=>512,"reset_partition"=>"explicit_FlowPath_key_splitting_continuous_RF","encoding_runtime_s"=>runtimes,"total_encoding_runtime_s"=>sum(values(runtimes)),
        "pilot_only"=>true,"production_rerun_authorized"=>false,"script_sha256"=>filehash(@__FILE__),
        "reconstruction_sha256"=>filehash(target),"reset_preflight_sha256"=>filehash(joinpath(output,"reset_preflight.toml"))))
end

function reference_viability(reference,sequence,output)
    gridfile=joinpath(ROOT,"outputs","aorta_inlet_replenishment","e034_inlet_quadrature_v1.h5")
    points,area=h5open(gridfile,"r") do f
        read(f,"centroids_source_m"),read(f,"area_mm2")
    end
    fov=Float64.(sequence.DEF["FOV"]);shape=size(reference)[1:3]
    magnitude=abs.(reference);common=trues(shape...)
    for b in axes(reference,4)
        common .&= magnitude[:,:,:,b].>=.05f0*maximum(magnitude[:,:,:,b])
    end
    maxima=Float64[maximum(@view magnitude[:,:,:,b]) for b in 1:8]
    means=zeros(Float64,8);covered=0.0
    for q in eachindex(area)
        pos=(Float64.(points[:,q]).-Float64.(CENTER))[ORDER]
        index=pos./fov.*collect(shape).+collect(shape)./2 .+1
        low=floor.(Int,index);frac=index.-low
        all(1 .<= low .< collect(shape)) || error("Face quadrature outside preserved MRI grid")
        support=0.0
        for ix in 0:1,iy in 0:1,iz in 0:1
            xyz=low.+[ix,iy,iz]
            weight=(ix==0 ? 1-frac[1] : frac[1])*(iy==0 ? 1-frac[2] : frac[2])*(iz==0 ? 1-frac[3] : frac[3])
            support+=weight*common[xyz...]
            for b in 1:8
                means[b]+=area[q]*weight*magnitude[xyz...,b]/maxima[b]
            end
        end
        support>=1-1e-10 && (covered+=area[q])
    end
    means./=sum(area);coverage=covered/sum(area)
    passed=all(means.>=.05)&&coverage>=.25
    write_toml(joinpath(output,"reference_viability.toml"),Dict("status"=>passed ? "PASS_REFERENCE_VIABILITY" : "FAIL_REFERENCE_VIABILITY_STOP",
        "area_weighted_mean_magnitude_fraction"=>means,"fixed05_complete_stencil_area_fraction"=>coverage,
        "quadrature_sha256"=>filehash(gridfile),"thresholds_fixed_before_raw_data"=>true))
    passed || error("Reference inlet signal/coverage gate failed: not running velocity encodings")
end

if abspath(PROGRAM_FILE)==@__FILE__
    length(ARGS)==3 || error("Usage: inlet_replenishment_pilot.jl build|run TRACK_DIR NEW_OUTPUT")
    mode,tracks,output=ARGS
    mode in ("build","run") || error("Unknown mode")
    phantom,sequences=build_phantom(tracks,output)
    params=reset_preflight(phantom,sequences,output)
    mode=="run" && simulate_pilot(phantom,sequences,params,output)
end
