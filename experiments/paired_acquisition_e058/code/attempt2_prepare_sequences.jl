# E-058 S0: certify the builder against the original real sequences, then build and
# audit the 700 Hz/pixel, TE 2.1 ms candidate. CPU only; no Koma signal.
using KomaMRI, KomaMRIBase, HDF5, TOML, SHA

const HELPERS=("gradient_design.jl","utils.jl","cartesian.jl","excitation.jl","gre.jl","PC.jl")
for name in HELPERS
    include(joinpath(@__DIR__,"sequence_helpers",name))
end
const E056=raw"D:\MRSim\outputs\acquisition_physics_e056"
const E055INPUTS=raw"D:\MRSim\outputs\phantom_density_e055\inputs_v1"
const ORIGINAL=raw"D:\MRSim\MRSIM\MRSimTracks\koma_dp150\pc_3d_scanner_matched_venc5"
const VOXEL=(1.25e-3,1.25e-3,2.5e-3)
const MATRIX=(400,64,32)
const FOV=Tuple(MATRIX[i]*VOXEL[i] for i in eachindex(MATRIX))  # same expression as the original generator
const ENC=("REF","X","Y","Z")
const NPROFILES=MATRIX[2]*MATRIX[3]*15
digest(path)=bytes2hex(sha256(read(path)))

function write_toml(path,value)
    ispath(path)&&error("Refuse existing file: "*path)
    open(path,"w") do io;TOML.print(io,value;sorted=true);end
end

event_vector(value)=value isa Number ? [value] : collect(value)

# --- copied verbatim from E056 code/prepare_sequences_v2.jl ---------------------
function rf_metadata(seq)
    starts = get_block_start_times(seq)
    indices = findall(i -> any(!iszero, event_vector(seq.RF[1, i].A)), eachindex(seq.DUR))
    full = maximum(maximum(abs, event_vector(seq.RF[1, i].A)) for i in indices)
    phases = [angle(event_vector(seq.RF[1, i].A)[argmax(abs.(event_vector(seq.RF[1, i].A)))]) for i in indices]
    scales = [maximum(abs, event_vector(seq.RF[1, i].A)) / full for i in indices]
    centers = [starts[i] + seq.RF[1, i].delay + rf_center(seq.RF[1, i]) for i in indices]
    return indices, starts[indices], centers, scales, phases
end

function exact_moments(times, gradients, adc, center)
    count = length(times)
    m0 = zeros(3, count); m1 = zeros(3, count); m2 = zeros(3, count)
    first = findfirst(t -> abs(t-center) < 1e-13, times)
    isnothing(first) && error("RF center absent from integration grid")
    for j in first:(count-1)
        dt = times[j+1]-times[j]
        dt > 0 || error("Nonincreasing gradient knots")
        a = times[j]-center
        v = gradients[:,j]; slope = (gradients[:,j+1]-v)/dt
        m0[:,j+1] = m0[:,j] + v*dt + slope*(dt^2/2)
        m1[:,j+1] = m1[:,j] + v*(a*dt+dt^2/2) + slope*(a*dt^2/2+dt^3/3)
        m2[:,j+1] = m2[:,j] + v*(a*a*dt+a*dt^2+dt^3/3) + slope*(a*a*dt^2/2+2a*dt^3/3+dt^4/4)
    end
    indices = [something(findfirst(t -> abs(t-x)<1e-13, times)) for x in adc]
    return m0[:,indices], m1[:,indices], m2[:,indices]
end

function shot_waveform(seq, rfblock, adcblock)
    shot = copy(seq[rfblock:adcblock])
    samples = KomaMRIBase.get_samples(shot)
    descriptor_center = shot.RF[1,1].delay + rf_center(shot.RF[1,1])
    adc = get_adc_sampling_times(shot)
    times = sort!(unique(vcat([0., descriptor_center, dur(shot)], adc,
        samples.gx.t, samples.gy.t, samples.gz.t, samples.rf.t, samples.Δf.t, samples.ψ.t)))
    gx,gy,gz = KomaMRIBase.get_grads(shot,times)
    gradients = permutedims(hcat(gx,gy,gz))
    b1,df,psi = KomaMRIBase.get_rfs(shot,times)
    dt=diff(times);a=abs.(b1[1:end-1]);slope=diff(abs.(b1))./dt
    area=a.*dt+slope.*dt.^2/2
    center=sum(times[1:end-1].*area+a.*dt.^2/2+slope.*dt.^3/3)/sum(area)
    abs(center-descriptor_center)<=1.01e-6||error("RF center descriptor differs by more than one RF raster")
    times=sort!(unique(vcat(times,[center])))
    gx,gy,gz=KomaMRIBase.get_grads(shot,times);gradients=permutedims(hcat(gx,gy,gz))
    b1,df,psi=KomaMRIBase.get_rfs(shot,times)
    raw = event_vector(shot.RF[1,1].A)
    phase = angle(raw[argmax(abs.(raw))])
    b1 = b1 ./ cis(phase)
    m0,m1,m2 = exact_moments(times,gradients,adc,center)
    return (; times,gradients,b1,df,psi,center,descriptor_center,adc,m0,m1,m2)
end
# --------------------------------------------------------------------------------

# Generalization of E056 export_encoding to the real matrix; audits only, no export.
function audit_encoding(seq)
    starts=get_block_start_times(seq);labels=get_labels(seq)
    rfblocks,rfstarts,rfcenters,rfscales,rfphases=rf_metadata(seq)
    blocks=findall(a->a.N>0,seq.ADC)
    length(blocks)==NPROFILES||error("Incorrect profile count")
    length(rfblocks)==length(blocks)+10||error("Ramp RF count differs")
    all(seq.ADC[b].N==MATRIX[1] for b in blocks)||error("Readout sample count differs")
    profile_rf=[searchsortedlast(rfblocks,b)-1 for b in blocks]
    labels0=hcat([[Int(labels[b].LIN),Int(labels[b].PAR),Int(labels[b].PHS)] for b in blocks]...)
    all((0 .<= labels0[1,:]) .& (labels0[1,:] .< MATRIX[2]))||error("ky labels invalid")
    all((0 .<= labels0[2,:]) .& (labels0[2,:] .< MATRIX[3]))||error("kz labels invalid")
    all((0 .<= labels0[3,:]) .& (labels0[3,:] .< 15))||error("phase labels invalid")
    firstadc=[starts[b]+seq.ADC[b].delay for b in blocks]
    dwells=[dwell(seq.ADC[b]) for b in blocks]
    counts=[seq.ADC[b].N for b in blocks]
    unique_index=labels0[1,:] .+ MATRIX[2] .* labels0[2,:]
    nshots=MATRIX[2]*MATRIX[3]
    m0all=zeros(3,MATRIX[1],nshots);m1all=similar(m0all);m2all=similar(m0all)
    first_profile=Dict{Int,Int}()
    for i in eachindex(blocks)
        labels0[3,i]==0&&!haskey(first_profile,unique_index[i])&&(first_profile[unique_index[i]]=i)
    end
    length(first_profile)==nshots||error("Missing spatial shot")
    expected_m0(p)=vcat(reshape(collect(-MATRIX[1]÷2:MATRIX[1]÷2-1)./FOV[1],1,:),
        fill((labels0[1,p]-MATRIX[2]÷2)/FOV[2],1,MATRIX[1]),fill((labels0[2,p]-MATRIX[3]÷2)/FOV[3],1,MATRIX[1]))
    # Common finite-RF slice offset from shot 0, exactly as in the serial E056 audit.
    p0=first_profile[0]
    wave0=shot_waveform(seq,rfblocks[profile_rf[p0]+1],blocks[p0])
    rf_offset=(γ.*wave0.m0.-expected_m0(p0))[:,1]
    maximum(abs.(rf_offset[1:2]))<2e-5||error("Unexpected readout/phase grid offset")
    nz=findall(x->abs(x)>1e-15,wave0.b1)
    center_index=argmin(abs.(wave0.times.-wave0.center))
    rf_footprint=abs(γ*wave0.gradients[3,center_index]*(wave0.times[nz[end]]-wave0.times[nz[1]])/2)
    abs(rf_offset[3])<=rf_footprint+2e-5||error("RF rephasing offset exceeds excitation footprint")
    # Independent shots audited in parallel; each writes only its own slices.
    kerror=zeros(nshots)
    Threads.@threads for key in 0:(nshots-1)
        p=first_profile[key]
        wave=shot_waveform(seq,rfblocks[profile_rf[p]+1],blocks[p])
        m0all[:,:,key+1]=wave.m0;m1all[:,:,key+1]=wave.m1;m2all[:,:,key+1]=wave.m2
        kerror[key+1]=maximum(abs.(γ.*wave.m0.-expected_m0(p).-rf_offset))
    end
    max_kerror=maximum(kerror)
    max_kerror<2e-5||error("Readback k-space grid differs: "*string(max_kerror))
    acquire_rfstarts=rfstarts[profile_rf.+1]
    maximum(abs.(diff(acquire_rfstarts).-1.012/15))<2e-9||error("RF cadence changed")
    firstprofile=findfirst(==(0),labels0[3,:])
    echo=firstadc[firstprofile]+(MATRIX[1]÷2)*dwells[firstprofile]
    metrics=Dict("profiles"=>length(blocks),"physical_rf_count"=>length(rfblocks),
        "max_kspace_error_cycles_m"=>max_kerror,"rf_rephasing_offset_cycles_m"=>rf_offset,
        "first_acquired_rf_start_s"=>acquire_rfstarts[1],"first_acquired_echo_s"=>echo,
        "echo_after_rf_center_s"=>echo-rfcenters[profile_rf[firstprofile]+1],
        "adc_dwell_s"=>dwells[1],"sequence_duration_s"=>dur(seq),
        "TE_s"=>Float64(seq.DEF["TE"]),"TR_s"=>Float64(seq.DEF["TR"]),
        "bandwidth_hz_pixel"=>Float64(seq.DEF["BandwidthPerPixel"]))
    timing=(adc_first_s=firstadc,dwell_s=dwells,n=counts,labels=labels0)
    return metrics,(m0all,m1all,m2all),(rfstarts,rfcenters,rfscales,rfphases),timing
end

function venc_audit!(report,moments)
    reference=moments["REF"];m1=1/(2γ*5.)
    for (axis,encoding) in enumerate(("X","Y","Z"))
        dm0=moments[encoding][1]-reference[1];dm1=moments[encoding][2]-reference[2]
        expected=zeros(size(dm1));expected[axis,:,:].=-m1
        maximum(abs.(dm0))<2e-12||error("Nonzero differential M0 "*encoding)
        maximum(abs.(dm1.-expected))<2e-15||error("VENC moment differs "*encoding)
        report[encoding]["max_differential_m0_T_s_m"]=maximum(abs.(dm0))
        report[encoding]["max_first_moment_error_T_s2_m"]=maximum(abs.(dm1.-expected))
    end
end

# Approximate structural equality for Koma sequence events.
function close_events(a,b,path,mismatches)
    if a isa Number&&b isa Number
        ok=a==b||abs(a-b)<=1e-9*max(abs(a),abs(b))+1e-18
        ok||push!(mismatches,path*" "*string(a)*" vs "*string(b))
        return ok
    elseif a isa AbstractArray&&b isa AbstractArray
        size(a)==size(b)||(push!(mismatches,path*" size");return false)
        return all(close_events(x,y,path*"[]",mismatches) for (x,y) in zip(a,b))
    elseif a isa Number||b isa Number
        # A constant waveform may be stored as a scalar on one side and a vector on the other.
        x=event_vector(a);y=event_vector(b)
        (length(x)==1||length(y)==1)||(push!(mismatches,path*" scalar/vector");return false)
        return all(close_events(u,v,path,mismatches) for u in x for v in y)
    elseif typeof(a)==typeof(b)&&!isempty(fieldnames(typeof(a)))
        return all(close_events(getfield(a,f),getfield(b,f),path*"."*string(f),mismatches) for f in fieldnames(typeof(a)))
    end
    ok=isequal(a,b);ok||push!(mismatches,path*" "*string(a)*" vs "*string(b));return ok
end

function equivalence(rebuilt,original)
    mismatches=String[]
    length(rebuilt)==length(original)||return false,["block count "*string(length(rebuilt))*" vs "*string(length(original))]
    lr=get_labels(rebuilt);lo=get_labels(original)
    for i in 1:length(original)
        close_events(rebuilt.DUR[i],original.DUR[i],"DUR[$i]",mismatches)
        close_events(rebuilt.RF[1,i],original.RF[1,i],"RF[$i]",mismatches)
        for j in 1:3
            close_events(rebuilt.GR[j,i],original.GR[j,i],"GR[$j,$i]",mismatches)
        end
        close_events(rebuilt.ADC[i],original.ADC[i],"ADC[$i]",mismatches)
        close_events(lr[i],lo[i],"LABEL[$i]",mismatches)
        length(mismatches)>=50&&break
    end
    return isempty(mismatches),mismatches
end

function build(bw,te)
    scanner=Scanner(limits=HardwareLimits(B0=3.,Gmax=40e-3,Smax=180.,B1=10e-6,ADC_Δt=1e-9,DUR_Δt=1e-9))
    seqs=build_pc_encoding_scans(FOV,MATRIX,scanner;venc=5.0,cardiac_bins=15,RR=1.012,BWpp=bw,
        flip_angle=deg2rad(8),rf_time_bandwidth_product=1,rf_window=0.0,exact_rf_flip=true,spoil_phase=0.0,
        n_ramp_shots=10,trigger_channel=nothing,rf_spoil_increment=deg2rad(117),target_TE=te,
        target_TR=46.88e-3,exact_rr=true,merge_velocity_prephaser=true,prephaser_duration=0.8e-3)
    return seqs,scanner
end

function write_readback(seq,scanner,path)
    check_timing(seq,scanner);check_hw_limits(seq,scanner)
    write_seq(seq,path)
    readback=read_seq(path;verbose=false,verify_signature=true)
    check_timing(readback,scanner);check_hw_limits(readback,scanner)
    return readback
end

function write_timing(path,timings)
    h5open(path,"w") do f
        for e in ENC
            g=create_group(f,e);t=timings[e]
            g["adc_first_s"]=t.adc_first_s;g["dwell_s"]=t.dwell_s;g["n"]=Int64.(t.n);g["labels"]=Int64.(t.labels)
        end
    end
end

function main(output)
    ispath(output)&&error("Refuse existing output")
    stage=output*".building";ispath(stage)&&error("Stage exists");mkpath(stage)
    e056reg=TOML.parsefile(joinpath(E056,"sequences_v1/registration.toml"))
    helpers=Dict(name=>digest(joinpath(@__DIR__,"sequence_helpers",name)) for name in HELPERS)
    helpers==e056reg["helpers_sha256"]||error("Copied helpers differ from E056")
    info=TOML.parsefile(joinpath(E055INPUTS,"case_info.toml"))
    report=Dict{String,Any}("experiment"=>"E-058","stage"=>"S0_sequences","new_signal_simulated"=>false,"new_1m"=>false,
        "julia_threads"=>Threads.nthreads(),"attempt_note"=>"attempt1 (single-threaded, ~38 min per encoding audit) stopped and preserved; checks unchanged",
        "script_sha256"=>digest(@__FILE__),"protocol_sha256"=>digest(joinpath(@__DIR__,"PROTOCOL.md")),
        "helpers_sha256"=>helpers,"matrix"=>collect(MATRIX),"fov_m"=>collect(FOV))
    originals=Dict{String,Any}()
    for e in ENC
        path=joinpath(ORIGINAL,lowercase(e)*".seq")
        digest(path)==info["sequence_sha256"][e]||error("Original sequence changed: "*e)
        originals[e]=read_seq(path;verbose=false,verify_signature=true)
    end
    # (a) builder certification
    rebuilt,scanner=build(350.0,2.74e-3)
    mkpath(joinpath(stage,"baseline_rebuilt"));equiv=Dict{String,Any}()
    for e in ENC
        readback=write_readback(getproperty(rebuilt,Symbol(e)),scanner,joinpath(stage,"baseline_rebuilt",lowercase(e)*".seq"))
        ok,mismatches=equivalence(readback,originals[e])
        equiv[e]=Dict("event_equivalent"=>ok,"first_mismatches"=>mismatches,
            "rebuilt_sha256"=>digest(joinpath(stage,"baseline_rebuilt",lowercase(e)*".seq")),"original_sha256"=>info["sequence_sha256"][e])
        println("E058_EQUIVALENCE ",e," ",ok," ",isempty(mismatches) ? "" : first(mismatches));flush(stdout)
    end
    report["builder_certification"]=equiv
    if !all(equiv[e]["event_equivalent"] for e in ENC)
        report["status"]="FAIL_E058_BUILDER_NOT_EQUIVALENT_TO_ORIGINAL"
        write_toml(joinpath(stage,"report.toml"),report);mv(stage,output*".failed")
        error("Builder certification failed; candidate not built")
    end
    # Original audits, and exporter identity against the frozen legacy timing.
    arms=Dict{String,Any}();moments=Dict{String,Any}();rfs=Dict{String,Any}();timings=Dict{String,Any}()
    for e in ENC
        metrics,m,rf,t=audit_encoding(originals[e]);moments[e]=m;rfs[e]=rf;timings[e]=t;arms[e]=metrics
        println("E058_AUDIT original ",e," PASS");flush(stdout)
    end
    venc_audit!(arms,moments)
    h5open(joinpath(E055INPUTS,"legacy_timing.h5"),"r") do f
        for e in ENC
            g=f[e];t=timings[e]
            maximum(abs.(t.adc_first_s.-read(g,"adc_first_s")))<1e-9||error("Timing exporter differs: adc_first_s")
            maximum(abs.(t.dwell_s.-read(g,"dwell_s")))<1e-13||error("Timing exporter differs: dwell")
            t.n==read(g,"n")&&t.labels==read(g,"labels")||error("Timing exporter differs: n/labels")
        end
    end
    report["original"]=arms;report["timing_exporter_reproduces_legacy"]=true
    baseline_rf=rfs["REF"];baseline_labels=timings["REF"].labels
    # (b) candidate
    candidate,scanner=build(700.0,2.1e-3)
    mkpath(joinpath(stage,"short_echo"));carms=Dict{String,Any}();cmoments=Dict{String,Any}();ctimings=Dict{String,Any}()
    for e in ENC
        path=joinpath(stage,"short_echo",lowercase(e)*".seq")
        readback=write_readback(getproperty(candidate,Symbol(e)),scanner,path)
        metrics,m,rf,t=audit_encoding(readback);cmoments[e]=m;ctimings[e]=t
        t.labels==baseline_labels||error("Candidate labels differ from baseline")
        for i in 1:3
            length(rf[i])==length(baseline_rf[i])||error("RF count differs")
            maximum(abs.(rf[i].-baseline_rf[i]))<2e-9||error("RF history differs")
        end
        maximum(abs.(angle.(cis.(rf[4].-baseline_rf[4]))))<2e-9||error("RF phase schedule differs")
        abs(metrics["bandwidth_hz_pixel"]-700)<=.01||error("Candidate bandwidth differs")
        abs(metrics["TE_s"]-2.1e-3)<=1e-9||error("Candidate TE differs")
        metrics["sequence_sha256"]=digest(path);carms[e]=metrics
        println("E058_AUDIT short_echo ",e," PASS TE=",metrics["TE_s"]," echo=",metrics["first_acquired_echo_s"]);flush(stdout)
    end
    venc_audit!(carms,cmoments)
    write_timing(joinpath(stage,"short_echo_timing.h5"),ctimings)
    report["short_echo"]=carms
    report["short_echo_timing_sha256"]=digest(joinpath(stage,"short_echo_timing.h5"))
    report["status"]="PASS_E058_SEQUENCE_CERTIFICATION_AND_CANDIDATE_AUDIT"
    write_toml(joinpath(stage,"report.toml"),report)
    mv(stage,output)
    println(report["status"])
end

length(ARGS)==1||error("NEW_OUTPUT required")
main(ARGS[1])
