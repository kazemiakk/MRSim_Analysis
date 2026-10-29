using KomaMRI, KomaMRIBase, HDF5, TOML, SHA

const HELPERS = ("gradient_design.jl", "utils.jl", "cartesian.jl", "excitation.jl", "gre.jl", "PC.jl")
for name in HELPERS
    include(joinpath(@__DIR__, "sequence_helpers", name))
end
const ARMS = (("baseline", 350.0, .00274), ("bw700_te274", 700.0, .00274), ("bw700_te200", 700.0, .002))
const MATRIX = (80, 16, 32)
const FOV = (.1, .02, .08)
digest(path) = bytes2hex(sha256(read(path)))

function write_toml(path, value)
    ispath(path) && error("Refuse existing file: " * path)
    open(path, "w") do io
        TOML.print(io, value; sorted=true)
    end
end

function event_vector(value)
    value isa Number ? [value] : collect(value)
end

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
    # Independent exact polynomial integration of exported piecewise-linear
    # gradient samples, referenced to the physical RF center.
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
    # Independently integrate the amplitude centroid of the actual symmetric
    # RF waveform. The Pulseq center descriptor differs by one RF raster here.
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

function export_encoding(parent, seq, encoding)
    group = create_group(parent,encoding)
    starts=get_block_start_times(seq); labels=get_labels(seq)
    rfblocks,rfstarts,rfcenters,rfscales,rfphases=rf_metadata(seq)
    blocks=findall(a->a.N>0,seq.ADC)
    length(blocks)==16*32*15 || error("Incorrect profile count")
    length(rfblocks)==length(blocks)+10 || error("Ramp RF count differs")
    profile_rf=[searchsortedlast(rfblocks,b)-1 for b in blocks]
    labels0=hcat([[Int(labels[b].LIN),Int(labels[b].PAR),Int(labels[b].PHS)] for b in blocks]...)
    all((0 .<= labels0[1,:]) .& (labels0[1,:] .< 16)) || error("ky labels invalid")
    all((0 .<= labels0[2,:]) .& (labels0[2,:] .< 32)) || error("kz labels invalid")
    all((0 .<= labels0[3,:]) .& (labels0[3,:] .< 15)) || error("phase labels invalid")
    firstadc=[starts[b]+seq.ADC[b].delay for b in blocks]
    dwells=[dwell(seq.ADC[b]) for b in blocks]
    adccomp=reshape(KomaMRIBase.get_adc_phase_compensation(seq),MATRIX[1],:)
    size(adccomp)==(80,length(blocks)) || error("ADC phase compensation shape differs")
    unique_index=labels0[1,:] .+ 16 .* labels0[2,:]
    for (name,value) in (("rf_block_index0",rfblocks.-1),("rf_start_s",rfstarts),
        ("rf_center_s",rfcenters),("rf_scale",rfscales),("rf_phase_rad",rfphases),
        ("profile_rf_index0",profile_rf),("adc_block_index0",blocks.-1),("adc_first_s",firstadc),("dwell_s",dwells),
        ("labels0",labels0),("profile_unique_shot_index0",unique_index),
        ("adc_phase_compensation",adccomp))
        write(group,name,value)
    end
    shots=create_group(group,"shots")
    m0all=zeros(3,80,512);m1all=similar(m0all);m2all=similar(m0all)
    max_kerror=0.;rf_offset=nothing
    for key in 0:511
        p=findfirst(i->unique_index[i]==key && labels0[3,i]==0,eachindex(blocks))
        isnothing(p)&&error("Missing spatial shot")
        adcblock=blocks[p];rfblock=rfblocks[profile_rf[p]+1]
        wave=shot_waveform(seq,rfblock,adcblock)
        target=create_group(shots,lpad(string(key),4,'0'))
        for (name,value) in (("t_s",wave.times),("g_T_m",wave.gradients),
            ("b1_T",wave.b1),("rf_frequency_Hz",wave.df),("rf_phase_rad_at_knots",wave.psi),
            ("rf_center_s",wave.center),("adc_time_s",wave.adc),
            ("rf_center_descriptor_s",wave.descriptor_center),
            ("moment0_adc_T_s_m",wave.m0),("moment1_adc_T_s2_m",wave.m1),
            ("moment2_adc_T_s3_m",wave.m2))
            write(target,name,value)
        end
        m0all[:,:,key+1]=wave.m0;m1all[:,:,key+1]=wave.m1;m2all[:,:,key+1]=wave.m2
        expected=vcat(reshape(collect(-40:39)./FOV[1],1,:),
            fill((labels0[1,p]-8)/FOV[2],1,80),fill((labels0[2,p]-16)/FOV[3],1,80))
        difference=γ.*wave.m0.-expected
        if isnothing(rf_offset)
            rf_offset=copy(difference[:,1])
            maximum(abs.(rf_offset[1:2]))<2e-5||error("Unexpected readout/phase grid offset")
            maximum(abs.(rf_offset))<2e-5||error("RF-waveform-center Cartesian grid offset")
        end
        max_kerror=max(max_kerror,maximum(abs.(difference.-rf_offset)))
    end
    # Require absolute grid and increments using the independently integrated
    # RF amplitude centroid. Descriptor and waveform centers are both retained.
    max_kerror<2e-5 || error("Readback k-space grid differs: "*string(max_kerror))
    acquire_rfstarts=rfstarts[profile_rf.+1]
    maximum(abs.(diff(acquire_rfstarts) .- 1.012/15))<2e-9 || error("RF cadence changed")
    firstprofile=findfirst(==(0),labels0[3,:])
    echo=firstadc[firstprofile]+40dwells[firstprofile]
    return Dict("profiles"=>length(blocks),"physical_rf_count"=>length(rfblocks),
        "max_kspace_error_cycles_m"=>max_kerror,"first_acquired_rf_start_s"=>acquire_rfstarts[1],
        "rf_rephasing_offset_cycles_m"=>rf_offset,
        "first_acquired_echo_s"=>echo,"adc_dwell_s"=>dwells[1],
        "rf_acquisition_interval_s"=>1.012/15,"sequence_duration_s"=>dur(seq)),
        (m0all,m1all,m2all), (rfstarts,rfcenters,rfscales,rfphases)
end

function main(repo,output)
    ispath(output)&&error("Refuse existing output")
    stage=output*".building";ispath(stage)&&error("Stage exists");mkpath(stage)
    provenance=Dict("experiment"=>"E-056","status"=>"REGISTERED_BEFORE_SEQUENCE_OUTPUT",
        "script_sha256"=>digest(@__FILE__),"manifest_sha256"=>digest(joinpath(repo,"koma_migration_e046","Manifest.toml")),
        "protocol_sha256"=>digest(joinpath(@__DIR__,"PROTOCOL.md")),
        "helpers_sha256"=>Dict(name=>digest(joinpath(@__DIR__,"sequence_helpers",name)) for name in HELPERS),
        "helper_origin"=>"Copied scanner-derived helpers; only sys.hardware_field to sys.limits.hardware_field compatibility adaptation",
        "new_1m"=>false,"matrix"=>collect(MATRIX),"fov_m"=>collect(FOV))
    write_toml(joinpath(stage,"registration.toml"),provenance)
    scanner=Scanner(limits=HardwareLimits(B0=3.,Gmax=.040,Smax=180.,B1=10e-6,ADC_Δt=1e-9,DUR_Δt=1e-9))
    reports=Dict{String,Any}();baseline_rf=nothing
    h5open(joinpath(stage,"waveforms.h5.partial"),"w") do waves
        write(waves,"gamma_Hz_T",Float64(γ))
        for (arm,bw,te) in ARMS
            seqs=build_pc_encoding_scans(FOV,MATRIX,scanner;venc=5.,cardiac_bins=15,RR=1.012,
                BWpp=bw,flip_angle=deg2rad(8),rf_time_bandwidth_product=1,rf_window=0.,exact_rf_flip=true,
                spoil_phase=0.,n_ramp_shots=10,trigger_channel=nothing,rf_spoil_increment=deg2rad(117),
                target_TE=te,target_TR=.04688,exact_rr=true,merge_velocity_prephaser=true,prephaser_duration=.0008)
            folder=joinpath(stage,arm);mkpath(folder);ag=create_group(waves,arm)
            armreport=Dict{String,Any}();moments=Dict{String,Any}()
            for encoding in ("REF","X","Y","Z")
                seq=getproperty(seqs,Symbol(encoding))
                check_timing(seq,scanner);check_hw_limits(seq,scanner)
                path=joinpath(folder,lowercase(encoding)*".seq")
                write_seq(seq,path)
                readback=read_seq(path;verbose=false,verify_signature=true)
                check_timing(readback,scanner);check_hw_limits(readback,scanner)
                metrics,m,rf=export_encoding(ag,readback,encoding)
                moments[encoding]=m
                if isnothing(baseline_rf)
                    baseline_rf=rf
                else
                    for i in 1:3
                        length(rf[i])==length(baseline_rf[i])||error("RF count differs between arms")
                        maximum(abs.(rf[i].-baseline_rf[i]))<2e-9||error("RF history differs between arms")
                    end
                    maximum(abs.(angle.(cis.(rf[4].-baseline_rf[4]))))<2e-9||error("RF phase schedule differs")
                end
                metrics["sequence_sha256"]=digest(path)
                metrics["TE_s"]=Float64(readback.DEF["TE"])
                metrics["TR_s"]=Float64(readback.DEF["TR"])
                metrics["bandwidth_hz_pixel"]=Float64(readback.DEF["BandwidthPerPixel"])
                armreport[encoding]=metrics
                println("E056_SEQUENCE ",arm," ",encoding," PASS");flush(stdout)
            end
            reference=moments["REF"]
            m1=1/(2γ*5.)
            for (axis,encoding) in enumerate(("X","Y","Z"))
                dm0=moments[encoding][1]-reference[1]
                dm1=moments[encoding][2]-reference[2]
                expected=zeros(size(dm1));expected[axis,:,:].=-m1
                maximum(abs.(dm0))<2e-12||error("Nonzero differential M0")
                maximum(abs.(dm1.-expected))<2e-15||error("VENC moment differs")
                armreport[encoding]["max_differential_m0_T_s_m"]=maximum(abs.(dm0))
                armreport[encoding]["max_first_moment_error_T_s2_m"]=maximum(abs.(dm1.-expected))
            end
            reports[arm]=armreport
            write_toml(joinpath(folder,"metadata.toml"),Dict("arm"=>arm,"requested_bw_hz_pixel"=>bw,
                "requested_TE_s"=>te,"matrix"=>collect(MATRIX),"fov_m"=>collect(FOV),"encodings"=>armreport))
        end
    end
    mv(joinpath(stage,"waveforms.h5.partial"),joinpath(stage,"waveforms.h5"))
    write_toml(joinpath(stage,"report.toml"),merge(provenance,Dict("status"=>"PASS_E056_THREE_ARM_SEQUENCE_AND_MOMENT_GATES",
        "arms"=>reports,"waveforms_sha256"=>digest(joinpath(stage,"waveforms.h5")),
        "baseline_echo_s"=>reports["baseline"]["REF"]["first_acquired_echo_s"],
        "rf_period_s"=>1.012/15,
        "hardware"=>Dict("Gmax_T_m"=>.04,"Smax_T_m_s"=>180.,"B1_T"=>10e-6),
        "waveform_contract"=>"HDF5 Julia arrays reverse dimensions in Python; g_T_m is Python(K,3), labels0 is Python(P,3); adc_phase_compensation is Python(P,N). Shot B1 neutralizes RF complex phase; reapply perRF rf_phase_rad. ADC times include half dwell.")))
    mv(stage,output)
    println("PASS_E056_THREE_ARM_SEQUENCE_AND_MOMENT_GATES")
end

length(ARGS)==2||error("REPO NEW_OUTPUT required")
main(ARGS...)
