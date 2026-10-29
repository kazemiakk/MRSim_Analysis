# E-060 S0b probe (after the slab fold-in gate failed): minimum TE and first-shot
# excitation for sharper slab pulses. No signal; excitations go to a scratch folder.
using KomaMRI, KomaMRIBase, HDF5
for name in ("gradient_design.jl", "utils.jl", "cartesian.jl", "excitation.jl", "gre.jl", "PC_e060.jl")
    include(joinpath(@__DIR__, "sequence_helpers", name))
end
const FOV = (0.08, 0.08, 0.075)
const MATRIX = (64, 64, 30)
const VENC = (1.5, 1.5, 3.0)
event_vector(value)=value isa Number ? [value] : collect(value)

# Sampling part of prepare_sequences_e060.jl shot_waveform (moments not needed).
function excitation_waveform(seq)
    rfblock=findfirst(i->any(!iszero,event_vector(seq.RF[1,i].A)),eachindex(seq.DUR))
    adcblock=findfirst(i->i>rfblock&&seq.ADC[i].N>0,eachindex(seq.DUR))
    shot=copy(seq[rfblock:adcblock])
    samples=KomaMRIBase.get_samples(shot)
    times=sort!(unique(vcat([0.,dur(shot)],samples.gx.t,samples.gy.t,samples.gz.t,samples.rf.t,samples.Δf.t)))
    gx,gy,gz=KomaMRIBase.get_grads(shot,times)
    b1,df,_=KomaMRIBase.get_rfs(shot,times)
    raw=event_vector(shot.RF[1,1].A)
    return times,permutedims(hcat(gx,gy,gz)),b1./cis(angle(raw[argmax(abs.(raw))])),df
end

out=ARGS[1];mkpath(out)
sys=Scanner(limits=HardwareLimits(B0=3.,Gmax=45e-3,Smax=200.,B1=10e-6,ADC_Δt=1e-9,DUR_Δt=1e-9))
for tbp in (4,6,8,10,12,16)
    tes=Float64[];ref=nothing
    for axis in (:REF,:RO,:PE,:SS)
        seq=build_pc(FOV,MATRIX,sys;venc=VENC,cardiac_bins=2,RR=1.012,BWpp=350.0,
            flip_angle=deg2rad(8),rf_time_bandwidth_product=tbp,rf_window=0.46,exact_rf_flip=true,
            spoil_phase=0.0,n_ramp_shots=0,trigger_channel=nothing,rf_spoil_increment=deg2rad(117),
            target_TE=nothing,target_TR=23.44e-3,exact_rr=false,merge_velocity_prephaser=true,
            prephaser_duration=0.95e-3,slab_thickness=0.06,_encoding_axis=axis)
        check_hw_limits(seq,sys);push!(tes,seq.DEF["TE"]);axis===:REF&&(ref=seq)
    end
    t,g,b1,df=excitation_waveform(ref)
    h5open(joinpath(out,"tbp$(tbp).h5"),"w") do f
        f["t_s"]=t;f["g_T_m"]=g;f["b1_T"]=ComplexF64.(b1);f["rf_frequency_Hz"]=Float64.(df)
    end
    println("E060_SLAB_PROBE TBP",tbp," min_TE_ms=",round(1e3*maximum(tes),digits=4))
end
