using KomaMRI,KomaMRIBase,TOML
for name in ("gradient_design.jl","utils.jl","cartesian.jl","excitation.jl","gre.jl","PC.jl")
    include(joinpath(@__DIR__,"sequence_helpers",name))
end
scanner=Scanner(limits=HardwareLimits(B0=3.,Gmax=.04,Smax=180.,B1=10e-6,ADC_Δt=1e-9,DUR_Δt=1e-9))
try
    s=build_pc((.1,.02,.08),(80,16,32),scanner;venc=5.,cardiac_bins=15,RR=1.012,
        BWpp=700.,flip_angle=deg2rad(8),rf_time_bandwidth_product=1,rf_window=0.,exact_rf_flip=true,
        spoil_phase=0.,n_ramp_shots=10,trigger_channel=nothing,rf_spoil_increment=deg2rad(117),
        target_TE=.002,target_TR=.04688,exact_rr=true,merge_velocity_prephaser=true,prephaser_duration=.0008,_encoding_axis=:REF)
    println("TE2MS_FEASIBLE ",s.DEF["TE"])
catch e
    println("TE_FEASIBILITY_STOP ",sprint(showerror,e))
end
