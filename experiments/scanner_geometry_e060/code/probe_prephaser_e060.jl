# E-060 S0a2: shortest feasible prephaser and resulting minimum TE (no signal, no files kept).
using KomaMRI, KomaMRIBase
for name in ("gradient_design.jl", "utils.jl", "cartesian.jl", "excitation.jl", "gre.jl", "PC_e060.jl")
    include(joinpath(@__DIR__, "sequence_helpers", name))
end
const FOV = (0.08, 0.08, 0.075)
const MATRIX = (64, 64, 30)
const VENC = (1.5, 1.5, 3.0)

function feasible(gmax, smax, tbp, window, prephaser)
    sys = Scanner(limits=HardwareLimits(B0=3., Gmax=gmax, Smax=smax, B1=10e-6, ADC_Δt=1e-9, DUR_Δt=1e-9))
    tes = Float64[]
    for axis in (:REF, :RO, :PE, :SS)
        try
            seq = build_pc(FOV, MATRIX, sys; venc=VENC, cardiac_bins=2, RR=1.012, BWpp=350.0,
                flip_angle=deg2rad(8), rf_time_bandwidth_product=tbp, rf_window=window, exact_rf_flip=true,
                spoil_phase=0.0, n_ramp_shots=0, trigger_channel=nothing, rf_spoil_increment=deg2rad(117),
                target_TE=nothing, target_TR=23.44e-3, exact_rr=false, merge_velocity_prephaser=true,
                prephaser_duration=prephaser, slab_thickness=0.06, _encoding_axis=axis)
            check_hw_limits(seq, sys)
            push!(tes, seq.DEF["TE"])
        catch
            return nothing
        end
    end
    return maximum(tes)
end

for (label, gmax, smax) in (("40mTm_180", 40e-3, 180.), ("45mTm_200_Skyra", 45e-3, 200.))
    for (tbp, window) in ((4, 0.46), (1, 0.0))
        for prephaser in 0.8e-3:0.05e-3:1.6e-3
            te = feasible(gmax, smax, tbp, window, prephaser)
            if !isnothing(te)
                println("E060_FEASIBLE ", label, " TBP", tbp, " prephaser_ms=", round(1e3*prephaser, digits=2),
                        " min_TE_ms=", round(1e3*te, digits=4))
                break
            end
        end
    end
end
