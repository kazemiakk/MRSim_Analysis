# E-060 S0a: minimum achievable TE for the scanner-geometry protocol (no signal, no files kept).
using KomaMRI, KomaMRIBase
for name in ("gradient_design.jl", "utils.jl", "cartesian.jl", "excitation.jl", "gre.jl", "PC_e060.jl")
    include(joinpath(@__DIR__, "sequence_helpers", name))
end
const FOV = (0.08, 0.08, 0.075)
const MATRIX = (64, 64, 30)
const VENC = (1.5, 1.5, 3.0)

function probe(label, gmax, smax, tbp, window; target_TE=nothing)
    sys = Scanner(limits=HardwareLimits(B0=3., Gmax=gmax, Smax=smax, B1=10e-6, ADC_Δt=1e-9, DUR_Δt=1e-9))
    for axis in (:REF, :RO, :PE, :SS)
        try
            seq = build_pc(FOV, MATRIX, sys; venc=VENC, cardiac_bins=2, RR=1.012, BWpp=350.0,
                flip_angle=deg2rad(8), rf_time_bandwidth_product=tbp, rf_window=window, exact_rf_flip=true,
                spoil_phase=0.0, n_ramp_shots=0, trigger_channel=nothing, rf_spoil_increment=deg2rad(117),
                target_TE=target_TE, target_TR=23.44e-3, exact_rr=false, merge_velocity_prephaser=true,
                prephaser_duration=0.8e-3, slab_thickness=0.06, _encoding_axis=axis)
            check_hw_limits(seq, sys)
            println("E060_TE ", label, " ", axis, " TE_ms=", round(1e3*seq.DEF["TE"], digits=4), " hw=PASS")
        catch err
            println("E060_TE ", label, " ", axis, " FAIL ", sprint(showerror, err)[1:min(end, 160)])
        end
    end
end

probe("min_TBP4_a046_40mTm", 40e-3, 180., 4, 0.46)
probe("min_TBP1_a0_40mTm", 40e-3, 180., 1, 0.0)
probe("min_TBP4_a046_45mTm_200", 45e-3, 200., 4, 0.46)
probe("target274_TBP4_a046_40mTm", 40e-3, 180., 4, 0.46; target_TE=2.74e-3)
