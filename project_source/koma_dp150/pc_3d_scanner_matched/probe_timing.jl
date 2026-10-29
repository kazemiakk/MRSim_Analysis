using KomaMRI

root = dirname(@__DIR__)
include(joinpath(root, "gradient_design.jl"))
include(joinpath(root, "utils.jl"))
include(joinpath(root, "cartesian.jl"))
include(joinpath(root, "excitation.jl"))
include(joinpath(root, "gre.jl"))
include(joinpath(root, "PC.jl"))

matrix = (400, 64, 32)
FOV = (0.5, 0.08, 0.08)
sys = Scanner(B0=3.0, Gmax=40e-3, Smax=180, B1=10e-6,
    ADC_Δt=1e-9, DUR_Δt=1e-9)
excitation = slice_selective_sinc(
    deg2rad(8), FOV[3], sys; TBP=1)
fixed_area = ntuple(axis -> _pc_gradient_area(excitation.GR[axis, 2]), 3)
readout_kernel = sgre_base(FOV, matrix, sys, 350.0;
    fixed_area, spoil_area=(0.0, 0.0, 0.0))
readout = readout_kernel(-(matrix[2] ÷ 2), -(matrix[3] ÷ 2))
readout = _pc_retime_prephaser(readout, 0.8e-3, sys)
first_moment = _pc_first_moment(4.5)
flat, ramp = _pc_bipolar_timing(first_moment, sys)
velocity, trailing, merged_area = _pc_merged_velocity_events(
    :RO, first_moment, flat, ramp, readout.DUR[1], sys)
readout = _pc_merge_prephaser(readout, :RO, trailing)
shot = _pc_shot(excitation, readout, velocity, 1, 1, 1,
    deg2rad(117); write_set_label=false, write_phs_label=true)
check_timing(shot, sys)
check_hw_limits(shot, sys)

println("excitation block durations = ", excitation.DUR)
println("velocity block durations = ", velocity.DUR)
println("readout block durations = ", readout.DUR)
println("shot block durations = ", shot.DUR)
println("minimum TE ms = ", 1e3 * _pc_echo_time(shot, matrix[1]))
println("minimum TR ms = ", 1e3 * dur(shot))
println("velocity flat/ramp us = ", 1e6 .* (flat, ramp))
println("fixed area = ", fixed_area)
println("merged velocity lobe area = ", merged_area)
