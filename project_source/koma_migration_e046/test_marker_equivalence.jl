include(joinpath(@__DIR__,"export_controls.jl"))
using Test
seq=Sequence()
for b in 1:3
    global seq+=Sequence(reshape([Grad(0.,1.),Grad(0.,1.),Grad(0.,1.)],3,1),
        reshape([RF(0.,1.)],1,1),[ADC(0,0.,0.)],[1.])
end
original=insert_periodic_seam_reset_markers(seq,.5,1.,3).sequence
candidate=insert_verified_zero_markers!(deepcopy(seq),.5,1.,3)
@test original.DUR==candidate.DUR
@test _event_arrays_exactly_equal(original.RF,candidate.RF)
@test _event_arrays_exactly_equal(original.GR,candidate.GR)
@test _event_arrays_exactly_equal(original.ADC,candidate.ADC)
@test original.EXT==candidate.EXT
println("PASS_ZERO_MARKER_EQUIVALENCE_5_CHECKS")
