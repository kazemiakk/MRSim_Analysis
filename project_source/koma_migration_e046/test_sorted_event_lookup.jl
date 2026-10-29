include(joinpath(@__DIR__,"export_controls.jl"))
include(joinpath(@__DIR__,"sorted_event_lookup.jl"))
using Test, Random
length(ARGS)==1||error("NEW_OUTPUT required")
output=ARGS[1];ispath(output)&&error("Refuse overwrite");mkpath(output)
info=TOML.parsefile(joinpath(E046ROOT,"outputs/upstream_migration_e046/phantom_inputs_v3/case_info.toml"))
seq=read_seq(joinpath(info["sequence_directory"],"ref.seq");verbose=false,verify_signature=true)
insert_verified_zero_markers!(seq,info["seam_time_s"],info["period_s"],2048)
timecurve,flags=h5open(joinpath(E046ROOT,"outputs/upstream_migration_e046/phantom_inputs_v3/case_inputs.h5"),"r") do f
    TimeCurve(t=read(f,"time"),t_unit=read(f,"time_unit"),periodic=true,periods=1f0,
        t_start=Float32(info["t_start_s"]),t_end=Float32(info["t_end_s"])),vec(any(Bool.(read(f,"reset"));dims=1))
end
motion=Motion(action=FlowPath(zeros(Float32,1,length(flags)),zeros(Float32,1,length(flags)),
    zeros(Float32,1,length(flags)),reshape(flags,1,:)),time=timecurve,spins=AllSpins())
events=KomaMRIBase.merge_sampling_times(KomaMRIBase.sequence_boundary_sampling_times(seq),KomaMRIBase.motion_sampling_times(seq,motion))
starts=get_block_start_times(seq);@test issorted(events)
Random.seed!(460046);blocks=unique(vcat([1,length(seq)],rand(1:length(seq),128)))
old_time=@elapsed old=[invoke(KomaMRIBase.block_global_event_times,Tuple{Any,Any,Any},starts,b,events) for b in blocks]
new_time=@elapsed candidate=[KomaMRIBase.block_global_event_times(starts,b,events) for b in blocks]
@test old==candidate
@test KomaMRIBase.block_global_event_times([0.,1.,2.],1,[0.,0.,1.,1.,2.])==[0.,0.,1.,1.]
@test isempty(KomaMRIBase.block_global_event_times([.1,.2],1,[0.,1.]))
# Complete discrete-sequence equivalence on a real source prefix and motion.
prefix=seq[1:120]
E046_SORTED_LOOKUP_ENABLED[]=false
baseline=discretize(prefix;motion,sampling_rule=MaxStepSizeRule(Inf,5e-5))
E046_SORTED_LOOKUP_ENABLED[]=true
optimized=discretize(prefix;motion,sampling_rule=MaxStepSizeRule(Inf,5e-5))
for name in fieldnames(typeof(baseline))
    @test isequal(getfield(baseline,name),getfield(optimized,name))
end
report=Dict("status"=>"PASS_EXACT_SORTED_EVENT_LOOKUP","full_sequence_blocks"=>length(seq),
    "motion_event_count"=>length(events),"sampled_real_blocks_exact"=>length(blocks),
    "full_prefix_discrete_fields_exact"=>true,"closed_endpoints_and_duplicates_exact"=>true,
    "sampled_generic_seconds"=>old_time,"sampled_binary_seconds"=>new_time,
    "helper_sha256"=>hashfile(joinpath(@__DIR__,"sorted_event_lookup.jl")),
    "test_sha256"=>hashfile(@__FILE__),"source_sequence_sha256"=>hashfile(joinpath(info["sequence_directory"],"ref.seq")))
open(joinpath(output,"lookup_equivalence.toml"),"w") do io;TOML.print(io,report);end
println("PASS_EXACT_SORTED_EVENT_LOOKUP ",report)
