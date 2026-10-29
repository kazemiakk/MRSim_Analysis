using TOML, SHA
length(ARGS)==4||error("ARM CASE_DIR MAP_DIR NEW_OUTPUT required")
gate=joinpath(@__DIR__,"../outputs/upstream_migration_e046/event_lookup_v1/lookup_equivalence.toml")
report=TOML.parsefile(gate)
report["status"]=="PASS_EXACT_SORTED_EVENT_LOOKUP"||error("Lookup equivalence gate missing")
hash(p)=bytes2hex(sha256(read(p)))
helper=joinpath(@__DIR__,"sorted_event_lookup.jl")
hash(helper)==report["helper_sha256"]||error("Lookup helper changed")
ispath(ARGS[4])&&error("Refuse overwrite")
TOML.parsefile(joinpath(ARGS[2],"case_info.toml"))["case"]=="phantom"||error("Phantom only wrapper")
registration=ARGS[4]*"_sampling_registration.toml"
ispath(registration)&&error("Refuse registration overwrite")
open(registration,"w") do io
    TOML.print(io,Dict("status"=>"REGISTERED_BEFORE_SIGNAL","helper_sha256"=>hash(helper),
        "wrapper_sha256"=>hash(@__FILE__),"equivalence_gate_sha256"=>hash(gate),
        "worker_sha256"=>hash(joinpath(@__DIR__,"run_control.jl")),
        "input_sha256"=>hash(joinpath(ARGS[2],"case_inputs.h5"))))
end
include(helper)
include(joinpath(@__DIR__,"run_control.jl"))
# Bind this exact-equivalent local acceleration separately from the stock worker.
binding=Dict("status"=>"COMPLETE_PHANTOM_EXACT_LOOKUP_CONTROL","helper_sha256"=>hash(helper),
    "wrapper_sha256"=>hash(@__FILE__),"equivalence_gate_sha256"=>hash(gate),
    "sampling_registration_sha256"=>hash(registration),
    "control_report_sha256"=>hash(joinpath(ARGS[4],"control_report.toml")),
    "installed_upstream_sources_modified"=>false,"numerical_sampling_rule_changed"=>false)
open(joinpath(ARGS[4],"sampling_control.toml"),"w") do io;TOML.print(io,binding);end
