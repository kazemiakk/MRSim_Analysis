using KomaMRI, TOML
include(joinpath(@__DIR__,"export_controls.jl"))
println("PROBE_IMPORT_COMPLETE");flush(stdout)
start=time()
seq=read_seq(joinpath(E046ROOT,"koma_dp150/pc_3d_scanner_matched_venc5/ref.seq");verbose=false,verify_signature=true)
println("PROBE_READ_SECONDS ",time()-start," blocks=",length(seq));flush(stdout)
start=time()
insert_verified_zero_markers!(seq,0.4379376769065857,1.0120000839233398,2048)
println("PROBE_MARKER_SECONDS ",time()-start);flush(stdout)
