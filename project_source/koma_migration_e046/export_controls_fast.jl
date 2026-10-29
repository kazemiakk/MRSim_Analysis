include(joinpath(@__DIR__,"export_controls.jl"))
include(joinpath(@__DIR__,"direct_sort.jl"))
length(ARGS)==2||error("CASE NEW_OUTPUT required")
export_case(ARGS...)
infofile=joinpath(ARGS[2],"case_info.toml")
info=TOML.parsefile(infofile)
info["direct_sort_sha256"]=hashfile(joinpath(@__DIR__,"direct_sort.jl"))
info["export_wrapper_sha256"]=hashfile(@__FILE__)
open(infofile,"w") do io;TOML.print(io,info);end
