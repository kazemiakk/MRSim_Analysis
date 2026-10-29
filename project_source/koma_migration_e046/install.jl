using Pkg, TOML, SHA
length(ARGS)==1 || error("Usage: install.jl READ_ONLY_LEGACY_DEPOT")
length(DEPOT_PATH)==1 || error("Start with the new isolated depot only")
primary=abspath(DEPOT_PATH[1]);legacy=abspath(ARGS[1])
primary!=legacy || error("Refuse installing into legacy depot")
Pkg.Registry.add("General")
push!(DEPOT_PATH,legacy) # read-through package/artifact cache; new registry wins
Pkg.resolve()
Pkg.instantiate(;update_registry=false,allow_autoprecomp=false)
dependencies=Pkg.dependencies()
report=Dict("status"=>"PASS_ISOLATED_UPSTREAM_INSTALL", "julia_version"=>string(VERSION),
    "primary_depot"=>primary,"legacy_read_through_depot"=>legacy,
    "project_sha256"=>bytes2hex(sha256(read(joinpath(@__DIR__,"Project.toml")))),
    "manifest_sha256"=>bytes2hex(sha256(read(joinpath(@__DIR__,"Manifest.toml")))),
    "packages"=>Dict(info.name=>Dict("version"=>string(info.version),"source"=>info.source)
        for info in values(dependencies) if startswith(info.name,"Koma") || info.name=="CUDA"))
path=joinpath(@__DIR__,"install_report.toml")
isfile(path)&&error("Refuse overwrite install report")
open(path,"w") do io;TOML.print(io,report);end
println("PASS_ISOLATED_UPSTREAM_INSTALL")
