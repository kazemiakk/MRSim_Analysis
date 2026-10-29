import Pkg

if isempty(Pkg.Registry.reachable_registries())
    Pkg.Registry.add("General")
end

Pkg.resolve()
Pkg.add(["CUDA", "Unitful"])
Pkg.instantiate()
Pkg.precompile()
