# E-063: write the stationary aortic tissue block as a Koma phantom (koma_aorta environment, as E-062).
# Positions from geometry_v1/aorta_static_spins.h5 (MRI frame, metres). rho = 1 (the PD scale is applied in S2),
# T1 0.9 s, T2 0.045 s, no motion.
using KomaMRI, HDF5, SHA, TOML

const E063 = raw"D:\MRSim\outputs\static_background_e063"
out = ARGS[1]
ispath(out) && error("Refuse existing output")
report = read(joinpath(E063, "geometry_v1", "report.json"), String)
occursin("\"status\": \"PASS_E063_S0_GEOMETRY\"", report) || error("E-063 geometry missing")
x, y, z = h5open(joinpath(E063, "geometry_v1", "aorta_static_spins.h5"), "r") do f
    Float32.(read(f, "x")), Float32.(read(f, "y")), Float32.(read(f, "z"))
end
n = length(x)
phantom = Phantom(name="E063_aorta_static_tissue", x=x, y=y, z=z, ρ=ones(Float32, n),
                  T1=fill(Float32(0.9), n), T2=fill(Float32(0.045), n))
mkpath(out)
file = joinpath(out, "aorta_static_tissue.phantom")
write_phantom(phantom, file)
saved = read_phantom(file)
length(saved) == n && saved.motion isa NoMotion || error("Phantom round trip failed")
open(joinpath(out, "report.toml"), "w") do io
    TOML.print(io, Dict("status" => "PASS_E063_AORTA_STATIC_PHANTOM", "spins" => n, "T1_s" => 0.9, "T2_s" => 0.045,
        "phantom_sha256" => bytes2hex(open(sha256, file)), "script_sha256" => bytes2hex(open(sha256, @__FILE__))))
end
println("PASS_E063_AORTA_STATIC_PHANTOM spins=", n)
