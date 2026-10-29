# E-064 (copy of E-063 builder): write the stationary aortic tissue block as a Koma phantom (koma_aorta environment, as E-062).
# Positions from frame_v1/tissue_spins.h5 (candy-cane scanner frame, metres). rho = 1 (the PD scale is applied in S2),
# T1 0.9 s, T2 0.045 s, no motion.
using KomaMRI, HDF5, SHA, TOML

const E064 = raw"D:\MRSim\outputs\aorta_aligned_e064"
out = ARGS[1]
ispath(out) && error("Refuse existing output")
report = read(joinpath(E064, "frame_v1", "frame.json"), String)
occursin("\"status\": \"PASS_E064_S0A_FRAME\"", report) || error("E-064 frame missing")
x, y, z = h5open(joinpath(E064, "frame_v1", "tissue_spins.h5"), "r") do f
    Float32.(read(f, "x")), Float32.(read(f, "y")), Float32.(read(f, "z"))
end
n = length(x)
phantom = Phantom(name="E064_aorta_static_tissue_candycane", x=x, y=y, z=z, ρ=ones(Float32, n),
                  T1=fill(Float32(0.9), n), T2=fill(Float32(0.045), n))
mkpath(out)
file = joinpath(out, "aorta_static_tissue.phantom")
write_phantom(phantom, file)
saved = read_phantom(file)
length(saved) == n && saved.motion isa NoMotion || error("Phantom round trip failed")
open(joinpath(out, "report.toml"), "w") do io
    TOML.print(io, Dict("status" => "PASS_E064_TISSUE_PHANTOM", "spins" => n, "T1_s" => 0.9, "T2_s" => 0.045,
        "phantom_sha256" => bytes2hex(open(sha256, file)), "script_sha256" => bytes2hex(open(sha256, @__FILE__))))
end
println("PASS_E064_TISSUE_PHANTOM spins=", n)
