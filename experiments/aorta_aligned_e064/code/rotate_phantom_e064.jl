# E-064: rigidly rotate the E-062 1M-spin aorta FlowPath phantom into the candy-cane scanner frame.
# Base positions are zero and dx/dy/dz are absolute paths (koma_aorta build_motion_phantom.jl), so
# new paths = R * (path - c); resets, time curve, spin range and tissue values are unchanged.
using KomaMRI, LinearAlgebra, SHA, TOML   # KomaMRIBase types are re-exported by KomaMRI (not a direct dependency)

const E064 = raw"D:\MRSim\outputs\aorta_aligned_e064"
const SOURCE = raw"D:\MRSim\MRSIM\MRSimTracks\outputs\aorta_dp2_mri_pipeline\koma\multibin_8x4mm\aorta_motion_1m_8bin.phantom"
const SOURCE_SHA = "d23241331c8ec57e0e0396ef5468e1902583e7d2095c6ece83e3c71854242720"

out = ARGS[1]
ispath(out) && error("Refuse existing output")
frame = read(joinpath(E064, "frame_v1", "frame.json"), String)
occursin("\"status\": \"PASS_E064_S0A_FRAME\"", frame) || error("E-064 frame missing")
grab(key) = parse.(Float64, split(strip(match(Regex("\"$key\": \\[([^\\]]*)\\]"), frame).captures[1]), ","))
rows = [parse.(Float64, split(strip(m.captures[1]), ",")) for m in
        eachmatch(r"\[\s*(-?[0-9.eE+-]+,\s*-?[0-9.eE+-]+,\s*-?[0-9.eE+-]+)\s*\]", split(split(frame, "\"rotation_rows_e2_e1_n\":")[2], "\"det\"")[1])]
length(rows) == 3 || error("rotation rows not parsed")
R = permutedims(hcat(rows...))                                              # rows e2, e1, n
c = grab("centre_mri_m"); fov = grab("fov_m")
abs(det(R) - 1) < 1e-9 || error("not a proper rotation")
bytes2hex(open(sha256, SOURCE)) == SOURCE_SHA || error("source phantom changed")
@info "Reading source phantom" SOURCE
p = read_phantom(SOURCE)
fp = p.motion.action
all(iszero, p.x) && all(iszero, p.y) && all(iszero, p.z) || error("expected zero base positions")
n, nt = size(fp.dx)
Rf = Float32.(R); cf = Float32.(c)
new = [similar(fp.dx) for _ in 1:3]
for k in 1:3
    new[k] .= Rf[k, 1] .* (fp.dx .- cf[1]) .+ Rf[k, 2] .* (fp.dy .- cf[2]) .+ Rf[k, 3] .* (fp.dz .- cf[3])
end
# Inverse check on every spin at every 40th sample.
idx = 1:40:nt
back = [Rf[1, k] .* new[1][:, idx] .+ Rf[2, k] .* new[2][:, idx] .+ Rf[3, k] .* new[3][:, idx] .+ cf[k] for k in 1:3]
inv_err = maximum(maximum(abs.(back[k] .- (k == 1 ? fp.dx : k == 2 ? fp.dy : fp.dz)[:, idx])) for k in 1:3)
maxabs = [maximum(abs, new[k]) for k in 1:3]
margins = fov ./ 2 .- maxabs
inv_err <= 1e-6 || error("inverse rotation error $(inv_err)")
all(margins .>= 4e-3) || error("positions too close to the FOV edge: $(margins)")
flow = FlowPath(dx=new[1], dy=new[2], dz=new[3], spin_reset=fp.spin_reset)
motion = Motion(action=flow, time=p.motion.time, spins=p.motion.spins)
q = Phantom(name="aorta_motion_1m_candycane_frame", x=p.x, y=p.y, z=p.z, ρ=p.ρ, T1=p.T1, T2=p.T2, motion=motion)
mkpath(out)
file = joinpath(out, "aorta_motion_1m_candycane.phantom")
write_phantom(q, file)
open(joinpath(out, "report.toml"), "w") do io
    TOML.print(io, Dict("status" => "PASS_E064_PHANTOM_ROTATION", "spins" => n, "samples" => nt, "inverse_max_error_m" => inv_err,
        "max_abs_new_m" => maxabs, "half_margin_m" => margins, "rotation_rows_e2_e1_n" => [R[k, :] for k in 1:3], "centre_mri_m" => c,
        "source_sha256" => SOURCE_SHA, "phantom_sha256" => bytes2hex(open(sha256, file)), "script_sha256" => bytes2hex(open(sha256, @__FILE__))))
end
println("PASS_E064_PHANTOM_ROTATION spins=", n, " margins_m=", margins, " inverse_err=", inv_err)
