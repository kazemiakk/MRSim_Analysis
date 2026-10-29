"""Derive sequence_helpers/PC_e060.jl from the hash-checked E058 copy of PC.jl.

Two changes only: per-axis `venc` (scalar still allowed) and a `slab_thickness`
that may differ from the partition FOV. Every replacement must match exactly once.
"""
import hashlib
from pathlib import Path
import shutil

SOURCE = Path("D:/MRSim/outputs/paired_acquisition_e058/code/sequence_helpers")
TARGET = Path("D:/MRSim/outputs/scanner_geometry_e060/code/sequence_helpers")
E056_HASH = "5e00682f869e9474dfe69e6018f5d886bdddd29e1c072fef16ea269db2c9ef25"
UNCHANGED = ("gradient_design.jl", "utils.jl", "cartesian.jl", "excitation.jl", "gre.jl")

EDITS = [
    ("    exact_rf_flip=false,\n    _encoding_axis=nothing,\n)",
     "    exact_rf_flip=false,\n    slab_thickness=nothing,\n    _encoding_axis=nothing,\n)"),
    ('    venc > 0 || error("venc must be positive.")\n',
     '    vencs = venc isa Number ? (Float64(venc), Float64(venc), Float64(venc)) : Tuple(Float64.(venc))\n'
     '    length(vencs) == 3 && all(>(0), vencs) || error("venc must be positive: scalar or (RO, PE, SS).")\n'
     '    isnothing(slab_thickness) || slab_thickness > 0 || error("slab_thickness must be positive.")\n'),
    ("        flip_angle,\n        FOV[3],\n",
     "        flip_angle,\n        isnothing(slab_thickness) ? FOV[3] : Float64(slab_thickness),\n"),
    ("    first_moment = _pc_first_moment(venc)\n    flat, ramp = _pc_bipolar_timing(first_moment, sys)\n",
     "    # Shared lobe timing from the largest first moment; each axis keeps pi/VENC_axis.\n"
     "    timing_moment = _pc_first_moment(minimum(vencs))\n"
     "    flat, ramp = _pc_bipolar_timing(timing_moment, sys)\n"
     "    axis_moment(axis) = _pc_first_moment(vencs[_pc_axis_index(axis)])\n"
     "    first_moment = isnothing(_encoding_axis) || _encoding_axis === :REF ?\n"
     "        axis_moment(velocity_axis) : axis_moment(_encoding_axis)\n"),
    ('    sequence.DEF["Venc"] = Float64(venc)\n',
     '    sequence.DEF["Venc"] = isnothing(_encoding_axis) || _encoding_axis === :REF ?\n'
     '        minimum(vencs) : vencs[_pc_axis_index(_encoding_axis)]\n'
     '    sequence.DEF["VencRO"] = vencs[1]\n'
     '    sequence.DEF["VencPE"] = vencs[2]\n'
     '    sequence.DEF["VencSS"] = vencs[3]\n'
     '    sequence.DEF["SlabThickness"] = isnothing(slab_thickness) ? FOV[3] : Float64(slab_thickness)\n'),
]


def main():
    if TARGET.exists():
        raise FileExistsError(TARGET)
    text = (SOURCE/"PC.jl").read_text(encoding="utf-8")
    if hashlib.sha256(text.encode("utf-8")).hexdigest() != E056_HASH:
        raise ValueError("Source PC.jl differs from the E056/E058 hash")
    for old, new in EDITS:
        if text.count(old) != 1:
            raise ValueError("Expected exactly one match: "+old[:60])
        text = text.replace(old, new)
    TARGET.mkdir(parents=True)
    for name in UNCHANGED:
        shutil.copy2(SOURCE/name, TARGET/name)
    (TARGET/"PC_e060.jl").write_text(text, encoding="utf-8", newline="\n")
    for p in sorted(TARGET.iterdir()):
        print(p.name, hashlib.sha256(p.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
