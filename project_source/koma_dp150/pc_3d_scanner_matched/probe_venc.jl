using KomaMRI

function first_shot(sequence)
    adc_block = findfirst(adc -> adc.N > 0, sequence.ADC)
    rf_blocks = [
        block for block in 1:adc_block
        if any(rf -> dur(rf) > 0, view(sequence.RF, :, block))
    ]
    return sequence[last(rf_blocks):adc_block]
end

sequences = Dict(
    encoding => read_seq(joinpath(@__DIR__, "$(lowercase(encoding)).seq"))
    for encoding in ("REF", "X", "Y", "Z")
)

for step in (5e-6, 1e-6, 2e-7)
    moments = Dict{String,Vector{Float64}}()
    for (encoding, sequence) in sequences
        _, m1_adc = get_M1(
            first_shot(sequence);
            sampling_rule=MaxStepSizeRule(1, step),
        )
        center = Int(sequence.DEF["Nx"]) ÷ 2 + 1
        moments[encoding] = vec(m1_adc[center, :])
    end
    println("sampling step = $step s")
    for (encoding, axis) in (("X", 1), ("Y", 2), ("Z", 3))
        delta = moments[encoding] .- moments["REF"]
        derived_venc = 1 / (2abs(delta[axis]))
        println("  $encoding: delta M1 = $delta, VENC = $derived_venc m/s")
    end
end
