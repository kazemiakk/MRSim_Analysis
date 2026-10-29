"""Sort unchanged Cartesian ADC samples by exact Pulseq labels, without computing an unused trajectory."""
function sorted_image(signal,seq)
    shape=Int.([seq.DEF["Nx"],seq.DEF["Ny"],seq.DEF["Nz"],seq.DEF["CardiacBins"]])
    k=zeros(ComplexF32,shape...);filled=falses(shape[2:end]...)
    labels=get_labels(seq);offset=0
    for b in eachindex(seq.ADC)
        samples=seq.ADC[b].N
        samples==0&&continue
        samples==shape[1]||error("Non-Cartesian sample count")
        y=Int(labels[b].LIN)+1;z=Int(labels[b].PAR)+1;phase=Int(labels[b].PHS)+1
        filled[y,z,phase]&&error("Duplicate profile")
        filled[y,z,phase]=true
        k[:,y,z,phase].=vec(signal[offset+1:offset+samples])
        offset+=samples
    end
    offset==length(signal)&&all(filled)||error("Incomplete ADC profiles")
    image=ComplexF32.(fftshift(ifft(ifftshift(k,(1,2,3)),(1,2,3)),(1,2,3)))
    return k,image
end
