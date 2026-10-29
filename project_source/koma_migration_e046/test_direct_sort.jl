using KomaMRI, KomaMRICore, FFTW, Test, HDF5, SHA, TOML
include(joinpath(@__DIR__,"direct_sort.jl"))
seq=Sequence()
for b in 0:1,z in 0:1,y in 0:1
    part=Sequence(reshape([Grad(0.01,1e-3),Grad(0.0,1e-3),Grad(0.0,1e-3)],3,1),
        reshape([RF(0.0,1e-3)],1,1),[ADC(4,1e-3,0.0)],[1e-3])
    part.EXT[1]=[LabelSet(y,"LIN"),LabelSet(z,"PAR"),LabelSet(b,"PHS")]
    global seq+=part
end
seq.DEF["Nx"]=4;seq.DEF["Ny"]=2;seq.DEF["Nz"]=2;seq.DEF["CardiacBins"]=2
seq.DEF["FOV"]=[.1,.1,.1]
signal=reshape(ComplexF32.(collect(1:32),collect(32:-1:1)),:,1,1)
k,image=sorted_image(signal,seq)
raw=KomaMRICore.signal_to_raw_data(signal,seq)
reference=zeros(ComplexF32,4,2,2,2)
for p in raw.profiles
    reference[:,Int(p.head.idx.kspace_encode_step_1)+1,Int(p.head.idx.kspace_encode_step_2)+1,Int(p.head.idx.phase)+1].=p.data[:,1]
end
@test k==reference
@test image==ComplexF32.(fftshift(ifft(ifftshift(reference,(1,2,3)),(1,2,3)),(1,2,3)))
@test_throws Exception sorted_image(signal[1:end-1],seq)
println("PASS_DIRECT_CARTESIAN_SORT_MATCHES_KOMA_RAW_SORT")
