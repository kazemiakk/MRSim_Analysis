using KomaMRI, SHA, TOML, Unitful
const NROOT=normpath(joinpath(@__DIR__,".."))
include(joinpath(@__DIR__,"gradient_design.jl"))
include(joinpath(@__DIR__,"utils.jl"))
include(joinpath(@__DIR__,"cartesian.jl"))
include(joinpath(@__DIR__,"excitation.jl"))
include(joinpath(@__DIR__,"gre.jl"))
include(joinpath(@__DIR__,"PC.jl"))

"""Unchanged shot physics; native0.72s triggers,16 bins and explicit idle."""
function native_sequence(axis::Symbol,sys;bins=16,period=.72,guard=.001)
    fov=(.28,.128,.128);matrix=(72,32,32);venc=2.5;flip=deg2rad(10);spoil=deg2rad(117)
    dwell=max(round(Int,.001/(matrix[1]-1)/sys.ADC_Δt),1)*sys.ADC_Δt;bw=1/(matrix[1]*dwell)
    excitation=slice_selective_sinc(flip,fov[3],sys;BW=nothing)
    fixed_area=ntuple(k->_pc_gradient_area(excitation.GR[k,2]),3)
    readout=sgre_base(fov,matrix,sys,bw;fixed_area,spoil_area=(0.0,0.0,0.0))
    moment=_pc_first_moment(venc);flat,ramp=_pc_bipolar_timing(moment,sys)
    reference_module=_pc_velocity_module(:SS,0.0,flat,ramp,sys)
    module_=axis===:REF ? reference_module : _pc_velocity_module(axis,moment,flat,ramp,sys)
    spatial=[(ky,kz) for kz in -16:15 for ky in -16:15]
    kernels=Dict(p=>readout(p...) for p in spatial)
    shot(k,b,rf;scale=1.0)=_pc_shot(excitation,kernels[k],module_,1,b,rf,spoil;flip_scale=scale,write_set_label=false,write_phs_label=true)
    tr=Float64(dur(shot(first(spatial),1,1)))
    interval=period/bins;lines=floor(Int,(interval-guard)/tr)
    lines>0||error("No shots fit native bin")
    pad=interval-lines*tr
    pad>guard||error("Insufficient boundary idle guard")
    # Exactly representable on the sequence block raster, no hidden RR rounding.
    abs(round(pad/sys.DUR_Δt)*sys.DUR_Δt-pad)<1e-10||error("Bin idle is off raster")
    seq=Sequence(sys);rf=1
    @addblock begin
        for scale in range(min(1.0,deg2rad(3)/flip),1.0;length=10)
            ramp_shot=_pc_shot(excitation,kernels[first(spatial)],reference_module,1,1,rf,spoil;flip_scale=scale,write_set_label=false,write_phs_label=true)
            seq+=_pc_disable_adc!(ramp_shot);rf=rf+1
        end
        initial_idle=period-Float64(dur(seq))
        initial_idle>0||error("Ramp exceeds initial native cycle")
        seq+=make_delay(initial_idle*u"s")
        for firstindex_ in 1:lines:length(spatial)
            encodings=spatial[firstindex_:min(firstindex_+lines-1,length(spatial))]
            seq+=make_delay(guard*u"s")
            for b in 1:bins
                for k in encodings
                    seq+=shot(k,b,rf);rf=rf+1
                end
                for _ in length(encodings)+1:lines
                    seq+=_pc_disable_adc!(shot(last(encodings),b,rf));rf=rf+1
                end
                seq+=make_delay((pad-(b==bins ? guard : 0.0))*u"s")
            end
        end
    end
    seq.DEF["FOV"]=collect(fov);seq.DEF["Nx"]=72;seq.DEF["Ny"]=32;seq.DEF["Nz"]=32
    seq.DEF["TR"]=tr;seq.DEF["RR"]=period;seq.DEF["CardiacBins"]=bins;seq.DEF["LinesPerCardiacBin"]=lines
    seq.DEF["BandwidthPerPixel"]=bw;seq.DEF["Venc"]=venc;seq.DEF["VelocityEncodingAxis"]=String(axis)
    seq.DEF["Encoding"]=axis===:RO ? "X" : axis===:PE ? "Y" : axis===:SS ? "Z" : "REF"
    seq.DEF["RfSpoilIncrement"]=spoil;seq.DEF["NativeSourceStartFrame"]=0;seq.DEF["CycleStartGuard"]=guard
    return seq,Dict("TR_s"=>tr,"period_s"=>period,"phase_interval_s"=>interval,"lines_per_bin"=>lines,"bin_idle_s"=>pad,"cycle_start_guard_s"=>guard,"initial_ramp_cycle_s"=>period)
end

function generate_native_sequences(output)
    ispath(output)&&error("Refuse overwrite: $output");mkpath(output)
    sys=Scanner(Gmax=40e-3,Smax=180,B1=10e-6);details=Dict{String,Any}()
    for (name,axis) in zip(("REF","X","Y","Z"),(:REF,:RO,:PE,:SS))
        seq,timing=native_sequence(axis,sys)
        check_timing(seq,sys);check_hw_limits(seq,sys)
        labels=get_labels(seq);adcs=[i for (i,event) in enumerate(seq) if is_ADC_on(event)]
        tuples=[(Int(labels[i].LIN),Int(labels[i].PAR),Int(labels[i].PHS)) for i in adcs]
        length(tuples)==length(unique(tuples))==32*32*16||error("Incomplete/duplicate acquisition labels")
        path=joinpath(output,lowercase(name)*".seq");write_seq(seq,path)
        details[name]=merge(timing,Dict("sha256"=>bytes2hex(sha256(read(path))),"profiles"=>length(tuples)))
    end
    report=Dict("experiment"=>"E-045","status"=>"PASS_NATIVE_SEQUENCE_LABELS_AND_HARDWARE","source_start_frame"=>0,"native_period_s"=>.72,"cardiac_bins"=>16,"matrix"=>[72,32,32],"FOV_m"=>[.28,.128,.128],"Venc_m_per_s"=>2.5,"sequence"=>details,"code_sha256"=>bytes2hex(sha256(read(@__FILE__))))
    open(joinpath(output,"native_sequence_report.toml"),"w") do io;TOML.print(io,report);end
    println("PASS_NATIVE_SEQUENCE_GENERATION ",output)
end

if abspath(PROGRAM_FILE)==@__FILE__
    length(ARGS)==1||error("Usage: native_zero_sequences.jl NEW_SEQUENCE_DIR")
    generate_native_sequences(ARGS[1])
end
