using KomaMRI, KomaMRICore, KomaMRIBase, CUDA, HDF5, TOML, SHA

const REPO=raw"D:\MRSim\MRSIM\MRSimTracks"
const HELPER=joinpath(REPO,"koma_migration_e046")
include(joinpath(HELPER,"direct_sort.jl"))
include(joinpath(HELPER,"perfect_spoiling.jl"))
const MODELS=("static","uniform_osc","parabolic_osc")
const ARMS=("baseline","bw700_te274","bw700_te210")
digest(p)=bytes2hex(sha256(read(p)))

function atomic_toml(path,value)
    ispath(path)&&error("Refuse overwrite: "*path)
    open(path*".partial","w") do io;TOML.print(io,value);end
    mv(path*".partial",path)
end

function phantom(input,model)
    h5open(input,"r") do f
        positions=read(f,"positions");rho=read(f,"rho");u=read(f,"u_"*model)
        basis=read(f,"path_basis_s");period=read(f,"motion_period_s")
        length(basis)==4097||error("Path knots changed")
        dx=u.*transpose(basis);dy=zeros(size(dx));dz=zeros(size(dx))
        maximum(abs.(dx[:,1].-dx[:,end]))<1e-14||error("Discontinuous control path")
        motion=Motion(action=KomaMRIBase.Path(dx,dy,dz),
            time=TimeCurve(t=[0.,period],t_unit=[0.,1.],periodic=true),spins=AllSpins())
        return Phantom(name="E056_"*model,x=vec(positions[1,:]),y=vec(positions[2,:]),z=vec(positions[3,:]),
            ρ=vec(rho),T1=vec(read(f,"T1")),T2=vec(read(f,"T2")),motion=motion)
    end
end

function main(mode,run,output)
    mode in ("prefix","run")||error("prefix|run required")
    ispath(output)&&error("Refuse existing output")
    get(ENV,"CUDA_VISIBLE_DEVICES","")=="GPU-8be27592-e325-7b7a-f432-b84f69cb18c6"||error("Reviewed GPU1 required")
    input=joinpath(run,"inputs_v1/controls.h5")
    manifest=joinpath(HELPER,"Manifest.toml")
    previous=TOML.parsefile(joinpath(REPO,"outputs/phantom_precision_e050/mapped_f64_v1/control_report.toml"))
    digest(manifest)==previous["manifest_sha256"]||error("Pinned runtime changed")
    for name in ("direct_sort.jl","perfect_spoiling.jl")
        digest(joinpath(HELPER,name))==previous["helper_sha256"][name]||error("Helper changed")
    end
    sequences=TOML.parsefile(joinpath(run,"sequences_v1/report.toml"))
    sequences["status"]=="PASS_E056_THREE_ARM_SEQUENCE_AND_MOMENT_GATES"||error("Sequence gates missing")
    registration=Dict("experiment"=>"E-056","status"=>"REGISTERED_BEFORE_KOMA_SIGNAL","mode"=>mode,
        "input_sha256"=>digest(input),"script_sha256"=>digest(@__FILE__),
        "protocol_sha256"=>digest(joinpath(@__DIR__,"PROTOCOL.md")),
        "sequence_report_sha256"=>digest(joinpath(run,"sequences_v1/report.toml")),
        "waveform_sha256"=>digest(joinpath(run,"sequences_v1/waveforms.h5")),
        "manifest_sha256"=>digest(manifest),"precision"=>"f64","new_1m"=>false,
        "primary_RF_step_s"=>.5e-6,"gradient_step"=>"Inf",
        "production_RF_step_negative_control_s"=>5e-5,
        "physical_gpu_uuid"=>ENV["CUDA_VISIBLE_DEVICES"],"visible_gpu"=>0)
    mkpath(output);atomic_toml(joinpath(output,"registration.toml"),registration)
    scanner=Scanner(limits=HardwareLimits(B0=3.,Gmax=.04,Smax=180.,B1=10e-6))
    CUDA.device!(0)
    params=Dict{String,Any}("gpu"=>true,"gpu_device"=>0,"Nthreads"=>1,"precision"=>"f64",
        "return_type"=>"mat","Δt"=>Inf,"Δt_rf"=>.5e-6,"max_block_length"=>512,"sim_method"=>MigrationPerfectSpoiling())
    if mode=="run"
        prefix=TOML.parsefile(joinpath(run,"prefix_v1/report.toml"))
        prefix["status"]=="PASS_E056_COMPLETE_POPULATION_PREFIX"||error("Resource prefix incomplete")
        gate=TOML.parsefile(joinpath(run,"prefix_v1/independent_gate.toml"))
        gate["status"]=="PASS_INDEPENDENT_REPEATED_RF_PREFIX"||error("Independent prefix failed")
        for key in ("input_sha256","script_sha256","protocol_sha256","sequence_report_sha256","manifest_sha256")
            prefix[key]==registration[key]||error("Prefix identity mismatch: "*key)
        end
    end
    identities=Dict{String,String}();runtimes=Dict{String,Float64}()
    sequence_cache=Dict{String,Any}()
    models=mode=="prefix" ? ("parabolic_osc",) : MODELS
    arms=mode=="prefix" ? ("baseline",) : ARMS
    encodings=mode=="prefix" ? ("REF","X") : ("REF","X","Y","Z")
    for model in models
        p=phantom(input,model)
        steps=mode=="prefix" ? (("production",5e-5),("fine",.5e-6)) : (("primary",.5e-6),)
        for arm in arms,encoding in encodings,(stepname,rfstep) in steps
            seqfile=joinpath(run,"sequences_v1",arm,lowercase(encoding)*".seq")
            digest(seqfile)==sequences["arms"][arm][encoding]["sequence_sha256"]||error("Sequence changed")
            seq=get!(sequence_cache,seqfile) do
                read_seq(seqfile;verbose=false,verify_signature=true)
            end
            if mode=="prefix"
                adc=findall(a->a.N>0,seq.ADC);seq=copy(seq[1:adc[15]])
            end
            key=model*"_"*arm*"_"*lowercase(encoding)
            mode=="prefix"&&(key*="_"*stepname)
            println("START_E056 ",key);flush(stdout)
            MIGRATION_PREVIOUS_RF[]=false;start=time()
            params["Δt_rf"]=rfstep
            signal=vec(simulate(p,seq,scanner;sim_params=copy(params),verbose=false))
            CUDA.synchronize();runtimes[key]=time()-start
            length(signal)==(mode=="prefix" ? 1200 : 614400)||error("Unexpected sample count")
            all(isfinite,signal)&&eltype(signal)==ComplexF64||error("Invalid signal")
            path=joinpath(output,key*".h5")
            h5open(path*".partial","w") do f
                write(f,"raw_signal_f64",signal)
                if mode=="run"
                    k,image=sorted_image(signal,seq);write(f,"rawsort",k);write(f,"image",image)
                end
                HDF5.attributes(f)["model"]=model;HDF5.attributes(f)["arm"]=arm;HDF5.attributes(f)["encoding"]=encoding
            end
            mv(path*".partial",path);identities[key]=digest(path)
            atomic_toml(joinpath(output,key*"_checkpoint.toml"),Dict("status"=>"COMPLETE_ENCODING",
                "seconds"=>runtimes[key],"sha256"=>identities[key]))
            println("FINISHED_E056 ",key," seconds=",runtimes[key]);flush(stdout)
            GC.gc();CUDA.reclaim()
        end
    end
    report=merge(registration,Dict("status"=>mode=="prefix" ? "PASS_E056_COMPLETE_POPULATION_PREFIX" : "COMPLETE_E056_SYNTHETIC_3D_ACQUISITIONS",
        "result_sha256"=>identities,"runtime_s"=>runtimes))
    atomic_toml(joinpath(output,"report.toml"),report)
    println(report["status"])
end

length(ARGS)==3||error("prefix|run RUN_ROOT NEW_OUTPUT required")
main(ARGS...)
