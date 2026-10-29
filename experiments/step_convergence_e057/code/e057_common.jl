# Shared E-057 identities. The phantom and simulation parameters are copied
# verbatim from E056 code/run_acquisition.jl; only the sampling rule may differ.
const REPO=raw"D:\MRSim\MRSIM\MRSimTracks"
const HELPER=joinpath(REPO,"koma_migration_e046")
const E056=raw"D:\MRSim\outputs\acquisition_physics_e056"
const ROOT=raw"D:\MRSim\outputs\step_convergence_e057"
const GPU1="GPU-8be27592-e325-7b7a-f432-b84f69cb18c6"
include(joinpath(HELPER,"perfect_spoiling.jl"))
include(joinpath(@__DIR__,"precession_step_rule.jl"))
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

# E056 primary parameters; Δt_prec=nothing reproduces E056 with no custom rule.
function e057_params(Δt_prec)
    params=Dict{String,Any}("gpu"=>true,"gpu_device"=>0,"Nthreads"=>1,"precision"=>"f64",
        "return_type"=>"mat","Δt"=>Inf,"Δt_rf"=>.5e-6,"max_block_length"=>512,"sim_method"=>MigrationPerfectSpoiling())
    isnothing(Δt_prec)||(params["sampling_rule"]=PrecessionMaxStepRule(Δt_prec,.5e-6))
    return params
end

# Bind the frozen runtime, helpers, inputs and sequences to the E056 receipts.
function verify_e056_identities()
    manifest=joinpath(HELPER,"Manifest.toml")
    previous=TOML.parsefile(joinpath(REPO,"outputs/phantom_precision_e050/mapped_f64_v1/control_report.toml"))
    digest(manifest)==previous["manifest_sha256"]||error("Pinned runtime changed")
    for name in ("direct_sort.jl","perfect_spoiling.jl")
        digest(joinpath(HELPER,name))==previous["helper_sha256"][name]||error("Helper changed")
    end
    acquisition=TOML.parsefile(joinpath(E056,"acquisition_v1/report.toml"))
    acquisition["status"]=="COMPLETE_E056_SYNTHETIC_3D_ACQUISITIONS"||error("E056 acquisition receipt changed")
    input=joinpath(E056,"inputs_v1/controls.h5")
    digest(input)==acquisition["input_sha256"]||error("E056 control input changed")
    sequences=TOML.parsefile(joinpath(E056,"sequences_v1/report.toml"))
    digest(joinpath(E056,"sequences_v1/report.toml"))==acquisition["sequence_report_sha256"]||error("Sequence report changed")
    return (manifest_sha256=digest(manifest),input=input,input_sha256=acquisition["input_sha256"],
        sequences=sequences,acquisition=acquisition)
end

function load_sequence(sequences,arm,encoding)
    seqfile=joinpath(E056,"sequences_v1",arm,lowercase(encoding)*".seq")
    digest(seqfile)==sequences["arms"][arm][encoding]["sequence_sha256"]||error("Sequence changed")
    return read_seq(seqfile;verbose=false,verify_signature=true),digest(seqfile)
end

# Exactly the discretization simulate() performs for these parameters.
function simulated_grid(seq,obj,params)
    p=KomaMRICore.default_sim_params(copy(params))
    rule=KomaMRICore.simulation_sampling_rule(p["sim_method"],p)
    return discretize(seq;sampling_rule=rule,motion=obj.motion,freq_in_phase=p["freq_in_phase"])
end
