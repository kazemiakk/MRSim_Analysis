# E-057 Koma runner. Same phantom, sequences, Float64, RF step, spoiling, GPU and
# reconstruction as E056 run_acquisition.jl; only the free-precession step changes.
using KomaMRI, KomaMRICore, KomaMRIBase, CUDA, HDF5, TOML, SHA
using FFTW  # the frozen direct_sort helper expects FFTW exports in its calling module
include(joinpath(@__DIR__,"e057_common.jl"))
include(joinpath(HELPER,"direct_sort.jl"))

const MODELS=("static","uniform_osc","parabolic_osc")
const ARMS=("baseline","bw700_te274","bw700_te210")
step_label(Δt)=isfinite(Δt) ? string(round(Int,Δt*1e6))*"us" : "inf"

function plan(stage,preflight)
    ladder=Float64.(preflight["ladder_retained_s"])
    finest=minimum(ladder)
    if stage=="stage1"
        items=[("uniform_osc","bw700_te274",e,Inf) for e in ("REF","X")]
        for Δt in sort(ladder;rev=true),e in ("REF","X")
            push!(items,("uniform_osc","bw700_te274",e,Δt))
        end
        append!(items,[("static","bw700_te274",e,finest) for e in ("REF","X")])
        return items
    elseif stage=="stage2"
        gate=JSON_STATUS(joinpath(ROOT,"ladder_analysis_v1/report.json"))
        gate=="PASS_E057_STAGE1_STEP_CONVERGENCE"||error("Stage 2 requires a passing Stage 1 analysis")
        return [(m,a,e,finest) for m in MODELS for a in ARMS for e in ("REF","X","Y","Z")]
    end
    error("stage1|stage2 required")
end

# Minimal status read without a JSON dependency: the analysis writes one status line.
function JSON_STATUS(path)
    for line in eachline(path)
        m=match(r"^\s*\"status\":\s*\"([A-Z0-9_]+)\"",line)
        isnothing(m)||return m.captures[1]
    end
    error("No status in "*path)
end

function main(stage,output)
    ispath(output)&&error("Refuse existing output")
    get(ENV,"CUDA_VISIBLE_DEVICES","")==GPU1||error("Reviewed GPU1 required")
    ids=verify_e056_identities()
    preflight_path=joinpath(ROOT,"grid_preflight_v1/report.toml")
    preflight=TOML.parsefile(preflight_path)
    preflight["status"]=="PASS_E057_GRID_CERTIFICATION"||error("Stage 0 grid certification missing")
    preflight["rule_sha256"]==digest(joinpath(@__DIR__,"precession_step_rule.jl"))||error("Rule changed after certification")
    preflight["common_sha256"]==digest(joinpath(@__DIR__,"e057_common.jl"))||error("Common source changed after certification")
    preflight["protocol_sha256"]==digest(joinpath(@__DIR__,"PROTOCOL.md"))||error("Protocol changed after certification")
    items=plan(stage,preflight)
    registration=Dict{String,Any}("experiment"=>"E-057","status"=>"REGISTERED_BEFORE_KOMA_SIGNAL","stage"=>stage,
        "input_sha256"=>ids.input_sha256,"script_sha256"=>digest(@__FILE__),
        "rule_sha256"=>preflight["rule_sha256"],"common_sha256"=>preflight["common_sha256"],
        "protocol_sha256"=>preflight["protocol_sha256"],"preflight_sha256"=>digest(preflight_path),
        "manifest_sha256"=>ids.manifest_sha256,"precision"=>"f64","new_1m"=>false,
        "RF_step_s"=>.5e-6,"max_block_length"=>512,"physical_gpu_uuid"=>ENV["CUDA_VISIBLE_DEVICES"],"visible_gpu"=>0,
        "plan"=>[Dict("model"=>m,"arm"=>a,"encoding"=>e,"dt_prec"=>step_label(d)) for (m,a,e,d) in items])
    mkpath(output);atomic_toml(joinpath(output,"registration.toml"),registration)
    scanner=Scanner(limits=HardwareLimits(B0=3.,Gmax=.04,Smax=180.,B1=10e-6))
    CUDA.device!(0)
    identities=Dict{String,String}();runtimes=Dict{String,Float64}();nodes=Dict{String,Int}()
    sequence_cache=Dict{String,Any}();phantom_cache=Dict{String,Any}()
    for (model,arm,encoding,Δt) in items
        seq,seqsha=get!(sequence_cache,arm*encoding) do
            load_sequence(ids.sequences,arm,encoding)
        end
        p=get!(()->phantom(ids.input,model),phantom_cache,model)
        key=model*"_"*arm*"_"*lowercase(encoding)*"_"*step_label(Δt)
        params=e057_params(Δt)
        nodes[key]=length(simulated_grid(seq,p,params).t)
        println("START_E057 ",key," nodes=",nodes[key]);flush(stdout)
        MIGRATION_PREVIOUS_RF[]=false;start=time()
        signal=vec(simulate(p,seq,scanner;sim_params=copy(params),verbose=false))
        CUDA.synchronize();runtimes[key]=time()-start
        length(signal)==614400||error("Unexpected sample count")
        all(isfinite,signal)&&eltype(signal)==ComplexF64||error("Invalid signal")
        path=joinpath(output,key*".h5")
        h5open(path*".partial","w") do f
            write(f,"raw_signal_f64",signal)
            k,image=sorted_image(signal,seq);write(f,"rawsort",k);write(f,"image",image)
            for (name,value) in (("model",model),("arm",arm),("encoding",encoding),("dt_prec",step_label(Δt)),("sequence_sha256",seqsha))
                HDF5.attributes(f)[name]=value
            end
        end
        mv(path*".partial",path);identities[key]=digest(path)
        atomic_toml(joinpath(output,key*"_checkpoint.toml"),Dict("status"=>"COMPLETE_ENCODING",
            "seconds"=>runtimes[key],"nodes"=>nodes[key],"sha256"=>identities[key]))
        println("FINISHED_E057 ",key," seconds=",runtimes[key]);flush(stdout)
        GC.gc();CUDA.reclaim()
    end
    report=merge(registration,Dict("status"=>"COMPLETE_E057_"*uppercase(stage)*"_ACQUISITIONS",
        "result_sha256"=>identities,"runtime_s"=>runtimes,"nodes"=>nodes))
    atomic_toml(joinpath(output,"report.toml"),report)
    println(report["status"])
end

length(ARGS)==2||error("stage1|stage2 NEW_OUTPUT required")
main(ARGS...)
