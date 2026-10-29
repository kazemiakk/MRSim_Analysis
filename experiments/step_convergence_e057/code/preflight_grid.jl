# E-057 Stage 0: certify the isolated rule against the E056 grid and size the ladder.
# CPU discretization only; no Koma signal is simulated.
using KomaMRI, KomaMRICore, KomaMRIBase, HDF5, TOML, SHA
include(joinpath(@__DIR__,"e057_common.jl"))

const LADDER=(16e-6,8e-6,4e-6,2e-6,1e-6)
const NODE_LIMIT_FACTOR=20
const PART_MEMORY_LIMIT_BYTES=2e9

fields_equal(a,b)=all(isequal(getfield(a,f),getfield(b,f)) for f in fieldnames(typeof(a)))

function excitation_nodes(s)
    idx=findall(s.excitation_bool)
    return sort!(unique!(vcat(s.t[idx],s.t[idx.+1])))
end

function longest_gradient_interval(s)
    active=(abs.(s.Gx).+abs.(s.Gy).+abs.(s.Gz)).>0
    longest=0.
    for i in eachindex(s.Δt)
        s.excitation_bool[i]&&continue
        (active[i]||active[i+1])||continue
        longest=max(longest,s.t[i+1]-s.t[i])
    end
    return longest
end

function rf_span_node_count(s)
    count(s.excitation_bool)
end

function main(output)
    ispath(output)&&error("Refuse existing output")
    ids=verify_e056_identities()
    obj=phantom(ids.input,"uniform_osc")
    stage=output*".building";ispath(stage)&&error("Refuse existing stage")
    mkpath(stage)
    report=Dict{String,Any}("experiment"=>"E-057","stage"=>"0_grid_certification_and_resource_preflight",
        "script_sha256"=>digest(@__FILE__),"rule_sha256"=>digest(joinpath(@__DIR__,"precession_step_rule.jl")),
        "common_sha256"=>digest(joinpath(@__DIR__,"e057_common.jl")),
        "protocol_sha256"=>digest(joinpath(@__DIR__,"PROTOCOL.md")),
        "manifest_sha256"=>ids.manifest_sha256,"input_sha256"=>ids.input_sha256,
        "new_signal_simulated"=>false,"new_1m"=>false,"spins"=>length(obj.x))
    checks=0;encodings=Dict{String,Any}()
    h5open(joinpath(stage,"primary_grid.h5"),"w") do saved
        for encoding in ("REF","X")
            seq,seqsha=load_sequence(ids.sequences,"bw700_te274",encoding)
            primary=simulated_grid(seq,obj,e057_params(nothing))
            custom=simulated_grid(seq,obj,e057_params(Inf))
            fields_equal(primary,custom)||error("C0a failed: custom Inf grid differs from E056 primary for "*encoding)
            checks+=1
            primary_adc=primary.t[primary.ADC]
            length(primary_adc)==614400||error("Unexpected primary ADC count")
            primary_exc=excitation_nodes(primary)
            n0=length(primary.t)
            ranges,_=KomaMRICore.get_sim_ranges(primary;max_block_length=512,max_rf_block_length=Inf,eval_intervals_per_step=1)
            # Export the free-precession grid: every endpoint of a non-excitation interval.
            keep=falses(length(primary.t))
            for i in eachindex(primary.Δt)
                primary.excitation_bool[i]&&continue
                keep[i]=true;keep[i+1]=true
            end
            g=create_group(saved,encoding)
            g["t_s"]=primary.t[keep];g["Gx_T_m"]=primary.Gx[keep];g["Gy_T_m"]=primary.Gy[keep];g["Gz_T_m"]=primary.Gz[keep]
            g["ADC"]=UInt8.(primary.ADC[keep])
            entry=Dict{String,Any}("sequence_sha256"=>seqsha,"primary_nodes"=>n0,"primary_parts"=>length(ranges),
                "primary_excitation_intervals"=>rf_span_node_count(primary),
                "primary_longest_gradient_interval_s"=>longest_gradient_interval(primary),
                "C0a_custom_inf_identical"=>true,"steps"=>Dict{String,Any}())
            for Δt in LADDER
                label=string(round(Int,Δt*1e6))*"us"
                refined=simulated_grid(seq,obj,e057_params(Δt))
                n=length(refined.t)
                retained=all(t->insorted(t,refined.t),primary.t)
                same_exc=excitation_nodes(refined)==primary_exc&&count(refined.excitation_bool)==count(primary.excitation_bool)
                same_adc=refined.t[refined.ADC]==primary_adc
                longest=longest_gradient_interval(refined)
                bounded=longest<=Δt*(1+1e-9)+1e-12
                ranges,_=KomaMRICore.get_sim_ranges(refined;max_block_length=512,max_rf_block_length=Inf,eval_intervals_per_step=1)
                part=maximum(length,ranges)
                memory=part*length(obj.x)*3*8.
                retain_step=n<=NODE_LIMIT_FACTOR*n0&&memory<=PART_MEMORY_LIMIT_BYTES
                (retained&&same_exc&&same_adc&&bounded)||error("C0b failed at "*label*" for "*encoding)
                checks+=4
                entry["steps"][label]=Dict("dt_prec_s"=>Δt,"nodes"=>n,"node_ratio"=>n/n0,"parts"=>length(ranges),
                    "longest_part_nodes"=>part,"part_position_bytes"=>memory,
                    "longest_gradient_interval_s"=>longest,"primary_nodes_retained"=>retained,
                    "excitation_nodes_identical"=>same_exc,"adc_times_identical"=>same_adc,
                    "gradient_interval_bounded"=>bounded,"retained_for_ladder"=>retain_step)
                println("E057_GRID ",encoding," ",label," nodes=",n," ratio=",n/n0," longest=",longest);flush(stdout)
            end
            encodings[encoding]=entry
        end
    end
    report["encodings"]=encodings;report["checks"]=checks
    report["status"]="PASS_E057_GRID_CERTIFICATION"
    report["ladder_retained_s"]=[Δt for Δt in LADDER if all(encodings[e]["steps"][string(round(Int,Δt*1e6))*"us"]["retained_for_ladder"] for e in ("REF","X"))]
    report["primary_grid_sha256"]=digest(joinpath(stage,"primary_grid.h5"))
    atomic_toml(joinpath(stage,"report.toml"),report)
    mv(stage,output)
    println(report["status"]," checks=",checks)
end

length(ARGS)==1||error("OUTPUT required")
main(ARGS[1])
