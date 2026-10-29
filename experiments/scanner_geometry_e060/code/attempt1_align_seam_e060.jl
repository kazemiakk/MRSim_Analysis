# E-060 S1b (amendment A3): periodic motion seam at the half-bin gap of the E-060
# sequences, with the frozen ADC seam audit and verified seam markers. CPU only.
const ORIGINAL_HELPERS=raw"D:\MRSim\MRSIM\MRSimTracks\koma_migration_e046"
const E055=raw"D:\MRSim\outputs\phantom_density_e055"
const ROOT=raw"D:\MRSim\outputs\scanner_geometry_e060"
using KomaMRI, KomaMRIBase, TOML, SHA
include(joinpath(ORIGINAL_HELPERS,"export_controls.jl"))  # frozen: koma_dp150/motion_timing.jl and verified markers
const ENCODINGS=("REF","X","Y","Z")
const BINS=30
const RR=1.012
const MARGIN_FRACTION=0.40
const EXPECTED_SEAMS=64*30
event_vector(value)=value isa Number ? [value] : collect(value)

function main(output)
    ispath(output)&&error("Refuse existing output")
    info=TOML.parsefile(joinpath(E055,"inputs_v1","case_info.toml"))
    period=info["period_s"]
    s0=TOML.parsefile(joinpath(ROOT,"sequences_v2","report.toml"))
    s0["status"]=="PASS_E060_SEQUENCES_AND_AUDITS"&&s0["hardware"]["TBP"]==12||error("S0 v2 missing")
    seqs=Dict{String,Any}()
    for e in ENCODINGS
        path=joinpath(ROOT,"sequences_v2","scanner_geometry",lowercase(e)*".seq")
        hashfile(path)==s0["scanner_geometry"][e]["sequence_sha256"]||error("Sequence changed: "*e)
        seqs[e]=read_seq(path;verbose=false,verify_signature=true)
    end
    centers=Dict(e=>first_adc_kspace_center(seqs[e]) for e in ENCODINGS)
    all(c==centers["REF"] for c in values(centers))||error("First ADC k-space centres differ between encodings")
    requested=centers["REF"]-RR/(2*BINS)
    grid=2.0^-23
    seam=round(requested/grid)*grid
    Float64(Float32(seam))==seam&&Float64(Float32(seam+period))==seam+period||error("Seam or seam+period not exact in Float32")
    count=floor(Int,(dur(seqs["REF"])-(seam-350e-6)-2e-6)/period)+1
    count==EXPECTED_SEAMS||error("Seam count differs from acquired heartbeats: "*string(count))
    margin=MARGIN_FRACTION*RR/BINS
    audits=Dict{String,Any}()
    for e in ENCODINGS
        a=audit_adc_sample_windows_against_periodic_seam(seqs[e],seam,period;minimum_margin_s=margin)
        audits[e]=Dict("seam_crossing_window_count"=>a.seam_crossing_window_count,"minimum_seam_margin_s"=>a.minimum_seam_margin_s,
            "minimum_seam_margin_fraction_of_bin"=>a.minimum_seam_margin_fraction_of_bin,"adc_window_count"=>a.adc_window_count,
            "required_minimum_seam_margin_s"=>a.required_minimum_seam_margin_s)
        # Frozen marker insertion must accept every marker (on a copy; the sequence files are untouched).
        insert_verified_zero_markers!(copy(seqs[e]),seam,period,count)
        # A3 check: no ADC sample between any marker and the next RF.
        starts=get_block_start_times(seqs[e])
        rf_starts=sort([starts[i]+seqs[e].RF[1,i].delay for i in eachindex(seqs[e].DUR) if any(!iszero,event_vector(seqs[e].RF[1,i].A))])
        adc_starts=sort([starts[i]+seqs[e].ADC[i].delay for i in eachindex(seqs[e].DUR) if seqs[e].ADC[i].N>0])
        worst=Inf
        for k in 0:count-1
            marker=seam+k*period-350e-6
            next_rf=searchsortedfirst(rf_starts,marker)
            next_adc=searchsortedfirst(adc_starts,marker)
            next_adc>length(adc_starts)&&continue
            next_rf<=length(rf_starts)&&rf_starts[next_rf]<adc_starts[next_adc]||error("ADC between marker $k and the next RF: "*e)
            worst=min(worst,adc_starts[next_adc]-rf_starts[next_rf])
        end
        audits[e]["markers_verified"]=count
        audits[e]["minimum_next_rf_lead_over_next_adc_s"]=worst
        println("E060_SEAM ",e," crossings=",audits[e]["seam_crossing_window_count"]," margin_ms=",round(1e3*audits[e]["minimum_seam_margin_s"],digits=3),
                " fraction=",round(audits[e]["minimum_seam_margin_fraction_of_bin"],digits=4));flush(stdout)
    end
    report=Dict{String,Any}("experiment"=>"E-060","stage"=>"S1b_seam_alignment","status"=>"PASS_E060_SEAM_ALIGNMENT",
        "rule"=>"frozen align_periodic_motion_seam_to_first_adc_midbin: seam = first ADC k-space centre - RR/(2*30), on a 2^-23 s grid",
        "first_adc_kspace_center_s"=>centers["REF"],"requested_seam_s"=>requested,"seam_time_s"=>seam,"rounding_s"=>seam-requested,
        "period_s"=>period,"t_end_s"=>seam+period,"e055_seam_time_s"=>info["seam_time_s"],"seam_markers"=>count,
        "required_minimum_seam_margin_s"=>margin,"margin_fraction_of_bin"=>MARGIN_FRACTION,"audits"=>audits,
        "script_sha256"=>hashfile(@__FILE__),"amendment_a3_sha256"=>hashfile(joinpath(ROOT,"code","AMENDMENT_A3_SEAM.md")),
        "sequence_report_sha256"=>hashfile(joinpath(ROOT,"sequences_v2","report.toml")))
    stage=output*".building";ispath(stage)&&error("Stage exists");mkpath(stage)
    open(joinpath(stage,"report.toml"),"w") do io;TOML.print(io,report;sorted=true);end
    mv(stage,output)
    println(report["status"]," seam=",seam)
end

length(ARGS)==1||error("NEW_OUTPUT required")
main(ARGS[1])
