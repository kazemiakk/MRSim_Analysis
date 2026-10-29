# Reuse-only E-034 recovery. No simulation, GPU allocation or acquisition edits.
using KomaMRI, JLD2, HDF5, FFTW, TOML, SHA, Test

filehash(path) = open(io -> bytes2hex(sha256(io)), path)

function reconstruct_inlet_pilot(acquisition, output)
    ispath(output) && error("Refusing output reuse: $output")
    root = normpath(joinpath(@__DIR__, ".."))
    seqdir = joinpath(root, "outputs", "aorta_dp2_mri_pipeline", "koma", "multibin_8x4mm", "sequences")
    phantom_report = TOML.parsefile(joinpath(acquisition, "phantom_report.toml"))
    reset = TOML.parsefile(joinpath(acquisition, "reset_preflight.toml"))
    viability = TOML.parsefile(joinpath(acquisition, "reference_viability.toml"))
    phantom_report["particle_count"] in (20000, 100000) || error("Pilot only")
    reset["status"] == "PASS_RESET_PREFLIGHT" || error("Reset preflight failed")
    viability["status"] == "PASS_REFERENCE_VIABILITY" || error("REF gate failed")
    phantom_hash = filehash(joinpath(acquisition, "inlet_pilot.phantom"))
    phantom_hash == phantom_report["phantom_sha256"] || error("Phantom hash mismatch")
    images = Dict{String,Array{ComplexF32,4}}()
    runtimes = Dict{String,Float64}()
    raw_hashes = Dict{String,String}()
    seq_hashes = Dict{String,String}()
    checks = 0
    for encoding in ("REF", "X", "Y", "Z")
        path = joinpath(acquisition, lowercase(encoding)*"_result.jld2")
        seqpath = joinpath(seqdir, lowercase(encoding)*".seq")
        data = JLD2.load(path)
        @test data["encoding"] == encoding
        @test data["phantom_sha256"] == phantom_hash
        @test data["sequence_sha256"] == filehash(seqpath)
        @test data["sequence_sha256"] == reset["encodings"][encoding]["sequence_sha256"]
        raw = data["rawsort"]; image = data["image"]
        @test size(raw) == size(image) == (72, 32, 32, 8)
        @test all(isfinite, raw) && all(isfinite, image)
        rebuilt = ComplexF32.(fftshift(ifft(ifftshift(raw,(1,2,3)),(1,2,3)),(1,2,3)))
        @test maximum(abs, rebuilt .- image) <= 1e-6 * maximum(abs, image)
        @test isfinite(data["elapsed_seconds"]) && data["elapsed_seconds"] > 0
        images[encoding] = image
        runtimes[encoding] = data["elapsed_seconds"]
        raw_hashes[encoding] = filehash(path); seq_hashes[encoding] = filehash(seqpath)
        checks += 8
    end
    seq = read_seq(joinpath(seqdir,"ref.seq"); verbose=false, verify_signature=true)
    println("FOV_DEFINITION_TYPE=", typeof(seq.DEF["FOV"]))
    reference=images["REF"]; magnitude=abs.(reference); mask=falses(size(magnitude))
    for b in axes(magnitude,4)
        mask[:,:,:,b] .= magnitude[:,:,:,b] .>= .05f0*maximum(magnitude[:,:,:,b])
    end
    venc=Float32(seq.DEF["Venc"]); velocity=zeros(Float32,size(reference)...,3)
    for (k,encoding) in enumerate(("X","Y","Z"))
        velocity[:,:,:,:,k] .= angle.(images[encoding].*conj.(reference)) .* (venc/Float32(pi))
        component=@view velocity[:,:,:,:,k]; component[.!mask] .= 0
    end
    # Retain the failed unpublished file. Compare its numerical payload before
    # publishing a new complete reconstruction with explicit vector attributes.
    failed = joinpath(acquisition,"pc_reconstruction.h5.partial")
    if isfile(failed)
        h5open(failed,"r") do f
            @test read(f,"reference_magnitude") == magnitude
            @test read(f,"velocity") == velocity
            @test read(f,"mask") == UInt8.(mask)
            @test read(f,"speed") == dropdims(sqrt.(sum(abs2,velocity;dims=5));dims=5)
        end
        checks += 4
    end
    staging = output*".building"
    ispath(staging) && error("Staging exists: $staging")
    mkpath(staging)
    target=joinpath(staging,"pc_reconstruction.h5")
    h5open(target,"w") do f
        write(f,"reference_magnitude",magnitude); write(f,"velocity",velocity); write(f,"mask",UInt8.(mask))
        write(f,"speed",dropdims(sqrt.(sum(abs2,velocity;dims=5));dims=5))
        HDF5.attributes(f)["FOV_m"] = Float32.(collect(seq.DEF["FOV"]))
        HDF5.attributes(f)["RR_s"] = Float32(seq.DEF["RR"])
        HDF5.attributes(f)["Venc_m_per_s"] = venc
    end
    report=Dict("status"=>"COMPLETE_EXPERIMENTAL_INLET_PILOT_REUSE_ONLY_RECONSTRUCTION",
        "acquisition_directory"=>abspath(acquisition), "particle_count"=>phantom_report["particle_count"],
        "acquisition_script_sha256"=>phantom_report["builder_sha256"],
        "reconstruction_script_sha256"=>filehash(@__FILE__), "phantom_sha256"=>phantom_hash,
        "raw_result_sha256"=>raw_hashes, "sequence_sha256"=>seq_hashes,
        "encoding_runtime_s"=>runtimes, "total_encoding_runtime_s"=>sum(values(runtimes)),
        "raw_integrity_fft_and_payload_checks_passed"=>checks,
        "reset_preflight_sha256"=>filehash(joinpath(acquisition,"reset_preflight.toml")),
        "reference_viability_sha256"=>filehash(joinpath(acquisition,"reference_viability.toml")),
        "phantom_report_sha256"=>filehash(joinpath(acquisition,"phantom_report.toml")),
        "reconstruction_sha256"=>filehash(target), "pilot_only"=>true,
        "physical_gpu_index_for_acquisition"=>1, "production_rerun_authorized"=>false,
        "recovery_only_no_simulation"=>true,
        "FOV_definition_type"=>string(typeof(seq.DEF["FOV"])),
        "failed_unpublished_payload_retained"=>isfile(failed))
    if isfile(failed)
        report["failed_unpublished_payload_sha256"]=filehash(failed)
    end
    open(joinpath(staging,"simulation_report.toml"),"w") do io
        TOML.print(io,report)
    end
    mv(staging,output)
    println("PASS_REUSE_ONLY_RECONSTRUCTION checks=",checks," output=",output)
end

if abspath(PROGRAM_FILE)==@__FILE__
    length(ARGS)==2 || error("Usage: reconstruct_inlet_pilot.jl ACQUISITION NEW_OUTPUT")
    reconstruct_inlet_pilot(ARGS...)
end
