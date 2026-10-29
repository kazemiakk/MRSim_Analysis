param(
    [int]$PhysicalGpuIndex = 0,
    [int]$BatchSize = 0,
    [string]$RunName = 'motion_1m_8bin_4mm',
    [int]$MinimumFreeMiB = 12000,
    [int]$MaximumUtilizationPercent = 25,
    [switch]$PreflightOnly,
    [switch]$AllowLegacyCompletedRun
)

$ErrorActionPreference = 'Stop'
if ($PhysicalGpuIndex -lt 0) { throw 'PhysicalGpuIndex must be nonnegative.' }
if ($BatchSize -lt 0) { throw 'BatchSize must be nonnegative.' }
$requiredRunPrefix = 'motion_1m_8bin_4mm'
if (
    [System.IO.Path]::GetFileName($RunName) -cne $RunName -or
    $RunName -cnotmatch '^[A-Za-z0-9][A-Za-z0-9_-]*$' -or
    -not (
        $RunName -ceq $requiredRunPrefix -or
        $RunName.StartsWith(
            "${requiredRunPrefix}_",
            [System.StringComparison]::Ordinal
        )
    )
) {
    throw (
        "RunName must be a path-safe name beginning with " +
        "'$requiredRunPrefix': $RunName"
    )
}
if ($MinimumFreeMiB -lt 1) { throw 'MinimumFreeMiB must be positive.' }
if ($MaximumUtilizationPercent -lt 0 -or $MaximumUtilizationPercent -gt 100) {
    throw 'MaximumUtilizationPercent must be between 0 and 100.'
}
if ($AllowLegacyCompletedRun -and -not $PreflightOnly) {
    throw 'AllowLegacyCompletedRun is permitted only with PreflightOnly.'
}
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$pipelineRoot = Join-Path $repoRoot 'outputs\aorta_dp2_mri_pipeline\koma\multibin_8x4mm'
$phantom = Join-Path $pipelineRoot 'aorta_motion_1m_8bin.phantom'
$phantomReport = Join-Path $pipelineRoot 'motion_phantom_report.toml'
$sequenceDir = Join-Path $pipelineRoot 'sequences'
$sequenceReport = Join-Path $sequenceDir 'sequence_report.toml'
$simulationScript = Join-Path $PSScriptRoot 'simulate_pc.jl'
$perfectSpoilingScript = Join-Path $PSScriptRoot 'perfect_spoiling.jl'
$runDir = Join-Path $pipelineRoot "simulation_$RunName"
$finalReport = Join-Path $runDir 'simulation_report.toml'
$reconstructionFile = Join-Path $runDir 'pc_reconstruction.h5'
$completeArtifacts = @(
    $reconstructionFile,
    (Join-Path $runDir 'reference_reconstruction.h5'),
    (Join-Path $runDir 'ref_result.jld2'),
    (Join-Path $runDir 'x_result.jld2'),
    (Join-Path $runDir 'y_result.jld2'),
    (Join-Path $runDir 'z_result.jld2'),
    (Join-Path $runDir 'reference_reconstruction_qc.png'),
    (Join-Path $runDir 'pc_reconstruction_qc.png')
)

$minimumJuliaVersion = [version]'1.12.0'

function Get-JuliaVersion([string]$Path) {
    $text = (@(& $Path --startup-file=no --version 2>&1) -join "`n").Trim()
    if ($LASTEXITCODE -ne 0 -or $text -notmatch '^julia version (?<version>\d+\.\d+\.\d+)$') {
        throw "Cannot determine a stable Julia version from $Path`: $text"
    }
    return [version]$Matches['version']
}

function Test-CompatibleJuliaVersion([version]$Version) {
    return $Version.Major -eq 1 -and $Version -ge $minimumJuliaVersion
}

function Resolve-JuliaExecutable {
    if ($env:MRSIM_JULIA_EXE) {
        $path = (Resolve-Path -LiteralPath $env:MRSIM_JULIA_EXE -ErrorAction Stop).Path
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "MRSIM_JULIA_EXE is not a file: $path"
        }
        $version = Get-JuliaVersion $path
        if (-not (Test-CompatibleJuliaVersion $version)) {
            throw "Julia 1.12 or newer within major version 1 is required; $path is $version"
        }
        return [pscustomobject]@{ Path = $path; Version = $version }
    }

    $candidatePaths = @(
        Get-Command julia -All -ErrorAction SilentlyContinue |
            Where-Object CommandType -eq 'Application' |
            ForEach-Object Source
    )
    if ($env:LOCALAPPDATA) {
        $candidatePaths += @(
            Get-ChildItem -Path (Join-Path $env:LOCALAPPDATA 'Programs\Julia-*\bin\julia.exe') -File -ErrorAction SilentlyContinue |
                ForEach-Object FullName
        )
    }
    $validCandidates = foreach ($candidate in @(
        $candidatePaths | Where-Object { $_ } | Select-Object -Unique
    )) {
        try {
            $path = (Resolve-Path -LiteralPath $candidate -ErrorAction Stop).Path
            $version = Get-JuliaVersion $path
            if (Test-CompatibleJuliaVersion $version) {
                [pscustomobject]@{ Path = $path; Version = $version }
            }
            else {
                Write-Warning "Ignoring incompatible Julia $version at $path"
            }
        }
        catch {
            Write-Warning $_.Exception.Message
        }
    }
    if (-not $validCandidates) {
        throw 'Julia >=1.12.0 and <2.0.0 was not found. Set MRSIM_JULIA_EXE.'
    }
    return ($validCandidates | Sort-Object Version -Descending | Select-Object -First 1)
}

function Get-FileSha256Lower([string]$Path) {
    return (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
}

function Get-TomlScalar(
    [string]$Path,
    [string]$Key,
    [string]$Section = ''
) {
    $currentSection = ''
    $escapedKey = [regex]::Escape($Key)
    foreach ($line in Get-Content -LiteralPath $Path) {
        if ($line -match '^\s*\[(?<section>[^]]+)\]\s*$') {
            $currentSection = $Matches['section']
            continue
        }
        if ($currentSection -ne $Section) { continue }
        if ($line -match "^\s*$escapedKey\s*=\s*(?<value>.+?)\s*$") {
            $value = $Matches['value'].Trim()
            if ($value.Length -ge 2 -and $value[0] -eq '"' -and $value[$value.Length - 1] -eq '"') {
                return $value.Substring(1, $value.Length - 2)
            }
            return $value
        }
    }
    return $null
}

$juliaResolution = Resolve-JuliaExecutable
$julia = $juliaResolution.Path
$juliaVersion = $juliaResolution.Version
$requiredInputs = @(
    $julia,
    $phantom,
    $phantomReport,
    $sequenceReport,
    $simulationScript,
    $perfectSpoilingScript,
    (Join-Path $sequenceDir 'ref.seq'),
    (Join-Path $sequenceDir 'x.seq'),
    (Join-Path $sequenceDir 'y.seq'),
    (Join-Path $sequenceDir 'z.seq')
)
foreach ($path in $requiredInputs) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required aorta production input is missing: $path"
    }
}

$phantomSha256 = Get-FileSha256Lower $phantom
$simulationScriptSha256 = Get-FileSha256Lower $simulationScript
$perfectSpoilingSha256 = Get-FileSha256Lower $perfectSpoilingScript
$sequencePaths = [ordered]@{
    REF = (Join-Path $sequenceDir 'ref.seq')
    X = (Join-Path $sequenceDir 'x.seq')
    Y = (Join-Path $sequenceDir 'y.seq')
    Z = (Join-Path $sequenceDir 'z.seq')
}
$sequenceSha256 = @{}
foreach ($encoding in $sequencePaths.Keys) {
    $actualSha256 = Get-FileSha256Lower $sequencePaths[$encoding]
    $reportedSha256 = Get-TomlScalar `
        -Path $sequenceReport `
        -Section "sequences.$encoding" `
        -Key 'sha256'
    if (-not $reportedSha256) {
        throw "Sequence report lacks SHA-256 provenance for $encoding`: $sequenceReport"
    }
    if ($actualSha256 -ne $reportedSha256.ToLowerInvariant()) {
        throw "Sequence SHA-256 mismatch for $encoding`: expected $reportedSha256, found $actualSha256"
    }
    $sequenceSha256[$encoding] = $actualSha256
}
if (-not (Select-String -LiteralPath $phantomReport -Pattern '^status = "PASS_APPROXIMATE_POINT_CLOUD"$' -Quiet)) {
    throw "The one-million-spin phantom report is not PASS: $phantomReport"
}
if (-not (Select-String -LiteralPath $phantomReport -Pattern '^particle_count = 1000000$' -Quiet)) {
    throw "The production phantom does not contain exactly 1,000,000 spins: $phantomReport"
}
$requiredPhantomReportPatterns = @(
    '^particle_limit_requested = 0$',
    '^flowpath_sample_count = 361$',
    '^realized_sequence_rr_s = 0\.7261599898338318$',
    '^sequence_matrix = \[72, 32, 32\]$',
    '^source_cycle_s = 0\.72$',
    '^venc_m_per_s = 2\.5$'
)
foreach ($pattern in $requiredPhantomReportPatterns) {
    if (-not (Select-String -LiteralPath $phantomReport -Pattern $pattern -Quiet)) {
        throw "The phantom report does not match the RR-mapped eight-bin production phantom: $phantomReport"
    }
}
$phantomBytesLine = @(Select-String -LiteralPath $phantomReport -Pattern '^phantom_bytes = (?<bytes>[0-9]+)$')
if ($phantomBytesLine.Count -ne 1) {
    throw "The production phantom report has no unique byte count: $phantomReport"
}
$expectedPhantomBytes = [long]$phantomBytesLine[0].Matches[0].Groups['bytes'].Value
$actualPhantomBytes = (Get-Item -LiteralPath $phantom).Length
if ($actualPhantomBytes -ne $expectedPhantomBytes) {
    throw "The production phantom byte count is wrong (expected $expectedPhantomBytes, found $actualPhantomBytes): $phantom"
}
$requiredSequenceReportPatterns = @(
    '^status = "PASS_SEQUENCE_LABEL_VALIDATION"$',
    '^cardiac_bins = 8$',
    '^matrix = \[72, 32, 32\]$',
    '^profile_count_per_encoding = 8192$'
)
foreach ($pattern in $requiredSequenceReportPatterns) {
    if (-not (Select-String -LiteralPath $sequenceReport -Pattern $pattern -Quiet)) {
        throw "The sequence report does not match the production eight-bin protocol: $sequenceReport"
    }
}
if (Test-Path -LiteralPath $finalReport -PathType Leaf) {
    if (Select-String -LiteralPath $finalReport -Pattern '^status = "PASS"$' -Quiet) {
        $missingArtifacts = @($completeArtifacts | Where-Object {
            -not (Test-Path -LiteralPath $_ -PathType Leaf) -or
            (Get-Item -LiteralPath $_).Length -eq 0
        })
        if ($missingArtifacts.Count -eq 0) {
            $storedSchema = Get-TomlScalar `
                -Path $finalReport `
                -Key 'provenance_schema_version'
            if (-not $storedSchema) {
                if ($AllowLegacyCompletedRun) {
                    Write-Warning (
                        'The completed production run predates embedded SHA ' +
                        'provenance. Artifact presence and current input hashes ' +
                        'were checked, but checkpoint reuse remains disabled.'
                    )
                    Write-Output (
                        "PASS_AORTA_1M_8BIN_LEGACY_COMPLETE " +
                        "run=$RunName julia=$juliaVersion phantom_sha256=$phantomSha256 " +
                        "report=$finalReport"
                    )
                    exit 0
                }
                throw (
                    'The completed production report predates SHA provenance. ' +
                    'Refusing to treat its checkpoints as reusable; all existing ' +
                    'artifacts were left unchanged. For a read-only audit of the ' +
                    'preserved run, pass -PreflightOnly -AllowLegacyCompletedRun.'
                )
            }
            if ($storedSchema -ne '1') {
                throw "Unsupported simulation provenance schema $storedSchema in $finalReport"
            }
            $expectedReportValues = [ordered]@{
                run_name = $RunName
                physical_gpu_index = [string]$PhysicalGpuIndex
                particle_count = '1000000'
                spin_batch_size = [string]$BatchSize
                maximum_simulation_block_length = '512'
                precision = 'f32'
                simulation_method = 'Bloch with perfect transverse spoiling before RF'
                julia_version = [string]$juliaVersion
                gradient_max_step_s = 'Inf (event boundaries only, matching supplied script)'
                unvalidated_block_length_override = 'false'
                simulation_script_sha256 = $simulationScriptSha256
                perfect_spoiling_sha256 = $perfectSpoilingSha256
            }
            foreach ($key in $expectedReportValues.Keys) {
                $storedValue = Get-TomlScalar -Path $finalReport -Key $key
                if ($storedValue -ne $expectedReportValues[$key]) {
                    throw "Completed-run provenance mismatch for $key`: $finalReport"
                }
            }
            $storedPhantomSha256 = Get-TomlScalar `
                -Path $finalReport `
                -Key 'phantom_sha256'
            if ($storedPhantomSha256 -ne $phantomSha256) {
                throw "Completed-run phantom SHA-256 does not match current input: $finalReport"
            }
            foreach ($encoding in $sequencePaths.Keys) {
                $storedSequenceSha256 = Get-TomlScalar `
                    -Path $finalReport `
                    -Section "sequence.$encoding" `
                    -Key 'sha256'
                if ($storedSequenceSha256 -ne $sequenceSha256[$encoding]) {
                    throw "Completed-run sequence SHA-256 does not match $encoding input: $finalReport"
                }
            }
            Write-Output (
                "PASS_AORTA_1M_8BIN_COMPLETE_PROVENANCE_BOUND " +
                "run=$RunName julia=$juliaVersion report=$finalReport"
            )
            exit 0
        }
        throw (
            "The PASS report is incomplete. Refusing to overwrite or reuse its " +
            "artifacts; missing or empty: " + ($missingArtifacts -join ', ')
        )
    }
}

$gpuRow = & nvidia-smi `
    "--id=$PhysicalGpuIndex" `
    '--query-gpu=index,uuid,memory.free,utilization.gpu' `
    '--format=csv,noheader,nounits'
if ($LASTEXITCODE -ne 0) { throw 'nvidia-smi failed during aorta preflight.' }
$fields = $gpuRow -split ',' | ForEach-Object { $_.Trim() }
if ($fields.Count -ne 4) { throw "Unexpected nvidia-smi GPU row: $gpuRow" }
$gpuUuid = $fields[1]
$freeMiB = [int]$fields[2]
$utilization = [int]$fields[3]
if ($freeMiB -lt $MinimumFreeMiB -or $utilization -gt $MaximumUtilizationPercent) {
    throw "GPU $PhysicalGpuIndex is not idle enough (free ${freeMiB} MiB, utilization ${utilization}%)."
}

# A batch worker can briefly report low utilization and nearly all memory free
# while it writes a checkpoint. Reject an existing Julia/Python CUDA context so
# this point-in-time check cannot launch a second simulation into that gap.
$computeRows = @(& nvidia-smi `
    '--query-compute-apps=gpu_uuid,pid,process_name' `
    '--format=csv,noheader,nounits')
if ($LASTEXITCODE -ne 0) { throw 'nvidia-smi process query failed during aorta preflight.' }
$conflictingProcesses = @()
foreach ($row in $computeRows) {
    $columns = @($row -split ',', 3 | ForEach-Object { $_.Trim() })
    if ($columns.Count -ne 3 -or $columns[0] -ne $gpuUuid) { continue }
    $processId = 0
    if (-not [int]::TryParse($columns[1], [ref]$processId)) { continue }
    $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
    $processName = if ($process) {
        $process.ProcessName
    }
    else {
        [System.IO.Path]::GetFileNameWithoutExtension($columns[2])
    }
    if ($processName -match '^(julia|pythonw?|python\d+(?:\.\d+)?)$') {
        $conflictingProcesses += "${processName} (PID $processId)"
    }
}
if ($conflictingProcesses.Count -gt 0) {
    throw (
        "GPU $PhysicalGpuIndex already has a simulation-capable process: " +
        ($conflictingProcesses -join ', ')
    )
}

if ($PreflightOnly) {
    Write-Output (
        "PASS_AORTA_1M_8BIN_PREFLIGHT GPU=$PhysicalGpuIndex " +
        "run=$RunName " +
        "free_mib=$freeMiB utilization_percent=$utilization " +
        "batch_size=$BatchSize julia=$juliaVersion"
    )
    exit 0
}

$env:CUDA_VISIBLE_DEVICES = [string]$PhysicalGpuIndex
$env:AORTA_PHYSICAL_GPU_INDEX = [string]$PhysicalGpuIndex
$env:AORTA_PHANTOM = $phantom
$env:AORTA_SEQUENCE_DIR = $sequenceDir
$env:AORTA_SIM_OUTPUT_DIR = $pipelineRoot
$env:AORTA_RUN_NAME = $RunName
$env:AORTA_ENCODINGS = 'REF,X,Y,Z'
$env:AORTA_GPU_DEVICE = '0'
$env:AORTA_REUSE_RESULTS = 'true'
$env:AORTA_BATCH_SIZE = [string]$BatchSize
$env:AORTA_MAX_BLOCK_LENGTH = '512'
$env:AORTA_ALLOW_UNVALIDATED_BLOCK_LENGTH = 'false'

Push-Location $repoRoot
$exitCode = 1
try {
    & $julia --startup-file=no "--project=$repoRoot\koma_aorta" $simulationScript
    $exitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}
exit ([int]$exitCode)
