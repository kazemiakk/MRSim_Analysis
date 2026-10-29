param(
    [int]$MinimumFreeMiB = 12000,
    [int]$MaximumUtilizationPercent = 10,
    [int]$StableProbeCount = 3,
    [int]$ProbeIntervalSeconds = 15,
    [int]$RetryIntervalSeconds = 60,
    [double]$MinimumFreeDiskGB = 30.0,
    [string]$RunName = 'motion_1m_scanner_matched_venc5',
    [switch]$PreflightOnly,
    [switch]$AllowLegacyCompletedRun
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'launcher_common.ps1')

Assert-Dp150LauncherParameters `
    -MinimumFreeMiB $MinimumFreeMiB `
    -MaximumUtilizationPercent $MaximumUtilizationPercent `
    -StableProbeCount $StableProbeCount `
    -ProbeIntervalSeconds $ProbeIntervalSeconds `
    -RetryIntervalSeconds $RetryIntervalSeconds `
    -MinimumFreeDiskGB $MinimumFreeDiskGB
Assert-Dp150SafeRunName -RunName $RunName -RequiredPrefix 'motion_1m_scanner_matched_venc5'
if ($AllowLegacyCompletedRun -and -not $PreflightOnly) {
    throw 'AllowLegacyCompletedRun is permitted only with PreflightOnly.'
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$projectDirectory = Join-Path $repoRoot 'koma_dp150'
$simulationScript = Join-Path $PSScriptRoot 'simulate.jl'
$runDir = Join-Path $repoRoot "outputs\dp150_mri_pipeline\koma\simulation_$RunName"
$statusPath = Join-Path $runDir 'queue_status.json'
$singleLockPath = Join-Path $runDir 'queue_launcher.lock'
$dualLockPath = Join-Path $runDir 'dual_queue_launcher.lock'
$reportPath = Join-Path $runDir 'simulation_report.toml'
$phantomPath = Join-Path $repoRoot 'outputs\dp150_mri_pipeline\koma\dp150_motion_1m.phantom'
$phantomReportPath = Join-Path $repoRoot 'outputs\dp150_mri_pipeline\koma\motion_phantom_report.toml'
$protocolAuditPath = Join-Path $PSScriptRoot 'protocol_audit.toml'
$auditScript = Join-Path $projectDirectory 'audit_completed_run.jl'
$globalLockDir = Join-Path $repoRoot 'outputs\dp150_mri_pipeline\koma\.launcher_locks'
$juliaResolution = Resolve-Dp150JuliaExecutable

$sequenceFiles = @{
    REF = (Join-Path $PSScriptRoot 'ref.seq')
    X = (Join-Path $PSScriptRoot 'x.seq')
    Y = (Join-Path $PSScriptRoot 'y.seq')
    Z = (Join-Path $PSScriptRoot 'z.seq')
}
$sourceFiles = @{
    phantom_sha256 = $phantomPath
    simulation_script_sha256 = (Join-Path $projectDirectory 'simulate_pc.jl')
    simulation_wrapper_sha256 = $simulationScript
    perfect_spoiling_sha256 = (Join-Path $projectDirectory 'perfect_spoiling.jl')
    motion_timing_script_sha256 = (Join-Path $projectDirectory 'motion_timing.jl')
    project_sha256 = (Join-Path $projectDirectory 'Project.toml')
    manifest_sha256 = (Join-Path $projectDirectory 'Manifest.toml')
    protocol_audit_sha256 = $protocolAuditPath
}
foreach ($path in @(
    $juliaResolution.Path,
    $simulationScript,
    $phantomPath,
    $phantomReportPath,
    $protocolAuditPath,
    $auditScript
) + @($sourceFiles.Values) + @($sequenceFiles.Values)) {
    Assert-Dp150FileNonEmpty -Path $path -Description 'Required single-GPU launcher input'
}
Assert-Dp150ProtocolInputs -AuditPath $protocolAuditPath -SequenceFiles $sequenceFiles
Assert-Dp150PhantomReport `
    -PhantomPath $phantomPath `
    -ReportPath $phantomReportPath `
    -ParticleCount 1000000

$freeDiskGB = Get-Dp150FreeDiskGB -Path $runDir
if ($freeDiskGB -lt $MinimumFreeDiskGB) {
    throw (
        "Only {0:N2} GB is free; at least {1:N2} GB is required." -f
        $freeDiskGB, $MinimumFreeDiskGB
    )
}
$completion = Get-Dp150CompletedRunState `
    -ReportPath $reportPath `
    -RunDirectory $runDir `
    -RunName $RunName `
    -ParticleCount 1000000 `
    -SourceFiles $sourceFiles `
    -SequenceFiles $sequenceFiles `
    -JuliaVersion $juliaResolution.Version

if ($PreflightOnly) {
    if ($completion.State -eq 'LegacyComplete') {
        if (-not $AllowLegacyCompletedRun) {
            throw (
                'The completed 1M run predates provenance schema 1. Refusing ' +
                'checkpoint reuse; all artifacts were left unchanged. Use ' +
                '-PreflightOnly -AllowLegacyCompletedRun for an explicit ' +
                'read-only audit, or select a fresh -RunName.'
            )
        }
        Invoke-Dp150LegacyCompletedRunAudit `
            -JuliaPath $juliaResolution.Path `
            -ProjectDirectory $projectDirectory `
            -AuditScript $auditScript `
            -RunDirectory $runDir `
            -ParticleCount 1000000
        Write-Output (
            "PASS_DP150_1M_LEGACY_PREFLIGHT_READ_ONLY run=$RunName " +
            "free_disk_gb=$('{0:N2}' -f $freeDiskGB) julia=$($juliaResolution.Version)"
        )
        exit 0
    }
    if ($completion.State -eq 'ProvenanceBoundComplete') {
        Write-Output (
            "PASS_DP150_1M_COMPLETE_PROVENANCE_BOUND_READ_ONLY run=$RunName " +
            "free_disk_gb=$('{0:N2}' -f $freeDiskGB) julia=$($juliaResolution.Version)"
        )
        exit 0
    }
    if ($AllowLegacyCompletedRun) {
        throw "No legacy completed run exists for explicit audit: $runDir"
    }

    $snapshot = @(Get-Dp150GpuSnapshot)
    $idleCandidates = @($snapshot | Where-Object {
        Test-Dp150GpuIdle `
            -Gpu $_ `
            -MinimumFreeMiB $MinimumFreeMiB `
            -MaximumUtilizationPercent $MaximumUtilizationPercent
    })
    Write-Output (
        "PASS_DP150_1M_SINGLE_FRESH_PREFLIGHT_READ_ONLY run=$RunName " +
        "free_disk_gb=$('{0:N2}' -f $freeDiskGB) idle_gpu_count=$($idleCandidates.Count) " +
        "julia=$($juliaResolution.Version)"
    )
    exit 0
}

if ($completion.State -eq 'LegacyComplete') {
    throw (
        'The completed 1M run predates provenance schema 1 and cannot be ' +
        'reused or overwritten. Use the explicit read-only legacy preflight, ' +
        'or select a fresh -RunName.'
    )
}
if ($completion.State -eq 'ProvenanceBoundComplete') {
    Write-Output "PASS_DP150_1M_ALREADY_COMPLETE_PROVENANCE_BOUND run=$RunName report=$reportPath"
    exit 0
}

New-Item -ItemType Directory -Path $runDir -Force | Out-Null
New-Item -ItemType Directory -Path $globalLockDir -Force | Out-Null

function Write-QueueStatus {
    param(
        [string]$State,
        [string]$Message,
        [int]$PhysicalGpu = -1,
        [object]$GpuSnapshot = $null
    )

    $status = [ordered]@{
        timestamp = (Get-Date).ToString('o')
        state = $State
        message = $Message
        launcher_pid = $PID
        run_name = $RunName
        physical_gpu_index = $PhysicalGpu
        minimum_free_mib = $MinimumFreeMiB
        maximum_utilization_percent = $MaximumUtilizationPercent
        stable_probe_count = $StableProbeCount
        run_directory = $runDir
        gpu_snapshot = $GpuSnapshot
    }
    $temporary = "$statusPath.partial"
    $status | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $temporary -Encoding ASCII
    Move-Item -LiteralPath $temporary -Destination $statusPath -Force
    Write-Output "[$($status.timestamp)] $State - $Message"
}

$lockStreams = @()
$launcherLocksHeld = $false
try {
    $lockStreams = @(Open-Dp150ExclusiveLocks `
        -Paths (@($singleLockPath, $dualLockPath) + @(
            Get-Dp150EncodingLockPaths `
                -RunDirectory $runDir `
                -Encodings @('REF', 'X', 'Y', 'Z')
        )) `
        -ConflictDescription 'Another dp150 launcher or worker')
    $launcherLocksHeld = $true

    $physicalGpu = -1
    while ($physicalGpu -lt 0) {
        $snapshot = @(Get-Dp150GpuSnapshot)
        $candidate = $snapshot |
            Where-Object {
                Test-Dp150GpuIdle `
                    -Gpu $_ `
                    -MinimumFreeMiB $MinimumFreeMiB `
                    -MaximumUtilizationPercent $MaximumUtilizationPercent
            } |
            Sort-Object -Property free_mib -Descending |
            Select-Object -First 1
        if ($null -eq $candidate) {
            Write-QueueStatus `
                -State 'waiting_for_gpu' `
                -Message 'No GPU currently satisfies the idle thresholds.' `
                -GpuSnapshot $snapshot
            Start-Sleep -Seconds $RetryIntervalSeconds
            continue
        }

        $candidateIndex = [int]$candidate.index
        $stable = $true
        for ($probe = 1; $probe -le $StableProbeCount; $probe++) {
            if ($probe -gt 1) {
                Start-Sleep -Seconds $ProbeIntervalSeconds
            }
            $probeSnapshot = @(Get-Dp150GpuSnapshot)
            $probeGpu = @($probeSnapshot | Where-Object { $_.index -eq $candidateIndex })
            if (
                $probeGpu.Count -ne 1 -or
                -not (Test-Dp150GpuIdle `
                    -Gpu $probeGpu[0] `
                    -MinimumFreeMiB $MinimumFreeMiB `
                    -MaximumUtilizationPercent $MaximumUtilizationPercent)
            ) {
                $stable = $false
                break
            }
            Write-QueueStatus `
                -State 'confirming_gpu_idle' `
                -Message "GPU $candidateIndex passed idle probe $probe of $StableProbeCount." `
                -PhysicalGpu $candidateIndex `
                -GpuSnapshot $probeSnapshot
        }
        if (-not $stable) {
            Start-Sleep -Seconds $RetryIntervalSeconds
            continue
        }

        $finalSnapshot = @(Get-Dp150GpuSnapshot)
        $finalGpu = @($finalSnapshot | Where-Object { $_.index -eq $candidateIndex })
        if (
            $finalGpu.Count -eq 1 -and
            (Test-Dp150GpuIdle `
                -Gpu $finalGpu[0] `
                -MinimumFreeMiB $MinimumFreeMiB `
                -MaximumUtilizationPercent $MaximumUtilizationPercent)
        ) {
        $physicalGpu = $candidateIndex
            break
        }
        Write-QueueStatus `
            -State 'waiting_for_gpu' `
            -Message "GPU $candidateIndex failed the immediate pre-launch recheck." `
            -GpuSnapshot $finalSnapshot
        Start-Sleep -Seconds $RetryIntervalSeconds
    }

    $lockStreams += @(Open-Dp150ExclusiveLocks `
        -Paths @((Join-Path $globalLockDir "physical_gpu_$physicalGpu.lock")) `
        -ConflictDescription 'Another dp150 worker')

    $freeDiskGB = Get-Dp150FreeDiskGB -Path $runDir
    if ($freeDiskGB -lt $MinimumFreeDiskGB) {
        throw (
            "Only {0:N2} GB is free; at least {1:N2} GB is required." -f
            $freeDiskGB, $MinimumFreeDiskGB
        )
    }

    $env:CUDA_VISIBLE_DEVICES = [string]$physicalGpu
    $env:DP150_GPU_DEVICE = '0'
    $env:DP150_PHYSICAL_GPU_INDEX = [string]$physicalGpu
    $env:DP150_RUN_NAME = $RunName
    $env:DP150_PHANTOM = $phantomPath
    $env:DP150_EXPECTED_PARTICLE_COUNT = '1000000'
    $env:DP150_BATCH_SIZE = '20000'
    $env:DP150_ENCODINGS = 'REF,X,Y,Z'
    $env:DP150_REPORT_NAME = 'simulation_report.toml'
    $env:DP150_REUSE_RESULTS = 'true'
    $env:DP150_B0_T = '3.0'
    $env:DP150_RESCALE_MOTION_TO_SEQUENCE_RR = 'true'
    $env:DP150_MAX_BLOCK_LENGTH = '512'
    $env:DP150_ALLOW_UNVALIDATED_BLOCK_LENGTH = 'false'
    $env:DP150_SIM_OUTPUT_DIR = Join-Path $repoRoot 'outputs\dp150_mri_pipeline\koma'

    Write-QueueStatus `
        -State 'launching' `
        -Message ("Starting the 1M run on GPU $physicalGpu with {0:N2} GB free disk." -f $freeDiskGB) `
        -PhysicalGpu $physicalGpu `
        -GpuSnapshot (Get-Dp150GpuSnapshot)
    Push-Location $repoRoot
    try {
        & $juliaResolution.Path `
            '--startup-file=no' `
            "--project=$projectDirectory" `
            $simulationScript
        $simulationExitCode = [int]$LASTEXITCODE
    }
    finally {
        Pop-Location
    }
    if ($simulationExitCode -ne 0) {
        throw "Julia exited with code $simulationExitCode; validated checkpoints remain resumable."
    }

    $verifiedCompletion = Get-Dp150CompletedRunState `
        -ReportPath $reportPath `
        -RunDirectory $runDir `
        -RunName $RunName `
        -ParticleCount 1000000 `
        -SourceFiles $sourceFiles `
        -SequenceFiles $sequenceFiles `
        -JuliaVersion $juliaResolution.Version
    if ($verifiedCompletion.State -ne 'ProvenanceBoundComplete') {
        throw 'Fresh simulation did not produce a provenance-bound completed run.'
    }
    Write-QueueStatus `
        -State 'complete' `
        -Message 'The provenance-bound 1M simulation completed successfully.' `
        -PhysicalGpu $physicalGpu
}
catch {
    if ($launcherLocksHeld) {
        Write-QueueStatus -State 'launcher_error' -Message $_.Exception.Message
    }
    throw
}
finally {
    Close-Dp150Locks -Streams $lockStreams
}

exit 0
