param(
    [int]$MinimumFreeMiB = 12000,
    [int]$MaximumUtilizationPercent = 10,
    [int]$StableProbeCount = 3,
    [int]$ProbeIntervalSeconds = 15,
    [int]$RetryIntervalSeconds = 60,
    [double]$MinimumFreeDiskGB = 12.0,
    [string]$RunName = 'motion_100k_scanner_matched_venc5',
    [switch]$PreflightOnly,
    [switch]$AllowLegacyCompletedRun,
    [switch]$SeamSafeMotion
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
Assert-Dp150SafeRunName -RunName $RunName -RequiredPrefix 'motion_100k_scanner_matched_venc5'
if ($AllowLegacyCompletedRun -and -not $PreflightOnly) {
    throw 'AllowLegacyCompletedRun is permitted only with PreflightOnly.'
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$projectDirectory = Join-Path $repoRoot 'koma_dp150'
$simulationScriptName = if ($SeamSafeMotion) {
    'simulate_100k_seam_safe.jl'
}
else {
    'simulate_100k.jl'
}
$simulationScript = Join-Path $PSScriptRoot $simulationScriptName
$workerScript = Join-Path $PSScriptRoot 'run_100k_worker.ps1'
$runDir = Join-Path $repoRoot "outputs\dp150_mri_pipeline\koma\simulation_$RunName"
$statusPath = Join-Path $runDir 'dual_queue_status.json'
$lockPath = Join-Path $runDir 'dual_queue_launcher.lock'
$finalReportPath = Join-Path $runDir 'simulation_report.toml'
$executionReportPath = Join-Path $runDir 'dual_gpu_execution_report.json'
$phantomPath = Join-Path $repoRoot 'outputs\dp150_mri_pipeline\koma\pilot_100k\dp150_motion_100k.phantom'
$phantomReportPath = Join-Path (Split-Path -Parent $phantomPath) 'motion_100k_phantom_report.toml'
$protocolAuditPath = Join-Path $PSScriptRoot 'protocol_audit.toml'
$auditScript = Join-Path $projectDirectory 'audit_completed_run.jl'
$globalLockDir = Join-Path $repoRoot 'outputs\dp150_mri_pipeline\koma\.launcher_locks'
$requiredGpuIndices = @(0, 1)
$juliaResolution = Resolve-Dp150JuliaExecutable
$powershell = Get-Dp150PowerShellExecutable

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
    $workerScript,
    $phantomPath,
    $phantomReportPath,
    $protocolAuditPath,
    $auditScript
) + @($sourceFiles.Values) + @($sequenceFiles.Values)) {
    Assert-Dp150FileNonEmpty -Path $path -Description 'Required 100k launcher input'
}
Assert-Dp150ProtocolInputs -AuditPath $protocolAuditPath -SequenceFiles $sequenceFiles
Assert-Dp150PhantomReport `
    -PhantomPath $phantomPath `
    -ReportPath $phantomReportPath `
    -ParticleCount 100000

$freeDiskGB = Get-Dp150FreeDiskGB -Path $runDir
if ($freeDiskGB -lt $MinimumFreeDiskGB) {
    throw (
        "Only {0:N2} GB is free; at least {1:N2} GB is required." -f
        $freeDiskGB, $MinimumFreeDiskGB
    )
}
$completion = Get-Dp150CompletedRunState `
    -ReportPath $finalReportPath `
    -RunDirectory $runDir `
    -RunName $RunName `
    -ParticleCount 100000 `
    -SourceFiles $sourceFiles `
    -SequenceFiles $sequenceFiles `
    -JuliaVersion $juliaResolution.Version

if ($PreflightOnly) {
    if ($completion.State -eq 'LegacyComplete') {
        if (-not $AllowLegacyCompletedRun) {
            throw (
                'The completed 100k run predates provenance schema 1. Refusing ' +
                'checkpoint reuse; all existing artifacts were left unchanged. ' +
                'Use -PreflightOnly -AllowLegacyCompletedRun for an explicit ' +
                'read-only audit, or select a fresh -RunName.'
            )
        }
        Invoke-Dp150LegacyCompletedRunAudit `
            -JuliaPath $juliaResolution.Path `
            -ProjectDirectory $projectDirectory `
            -AuditScript $auditScript `
            -RunDirectory $runDir `
            -ParticleCount 100000
        Write-Output (
            "PASS_DP150_100K_LEGACY_PREFLIGHT_READ_ONLY run=$RunName " +
            "free_disk_gb=$('{0:N2}' -f $freeDiskGB) julia=$($juliaResolution.Version)"
        )
        exit 0
    }
    if ($completion.State -eq 'ProvenanceBoundComplete') {
        Write-Output (
            "PASS_DP150_100K_COMPLETE_PROVENANCE_BOUND_READ_ONLY run=$RunName " +
            "free_disk_gb=$('{0:N2}' -f $freeDiskGB) julia=$($juliaResolution.Version)"
        )
        exit 0
    }
    if ($AllowLegacyCompletedRun) {
        throw "No legacy completed run exists for explicit audit: $runDir"
    }

    $snapshot = @(Get-Dp150GpuSnapshot)
    foreach ($index in $requiredGpuIndices) {
        if (@($snapshot | Where-Object { $_.index -eq $index }).Count -ne 1) {
            throw "Required physical GPU $index was not found."
        }
    }
    $idle = Test-Dp150GpusIdle `
        -Snapshot $snapshot `
        -PhysicalGpuIndices $requiredGpuIndices `
        -MinimumFreeMiB $MinimumFreeMiB `
        -MaximumUtilizationPercent $MaximumUtilizationPercent
    Write-Output (
        "PASS_DP150_100K_FRESH_PREFLIGHT_READ_ONLY run=$RunName " +
        "free_disk_gb=$('{0:N2}' -f $freeDiskGB) gpus_idle=$($idle.ToString().ToLowerInvariant()) " +
        "julia=$($juliaResolution.Version)"
    )
    exit 0
}

if ($completion.State -eq 'LegacyComplete') {
    throw (
        'The completed 100k run predates provenance schema 1 and cannot be ' +
        'reused or overwritten. Use the explicit read-only legacy preflight, ' +
        'or select a fresh -RunName.'
    )
}
if ($completion.State -eq 'ProvenanceBoundComplete') {
    Write-Output (
        "PASS_DP150_100K_ALREADY_COMPLETE_PROVENANCE_BOUND run=$RunName " +
        "report=$finalReportPath"
    )
    exit 0
}

New-Item -ItemType Directory -Path $runDir -Force | Out-Null
New-Item -ItemType Directory -Path $globalLockDir -Force | Out-Null

function Write-QueueStatus {
    param(
        [string]$State,
        [string]$Message,
        [object]$GpuSnapshot = $null,
        [object]$Workers = $null
    )

    $status = [ordered]@{
        timestamp = (Get-Date).ToString('o')
        state = $State
        message = $Message
        launcher_pid = $PID
        run_name = $RunName
        required_physical_gpu_indices = $requiredGpuIndices
        minimum_free_mib_per_gpu = $MinimumFreeMiB
        maximum_utilization_percent = $MaximumUtilizationPercent
        stable_probe_count = $StableProbeCount
        run_directory = $runDir
        gpu_snapshot = $GpuSnapshot
        workers = $Workers
    }
    $temporary = "$statusPath.partial"
    $status | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $temporary -Encoding ASCII
    Move-Item -LiteralPath $temporary -Destination $statusPath -Force
    Write-Output "[$($status.timestamp)] $State - $Message"
}

$launcherLockStreams = @()
$resultLockStreams = @()
$gpuLockStreams = @()
$launcherLocksHeld = $false
try {
    $launcherLockStreams = @(Open-Dp150ExclusiveLocks `
        -Paths @($lockPath) `
        -ConflictDescription 'Another 100k dual-GPU launcher')
    $launcherLocksHeld = $true

    while ($true) {
        $snapshot = @(Get-Dp150GpuSnapshot)
        if (-not (Test-Dp150GpusIdle `
            -Snapshot $snapshot `
            -PhysicalGpuIndices $requiredGpuIndices `
            -MinimumFreeMiB $MinimumFreeMiB `
            -MaximumUtilizationPercent $MaximumUtilizationPercent)) {
            Write-QueueStatus `
                -State 'waiting_for_both_gpus' `
                -Message 'Both GPUs are required and at least one is currently busy.' `
                -GpuSnapshot $snapshot
            Start-Sleep -Seconds $RetryIntervalSeconds
            continue
        }

        $stable = $true
        for ($probe = 1; $probe -le $StableProbeCount; $probe++) {
            if ($probe -gt 1) {
                Start-Sleep -Seconds $ProbeIntervalSeconds
            }
            $probeSnapshot = @(Get-Dp150GpuSnapshot)
            if (-not (Test-Dp150GpusIdle `
                -Snapshot $probeSnapshot `
                -PhysicalGpuIndices $requiredGpuIndices `
                -MinimumFreeMiB $MinimumFreeMiB `
                -MaximumUtilizationPercent $MaximumUtilizationPercent)) {
                $stable = $false
                break
            }
            Write-QueueStatus `
                -State 'confirming_both_gpus_idle' `
                -Message "Both GPUs passed idle probe $probe of $StableProbeCount." `
                -GpuSnapshot $probeSnapshot
        }
        if (-not $stable) {
            Start-Sleep -Seconds $RetryIntervalSeconds
            continue
        }

        $finalSnapshot = @(Get-Dp150GpuSnapshot)
        if (Test-Dp150GpusIdle `
            -Snapshot $finalSnapshot `
            -PhysicalGpuIndices $requiredGpuIndices `
            -MinimumFreeMiB $MinimumFreeMiB `
            -MaximumUtilizationPercent $MaximumUtilizationPercent) {
            break
        }
        Write-QueueStatus `
            -State 'waiting_for_both_gpus' `
            -Message 'At least one GPU failed the immediate pre-launch recheck.' `
            -GpuSnapshot $finalSnapshot
        Start-Sleep -Seconds $RetryIntervalSeconds
    }

    $freeDiskGB = Get-Dp150FreeDiskGB -Path $runDir
    if ($freeDiskGB -lt $MinimumFreeDiskGB) {
        throw (
            "Only {0:N2} GB is free; at least {1:N2} GB is required." -f
            $freeDiskGB, $MinimumFreeDiskGB
        )
    }

    $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $worker0Out = Join-Path $runDir "worker_gpu0_$stamp.stdout.log"
    $worker0Err = Join-Path $runDir "worker_gpu0_$stamp.stderr.log"
    $worker1Out = Join-Path $runDir "worker_gpu1_$stamp.stdout.log"
    $worker1Err = Join-Path $runDir "worker_gpu1_$stamp.stderr.log"
    $worker0ReportName = 'simulation_report_gpu0.toml'
    $worker1ReportName = 'simulation_report_gpu1.toml'
    $worker0Args = (
        "-NoProfile -ExecutionPolicy Bypass -File `"$workerScript`" " +
        "-PhysicalGpuIndex 0 -Encodings REF,X -ReportName $worker0ReportName " +
        "-RunName $RunName"
    )
    $worker1Args = (
        "-NoProfile -ExecutionPolicy Bypass -File `"$workerScript`" " +
        "-PhysicalGpuIndex 1 -Encodings Y,Z -ReportName $worker1ReportName " +
        "-RunName $RunName"
    )
    if ($SeamSafeMotion) {
        $worker0Args += ' -SeamSafeMotion'
        $worker1Args += ' -SeamSafeMotion'
    }
    $started = Get-Date

    Write-QueueStatus `
        -State 'launching_dual_gpu' `
        -Message ("Starting the 100k run with {0:N2} GB free disk." -f $freeDiskGB) `
        -GpuSnapshot (Get-Dp150GpuSnapshot)
    $worker0 = Start-Process `
        -FilePath $powershell `
        -ArgumentList $worker0Args `
        -RedirectStandardOutput $worker0Out `
        -RedirectStandardError $worker0Err `
        -WindowStyle Hidden `
        -PassThru
    $worker1 = Start-Process `
        -FilePath $powershell `
        -ArgumentList $worker1Args `
        -RedirectStandardOutput $worker1Out `
        -RedirectStandardError $worker1Err `
        -WindowStyle Hidden `
        -PassThru
    $workers = @(
        [ordered]@{ gpu = 0; encodings = @('REF', 'X'); pid = $worker0.Id },
        [ordered]@{ gpu = 1; encodings = @('Y', 'Z'); pid = $worker1.Id }
    )
    Write-QueueStatus `
        -State 'running_dual_gpu' `
        -Message 'Both 100k encoding workers are running.' `
        -Workers $workers

    $worker0.WaitForExit()
    $worker1.WaitForExit()
    $worker0.Refresh()
    $worker1.Refresh()
    $worker0Exit = [int]$worker0.ExitCode
    $worker1Exit = [int]$worker1.ExitCode
    if ($worker0Exit -ne 0 -or $worker1Exit -ne 0) {
        throw "Worker exit codes were GPU0=$worker0Exit and GPU1=$worker1Exit."
    }

    $worker0Report = Join-Path $runDir $worker0ReportName
    $worker1Report = Join-Path $runDir $worker1ReportName
    Assert-Dp150SimulationReportCore `
        -Path $worker0Report `
        -RunName $RunName `
        -ParticleCount 100000 `
        -Encodings @('REF', 'X')
    Assert-Dp150SimulationReportCore `
        -Path $worker1Report `
        -RunName $RunName `
        -ParticleCount 100000 `
        -Encodings @('Y', 'Z')
    if (
        [string](Get-Dp150TomlScalar -Path $worker0Report -Key 'physical_gpu_index') -ne '0' -or
        [string](Get-Dp150TomlScalar -Path $worker1Report -Key 'physical_gpu_index') -ne '1' -or
        [string](Get-Dp150TomlScalar -Path $worker0Report -Key 'gpu_device_zero_based') -ne '0' -or
        [string](Get-Dp150TomlScalar -Path $worker1Report -Key 'gpu_device_zero_based') -ne '0'
    ) {
        throw 'Worker reports do not match their physical/visible GPU assignments.'
    }

    $resultLockStreams = @(Open-Dp150ExclusiveLocks `
        -Paths (Get-Dp150EncodingLockPaths `
            -RunDirectory $runDir `
            -Encodings @('REF', 'X', 'Y', 'Z')) `
        -ConflictDescription 'Another dp150 worker')
    $gpuLockStreams = @(Open-Dp150ExclusiveLocks `
        -Paths @((Join-Path $globalLockDir 'physical_gpu_0.lock')) `
        -ConflictDescription 'Another dp150 worker')
    foreach ($encoding in @('ref', 'x', 'y', 'z')) {
        Assert-Dp150FileNonEmpty `
            -Path (Join-Path $runDir "${encoding}_result.jld2") `
            -Description "$($encoding.ToUpperInvariant()) worker result"
    }

    Write-QueueStatus `
        -State 'reconstructing' `
        -Message 'All four 100k encodings passed worker validation; assembling reconstruction.'
    $env:CUDA_VISIBLE_DEVICES = '0'
    $env:DP150_GPU_DEVICE = '0'
    $env:DP150_PHYSICAL_GPU_INDEX = '0 (reuse-only reconstruction)'
    $env:DP150_RUN_NAME = $RunName
    $env:DP150_PHANTOM = $phantomPath
    $env:DP150_EXPECTED_PARTICLE_COUNT = '100000'
    $env:DP150_BATCH_SIZE = '20000'
    $env:DP150_ENCODINGS = 'REF,X,Y,Z'
    $env:DP150_REPORT_NAME = 'simulation_report.toml'
    $env:DP150_REUSE_RESULTS = 'true'
    $env:DP150_B0_T = '3.0'
    $env:DP150_RESCALE_MOTION_TO_SEQUENCE_RR = 'true'
    $env:DP150_SEAM_SAFE_MOTION = if ($SeamSafeMotion) { 'true' } else { 'false' }
    $env:DP150_MAX_BLOCK_LENGTH = '512'
    $env:DP150_ALLOW_UNVALIDATED_BLOCK_LENGTH = 'false'
    $env:DP150_SIM_OUTPUT_DIR = Join-Path $repoRoot 'outputs\dp150_mri_pipeline\koma'
    Push-Location $repoRoot
    try {
        & $juliaResolution.Path `
            '--startup-file=no' `
            "--project=$projectDirectory" `
            $simulationScript
        $reconstructionExit = [int]$LASTEXITCODE
    }
    finally {
        Pop-Location
    }
    if ($reconstructionExit -ne 0) {
        throw "Reuse-only reconstruction exited with code $reconstructionExit."
    }

    $verifiedCompletion = Get-Dp150CompletedRunState `
        -ReportPath $finalReportPath `
        -RunDirectory $runDir `
        -RunName $RunName `
        -ParticleCount 100000 `
        -SourceFiles $sourceFiles `
        -SequenceFiles $sequenceFiles `
        -JuliaVersion $juliaResolution.Version
    if ($verifiedCompletion.State -ne 'ProvenanceBoundComplete') {
        throw 'Fresh reconstruction did not produce a provenance-bound completed run.'
    }

    $finished = Get-Date
    $execution = [ordered]@{
        status = 'PASS'
        started = $started.ToString('o')
        finished = $finished.ToString('o')
        dual_gpu_wall_seconds = ($finished - $started).TotalSeconds
        run_name = $RunName
        particle_count = 100000
        batch_size = 20000
        seam_safe_motion = [bool]$SeamSafeMotion
        julia_version = [string]$juliaResolution.Version
        assignments = $workers
        worker_exit_codes = [ordered]@{ gpu0 = $worker0Exit; gpu1 = $worker1Exit }
        worker_logs = [ordered]@{
            gpu0_stdout = $worker0Out
            gpu0_stderr = $worker0Err
            gpu1_stdout = $worker1Out
            gpu1_stderr = $worker1Err
        }
        final_simulation_report = $finalReportPath
    }
    $temporaryExecutionReport = "$executionReportPath.partial"
    $execution | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $temporaryExecutionReport -Encoding ASCII
    Move-Item -LiteralPath $temporaryExecutionReport -Destination $executionReportPath -Force
    Write-QueueStatus `
        -State 'complete' `
        -Message ("The provenance-bound 100k simulation completed in {0:N1} minutes." -f (($finished - $started).TotalMinutes))
}
catch {
    if ($launcherLocksHeld) {
        Write-QueueStatus -State 'launcher_error' -Message $_.Exception.Message
    }
    throw
}
finally {
    Close-Dp150Locks -Streams $gpuLockStreams
    Close-Dp150Locks -Streams $resultLockStreams
    Close-Dp150Locks -Streams $launcherLockStreams
}

exit 0
