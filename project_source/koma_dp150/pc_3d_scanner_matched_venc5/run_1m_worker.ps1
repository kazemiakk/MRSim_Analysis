param(
    [Parameter(Mandatory = $true)]
    [ValidateSet(0, 1)]
    [int]$PhysicalGpuIndex,

    [Parameter(Mandatory = $true)]
    [string]$Encodings,

    [Parameter(Mandatory = $true)]
    [string]$ReportName,

    [string]$RunName = 'motion_1m_scanner_matched_venc5',

    [switch]$SeamSafeMotion
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'launcher_common.ps1')

$encodingList = @(ConvertTo-Dp150EncodingList -Value $Encodings)
Assert-Dp150SafeReportName -ReportName $ReportName
Assert-Dp150SafeRunName -RunName $RunName -RequiredPrefix 'motion_1m_scanner_matched_venc5'

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$projectDirectory = Join-Path $repoRoot 'koma_dp150'
$simulationScriptName = if ($SeamSafeMotion) {
    'simulate_1m_seam_safe.jl'
}
else {
    'simulate.jl'
}
$simulationScript = Join-Path $PSScriptRoot $simulationScriptName
$phantom = Join-Path $repoRoot 'outputs\dp150_mri_pipeline\koma\dp150_motion_1m.phantom'
$phantomReport = Join-Path $repoRoot 'outputs\dp150_mri_pipeline\koma\motion_phantom_report.toml'
$runDir = Join-Path $repoRoot "outputs\dp150_mri_pipeline\koma\simulation_$RunName"
$reportPath = Join-Path $runDir $ReportName
$finalReportPath = Join-Path $runDir 'simulation_report.toml'
$globalLockDir = Join-Path $repoRoot 'outputs\dp150_mri_pipeline\koma\.launcher_locks'
$physicalGpuLockPath = Join-Path $globalLockDir "physical_gpu_$PhysicalGpuIndex.lock"
$juliaResolution = Resolve-Dp150JuliaExecutable

foreach ($path in @(
    $juliaResolution.Path,
    (Join-Path $projectDirectory 'Project.toml'),
    (Join-Path $projectDirectory 'Manifest.toml'),
    $simulationScript,
    $phantom,
    $phantomReport,
    (Join-Path $PSScriptRoot 'protocol_audit.toml')
)) {
    Assert-Dp150FileNonEmpty -Path $path -Description 'Required 1M worker input'
}
foreach ($encoding in $encodingList) {
    Assert-Dp150FileNonEmpty `
        -Path (Join-Path $PSScriptRoot "$($encoding.ToLowerInvariant()).seq") `
        -Description "$encoding sequence"
}
$protocolAudit = Join-Path $PSScriptRoot 'protocol_audit.toml'
$allSequenceFiles = @{
    REF = (Join-Path $PSScriptRoot 'ref.seq')
    X = (Join-Path $PSScriptRoot 'x.seq')
    Y = (Join-Path $PSScriptRoot 'y.seq')
    Z = (Join-Path $PSScriptRoot 'z.seq')
}
Assert-Dp150ProtocolInputs -AuditPath $protocolAudit -SequenceFiles $allSequenceFiles
Assert-Dp150PhantomReport `
    -PhantomPath $phantom `
    -ReportPath $phantomReport `
    -ParticleCount 1000000
if (
    (Test-Path -LiteralPath $finalReportPath -PathType Leaf) -and
    [string](Get-Dp150TomlScalar -Path $finalReportPath -Key 'status') -eq 'PASS'
) {
    throw (
        'A final PASS report already protects this run directory. Direct worker ' +
        "execution cannot modify a completed run: $finalReportPath"
    )
}
$protectedFinalArtifacts = @(
    (Join-Path $runDir 'pc_reconstruction.h5'),
    (Join-Path $runDir 'pc_reconstruction_qc.png')
)
$existingFinalArtifacts = @($protectedFinalArtifacts | Where-Object {
    Test-Path -LiteralPath $_ -PathType Leaf
})
if ($existingFinalArtifacts.Count -gt 0) {
    throw (
        'Direct worker execution cannot modify a run with final artifacts: ' +
        ($existingFinalArtifacts -join ', ')
    )
}

New-Item -ItemType Directory -Path $runDir -Force | Out-Null
New-Item -ItemType Directory -Path $globalLockDir -Force | Out-Null
$lockStreams = @()
try {
    $lockStreams = @(Open-Dp150ExclusiveLocks `
        -Paths (@(
            Get-Dp150EncodingLockPaths -RunDirectory $runDir -Encodings $encodingList
        ) + @("$reportPath.lock", $physicalGpuLockPath)) `
        -ConflictDescription 'Another dp150 worker')

    $workerAlreadyComplete = $false
    if (Test-Path -LiteralPath $reportPath -PathType Leaf) {
        $status = [string](Get-Dp150TomlScalar -Path $reportPath -Key 'status')
        if ($status -eq 'PASS') {
            Assert-Dp150SimulationReportCore `
                -Path $reportPath `
                -RunName $RunName `
                -ParticleCount 1000000 `
                -Encodings $encodingList
            foreach ($encoding in $encodingList) {
                Assert-Dp150FileNonEmpty `
                    -Path (Join-Path $runDir "$($encoding.ToLowerInvariant())_result.jld2") `
                    -Description "$encoding worker result"
            }
            $workerAlreadyComplete = $true
        }
    }

    if (-not $workerAlreadyComplete) {
        $env:CUDA_VISIBLE_DEVICES = [string]$PhysicalGpuIndex
        $env:DP150_GPU_DEVICE = '0'
        $env:DP150_PHYSICAL_GPU_INDEX = [string]$PhysicalGpuIndex
        $env:DP150_RUN_NAME = $RunName
        $env:DP150_PHANTOM = $phantom
        $env:DP150_EXPECTED_PARTICLE_COUNT = '1000000'
        $env:DP150_BATCH_SIZE = '20000'
        $env:DP150_ENCODINGS = $encodingList -join ','
        $env:DP150_REPORT_NAME = $ReportName
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
            $workerExitCode = [int]$LASTEXITCODE
        }
        finally {
            Pop-Location
        }
        if ($workerExitCode -ne 0) {
            throw "Julia worker exited with code $workerExitCode."
        }

        Assert-Dp150SimulationReportCore `
            -Path $reportPath `
            -RunName $RunName `
            -ParticleCount 1000000 `
            -Encodings $encodingList
        foreach ($encoding in $encodingList) {
            Assert-Dp150FileNonEmpty `
                -Path (Join-Path $runDir "$($encoding.ToLowerInvariant())_result.jld2") `
                -Description "$encoding worker result"
        }
    }
    if ([string](Get-Dp150TomlScalar -Path $reportPath -Key 'physical_gpu_index') -ne [string]$PhysicalGpuIndex) {
        throw "Worker report physical GPU does not match GPU $PhysicalGpuIndex`: $reportPath"
    }
    if ([string](Get-Dp150TomlScalar -Path $reportPath -Key 'gpu_device_zero_based') -ne '0') {
        throw "Worker report did not use CUDA-visible device zero: $reportPath"
    }
}
finally {
    Close-Dp150Locks -Streams $lockStreams
}

Write-Output (
    "PASS_DP150_1M_WORKER encodings=$($encodingList -join ',') " +
    "run=$RunName gpu=$PhysicalGpuIndex julia=$($juliaResolution.Version) report=$reportPath"
)
exit 0
