param(
    [int]$StableProbeCount = 3,
    [int]$ProbeIntervalSeconds = 15,
    [int]$RetryIntervalSeconds = 60,
    [int]$MinimumFreeMiB = 12000,
    [int]$MaximumUtilizationPercent = 25,
    [double]$MinimumFreeDiskGB = 30.0,
    [string]$RunName = 'motion_1m_scanner_matched_venc5',
    [switch]$PreflightOnly,
    [switch]$AllowLegacyCompletedRun
)

$ErrorActionPreference = 'Stop'
$hardenedLauncher = Join-Path $PSScriptRoot 'launch_1m_dual_when_gpus_idle.ps1'
if (-not (Test-Path -LiteralPath $hardenedLauncher -PathType Leaf)) {
    throw "The hardened 1M launcher is missing: $hardenedLauncher"
}

Write-Output (
    'Migration resume delegates to the provenance-bound dual launcher; ' +
    'its worker locks and report checks safely reuse any validated partial results.'
)
& $hardenedLauncher `
    -StableProbeCount $StableProbeCount `
    -ProbeIntervalSeconds $ProbeIntervalSeconds `
    -RetryIntervalSeconds $RetryIntervalSeconds `
    -MinimumFreeMiB $MinimumFreeMiB `
    -MaximumUtilizationPercent $MaximumUtilizationPercent `
    -MinimumFreeDiskGB $MinimumFreeDiskGB `
    -RunName $RunName `
    -PreflightOnly:$PreflightOnly `
    -AllowLegacyCompletedRun:$AllowLegacyCompletedRun
exit ([int]$LASTEXITCODE)
