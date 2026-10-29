$ErrorActionPreference = 'Stop'
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$helper = Join-Path $repo 'scripts/initialize_d_storage.ps1'
$marker = 'D:\MRSim\storage.json'
$before = (Get-FileHash -LiteralPath $marker -Algorithm SHA256).Hash

$check = (& $helper -CheckOnly | Out-String | ConvertFrom-Json)
if ($check.status -ne 'PASS_STORAGE_CHECK' -or $check.verified_copies -ne 3 -or
    $check.checked_frozen_sources -ne 4 -or $check.simulation_started -or
    $check.legacy_launchers_redirected) { throw 'Read-only storage check failed.' }

$reuse = (& $helper | Out-String | ConvertFrom-Json)
if ($reuse.status -ne 'PASS_STORAGE_CHECK') { throw 'Idempotent setup failed.' }
if ((Get-FileHash -LiteralPath $marker -Algorithm SHA256).Hash -ne $before) {
    throw 'Reuse overwrote the storage registration.'
}

$previousTemp = $env:TEMP
$previousTmp = $env:TMP
$previousOutput = $env:MRSIM_OUTPUT_ROOT
try {
    $active = (& $helper -CheckOnly -Activate | Out-String | ConvertFrom-Json)
    if (-not $active.process_environment_activated -or
        $env:TEMP -cne 'D:\MRSim\temp' -or
        $env:TMP -cne 'D:\MRSim\temp' -or
        $env:MRSIM_OUTPUT_ROOT -cne 'D:\MRSim\outputs') {
        throw 'Process-local environment activation failed.'
    }
} finally {
    $env:TEMP = $previousTemp
    $env:TMP = $previousTmp
    $env:MRSIM_OUTPUT_ROOT = $previousOutput
}
if (@(Get-ChildItem -LiteralPath 'D:\MRSim\outputs' -Force).Count -ne 0) {
    throw 'Unexpected new output: storage setup must not start simulations.'
}
'PASS: read-only verification, idempotent reuse, unchanged registration, process-local activation/restoration, empty new output root.'
