param(
    [switch]$CheckOnly,
    [switch]$Activate
)

# Operational storage setup only: never launch a simulation or relocate evidence.
$ErrorActionPreference = 'Stop'
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..')).TrimEnd('\')
$root = 'D:\MRSim'
$marker = Join-Path $root 'storage.json'
$guide = Join-Path $repo 'docs/STORAGE_D_DRIVE.md'

function Assert-PlainDirectory([string]$path) {
    $item = Get-Item -LiteralPath $path -Force
    if (-not $item.PSIsContainer -or
        ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        throw "Not a plain directory: $path"
    }
}

function Copy-Verified([string]$source, [string]$destination) {
    $expected = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash
    if (Test-Path -LiteralPath $destination) {
        $item = Get-Item -LiteralPath $destination -Force
        if ($item.PSIsContainer -or
            ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
            throw "Refuse unexpected copy destination: $destination"
        }
    } else {
        [IO.File]::Copy($source, $destination, $false)
    }
    if ((Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash -ne $expected) {
        throw "Existing or copied file differs; do not overwrite: $destination"
    }
    return @{ path=$destination; source=$source; sha256=$expected.ToLowerInvariant() }
}

Assert-PlainDirectory 'D:\'
$drive = [IO.DriveInfo]::new('D:\')
if (-not $drive.IsReady -or $drive.DriveFormat -ne 'NTFS') {
    throw 'D: must be ready and NTFS; do not use an unexpected volume.'
}
$paths = [ordered]@{
    code_root=$repo
    legacy_outputs=(Join-Path $repo 'outputs')
    output_root=(Join-Path $root 'outputs')
    temp_root=(Join-Path $root 'temp')
    log_root=(Join-Path $root 'logs')
    verification_root=(Join-Path $root 'verification')
}

if (Test-Path -LiteralPath $root) {
    Assert-PlainDirectory $root
    if (-not (Test-Path -LiteralPath $marker -PathType Leaf)) {
        throw 'D:\MRSim already exists without this registration; inspect before reuse.'
    }
    $saved = Get-Content -LiteralPath $marker -Raw | ConvertFrom-Json
    if ($saved.schema -ne 'mrsim-split-storage-v1') { throw 'Unknown storage registration.' }
    foreach ($key in $paths.Keys) {
        if ($saved.paths.$key -cne $paths[$key]) { throw "Storage path changed: $key" }
    }
} elseif ($CheckOnly) {
    throw 'D-drive storage has not been initialized.'
} else {
    if ($drive.AvailableFreeSpace -lt 100GB) { throw 'Less than 100 GiB free on D:.' }
    New-Item -ItemType Directory -Path $root | Out-Null
}

foreach ($key in @('output_root', 'temp_root', 'log_root', 'verification_root')) {
    $path = $paths[$key]
    if (-not (Test-Path -LiteralPath $path)) {
        if ($CheckOnly) { throw "Missing storage directory: $path" }
        New-Item -ItemType Directory -Path $path | Out-Null
    }
    Assert-PlainDirectory $path
}

if (-not (Test-Path -LiteralPath $marker)) {
    $copies = @(
        (Copy-Verified $guide (Join-Path $root 'README.md')),
        (Copy-Verified (Join-Path $repo 'uv.lock') (Join-Path $paths.verification_root 'uv.lock')),
        (Copy-Verified (Join-Path $repo 'koma_migration_e046/Manifest.toml') (Join-Path $paths.verification_root 'Koma_Manifest.toml'))
    )
    $frozen = @{}
    foreach ($relative in @(
        'koma_migration_e046/precision_confirmation_e050.jl',
        'koma_migration_e046/known_flow_3d_e051.jl',
        'outputs/phantom_precision_e050/analysis_v1/report.json',
        'outputs/known_flow_3d_e051/analysis_v1/report.json'
    )) {
        $frozen[$relative] = (Get-FileHash -LiteralPath (Join-Path $repo $relative) -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    $record = [ordered]@{
        schema='mrsim-split-storage-v1'
        status='STORAGE_READY_NO_SIMULATION_LAUNCHED'
        created_utc=[DateTime]::UtcNow.ToString('o')
        paths=$paths
        free_bytes_at_setup=$drive.AvailableFreeSpace
        copies=$copies
        frozen_c_source_sha256=$frozen
        setup_script_sha256=(Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()
        legacy_launchers_redirected=$false
        existing_data_moved=$false
        new_simulation_started=$false
    }
    $partial = Join-Path $root ('storage.'+[Guid]::NewGuid().ToString('N')+'.partial')
    $record | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $partial -Encoding UTF8
    [IO.File]::Move($partial, $marker)
}

$saved = Get-Content -LiteralPath $marker -Raw | ConvertFrom-Json
foreach ($copy in $saved.copies) {
    if ((Get-FileHash -LiteralPath $copy.path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $copy.sha256) {
        throw "Storage verification copy changed: $($copy.path)"
    }
}
foreach ($entry in $saved.frozen_c_source_sha256.PSObject.Properties) {
    if ((Get-FileHash -LiteralPath (Join-Path $repo $entry.Name) -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.Value) {
        throw "Frozen C: evidence changed: $($entry.Name)"
    }
}

if ($Activate) {
    # Process-local only; old launchers do not automatically read MRSIM_OUTPUT_ROOT.
    $env:MRSIM_OUTPUT_ROOT=$paths.output_root
    $env:TEMP=$paths.temp_root
    $env:TMP=$paths.temp_root
}
[ordered]@{
    status='PASS_STORAGE_CHECK'
    paths=$paths
    free_bytes_now=$drive.AvailableFreeSpace
    verified_copies=$saved.copies.Count
    checked_frozen_sources=$saved.frozen_c_source_sha256.PSObject.Properties.Name.Count
    process_environment_activated=[bool]$Activate
    legacy_launchers_redirected=$false
    simulation_started=$false
} | ConvertTo-Json -Depth 5
