$ErrorActionPreference = 'Stop'
$repo = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$folder = [System.IO.Path]::GetFullPath((Join-Path $repo 'outputs/known_flow_3d_e051/acquisition_v2'))
$receipt = Join-Path $repo 'outputs/known_flow_3d_e051/filesystem_compression_v1.json'
if (Test-Path -LiteralPath $receipt) { throw 'Compression receipt already exists.' }
$names = @('static_landmarks_ref.h5', 'static_landmarks_x.h5', 'static_landmarks_y.h5', 'static_landmarks_z.h5', 'plug_positive_ref.h5')
$hashes = @{}
$paths = @()
foreach ($name in $names) {
    $path = (Resolve-Path -LiteralPath (Join-Path $folder $name)).Path
    if (-not $path.StartsWith(($folder + [System.IO.Path]::DirectorySeparatorChar), [System.StringComparison]::OrdinalIgnoreCase)) {
        throw 'Compression target outside the new E051 control folder.'
    }
    if ((Get-Item -LiteralPath $path).Attributes -band [System.IO.FileAttributes]::ReparsePoint) { throw 'Refuse reparse-point target.' }
    if (-not (Test-Path -LiteralPath (Join-Path $folder ($name.Replace('.h5', '_checkpoint.toml'))))) { throw 'Encoding is not checkpoint-complete.' }
    $hashes[$name] = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
    $paths += $path
}
$before = [System.IO.DriveInfo]::new('C:/').AvailableFreeSpace
& 'C:/Windows/System32/compact.exe' /C /Q $paths
$compressionExit = $LASTEXITCODE
foreach ($name in $names) {
    $afterHash = (Get-FileHash -LiteralPath (Join-Path $folder $name) -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($afterHash -ne $hashes[$name]) { throw ('Scientific file content changed: ' + $name) }
}
$after = [System.IO.DriveInfo]::new('C:/').AvailableFreeSpace
@{status='FILE_CONTENT_IDENTICAL_AFTER_LOSSLESS_FILESYSTEM_COMPRESSION';
  compact_exit_code=$compressionExit; free_disk_before_bytes=$before; free_disk_after_bytes=$after;
  files_sha256=$hashes; finished_utc=[DateTime]::UtcNow.ToString('o')} |
    ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $receipt -Encoding UTF8
Write-Output ('CONTENT_UNCHANGED free_disk_bytes=' + $after + ' observed_free_space_change=' + ($after-$before))
