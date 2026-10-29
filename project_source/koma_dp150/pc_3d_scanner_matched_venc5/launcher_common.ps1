$script:Dp150MinimumJuliaVersion = [version]'1.12.0'
$script:Dp150MaximumJuliaVersion = [version]'2.0.0'

function Get-Dp150JuliaVersion {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    $versionOutput = @(& $Path '--startup-file=no' '--version' 2>&1)
    $versionText = ($versionOutput -join "`n").Trim()
    if ($LASTEXITCODE -ne 0 -or $versionText -notmatch '^julia version (?<version>\d+\.\d+\.\d+)$') {
        throw "Cannot determine a stable Julia version from $Path`: $versionText"
    }
    return [version]$Matches['version']
}

function Test-Dp150JuliaVersion {
    param(
        [Parameter(Mandatory = $true)]
        [version]$Version
    )

    return (
        $Version -ge $script:Dp150MinimumJuliaVersion -and
        $Version -lt $script:Dp150MaximumJuliaVersion
    )
}

function Resolve-Dp150JuliaExecutable {
    if ($env:MRSIM_JULIA_EXE) {
        $resolved = Resolve-Path -LiteralPath $env:MRSIM_JULIA_EXE -ErrorAction Stop
        $path = $resolved.Path
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "MRSIM_JULIA_EXE is not a file: $path"
        }
        $version = Get-Dp150JuliaVersion -Path $path
        if (-not (Test-Dp150JuliaVersion -Version $version)) {
            throw "Julia >=1.12.0 and <2.0.0 is required; $path is $version"
        }
        return [pscustomobject]@{
            Path = $path
            Version = $version
        }
    }

    $candidatePaths = @(
        Get-Command julia -All -ErrorAction SilentlyContinue |
            Where-Object { $_.CommandType -eq 'Application' } |
            ForEach-Object { $_.Source }
    )
    if ($env:LOCALAPPDATA) {
        $candidatePaths += @(
            Get-ChildItem `
                -Path (Join-Path $env:LOCALAPPDATA 'Programs\Julia-*\bin\julia.exe') `
                -File `
                -ErrorAction SilentlyContinue |
                ForEach-Object { $_.FullName }
        )
    }

    $validCandidates = @()
    foreach ($candidate in @(
        $candidatePaths |
            Where-Object { $_ } |
            Select-Object -Unique
    )) {
        try {
            $path = (Resolve-Path -LiteralPath $candidate -ErrorAction Stop).Path
            $version = Get-Dp150JuliaVersion -Path $path
            if (Test-Dp150JuliaVersion -Version $version) {
                $validCandidates += [pscustomobject]@{
                    Path = $path
                    Version = $version
                }
            }
            else {
                Write-Warning "Ignoring incompatible Julia $version at $path"
            }
        }
        catch {
            Write-Warning $_.Exception.Message
        }
    }

    if ($validCandidates.Count -eq 0) {
        throw 'Julia >=1.12.0 and <2.0.0 was not found. Set MRSIM_JULIA_EXE.'
    }
    return (
        $validCandidates |
            Sort-Object -Property Version -Descending |
            Select-Object -First 1
    )
}

function Get-Dp150FileSha256Lower {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    return (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
}

function ConvertFrom-Dp150TomlValue {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Value
    )

    $trimmed = $Value.Trim()
    if (
        ($trimmed.StartsWith('"') -and $trimmed.EndsWith('"')) -or
        ($trimmed.StartsWith('[') -and $trimmed.EndsWith(']'))
    ) {
        try {
            return ($trimmed | ConvertFrom-Json -ErrorAction Stop)
        }
        catch {
            if ($trimmed.StartsWith('"') -and $trimmed.EndsWith('"')) {
                return $trimmed.Substring(1, $trimmed.Length - 2)
            }
            throw "Unsupported TOML value: $Value"
        }
    }
    return $trimmed
}

function Get-Dp150TomlScalar {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,

        [Parameter(Mandatory = $true)]
        [string]$Key,

        [string]$Section = ''
    )

    $currentSection = ''
    $escapedKey = [regex]::Escape($Key)
    $matchesFound = @()
    foreach ($line in Get-Content -LiteralPath $Path) {
        if ($line -match '^\s*\[(?<section>[^]]+)\]\s*$') {
            $currentSection = $Matches['section']
            continue
        }
        if ($currentSection -ne $Section) {
            continue
        }
        if ($line -match "^\s*$escapedKey\s*=\s*(?<value>.+?)\s*$") {
            $matchesFound += $Matches['value']
        }
    }

    if ($matchesFound.Count -eq 0) {
        return $null
    }
    if ($matchesFound.Count -ne 1) {
        throw "TOML key [$Section] $Key is not unique in $Path"
    }
    return ConvertFrom-Dp150TomlValue -Value $matchesFound[0]
}

function Assert-Dp150FileNonEmpty {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,

        [string]$Description = 'Required file'
    )

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Description is missing: $Path"
    }
    if ((Get-Item -LiteralPath $Path).Length -le 0) {
        throw "$Description is empty: $Path"
    }
}

function Assert-Dp150ProtocolInputs {
    param(
        [Parameter(Mandatory = $true)]
        [string]$AuditPath,

        [Parameter(Mandatory = $true)]
        [hashtable]$SequenceFiles
    )

    Assert-Dp150FileNonEmpty -Path $AuditPath -Description 'Protocol audit'
    $expectedRoot = [ordered]@{
        status = 'PASS'
        provenance_schema_version = '1'
        venc_target_m_per_s = '5.0'
    }
    foreach ($key in $expectedRoot.Keys) {
        $actual = [string](Get-Dp150TomlScalar -Path $AuditPath -Key $key)
        if ($actual -ne $expectedRoot[$key]) {
            throw "Protocol audit mismatch for $key`: $AuditPath"
        }
    }

    $protocolReportName = [string](Get-Dp150TomlScalar `
        -Path $AuditPath `
        -Key 'protocol_report_file')
    if (
        -not $protocolReportName -or
        [System.IO.Path]::GetFileName($protocolReportName) -cne $protocolReportName
    ) {
        throw "Protocol audit declares an unsafe protocol report path: $AuditPath"
    }
    $protocolReportPath = Join-Path (Split-Path -Parent $AuditPath) $protocolReportName
    Assert-Dp150FileNonEmpty -Path $protocolReportPath -Description 'Portable protocol report'
    $expectedProtocolReportSha = [string](Get-Dp150TomlScalar `
        -Path $AuditPath `
        -Key 'protocol_report_sha256')
    $actualProtocolReportSha = Get-Dp150FileSha256Lower -Path $protocolReportPath
    if ($expectedProtocolReportSha -cne $actualProtocolReportSha) {
        throw "Portable protocol report SHA-256 does not match its audit: $AuditPath"
    }

    foreach ($encoding in @('REF', 'X', 'Y', 'Z')) {
        if (-not $SequenceFiles.ContainsKey($encoding)) {
            throw "No current sequence path was supplied for $encoding"
        }
        $sequencePath = [string]$SequenceFiles[$encoding]
        Assert-Dp150FileNonEmpty -Path $sequencePath -Description "$encoding sequence"
        $section = "encodings.$encoding"
        $reportedFile = [string](Get-Dp150TomlScalar `
            -Path $AuditPath `
            -Section $section `
            -Key 'file')
        if ($reportedFile -cne [System.IO.Path]::GetFileName($sequencePath)) {
            throw "Protocol audit file mismatch for $encoding`: $AuditPath"
        }
        $actualBytes = (Get-Item -LiteralPath $sequencePath).Length
        $reportedBytes = [string](Get-Dp150TomlScalar `
            -Path $AuditPath `
            -Section $section `
            -Key 'bytes')
        if ($reportedBytes -ne [string]$actualBytes) {
            throw "Protocol audit byte-count mismatch for $encoding`: $AuditPath"
        }
        $actualSha = Get-Dp150FileSha256Lower -Path $sequencePath
        $reportedSha = [string](Get-Dp150TomlScalar `
            -Path $AuditPath `
            -Section $section `
            -Key 'sha256')
        if ($reportedSha -cne $actualSha) {
            throw "Protocol audit SHA-256 mismatch for $encoding`: $AuditPath"
        }
        $matrix = @(Get-Dp150TomlScalar `
            -Path $AuditPath `
            -Section $section `
            -Key 'matrix')
        if (($matrix -join ',') -ne '400,64,32') {
            throw "Protocol audit matrix mismatch for $encoding`: $AuditPath"
        }
        foreach ($passKey in @('hardware_check', 'timing_check')) {
            if ([string](Get-Dp150TomlScalar `
                -Path $AuditPath `
                -Section $section `
                -Key $passKey) -ne 'passed') {
                throw "Protocol audit $passKey failed for $encoding`: $AuditPath"
            }
        }
    }
}

function Assert-Dp150PhantomReport {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PhantomPath,

        [Parameter(Mandatory = $true)]
        [string]$ReportPath,

        [Parameter(Mandatory = $true)]
        [long]$ParticleCount
    )

    Assert-Dp150FileNonEmpty -Path $PhantomPath -Description 'Motion phantom'
    Assert-Dp150FileNonEmpty -Path $ReportPath -Description 'Motion phantom report'
    if ([string](Get-Dp150TomlScalar -Path $ReportPath -Key 'status') -ne 'PASS') {
        throw "The motion phantom report is not PASS: $ReportPath"
    }
    if ([string](Get-Dp150TomlScalar -Path $ReportPath -Key 'particle_count') -ne [string]$ParticleCount) {
        throw "The motion phantom report has the wrong particle count: $ReportPath"
    }
    $actualBytes = (Get-Item -LiteralPath $PhantomPath).Length
    $reportedBytes = [string](Get-Dp150TomlScalar -Path $ReportPath -Key 'phantom_bytes')
    if ($reportedBytes -ne [string]$actualBytes) {
        throw "The motion phantom byte count does not match its report: $ReportPath"
    }
    if ([string](Get-Dp150TomlScalar -Path $ReportPath -Key 'flowpath_sample_count') -ne '1441') {
        throw "The motion phantom report has the wrong FlowPath sample count: $ReportPath"
    }
}

function Test-Dp150SamePath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Left,

        [Parameter(Mandatory = $true)]
        [string]$Right
    )

    $trimCharacters = [char[]]@([char]'\', [char]'/')
    $leftFull = [System.IO.Path]::GetFullPath($Left).TrimEnd($trimCharacters)
    $rightFull = [System.IO.Path]::GetFullPath($Right).TrimEnd($trimCharacters)
    return [string]::Equals(
        $leftFull,
        $rightFull,
        [System.StringComparison]::OrdinalIgnoreCase
    )
}

function Assert-Dp150SimulationReportCore {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,

        [Parameter(Mandatory = $true)]
        [string]$RunName,

        [Parameter(Mandatory = $true)]
        [long]$ParticleCount,

        [Parameter(Mandatory = $true)]
        [string[]]$Encodings,

        [int]$BatchSize = 20000
    )

    Assert-Dp150FileNonEmpty -Path $Path -Description 'Simulation report'
    $expectedScalars = [ordered]@{
        status = 'PASS'
        run_name = $RunName
        particle_count = [string]$ParticleCount
        expected_particle_count = [string]$ParticleCount
        scanner_B0_T = '3.0'
        spin_batch_size = [string]$BatchSize
        precision = 'f32'
        maximum_simulation_block_length = '512'
        validated_simulation_block_length = '512'
        unvalidated_block_length_override = 'false'
        result_reuse_enabled = 'true'
        simulation_method = 'Bloch with perfect transverse spoiling before RF'
        gradient_max_step_s = 'Inf (event boundaries only, matching supplied script)'
        gradient_spoiling_simulated = 'false'
    }
    foreach ($key in $expectedScalars.Keys) {
        $actual = Get-Dp150TomlScalar -Path $Path -Key $key
        if ([string]$actual -ne $expectedScalars[$key]) {
            throw (
                "Simulation report mismatch for $key`: expected " +
                "'$($expectedScalars[$key])', found '$actual' in $Path"
            )
        }
    }

    $reportedEncodings = @(Get-Dp150TomlScalar -Path $Path -Key 'encodings')
    if (
        $reportedEncodings.Count -ne $Encodings.Count -or
        (($reportedEncodings -join ',') -ne ($Encodings -join ','))
    ) {
        throw (
            "Simulation report encodings mismatch: expected " +
            "[$($Encodings -join ',')], found [$($reportedEncodings -join ',')] in $Path"
        )
    }

    foreach ($encoding in $Encodings) {
        $section = "sequence.$encoding"
        if ([string](Get-Dp150TomlScalar -Path $Path -Section $section -Key 'Venc_m_per_s') -ne '5.0') {
            throw "Simulation report $section does not certify VENC 5.0 m/s: $Path"
        }
        if ([string](Get-Dp150TomlScalar -Path $Path -Section $section -Key 'cardiac_bins') -ne '15') {
            throw "Simulation report $section does not certify 15 cardiac bins: $Path"
        }
        $matrix = @(Get-Dp150TomlScalar -Path $Path -Section $section -Key 'matrix')
        if (($matrix -join ',') -ne '400,64,32') {
            throw "Simulation report $section has the wrong matrix: $Path"
        }
    }
}

function Assert-Dp150ProvenanceBoundCompletion {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ReportPath,

        [Parameter(Mandatory = $true)]
        [string]$RunDirectory,

        [Parameter(Mandatory = $true)]
        [hashtable]$SourceFiles,

        [Parameter(Mandatory = $true)]
        [hashtable]$SequenceFiles,

        [Parameter(Mandatory = $true)]
        [version]$JuliaVersion
    )

    $schema = [string](Get-Dp150TomlScalar -Path $ReportPath -Key 'provenance_schema_version')
    if ($schema -ne '1') {
        throw "Unsupported dp150 provenance schema '$schema' in $ReportPath"
    }
    $configurationSha = [string](Get-Dp150TomlScalar -Path $ReportPath -Key 'run_configuration_sha256')
    if ($configurationSha -cnotmatch '^[0-9a-f]{64}$') {
        throw "Invalid run_configuration_sha256 in $ReportPath"
    }
    if ([string](Get-Dp150TomlScalar -Path $ReportPath -Key 'julia_version') -ne [string]$JuliaVersion) {
        throw "Completed-run Julia version does not match resolved Julia $JuliaVersion`: $ReportPath"
    }

    foreach ($key in $SourceFiles.Keys) {
        $sourcePath = [string]$SourceFiles[$key]
        Assert-Dp150FileNonEmpty -Path $sourcePath -Description "Provenance source $key"
        $expectedSha = Get-Dp150FileSha256Lower -Path $sourcePath
        $reportedSha = [string](Get-Dp150TomlScalar -Path $ReportPath -Key $key)
        if ($reportedSha -cne $expectedSha) {
            throw "Completed-run provenance mismatch for $key`: $ReportPath"
        }
    }

    foreach ($encoding in @('REF', 'X', 'Y', 'Z')) {
        if (-not $SequenceFiles.ContainsKey($encoding)) {
            throw "No current sequence path was supplied for $encoding"
        }
        $sequencePath = [string]$SequenceFiles[$encoding]
        Assert-Dp150FileNonEmpty -Path $sequencePath -Description "$encoding sequence"
        $section = "sequence.$encoding"
        $reportedPath = [string](Get-Dp150TomlScalar -Path $ReportPath -Section $section -Key 'file')
        if (-not $reportedPath -or -not (Test-Dp150SamePath -Left $reportedPath -Right $sequencePath)) {
            throw "Completed-run sequence path mismatch for $encoding`: $ReportPath"
        }
        $expectedSha = Get-Dp150FileSha256Lower -Path $sequencePath
        $reportedSha = [string](Get-Dp150TomlScalar -Path $ReportPath -Section $section -Key 'sha256')
        if ($reportedSha -cne $expectedSha) {
            throw "Completed-run sequence SHA-256 mismatch for $encoding`: $ReportPath"
        }

        $resultPath = Join-Path $RunDirectory "$(($encoding).ToLowerInvariant())_result.jld2"
        Assert-Dp150FileNonEmpty -Path $resultPath -Description "$encoding result"
        $artifactSection = "result_artifacts.$encoding"
        $reportedResultPath = [string](Get-Dp150TomlScalar `
            -Path $ReportPath `
            -Section $artifactSection `
            -Key 'file')
        if (
            -not $reportedResultPath -or
            -not (Test-Dp150SamePath -Left $reportedResultPath -Right $resultPath)
        ) {
            throw "Completed-run result path mismatch for $encoding`: $ReportPath"
        }
        $actualResultBytes = (Get-Item -LiteralPath $resultPath).Length
        $reportedResultBytes = [string](Get-Dp150TomlScalar `
            -Path $ReportPath `
            -Section $artifactSection `
            -Key 'bytes')
        if ($reportedResultBytes -ne [string]$actualResultBytes) {
            throw "Completed-run result byte-count mismatch for $encoding`: $ReportPath"
        }
        $actualResultSha = Get-Dp150FileSha256Lower -Path $resultPath
        $reportedResultSha = [string](Get-Dp150TomlScalar `
            -Path $ReportPath `
            -Section $artifactSection `
            -Key 'sha256')
        if ($reportedResultSha -cne $actualResultSha) {
            throw "Completed-run result SHA-256 mismatch for $encoding`: $ReportPath"
        }
    }

    $finalArtifacts = @(
        [pscustomobject]@{
            Name = 'reconstruction'
            Path = (Join-Path $RunDirectory 'pc_reconstruction.h5')
            BytesKey = 'reconstruction_bytes'
            ShaKey = 'reconstruction_sha256'
        },
        [pscustomobject]@{
            Name = 'QC figure'
            Path = (Join-Path $RunDirectory 'pc_reconstruction_qc.png')
            BytesKey = 'qc_figure_bytes'
            ShaKey = 'qc_figure_sha256'
        }
    )
    foreach ($artifact in $finalArtifacts) {
        Assert-Dp150FileNonEmpty -Path $artifact.Path -Description "Completed $($artifact.Name)"
        $actualBytes = (Get-Item -LiteralPath $artifact.Path).Length
        $reportedBytes = [string](Get-Dp150TomlScalar `
            -Path $ReportPath `
            -Key $artifact.BytesKey)
        if ($reportedBytes -ne [string]$actualBytes) {
            throw "Completed-run $($artifact.Name) byte-count mismatch: $ReportPath"
        }
        $actualSha = Get-Dp150FileSha256Lower -Path $artifact.Path
        $reportedSha = [string](Get-Dp150TomlScalar `
            -Path $ReportPath `
            -Key $artifact.ShaKey)
        if ($reportedSha -cne $actualSha) {
            throw "Completed-run $($artifact.Name) SHA-256 mismatch: $ReportPath"
        }
    }
}

function Get-Dp150CompletedRunState {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ReportPath,

        [Parameter(Mandatory = $true)]
        [string]$RunDirectory,

        [Parameter(Mandatory = $true)]
        [string]$RunName,

        [Parameter(Mandatory = $true)]
        [long]$ParticleCount,

        [Parameter(Mandatory = $true)]
        [hashtable]$SourceFiles,

        [Parameter(Mandatory = $true)]
        [hashtable]$SequenceFiles,

        [Parameter(Mandatory = $true)]
        [version]$JuliaVersion
    )

    if (-not (Test-Path -LiteralPath $ReportPath -PathType Leaf)) {
        foreach ($finalArtifact in @(
            (Join-Path $RunDirectory 'pc_reconstruction.h5'),
            (Join-Path $RunDirectory 'pc_reconstruction_qc.png')
        )) {
            if (Test-Path -LiteralPath $finalArtifact -PathType Leaf) {
                throw "Final artifact exists without its simulation report; refusing overwrite: $finalArtifact"
            }
        }
        return [pscustomobject]@{
            State = 'NotComplete'
            ReportPath = $ReportPath
        }
    }

    $status = [string](Get-Dp150TomlScalar -Path $ReportPath -Key 'status')
    if ($status -ne 'PASS') {
        foreach ($finalArtifact in @(
            (Join-Path $RunDirectory 'pc_reconstruction.h5'),
            (Join-Path $RunDirectory 'pc_reconstruction_qc.png')
        )) {
            if (Test-Path -LiteralPath $finalArtifact -PathType Leaf) {
                throw "Final artifact exists with a non-PASS report; refusing overwrite: $finalArtifact"
            }
        }
        return [pscustomobject]@{
            State = 'NotComplete'
            ReportPath = $ReportPath
        }
    }

    Assert-Dp150SimulationReportCore `
        -Path $ReportPath `
        -RunName $RunName `
        -ParticleCount $ParticleCount `
        -Encodings @('REF', 'X', 'Y', 'Z')
    foreach ($encoding in @('ref', 'x', 'y', 'z')) {
        Assert-Dp150FileNonEmpty `
            -Path (Join-Path $RunDirectory "${encoding}_result.jld2") `
            -Description "Completed $($encoding.ToUpperInvariant()) result"
    }
    Assert-Dp150FileNonEmpty `
        -Path (Join-Path $RunDirectory 'pc_reconstruction.h5') `
        -Description 'Completed reconstruction'
    Assert-Dp150FileNonEmpty `
        -Path (Join-Path $RunDirectory 'pc_reconstruction_qc.png') `
        -Description 'Completed reconstruction QC figure'

    $schema = Get-Dp150TomlScalar -Path $ReportPath -Key 'provenance_schema_version'
    if ($null -eq $schema) {
        return [pscustomobject]@{
            State = 'LegacyComplete'
            ReportPath = $ReportPath
        }
    }

    Assert-Dp150ProvenanceBoundCompletion `
        -ReportPath $ReportPath `
        -RunDirectory $RunDirectory `
        -SourceFiles $SourceFiles `
        -SequenceFiles $SequenceFiles `
        -JuliaVersion $JuliaVersion
    return [pscustomobject]@{
        State = 'ProvenanceBoundComplete'
        ReportPath = $ReportPath
    }
}

function Invoke-Dp150LegacyCompletedRunAudit {
    param(
        [Parameter(Mandatory = $true)]
        [string]$JuliaPath,

        [Parameter(Mandatory = $true)]
        [string]$ProjectDirectory,

        [Parameter(Mandatory = $true)]
        [string]$AuditScript,

        [Parameter(Mandatory = $true)]
        [string]$RunDirectory,

        [Parameter(Mandatory = $true)]
        [long]$ParticleCount
    )

    Assert-Dp150FileNonEmpty -Path $AuditScript -Description 'Read-only legacy audit script'
    $auditOutput = @(
        & $JuliaPath `
            '--startup-file=no' `
            '--compiled-modules=no' `
            '--pkgimages=no' `
            "--project=$ProjectDirectory" `
            $AuditScript `
            $RunDirectory `
            ([string]$ParticleCount)
    )
    if ($LASTEXITCODE -ne 0) {
        throw "Read-only legacy completion audit exited with code $LASTEXITCODE."
    }
    $passLines = @($auditOutput | Where-Object {
        [string]$_ -match '^PASS_DP150_LEGACY_COMPLETED_RUN_READ_ONLY\b'
    })
    if ($passLines.Count -ne 1) {
        throw 'Read-only legacy completion audit did not emit its unique PASS marker.'
    }
    $auditOutput | ForEach-Object { Write-Output $_ }
}

function Open-Dp150ExclusiveLocks {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Paths,

        [string]$ConflictDescription = 'Another dp150 process'
    )

    $streams = @()
    try {
        foreach ($path in @(
            $Paths |
                ForEach-Object { [System.IO.Path]::GetFullPath($_) } |
                Sort-Object -Unique
        )) {
            try {
                $streams += [System.IO.File]::Open(
                    $path,
                    [System.IO.FileMode]::OpenOrCreate,
                    [System.IO.FileAccess]::ReadWrite,
                    [System.IO.FileShare]::None
                )
            }
            catch {
                throw "$ConflictDescription already holds $path"
            }
        }
        return $streams
    }
    catch {
        Close-Dp150Locks -Streams $streams
        throw
    }
}

function Close-Dp150Locks {
    param(
        [AllowEmptyCollection()]
        [object[]]$Streams = @()
    )

    foreach ($stream in $Streams) {
        if ($null -ne $stream) {
            $stream.Dispose()
        }
    }
}

function Get-Dp150EncodingLockPaths {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RunDirectory,

        [Parameter(Mandatory = $true)]
        [string[]]$Encodings
    )

    return @($Encodings | ForEach-Object {
        Join-Path $RunDirectory "worker_$($_.ToLowerInvariant()).lock"
    })
}

function ConvertTo-Dp150EncodingList {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Value
    )

    $encodings = @($Value.Split(',') | ForEach-Object { $_.Trim().ToUpperInvariant() })
    if ($encodings.Count -eq 0 -or @($encodings | Where-Object { -not $_ }).Count -gt 0) {
        throw 'Encodings must contain one or more comma-separated values.'
    }
    $invalid = @($encodings | Where-Object { $_ -notin @('REF', 'X', 'Y', 'Z') })
    if ($invalid.Count -gt 0) {
        throw "Unsupported encodings: $($invalid -join ', ')"
    }
    if (@($encodings | Select-Object -Unique).Count -ne $encodings.Count) {
        throw "Encodings contain duplicates: $Value"
    }
    return $encodings
}

function Assert-Dp150SafeReportName {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ReportName
    )

    if (
        [System.IO.Path]::GetFileName($ReportName) -cne $ReportName -or
        $ReportName -cnotmatch '^simulation_report_[A-Za-z0-9_-]+\.toml$'
    ) {
        throw "ReportName must be a safe simulation_report_*.toml file name: $ReportName"
    }
}

function Assert-Dp150SafeRunName {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RunName,

        [Parameter(Mandatory = $true)]
        [string]$RequiredPrefix
    )

    if (
        [System.IO.Path]::GetFileName($RunName) -cne $RunName -or
        $RunName -cnotmatch '^[A-Za-z0-9][A-Za-z0-9_-]*$' -or
        -not (
            $RunName -ceq $RequiredPrefix -or
            $RunName.StartsWith("${RequiredPrefix}_", [System.StringComparison]::Ordinal)
        )
    ) {
        throw (
            "RunName must be a path-safe name beginning with " +
            "'$RequiredPrefix': $RunName"
        )
    }
}

function Get-Dp150GpuSnapshot {
    $rows = @(& nvidia-smi `
        '--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu' `
        '--format=csv,noheader,nounits')
    if ($LASTEXITCODE -ne 0) {
        throw "nvidia-smi failed with exit code $LASTEXITCODE."
    }

    return @($rows | ForEach-Object {
        $columns = @($_ -split ',')
        if ($columns.Count -ne 6) {
            throw "Unexpected nvidia-smi row: $_"
        }
        [pscustomobject]@{
            index = [int]$columns[0].Trim()
            name = $columns[1].Trim()
            total_mib = [int]$columns[2].Trim()
            used_mib = [int]$columns[3].Trim()
            free_mib = [int]$columns[4].Trim()
            utilization_percent = [int]$columns[5].Trim()
        }
    })
}

function Test-Dp150GpuIdle {
    param(
        [Parameter(Mandatory = $true)]
        [object]$Gpu,

        [Parameter(Mandatory = $true)]
        [int]$MinimumFreeMiB,

        [Parameter(Mandatory = $true)]
        [int]$MaximumUtilizationPercent
    )

    return (
        $Gpu.free_mib -ge $MinimumFreeMiB -and
        $Gpu.utilization_percent -le $MaximumUtilizationPercent
    )
}

function Test-Dp150GpusIdle {
    param(
        [Parameter(Mandatory = $true)]
        [object[]]$Snapshot,

        [Parameter(Mandatory = $true)]
        [int[]]$PhysicalGpuIndices,

        [Parameter(Mandatory = $true)]
        [int]$MinimumFreeMiB,

        [Parameter(Mandatory = $true)]
        [int]$MaximumUtilizationPercent
    )

    foreach ($index in $PhysicalGpuIndices) {
        $matches = @($Snapshot | Where-Object { $_.index -eq $index })
        if (
            $matches.Count -ne 1 -or
            -not (Test-Dp150GpuIdle `
                -Gpu $matches[0] `
                -MinimumFreeMiB $MinimumFreeMiB `
                -MaximumUtilizationPercent $MaximumUtilizationPercent)
        ) {
            return $false
        }
    }
    return $true
}

function Get-Dp150FreeDiskGB {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    $root = [System.IO.Path]::GetPathRoot([System.IO.Path]::GetFullPath($Path))
    $drive = [System.IO.DriveInfo]::new($root)
    return [double]($drive.AvailableFreeSpace / 1GB)
}

function Assert-Dp150LauncherParameters {
    param(
        [Parameter(Mandatory = $true)]
        [int]$MinimumFreeMiB,

        [Parameter(Mandatory = $true)]
        [int]$MaximumUtilizationPercent,

        [Parameter(Mandatory = $true)]
        [int]$StableProbeCount,

        [Parameter(Mandatory = $true)]
        [int]$ProbeIntervalSeconds,

        [Parameter(Mandatory = $true)]
        [int]$RetryIntervalSeconds,

        [Parameter(Mandatory = $true)]
        [double]$MinimumFreeDiskGB
    )

    if ($MinimumFreeMiB -lt 1) {
        throw 'MinimumFreeMiB must be positive.'
    }
    if ($MaximumUtilizationPercent -lt 0 -or $MaximumUtilizationPercent -gt 100) {
        throw 'MaximumUtilizationPercent must be between 0 and 100.'
    }
    if ($StableProbeCount -lt 1) {
        throw 'StableProbeCount must be positive.'
    }
    if ($ProbeIntervalSeconds -lt 1 -or $RetryIntervalSeconds -lt 1) {
        throw 'Probe and retry intervals must be positive.'
    }
    if ($MinimumFreeDiskGB -le 0) {
        throw 'MinimumFreeDiskGB must be positive.'
    }
}

function Get-Dp150PowerShellExecutable {
    $executable = if ($PSVersionTable.PSEdition -eq 'Core') {
        Join-Path $PSHOME 'pwsh.exe'
    }
    else {
        Join-Path $PSHOME 'powershell.exe'
    }
    Assert-Dp150FileNonEmpty -Path $executable -Description 'PowerShell executable'
    return $executable
}
