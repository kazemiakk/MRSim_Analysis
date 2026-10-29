$ErrorActionPreference = 'Stop'
$repo = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
Set-Location -LiteralPath $repo
$base = Join-Path $repo 'outputs/phantom_precision_e050'
$workflow = Join-Path $base 'workflow_v1'
if (Test-Path -LiteralPath $workflow) { throw 'Refuse duplicate workflow; inspect the existing registration and checkpoints.' }
New-Item -ItemType Directory -Path $workflow | Out-Null

function Save-Stage([string]$name, $record) {
    $destination = Join-Path $workflow ($name + '.json')
    if (Test-Path -LiteralPath $destination) { throw ('Refuse stage overwrite: ' + $destination) }
    $record | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath ($destination + '.partial') -Encoding UTF8
    Move-Item -LiteralPath ($destination + '.partial') -Destination $destination
}

$stage = 'resource_gate'
try {
    $freeBytes = [System.IO.DriveInfo]::new('C:/').AvailableFreeSpace
    if ($freeBytes -lt 6GB) { throw 'Less than 6 GiB free for the new candidate, analysis and safety margin.' }
    $gpuUuid = 'GPU-8be27592-e325-7b7a-f432-b84f69cb18c6'
    $gpuStatus = & nvidia-smi --id=$gpuUuid --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits
    if ($LASTEXITCODE -ne 0) { throw 'GPU inspection failed.' }
    $gpuValues = $gpuStatus.Trim().Split(',')
    if ([int]$gpuValues[0].Trim() -gt 256 -or [int]$gpuValues[1].Trim() -gt 5) { throw 'Reviewed GPU1 is not idle; do not displace another workload.' }
    $files = @('koma_migration_e046/precision_confirmation_e050.jl', 'koma_migration_e046/run_precision_e050.ps1',
               'scripts/analyze_phantom_precision.py', 'scripts/validate_phantom_precision.py',
               'scripts/analyze_remapped_confirmation.py', 'scripts/compare_dp150_hemodynamics.py',
               'outputs/phantom_precision_e050/PROTOCOL.md')
    $hashes = @{}
    foreach ($file in $files) { $hashes[$file] = (Get-FileHash -LiteralPath (Join-Path $repo $file) -Algorithm SHA256).Hash.ToLowerInvariant() }
    Save-Stage 'registration' @{ status='REGISTERED_BEFORE_FULL_SIGNALS'; started_utc=[DateTime]::UtcNow.ToString('o');
        pid=$PID; gpu_uuid=$gpuUuid; free_disk_bytes=$freeBytes; source_sha256=$hashes; new_1m=$false }
    $env:JULIA_DEPOT_PATH=(Join-Path $repo 'outputs/upstream_migration_e046/depot')+';C:/Users/lambda-3/.julia'
    $env:JULIA_LOAD_PATH=(Join-Path $repo 'koma_migration_e046')+';@stdlib'
    $env:JULIA_PKG_OFFLINE='true'
    $env:JULIA_NUM_THREADS='2'
    $env:CUDA_VISIBLE_DEVICES=$gpuUuid
    $stage = 'acquisition'
    Save-Stage 'acquisition_started' @{ status='ACQUISITION_STARTED'; started_utc=[DateTime]::UtcNow.ToString('o') }
    & 'C:/Users/lambda-3/AppData/Local/Programs/Julia-1.12.7/bin/julia.exe' --startup-file=no `
        --project="$repo/koma_migration_e046" koma_migration_e046/precision_confirmation_e050.jl run `
        outputs/remapped_confirmation_e047/phantom_inputs_v1 outputs/remapped_confirmation_e047/phantom_map_v1 `
        outputs/phantom_precision_e050/mapped_f64_v1 `
        1> (Join-Path $workflow 'acquisition.stdout.log') 2> (Join-Path $workflow 'acquisition.stderr.log')
    if ($LASTEXITCODE -ne 0) { throw ('Acquisition exit code ' + $LASTEXITCODE) }
    Save-Stage 'acquisition_complete' @{ status='ACQUISITION_COMPLETE_ANALYSIS_PENDING'; finished_utc=[DateTime]::UtcNow.ToString('o') }
    foreach ($file in $files) {
        if ((Get-FileHash -LiteralPath (Join-Path $repo $file) -Algorithm SHA256).Hash.ToLowerInvariant() -ne $hashes[$file]) {
            throw ('Registered source changed during acquisition: ' + $file)
        }
    }
    $env:PYTHONPATH=(Join-Path $repo '.venv/Lib/site-packages')
    $python = 'C:/Users/lambda-3/AppData/Local/Programs/Python/Python312/python.exe'
    $stage = 'analysis'
    & $python -m scripts.analyze_phantom_precision --candidate outputs/phantom_precision_e050/mapped_f64_v1 `
        --output outputs/phantom_precision_e050/analysis_v1 `
        1> (Join-Path $workflow 'analysis.stdout.log') 2> (Join-Path $workflow 'analysis.stderr.log')
    if ($LASTEXITCODE -ne 0) { throw ('Analysis exit code ' + $LASTEXITCODE) }
    $stage = 'independent_validation'
    & $python -m scripts.validate_phantom_precision --analysis outputs/phantom_precision_e050/analysis_v1 `
        1> (Join-Path $workflow 'validation.stdout.log') 2> (Join-Path $workflow 'validation.stderr.log')
    if ($LASTEXITCODE -ne 0) { throw ('Independent validation exit code ' + $LASTEXITCODE) }
    Save-Stage 'complete' @{ status='COMPUTATION_AND_METRIC_CHECKS_COMPLETE_SCIENTIFIC_REVIEW_PENDING';
        finished_utc=[DateTime]::UtcNow.ToString('o'); new_1m=$false; accuracy_promoted=$false }
}
catch {
    Save-Stage 'failure' @{ status='STOPPED_WITH_PRESERVED_OUTPUTS'; stage=$stage; message=$_.Exception.Message;
        finished_utc=[DateTime]::UtcNow.ToString('o'); new_1m=$false }
    exit 1
}
