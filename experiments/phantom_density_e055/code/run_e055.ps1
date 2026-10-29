$ErrorActionPreference='Stop'
$repo='D:/MRSim/MRSIM/MRSimTracks'
$run='D:/MRSim/outputs/phantom_density_e055'
$code=Join-Path $run 'code'
$env:PYTHONPATH=(Join-Path $repo '.venv/Lib/site-packages')+';'+$repo
$env:PYTHONDONTWRITEBYTECODE='1';$env:OPENBLAS_NUM_THREADS='1'
$env:TEMP='D:/MRSim/temp';$env:TMP=$env:TEMP
$env:NUMBA_CACHE_DIR='D:/MRSim/temp/numba_e055';$env:MPLCONFIGDIR='D:/MRSim/temp/matplotlib_e055'
$env:JULIA_DEPOT_PATH='D:/MRSim/temp/julia_e055;'+(Join-Path $repo 'outputs/upstream_migration_e046/depot')+';C:/Users/lambda-3/.julia'
$env:JULIA_LOAD_PATH=(Join-Path $repo 'koma_migration_e046')+';@stdlib'
$env:JULIA_PKG_OFFLINE='true';$env:JULIA_NUM_THREADS='1'
$env:CUDA_VISIBLE_DEVICES='GPU-8be27592-e325-7b7a-f432-b84f69cb18c6'
$env:CUDA_DEVICE_ORDER='PCI_BUS_ID'
$python='C:/Users/lambda-3/AppData/Local/Programs/Python/Python312/python.exe'
$julia='C:/Users/lambda-3/AppData/Local/Programs/Julia-1.12.7/bin/julia.exe'
Set-Location -LiteralPath 'D:/MRSim'
function RequireIdleGPU {
    $row=(& nvidia-smi --id=1 --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits).Split(',').Trim()
    if($row.Count -ne 3 -or $row[0] -cne $env:CUDA_VISIBLE_DEVICES -or [int]$row[1] -gt 256 -or [int]$row[2] -gt 5){throw 'Reviewed GPU1 is not idle'}
}
function Logged([string]$exe,[string[]]$arguments,[string]$log) {
    $old=$ErrorActionPreference;$ErrorActionPreference='Continue'
    & $exe @arguments *> (Join-Path $run $log)
    $exitCode=$LASTEXITCODE;$ErrorActionPreference=$old
    if($exitCode -ne 0){throw ('E055 stage failed: '+$log+', exit '+$exitCode)}
    Write-Output ('COMPLETE_STAGE '+$log)
}
if([IO.DriveInfo]::new('D:\').AvailableFreeSpace -lt 30000000000){throw 'Insufficient D: storage'}
RequireIdleGPU
$argsCommon=@('--startup-file=no','--compiled-modules=existing',('--project='+(Join-Path $repo 'koma_migration_e046')),(Join-Path $code 'run_300k.jl'))
Logged $julia ($argsCommon+@('preflight',(Join-Path $run 'inputs_v1'),(Join-Path $run 'map_v1'),(Join-Path $run 'preflight_v1'))) 'preflight.log'
$pre=Get-Content -LiteralPath (Join-Path $run 'preflight_v1/report.toml') -Raw
if($pre -notmatch 'PASS_COUPLED300K_F64_RESOURCE_PREFIX'){throw 'Resource gate failed'}
RequireIdleGPU
Logged $julia ($argsCommon+@('run',(Join-Path $run 'inputs_v1'),(Join-Path $run 'map_v1'),(Join-Path $run 'mapped_f64_v1'))) 'acquisition.log'
Logged $python @((Join-Path $code 'analyze_300k.py'),'--repo',$repo,'--run',$run) 'analysis.log'
$report=Get-Content -LiteralPath (Join-Path $run 'analysis_v1/report.json') -Raw | ConvertFrom-Json
$receipt=@{experiment='E-055';status=$report.status;one_million_decision=$report.one_million_decision.status;
    utc=[DateTime]::UtcNow.ToString('o');new_1m_started=$false;report_sha256=(Get-FileHash -LiteralPath (Join-Path $run 'analysis_v1/report.json') -Algorithm SHA256).Hash.ToLowerInvariant()}
$receipt|ConvertTo-Json|Set-Content -LiteralPath (Join-Path $run 'workflow_complete.json') -Encoding UTF8
$receipt|ConvertTo-Json
