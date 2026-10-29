$ErrorActionPreference='Stop'
$repo='D:/MRSim/MRSIM/MRSimTracks'
$run='D:/MRSim/outputs/stenosis_controls_e054'
$code=Join-Path $run 'code'
$env:PYTHONPATH=(Join-Path $repo '.venv/Lib/site-packages')+';'+$repo
$env:PYTHONDONTWRITEBYTECODE='1'
$env:TEMP='D:/MRSim/temp';$env:TMP=$env:TEMP
$env:NUMBA_CACHE_DIR='D:/MRSim/temp/numba_e054'
$env:MPLCONFIGDIR='D:/MRSim/temp/matplotlib_e054'
$env:JULIA_DEPOT_PATH='D:/MRSim/temp/julia_e054;'+(Join-Path $repo 'outputs/upstream_migration_e046/depot')+';C:/Users/lambda-3/.julia'
$env:JULIA_LOAD_PATH=(Join-Path $repo 'koma_migration_e046')+';@stdlib'
$env:JULIA_PKG_OFFLINE='true';$env:JULIA_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$python='C:/Users/lambda-3/AppData/Local/Programs/Python/Python312/python.exe'
$julia='C:/Users/lambda-3/AppData/Local/Programs/Julia-1.12.7/bin/julia.exe'
Set-Location -LiteralPath 'D:/MRSim'
function Logged([string]$exe,[string[]]$arguments,[string]$log) {
    $old=$ErrorActionPreference;$ErrorActionPreference='Continue'
    & $exe @arguments *> (Join-Path $run $log)
    $exitCode=$LASTEXITCODE;$ErrorActionPreference=$old
    if($exitCode -ne 0){throw ('Control stage failed: '+$log+', exit '+$exitCode)}
    Write-Output ('COMPLETE_STAGE '+$log)
}
Logged $python @((Join-Path $code 'prepare_controls.py'),'--repo',$repo,'--code',$code,'--output',(Join-Path $run 'inputs_v1')) 'prepare.log'
$registration=Get-Content -LiteralPath (Join-Path $run 'inputs_v1/registration.json') -Raw | ConvertFrom-Json
Logged $julia @('--startup-file=no','--compiled-modules=existing',('--project='+(Join-Path $repo 'koma_migration_e046')),
    (Join-Path $code 'run_controls.jl'),$repo,(Join-Path $run 'inputs_v1'),$registration.sequence_directory,(Join-Path $run 'cpu_v1')) 'cpu.log'
Logged $python @((Join-Path $code 'analyze_controls.py'),'--inputs',(Join-Path $run 'inputs_v1'),'--cpu',(Join-Path $run 'cpu_v1'),
    '--output',(Join-Path $run 'analysis_v1')) 'analysis.log'
$report=Get-Content -LiteralPath (Join-Path $run 'analysis_v1/report.json') -Raw | ConvertFrom-Json
$receipt=@{experiment='E-054';status=$report.status;utc=[DateTime]::UtcNow.ToString('o');new_300k_started=$false;new_1m_started=$false;
    report_sha256=(Get-FileHash -LiteralPath (Join-Path $run 'analysis_v1/report.json') -Algorithm SHA256).Hash.ToLowerInvariant()}
$receipt|ConvertTo-Json|Set-Content -LiteralPath (Join-Path $run 'workflow_complete.json') -Encoding UTF8
$receipt|ConvertTo-Json
