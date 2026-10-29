# Process-local E-061 runtime, identical to the E056-E058 orchestration environment.
$ErrorActionPreference='Stop'
$taskRoot='D:/MRSim/outputs/venc_unwrap_e061'
$taskRepo='D:/MRSim/MRSIM/MRSimTracks'
$taskCode=$taskRoot+'/code'
$env:PYTHONPATH=$taskRepo+';'+$taskRepo+'/.venv/Lib/site-packages'
$env:PYTHONDONTWRITEBYTECODE='1';$env:OPENBLAS_NUM_THREADS='1'
$env:JULIA_DEPOT_PATH='D:/MRSim/temp/julia_e056;'+$taskRepo+'/outputs/upstream_migration_e046/depot;C:/Users/lambda-3/.julia'
$env:JULIA_LOAD_PATH=$taskRepo+'/koma_migration_e046;@stdlib'
$env:JULIA_PKG_OFFLINE='true';$env:JULIA_NUM_THREADS='1'
$env:CUDA_VISIBLE_DEVICES='GPU-8be27592-e325-7b7a-f432-b84f69cb18c6'
$env:CUDA_DEVICE_ORDER='PCI_BUS_ID'
$env:TEMP='D:/MRSim/temp';$env:TMP=$env:TEMP
$taskPython='C:/Users/lambda-3/AppData/Local/Programs/Python/Python312/python.exe'
$taskJulia='C:/Users/lambda-3/AppData/Local/Programs/Julia-1.12.7/bin/julia.exe'
$taskJuliaArgs=@('--startup-file=no','--compiled-modules=existing',('--project='+$taskRepo+'/koma_migration_e046'))
function Logged([string]$exe,[string[]]$arguments,[string]$log) {
    if(Test-Path -LiteralPath ($taskRoot+'/'+$log)){throw ('Refuse existing log '+$log)}
    $old=$ErrorActionPreference;$ErrorActionPreference='Continue'
    & $exe @arguments *> ($taskRoot+'/'+$log)
    $taskExit=$LASTEXITCODE;$ErrorActionPreference=$old
    if($taskExit -ne 0){throw ('E061 stage failed '+$log+' exit '+$taskExit)}
    Write-Output ('COMPLETE_STAGE '+$log)
}
function Assert-IdleGpu1 {
    if([IO.DriveInfo]::new('D:\').AvailableFreeSpace -lt 10000000000){throw 'Insufficient D storage'}
    $row=(& nvidia-smi --id=1 --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits).Split(',').Trim()
    if($row.Count -ne 3 -or $row[0] -cne $env:CUDA_VISIBLE_DEVICES -or [int]$row[1] -gt 256 -or [int]$row[2] -gt 5){throw 'Reviewed GPU1 is not idle'}
}
