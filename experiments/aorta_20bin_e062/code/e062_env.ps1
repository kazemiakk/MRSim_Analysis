# Process-local E-062 runtime: the koma_aorta environment of the 8-bin production run (env/ is a byte copy).
$ErrorActionPreference='Stop'
$taskRoot='D:/MRSim/outputs/aorta_20bin_e062'
$taskRepo='D:/MRSim/MRSIM/MRSimTracks'
$taskCode=$taskRoot+'/code'
$env:PYTHONPATH=$taskRepo+';'+$taskRepo+'/.venv/Lib/site-packages'
$env:PYTHONDONTWRITEBYTECODE='1';$env:OPENBLAS_NUM_THREADS='1'
$env:JULIA_DEPOT_PATH='C:/Users/lambda-3/.julia'
Remove-Item Env:JULIA_LOAD_PATH -ErrorAction SilentlyContinue
$env:JULIA_PKG_OFFLINE='true'
$env:CUDA_DEVICE_ORDER='PCI_BUS_ID'
$env:TEMP='D:/MRSim/temp';$env:TMP=$env:TEMP
$taskPython='C:/Users/lambda-3/AppData/Local/Programs/Python/Python312/python.exe'
$taskJulia='C:/Users/lambda-3/AppData/Local/Programs/Julia-1.12.7/bin/julia.exe'
$taskJuliaArgs=@('--startup-file=no',('--project='+$taskRoot+'/env'))
function Logged([string]$exe,[string[]]$arguments,[string]$log) {
    if(Test-Path -LiteralPath ($taskRoot+'/'+$log)){throw ('Refuse existing log '+$log)}
    $old=$ErrorActionPreference;$ErrorActionPreference='Continue'
    & $exe @arguments *> ($taskRoot+'/'+$log)
    $taskExit=$LASTEXITCODE;$ErrorActionPreference=$old
    if($taskExit -ne 0){throw ('E062 stage failed '+$log+' exit '+$taskExit)}
    Write-Output ('COMPLETE_STAGE '+$log)
}
