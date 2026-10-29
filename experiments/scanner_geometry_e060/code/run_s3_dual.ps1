. 'D:/MRSim/outputs/scanner_geometry_e060/code/e060_env.ps1'
# Amendment A2 S3: REF+X on GPU1 and Y+Z on GPU0 in parallel, then finalize.
$taskGpu0='GPU-3da2b149-c022-07f4-7fcb-7112684b5ac9'
$taskGpu1=$env:CUDA_VISIBLE_DEVICES
$taskOut=$taskRoot+'/acquisition_v1'
$taskPrefix=Get-Content -LiteralPath ($taskRoot+'/prefix_v1/report.toml') -Raw
if($taskPrefix -notmatch 'status = "PASS_E060_PREFIX"'){throw 'Prefix not passed'}
if([IO.DriveInfo]::new('D:\').AvailableFreeSpace -lt 30000000000){throw 'Insufficient D storage'}
Assert-IdleGpu1
$taskRow=(& nvidia-smi --id=0 --query-gpu=uuid,memory.used --format=csv,noheader,nounits).Split(',').Trim()
if($taskRow[0] -cne $taskGpu0 -or [int]$taskRow[1] -gt 16000){throw 'GPU0 not as reviewed or too loaded'}
Set-Location -LiteralPath 'D:/MRSim'
$taskArgs=$taskJuliaArgs+@(($taskCode+'/run_e060.jl'),'run',$taskOut)
function Start-Arm([string]$gpu,[string]$encodings,[string]$name) {
    foreach($suffix in @('.log','.err.log')){ if(Test-Path -LiteralPath ($taskRoot+'/'+$name+$suffix)){throw ('Refuse existing log '+$name)} }
    $env:CUDA_VISIBLE_DEVICES=$gpu
    $p=Start-Process -FilePath $taskJulia -ArgumentList ($taskArgs+@($encodings)) -NoNewWindow -PassThru `
        -RedirectStandardOutput ($taskRoot+'/'+$name+'.log') -RedirectStandardError ($taskRoot+'/'+$name+'.err.log')
    $null=$p.Handle
    return $p
}
$taskA=Start-Arm $taskGpu1 'REF,X' 'acquisition_gpu1'
# The GPU1 process registers the output first; the GPU0 process must match that registration.
$taskWait=0
while(-not (Test-Path -LiteralPath ($taskOut+'/registration.toml'))){
    if($taskA.HasExited){throw 'GPU1 arm exited before registering'}
    Start-Sleep -Seconds 5; $taskWait+=5; if($taskWait -gt 1800){throw 'Registration timeout'}
}
$taskB=Start-Arm $taskGpu0 'Y,Z' 'acquisition_gpu0'
$taskA.WaitForExit(); $taskB.WaitForExit()
$env:CUDA_VISIBLE_DEVICES=$taskGpu1
if($taskA.ExitCode -ne 0 -or $taskB.ExitCode -ne 0){throw ('E060 acquisition arm failed: GPU1 '+$taskA.ExitCode+' GPU0 '+$taskB.ExitCode)}
Logged $taskJulia ($taskJuliaArgs+@(($taskCode+'/run_e060.jl'),'finalize',$taskOut)) 'finalize.log'
Write-Output 'COMPLETE_E060_S3_ACQUISITION'
