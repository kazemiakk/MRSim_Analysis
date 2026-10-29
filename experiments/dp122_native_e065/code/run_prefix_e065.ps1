. 'D:/MRSim/outputs/dp122_native_e065/code/e065_env.ps1'
# E-065 prefix (step ladder, cross-GPU gate, estimate) with both reviewed GPUs visible (GPU1 first), as E-060 run_s2_prefix.ps1.
$taskGpu0='GPU-3da2b149-c022-07f4-7fcb-7112684b5ac9'
$taskS0=Get-Content -LiteralPath ($taskRoot+'/sequences_v1/report.toml') -Raw
if($taskS0 -notmatch 'status = "PASS_E065_SEQUENCES_AND_AUDITS"'){throw 'E-065 sequences missing'}
$taskSeam=Get-Content -LiteralPath ($taskRoot+'/seam_v1/report.toml') -Raw
if($taskSeam -notmatch 'status = "PASS_E065_SEAM_ALIGNMENT"'){throw 'E-065 seam alignment missing'}
Assert-IdleGpu1
$taskRow=(& nvidia-smi --id=0 --query-gpu=uuid,memory.used --format=csv,noheader,nounits).Split(',').Trim()
if($taskRow[0] -cne $taskGpu0 -or [int]$taskRow[1] -gt 16000){throw 'GPU0 not as reviewed or too loaded'}
$env:CUDA_VISIBLE_DEVICES=$env:CUDA_VISIBLE_DEVICES+','+$taskGpu0
Set-Location -LiteralPath 'D:/MRSim'
Logged $taskJulia ($taskJuliaArgs+@(($taskCode+'/run_e065.jl'),'prefix',($taskRoot+'/prefix_v1'))) 'prefix.log'
$taskPrefix=Get-Content -LiteralPath ($taskRoot+'/prefix_v1/report.toml') -Raw
if($taskPrefix -notmatch 'status = "PASS_E065_PREFIX"'){ Write-Output 'STOP_E065_AFTER_PREFIX_REVIEW_REQUIRED'; exit 0 }
Write-Output 'PASS_E065_PREFIX'
