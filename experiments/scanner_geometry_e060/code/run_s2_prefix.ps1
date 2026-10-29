. 'D:/MRSim/outputs/scanner_geometry_e060/code/e060_env.ps1'
# Amendment A2 S2: prefix (rotation invariance, step ladder, cross-GPU gate, estimate)
# with both reviewed GPUs visible (GPU1 first). Supersedes the unrun run_s2_s3.ps1.
$taskGpu0='GPU-3da2b149-c022-07f4-7fcb-7112684b5ac9'
$taskS0=Get-Content -LiteralPath ($taskRoot+'/sequences_v2/report.toml') -Raw
if($taskS0 -notmatch 'status = "PASS_E060_SEQUENCES_AND_AUDITS"'){throw 'S0 v2 sequences missing'}
$taskSlab=Get-Content -LiteralPath ($taskRoot+'/slab_profile_v2/report.json') -Raw
if($taskSlab -notmatch '"status": "PASS_E060_SLAB_FOLD_IN_GATE"'){throw 'Slab v2 fold-in gate missing'}
$taskS1=Get-Content -LiteralPath ($taskRoot+'/phantom_v1/report.json') -Raw
if($taskS1 -notmatch '"status": "PASS_E060_PHANTOM_ROTATION"'){throw 'S1 phantom rotation missing'}
$taskSeam=Get-Content -LiteralPath ($taskRoot+'/seam_v1/report.toml') -Raw
if($taskSeam -notmatch 'status = "PASS_E060_SEAM_ALIGNMENT"'){throw 'A3 seam alignment missing'}
Assert-IdleGpu1
$taskRow=(& nvidia-smi --id=0 --query-gpu=uuid,memory.used --format=csv,noheader,nounits).Split(',').Trim()
if($taskRow[0] -cne $taskGpu0 -or [int]$taskRow[1] -gt 16000){throw 'GPU0 not as reviewed or too loaded'}
$env:CUDA_VISIBLE_DEVICES=$env:CUDA_VISIBLE_DEVICES+','+$taskGpu0
Set-Location -LiteralPath 'D:/MRSim'
Logged $taskJulia ($taskJuliaArgs+@(($taskCode+'/run_e060.jl'),'prefix',($taskRoot+'/prefix_v1'))) 'prefix.log'
$taskPrefix=Get-Content -LiteralPath ($taskRoot+'/prefix_v1/report.toml') -Raw
if($taskPrefix -notmatch 'status = "PASS_E060_PREFIX"'){ Write-Output 'STOP_E060_AFTER_PREFIX_REVIEW_REQUIRED'; exit 0 }
Write-Output 'PASS_E060_PREFIX'
