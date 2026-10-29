. 'D:/MRSim/outputs/scanner_geometry_e060/code/e060_env.ps1'
# S2 prefix (rotation invariance, step ladder, estimate), then S3 acquisition.
# Requires S0 sequences, S0b slab fold-in gate and S1 phantom rotation to have passed.
$taskS0=Get-Content -LiteralPath ($taskRoot+'/sequences_v1/report.toml') -Raw
if($taskS0 -notmatch 'status = "PASS_E060_SEQUENCES_AND_AUDITS"'){throw 'S0 sequences missing'}
$taskSlab=Get-Content -LiteralPath ($taskRoot+'/slab_profile_v1/report.json') -Raw
if($taskSlab -notmatch '"status": "PASS_E060_SLAB_FOLD_IN_GATE"'){throw 'Slab fold-in gate missing'}
$taskS1=Get-Content -LiteralPath ($taskRoot+'/phantom_v1/report.json') -Raw
if($taskS1 -notmatch '"status": "PASS_E060_PHANTOM_ROTATION"'){throw 'S1 phantom rotation missing'}
if([IO.DriveInfo]::new('D:\').AvailableFreeSpace -lt 30000000000){throw 'Insufficient D storage'}
Assert-IdleGpu1
Set-Location -LiteralPath 'D:/MRSim'
$taskRun=$taskJuliaArgs+@(($taskCode+'/run_e060.jl'))
Logged $taskJulia ($taskRun+@('prefix',($taskRoot+'/prefix_v1'))) 'prefix.log'
$taskPrefix=Get-Content -LiteralPath ($taskRoot+'/prefix_v1/report.toml') -Raw
if($taskPrefix -notmatch 'status = "PASS_E060_PREFIX"'){
    Write-Output 'STOP_E060_AFTER_PREFIX_REVIEW_REQUIRED'; exit 0
}
Assert-IdleGpu1
Logged $taskJulia ($taskRun+@('run',($taskRoot+'/acquisition_v1'))) 'acquisition.log'
Write-Output 'COMPLETE_E060_S3_ACQUISITION'
