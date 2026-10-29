. 'D:/MRSim/outputs/paired_acquisition_e058/code/e058_env.ps1'
# S2b candidate prefix, then S3 acquisitions. Requires S0 PASS and S2a PASS.
$taskS0=Get-Content -LiteralPath ($taskRoot+'/sequences_v1/report.toml') -Raw
if($taskS0 -notmatch 'PASS_E058_SEQUENCE_CERTIFICATION_AND_CANDIDATE_AUDIT'){throw 'S0 certification missing'}
$taskPrefix=Get-Content -LiteralPath ($taskRoot+'/prefix_v1/report.toml') -Raw
if($taskPrefix -notmatch 'status = "PASS_E058_BASELINE_PREFIX_STEP_SELECTION"'){throw 'Baseline prefix not passed'}
if([IO.DriveInfo]::new('D:\').AvailableFreeSpace -lt 30000000000){throw 'Insufficient D storage'}
Assert-IdleGpu1
Set-Location -LiteralPath 'D:/MRSim'
$taskRun=$taskJuliaArgs+@(($taskCode+'/run_real.jl'))
Logged $taskJulia ($taskRun+@('prefix_candidate',($taskRoot+'/prefix_candidate_v1'))) 'prefix_candidate.log'
$taskCandidate=Get-Content -LiteralPath ($taskRoot+'/prefix_candidate_v1/report.toml') -Raw
if($taskCandidate -notmatch 'status = "PASS_E058_CANDIDATE_PREFIX"'){
    Write-Output 'STOP_E058_AFTER_CANDIDATE_PREFIX_REVIEW_REQUIRED'; exit 0
}
Assert-IdleGpu1
Logged $taskJulia ($taskRun+@('run','baseline',($taskRoot+'/acquisition_baseline_v1'))) 'acquisition_baseline.log'
Assert-IdleGpu1
Logged $taskJulia ($taskRun+@('run','short_echo',($taskRoot+'/acquisition_short_echo_v1'))) 'acquisition_short_echo.log'
Write-Output 'COMPLETE_E058_S3_ACQUISITIONS'
