. 'D:/MRSim/outputs/paired_acquisition_e058/code/e058_env.ps1'
# S2a: baseline-only numerical prefix; needs no S0 output (original real sequences).
if([IO.DriveInfo]::new('D:\').AvailableFreeSpace -lt 30000000000){throw 'Insufficient D storage'}
Assert-IdleGpu1
Set-Location -LiteralPath 'D:/MRSim'
Logged $taskJulia ($taskJuliaArgs+@(($taskCode+'/run_real.jl'),'prefix',($taskRoot+'/prefix_v1'))) 'prefix.log'
Write-Output 'COMPLETE_E058_S2A_BASELINE_PREFIX_REVIEW_REPORT'
