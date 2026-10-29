. 'D:/MRSim/outputs/paired_acquisition_e058/code/e058_env.ps1'
$taskS0=Get-Content -LiteralPath ($taskRoot+'/sequences_v1/report.toml') -Raw
if($taskS0 -notmatch 'PASS_E058_SEQUENCE_CERTIFICATION_AND_CANDIDATE_AUDIT'){throw 'S0 certification missing'}
$env:MPLCONFIGDIR='D:/MRSim/temp/matplotlib_e058';$env:NUMBA_CACHE_DIR='D:/MRSim/temp/numba_e058'
Set-Location -LiteralPath 'D:/MRSim/temp'
Logged $taskPython @(($taskCode+'/build_references.py')) 'build_references.log'
Write-Output 'COMPLETE_E058_S1_REFERENCES'
