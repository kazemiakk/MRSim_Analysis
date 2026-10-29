. 'D:/MRSim/outputs/step_convergence_e057/code/e057_env.ps1'
Assert-IdleGpu1
if(!(Test-Path -LiteralPath ($taskRoot+'/grid_preflight_v1/report.toml'))){throw 'Stage 0 grid certification missing'}
Set-Location -LiteralPath 'D:/MRSim'
Logged $taskJulia ($taskJuliaArgs+@(($taskCode+'/run_ladder.jl'),'stage1',($taskRoot+'/ladder_v1'))) 'ladder_stage1.log'
Logged $taskPython @(($taskCode+'/analyze_ladder.py')) 'ladder_analysis.log'
Write-Output 'COMPLETE_E057_STAGE1_WORKFLOW_REVIEW_GATES_BEFORE_CLAIMS'
