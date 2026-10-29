. 'D:/MRSim/outputs/step_convergence_e057/code/e057_env.ps1'
Assert-IdleGpu1
$taskGate=Get-Content -LiteralPath ($taskRoot+'/ladder_analysis_v1/report.json') -Raw -Encoding UTF8 | ConvertFrom-Json
if($taskGate.status -cne 'PASS_E057_STAGE1_STEP_CONVERGENCE'){throw 'Stage 2 requires a passing Stage 1 analysis'}
Set-Location -LiteralPath 'D:/MRSim'
Logged $taskJulia ($taskJuliaArgs+@(($taskCode+'/run_ladder.jl'),'stage2',($taskRoot+'/suite_v1'))) 'suite_stage2.log'
Logged $taskPython @(($taskCode+'/analyze_suite.py'),$taskRepo,'D:/MRSim/outputs/acquisition_physics_e056',($taskRoot+'/suite_analysis_v1')) 'suite_analysis.log'
Write-Output 'COMPLETE_E057_STAGE2_WORKFLOW_REVIEW_GATES_BEFORE_CLAIMS'
