# E-064 orchestrator: sequences (+ audit), rotated flow phantom, tissue phantom, then flow and tissue simulations
# with the frozen koma_aorta simulator (REF+X on GPU0, Y+Z on GPU1, provenance-checked combine), as E-062/E-063.
. 'D:/MRSim/outputs/aorta_aligned_e064/code/e064_env.ps1'
$taskGpu0='GPU-3da2b149-c022-07f4-7fcb-7112684b5ac9'
$taskGpu1='GPU-8be27592-e325-7b7a-f432-b84f69cb18c6'
$taskSim='D:/MRSim/MRSIM/MRSimTracks/koma_aorta/simulate_pc.jl'
if((Get-FileHash -LiteralPath $taskSim).Hash.ToLower() -ne 'a9cce28cf2107bafd986b8130e25eeaea5dfcb2a7ff49813b8d440d974ce4024'){throw 'Simulator changed'}
if([IO.DriveInfo]::new('D:\').AvailableFreeSpace -lt 40000000000){throw 'Insufficient D storage'}
function GpuRow([int]$i){ (& nvidia-smi --id=$i --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits).Split(',').Trim() }
function Assert-Gpus { $r0=GpuRow 0; $r1=GpuRow 1
    if($r0[0] -cne $taskGpu0 -or [int]$r0[1] -gt 16000 -or [int]$r0[2] -gt 40){throw 'GPU0 not as reviewed or too loaded'}
    if($r1[0] -cne $taskGpu1 -or [int]$r1[1] -gt 512 -or [int]$r1[2] -gt 5){throw 'GPU1 not idle'} }
function Start-Arm([string]$gpu,[string]$exe,[string[]]$arguments,[string]$name) {
    foreach($suffix in @('.log','.err.log')){ if(Test-Path -LiteralPath ($taskRoot+'/'+$name+$suffix)){throw ('Refuse existing log '+$name)} }
    $env:CUDA_VISIBLE_DEVICES=$gpu
    $p=Start-Process -FilePath $exe -ArgumentList $arguments -NoNewWindow -PassThru `
        -RedirectStandardOutput ($taskRoot+'/'+$name+'.log') -RedirectStandardError ($taskRoot+'/'+$name+'.err.log')
    $null=$p.Handle; return $p
}
Set-Location -LiteralPath 'D:/MRSim'
Logged $taskJulia ($taskJuliaArgs+@(($taskCode+'/generate_e064.jl'))) 'generate.log'
Set-Location -LiteralPath 'C:/Users/lambda-3/AppData/Local/Temp/claude'
Logged $taskPython @(($taskCode+'/audit_sequences_e064.py')) 'audit.log'
$audit=Get-Content -LiteralPath ($taskRoot+'/sequence_audit_v1/report.json') -Raw | ConvertFrom-Json
if($audit.status -ne 'PASS_E064_S0_G0D'){ Write-Output 'STOP_E064_AFTER_AUDIT'; exit 0 }
Set-Location -LiteralPath 'D:/MRSim'
Logged $taskJulia ($taskJuliaArgs+@(($taskCode+'/rotate_phantom_e064.jl'),($taskRoot+'/phantom_v1'))) 'rotate.log'
Logged $taskJulia ($taskJuliaArgs+@(($taskCode+'/build_tissue_phantom_e064.jl'),($taskRoot+'/tissue_phantom_v1'))) 'tissue_phantom.log'
$env:AORTA_SEQUENCE_DIR=$taskRoot+'/sequences_v1'; $env:AORTA_GPU_DEVICE='0'; $env:AORTA_BATCH_SIZE='0'
$env:AORTA_MAX_BLOCK_LENGTH='512'; $env:AORTA_ALLOW_UNVALIDATED_BLOCK_LENGTH='false'
function Run-Case([string]$phantom,[string]$run,[string]$prefix) {
    $env:AORTA_PHANTOM=$phantom; $env:AORTA_RUN_NAME=$run
    $parts=@()
    foreach($spec in @(@($taskGpu0,0,'REF,X',($prefix+'_part_gpu0')),@($taskGpu1,1,'Y,Z',($prefix+'_part_gpu1')))){
        $env:AORTA_PHYSICAL_GPU_INDEX=[string]$spec[1]; $env:AORTA_ENCODINGS=$spec[2]
        $env:AORTA_SIM_OUTPUT_DIR=$taskRoot+'/'+$spec[3]; $env:AORTA_REUSE_RESULTS='false'
        $parts+=Start-Arm $spec[0] $taskJulia ($taskJuliaArgs+@($taskSim)) $spec[3]
    }
    Write-Output ('START_E064_'+$prefix+' '+(Get-Date -Format o))
    foreach($p in $parts){ $p.WaitForExit() }
    foreach($p in $parts){ if($p.ExitCode -ne 0){throw ('E064 '+$prefix+' part failed '+$p.ExitCode)} }
    $rd='/simulation_'+$run
    New-Item -ItemType Directory -Path ($taskRoot+'/'+$prefix+'_combined'+$rd) | Out-Null
    foreach($pair in @(@('gpu0','ref'),@('gpu0','x'),@('gpu1','y'),@('gpu1','z'))){
        Copy-Item -LiteralPath ($taskRoot+'/'+$prefix+'_part_'+$pair[0]+$rd+'/'+$pair[1]+'_result.jld2') -Destination ($taskRoot+'/'+$prefix+'_combined'+$rd+'/') }
    $env:CUDA_VISIBLE_DEVICES=$taskGpu1; $env:AORTA_PHYSICAL_GPU_INDEX='1'; $env:AORTA_ENCODINGS='REF,X,Y,Z'
    $env:AORTA_SIM_OUTPUT_DIR=$taskRoot+'/'+$prefix+'_combined'; $env:AORTA_REUSE_RESULTS='true'
    Logged $taskJulia ($taskJuliaArgs+@($taskSim)) ($prefix+'_combine.log')
    Write-Output ('COMPLETE_E064_'+$prefix+' '+(Get-Date -Format o))
}
Assert-Gpus
Run-Case ($taskRoot+'/phantom_v1/aorta_motion_1m_candycane.phantom') 'motion_1m_20bin_candycane' 'flow'
Assert-Gpus
Run-Case ($taskRoot+'/tissue_phantom_v1/aorta_static_tissue.phantom') 'static_tissue_20bin_candycane' 'tissue'
Write-Output 'COMPLETE_E064_S1'
