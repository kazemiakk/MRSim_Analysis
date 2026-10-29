# E-063 S1 orchestrator. Phantom: G1 identity prefixes (GPU1), then full REF (GPU1) and X (GPU0) in parallel.
# Aorta: static phantom file, then REF+X (GPU0) and Y+Z (GPU1) with the frozen koma_aorta simulator, then combine.
. 'D:/MRSim/outputs/static_background_e063/code/e063_env.ps1'
$taskGpu0='GPU-3da2b149-c022-07f4-7fcb-7112684b5ac9'
$taskGpu1='GPU-8be27592-e325-7b7a-f432-b84f69cb18c6'
if([IO.DriveInfo]::new('D:\').AvailableFreeSpace -lt 20000000000){throw 'Insufficient D storage'}
function GpuRow([int]$i){ (& nvidia-smi --id=$i --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits).Split(',').Trim() }
function Assert-Gpus { $r0=GpuRow 0; $r1=GpuRow 1
    if($r0[0] -cne $taskGpu0 -or [int]$r0[1] -gt 16000 -or [int]$r0[2] -gt 40){throw 'GPU0 not as reviewed or too loaded'}
    if($r1[0] -cne $taskGpu1 -or [int]$r1[1] -gt 256 -or [int]$r1[2] -gt 5){throw 'GPU1 not idle'} }
Set-Location -LiteralPath 'D:/MRSim'
$run=$taskJuliaArgs+@(($taskCode+'/run_static_e063.jl'))
# --- phantom G1 ---
Assert-Gpus; $env:CUDA_VISIBLE_DEVICES=$taskGpu1
Logged $taskJulia ($run+@('identity',($taskRoot+'/identity_v1'))) 'identity.log'
$g1=Get-Content -LiteralPath ($taskRoot+'/identity_v1/report.toml') -Raw
if($g1 -notmatch 'status = "PASS_E063_G1_PREFIX_IDENTITY"'){ Write-Output 'STOP_E063_AFTER_G1'; exit 0 }
# --- phantom full REF (GPU1) and X (GPU0) ---
function Start-Arm([string]$gpu,[string]$exe,[string[]]$arguments,[string]$name) {
    foreach($suffix in @('.log','.err.log')){ if(Test-Path -LiteralPath ($taskRoot+'/'+$name+$suffix)){throw ('Refuse existing log '+$name)} }
    $env:CUDA_VISIBLE_DEVICES=$gpu
    $p=Start-Process -FilePath $exe -ArgumentList $arguments -NoNewWindow -PassThru `
        -RedirectStandardOutput ($taskRoot+'/'+$name+'.log') -RedirectStandardError ($taskRoot+'/'+$name+'.err.log')
    $null=$p.Handle; return $p
}
Assert-Gpus
$a=Start-Arm $taskGpu1 $taskJulia ($run+@('run',($taskRoot+'/phantom_ref_v1'),'REF')) 'phantom_ref'
$b=Start-Arm $taskGpu0 $taskJulia ($run+@('run',($taskRoot+'/phantom_x_v1'),'X')) 'phantom_x'
Write-Output ('START_E063_PHANTOM '+(Get-Date -Format o))
$a.WaitForExit(); $b.WaitForExit()
if($a.ExitCode -ne 0 -or $b.ExitCode -ne 0){throw ('E063 phantom arm failed: REF '+$a.ExitCode+' X '+$b.ExitCode)}
Write-Output ('COMPLETE_E063_PHANTOM '+(Get-Date -Format o))
# --- aorta (koma_aorta environment, as E-062) ---
. 'D:/MRSim/outputs/aorta_20bin_e062/code/e062_env.ps1'
$taskRoot='D:/MRSim/outputs/static_background_e063'; $taskCode=$taskRoot+'/code'
Logged $taskJulia ($taskJuliaArgs+@(($taskCode+'/build_aorta_static_phantom_e063.jl'),($taskRoot+'/aorta_phantom_v1'))) 'aorta_phantom.log'
$taskSim='D:/MRSim/MRSIM/MRSimTracks/koma_aorta/simulate_pc.jl'
if((Get-FileHash -LiteralPath $taskSim).Hash.ToLower() -ne 'a9cce28cf2107bafd986b8130e25eeaea5dfcb2a7ff49813b8d440d974ce4024'){throw 'Simulator changed'}
$env:AORTA_PHANTOM=$taskRoot+'/aorta_phantom_v1/aorta_static_tissue.phantom'
$env:AORTA_SEQUENCE_DIR='D:/MRSim/outputs/aorta_20bin_e062/sequences_v1'; $env:AORTA_RUN_NAME='static_tissue_20bin_4mm'
$env:AORTA_GPU_DEVICE='0'; $env:AORTA_BATCH_SIZE='0'; $env:AORTA_MAX_BLOCK_LENGTH='512'; $env:AORTA_ALLOW_UNVALIDATED_BLOCK_LENGTH='false'
function Start-Part([string]$gpu,[int]$index,[string]$encodings,[string]$name) {
    $env:AORTA_PHYSICAL_GPU_INDEX=[string]$index; $env:AORTA_ENCODINGS=$encodings
    $env:AORTA_SIM_OUTPUT_DIR=$taskRoot+'/'+$name; $env:AORTA_REUSE_RESULTS='false'
    return Start-Arm $gpu $taskJulia ($taskJuliaArgs+@($taskSim)) $name
}
Assert-Gpus
$c=Start-Part $taskGpu0 0 'REF,X' 'aorta_part_gpu0'
$d=Start-Part $taskGpu1 1 'Y,Z' 'aorta_part_gpu1'
Write-Output ('START_E063_AORTA '+(Get-Date -Format o))
$c.WaitForExit(); $d.WaitForExit()
if($c.ExitCode -ne 0 -or $d.ExitCode -ne 0){throw ('E063 aorta part failed: GPU0 '+$c.ExitCode+' GPU1 '+$d.ExitCode)}
$rd='/simulation_static_tissue_20bin_4mm'
New-Item -ItemType Directory -Path ($taskRoot+'/aorta_combined'+$rd) | Out-Null
foreach($pair in @(@('aorta_part_gpu0','ref'),@('aorta_part_gpu0','x'),@('aorta_part_gpu1','y'),@('aorta_part_gpu1','z'))){
    Copy-Item -LiteralPath ($taskRoot+'/'+$pair[0]+$rd+'/'+$pair[1]+'_result.jld2') -Destination ($taskRoot+'/aorta_combined'+$rd+'/') }
$env:CUDA_VISIBLE_DEVICES=$taskGpu1; $env:AORTA_PHYSICAL_GPU_INDEX='1'; $env:AORTA_ENCODINGS='REF,X,Y,Z'
$env:AORTA_SIM_OUTPUT_DIR=$taskRoot+'/aorta_combined'; $env:AORTA_REUSE_RESULTS='true'
Logged $taskJulia ($taskJuliaArgs+@($taskSim)) 'aorta_combine.log'
Write-Output 'COMPLETE_E063_S1'
