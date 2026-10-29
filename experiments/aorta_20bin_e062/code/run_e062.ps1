. 'D:/MRSim/outputs/aorta_20bin_e062/code/e062_env.ps1'
# E-062 S1: REF+X on GPU0 and Y+Z on GPU1 in parallel (separate output roots, no reuse), then a
# provenance-checked combine (result files copied, simulator re-run with reuse on) that writes pc_reconstruction.h5.
$taskGpu0='GPU-3da2b149-c022-07f4-7fcb-7112684b5ac9'
$taskGpu1='GPU-8be27592-e325-7b7a-f432-b84f69cb18c6'
$taskPhantom='D:/MRSim/MRSIM/MRSimTracks/outputs/aorta_dp2_mri_pipeline/koma/multibin_8x4mm/aorta_motion_1m_8bin.phantom'
$taskPhantomSha='d23241331c8ec57e0e0396ef5468e1902583e7d2095c6ece83e3c71854242720'
$taskSim='D:/MRSim/MRSIM/MRSimTracks/koma_aorta/simulate_pc.jl'
$taskSimSha='a9cce28cf2107bafd986b8130e25eeaea5dfcb2a7ff49813b8d440d974ce4024'
$taskRun='motion_1m_20bin_4mm'
$audit=Get-Content -LiteralPath ($taskRoot+'/sequence_audit_v1/report.json') -Raw | ConvertFrom-Json
if($audit.status -ne 'PASS_E062_S0_G0D'){throw 'Sequence audit not passed'}
if((Get-FileHash -LiteralPath $taskSim).Hash.ToLower() -ne $taskSimSha){throw 'Simulator changed'}
if((Get-FileHash -LiteralPath $taskPhantom).Hash.ToLower() -ne $taskPhantomSha){throw 'Phantom changed'}
foreach($d in 'part_gpu0','part_gpu1','combined'){ if(Test-Path -LiteralPath ($taskRoot+'/'+$d)){throw ('Refuse existing '+$d)} }
if([IO.DriveInfo]::new('D:\').AvailableFreeSpace -lt 30000000000){throw 'Insufficient D storage'}
if((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory*1KB -lt 40GB){throw 'Insufficient free RAM'}
function GpuRow([int]$i){ (& nvidia-smi --id=$i --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits).Split(',').Trim() }
$r0=GpuRow 0; $r1=GpuRow 1
if($r0[0] -cne $taskGpu0 -or [int]$r0[1] -gt 16000 -or [int]$r0[2] -gt 30){throw 'GPU0 not as reviewed or too loaded'}
if($r1[0] -cne $taskGpu1 -or [int]$r1[1] -gt 256 -or [int]$r1[2] -gt 5){throw 'GPU1 not idle'}
Set-Location -LiteralPath 'D:/MRSim'
$env:AORTA_PHANTOM=$taskPhantom; $env:AORTA_SEQUENCE_DIR=$taskRoot+'/sequences_v1'; $env:AORTA_RUN_NAME=$taskRun
$env:AORTA_GPU_DEVICE='0'; $env:AORTA_BATCH_SIZE='0'; $env:AORTA_MAX_BLOCK_LENGTH='512'; $env:AORTA_ALLOW_UNVALIDATED_BLOCK_LENGTH='false'
function Start-Part([string]$gpu,[int]$index,[string]$encodings,[string]$name) {
    foreach($suffix in @('.log','.err.log')){ if(Test-Path -LiteralPath ($taskRoot+'/'+$name+$suffix)){throw ('Refuse existing log '+$name)} }
    $env:CUDA_VISIBLE_DEVICES=$gpu; $env:AORTA_PHYSICAL_GPU_INDEX=[string]$index; $env:AORTA_ENCODINGS=$encodings
    $env:AORTA_SIM_OUTPUT_DIR=$taskRoot+'/'+$name; $env:AORTA_REUSE_RESULTS='false'
    $p=Start-Process -FilePath $taskJulia -ArgumentList ($taskJuliaArgs+@($taskSim)) -NoNewWindow -PassThru `
        -RedirectStandardOutput ($taskRoot+'/'+$name+'.log') -RedirectStandardError ($taskRoot+'/'+$name+'.err.log')
    $null=$p.Handle
    return $p
}
$taskA=Start-Part $taskGpu0 0 'REF,X' 'part_gpu0'
$taskB=Start-Part $taskGpu1 1 'Y,Z' 'part_gpu1'
Write-Output ('START_E062_PARTS '+(Get-Date -Format o))
$taskA.WaitForExit(); $taskB.WaitForExit()
if($taskA.ExitCode -ne 0 -or $taskB.ExitCode -ne 0){throw ('E062 part failed: GPU0 '+$taskA.ExitCode+' GPU1 '+$taskB.ExitCode)}
Write-Output ('COMPLETE_E062_PARTS '+(Get-Date -Format o))
$run='/simulation_'+$taskRun
New-Item -ItemType Directory -Path ($taskRoot+'/combined'+$run) | Out-Null
foreach($pair in @(@('part_gpu0','ref'),@('part_gpu0','x'),@('part_gpu1','y'),@('part_gpu1','z'))){
    Copy-Item -LiteralPath ($taskRoot+'/'+$pair[0]+$run+'/'+$pair[1]+'_result.jld2') -Destination ($taskRoot+'/combined'+$run+'/')
}
$env:CUDA_VISIBLE_DEVICES=$taskGpu1; $env:AORTA_PHYSICAL_GPU_INDEX='1'; $env:AORTA_ENCODINGS='REF,X,Y,Z'
$env:AORTA_SIM_OUTPUT_DIR=$taskRoot+'/combined'; $env:AORTA_REUSE_RESULTS='true'
Logged $taskJulia ($taskJuliaArgs+@($taskSim)) 'combine.log'
Write-Output 'COMPLETE_E062_S1'
