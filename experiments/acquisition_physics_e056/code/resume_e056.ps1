$ErrorActionPreference='Stop'
$taskRun='D:/MRSim/outputs/acquisition_physics_e056'
$taskSource='C:/Users/lambda-3/Desktop/MRSim/migration_e053_control/e056_science'
$taskOldBuilder=39584
if(Get-Process -Id $taskOldBuilder -ErrorAction SilentlyContinue){Wait-Process -Id $taskOldBuilder -Timeout 3600}
$taskOldLog=Get-Content -LiteralPath ($taskRun+'/prepare_sequences_attempt5.log') -Raw -Encoding UTF8
if($taskOldLog -notmatch 'target_TE cannot be reached'){throw 'Unexpected prior builder outcome; do not continue'}
foreach($taskArm in @('baseline','bw700_te274')){
 if(!(Test-Path -LiteralPath ($taskRun+'/sequences_v1.building/'+$taskArm+'/metadata.toml'))){throw 'No completed reusable arm'}
}
$taskFailed=Get-Item -LiteralPath ($taskRun+'/sequences_v1.building')
if($taskFailed.FullName -ne 'D:\MRSim\outputs\acquisition_physics_e056\sequences_v1.building'){throw 'Unexpected preserved-stage path'}
Move-Item -LiteralPath $taskFailed.FullName -Destination ($taskRun+'/attempt5_sequences_v1.building')
foreach($taskFile in @('PROTOCOL.md','run_acquisition.jl','analyze_physics.py','test_reference.py','check_prefix.py','run_e056.ps1')){
 Copy-Item -LiteralPath ($taskRun+'/code/'+$taskFile) -Destination ($taskRun+'/code/attempt5_'+$taskFile)
 Copy-Item -LiteralPath ($taskSource+'/'+$taskFile) -Destination ($taskRun+'/code/'+$taskFile)
}
Copy-Item -LiteralPath ($taskSource+'/prepare_sequences_v2.jl') -Destination ($taskRun+'/code/prepare_sequences_v2.jl')
$env:JULIA_DEPOT_PATH='D:/MRSim/temp/julia_e056;D:/MRSim/MRSIM/MRSimTracks/outputs/upstream_migration_e046/depot;C:/Users/lambda-3/.julia'
$env:JULIA_LOAD_PATH='D:/MRSim/MRSIM/MRSimTracks/koma_migration_e046;@stdlib'
$env:JULIA_PKG_OFFLINE='true';$env:JULIA_NUM_THREADS='1'
$ErrorActionPreference='Continue'
& 'C:/Users/lambda-3/AppData/Local/Programs/Julia-1.12.7/bin/julia.exe' --startup-file=no --compiled-modules=existing --project=D:/MRSim/MRSIM/MRSimTracks/koma_migration_e046 ($taskRun+'/code/prepare_sequences_v2.jl') 'D:/MRSim/MRSIM/MRSimTracks' ($taskRun+'/sequences_v1') ($taskRun+'/attempt5_sequences_v1.building') *> ($taskRun+'/prepare_sequences_v2.log')
$taskExit=$LASTEXITCODE;$ErrorActionPreference='Stop'
if($taskExit -ne 0){throw 'Sequence v2 failed; no GPU simulation is authorized by this failed gate'}
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File ($taskRun+'/code/run_e056.ps1')
if($LASTEXITCODE -ne 0){throw 'E056 stopped at a numerical/resource gate; preserve logs'}
