"""Freeze E-046 legacy identities and small source snapshots; never edit inputs."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import uuid
from scripts.build_aorta_inlet_picker import sha


def prepare(output):
    if output.exists():
        raise FileExistsError(output)
    root = Path(__file__).resolve().parents[1]
    source_names = ['koma_aorta/Project.toml', 'koma_aorta/Manifest.toml',
        'koma_dp150/Project.toml', 'koma_dp150/Manifest.toml',
        'koma_aorta/native_zero_pilot.jl', 'koma_aorta/perfect_spoiling.jl',
        'koma_dp150/motion_timing.jl', 'koma_dp150/simulate_pc.jl',
        'koma_dp150/perfect_spoiling.jl', 'mrsimtracks/__init__.py',
        'mrsimtracks/reseeding.py', 'mrsimtracks/pointcloud.py',
        'pyproject.toml', 'uv.lock']
    base = root/'outputs/aorta_inlet_replenishment'
    dp = root/'outputs/dp150_mri_pipeline/koma/simulation_motion_100k_scanner_matched_venc5_seam_safe_v3'
    inputs = [base/'e045_native0_100k_mri_v1'/name for name in [
        'simulation_report.toml', 'phantom_report.toml', 'reset_preflight.toml',
        'reference_viability.toml', 'inlet_pilot.phantom',
        'ref_result.jld2', 'x_result.jld2', 'y_result.jld2', 'z_result.jld2']]
    inputs += [dp/'simulation_report.toml', dp/'seam_safe_preflight.toml',
               root/'outputs/dp150_mri_pipeline/koma/pilot_100k/dp150_motion_100k.phantom']
    inputs += [dp/(e+'_batches')/'batch_0001_of_0005.jld2' for e in ['ref', 'x', 'y', 'z']]
    for folder in [base/'e045_native0_sequences_v4', root/'koma_dp150/pc_3d_scanner_matched_venc5']:
        inputs += [folder/(e+'.seq') for e in ['ref', 'x', 'y', 'z']]
    protocol = base/'E046_MIGRATION_PROTOCOL.md'
    inputs.append(protocol)
    staging = output.with_name(output.name+'.building-'+uuid.uuid4().hex[:8])
    staging.mkdir(parents=True)
    for name in source_names:
        target = staging/'source_snapshot'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root/name, target)
    all_paths = [root/name for name in source_names]+inputs
    identities = {str(p.relative_to(root)): dict(bytes=p.stat().st_size, sha256=sha(p)) for p in all_paths}
    result = dict(experiment='E-046', status='PASS_LEGACY_SNAPSHOT',
        created_utc=datetime.now(timezone.utc).isoformat(), code_sha256=sha(__file__),
        inputs=identities, copied_sources=source_names, originals_read_only=True,
        no_new_1M_authorization=True)
    (staging/'registration.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    staging.rename(output)
    print(json.dumps(dict(status=result['status'], bound_files=len(identities), copied_sources=len(source_names))))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    prepare(p.parse_args().output)
