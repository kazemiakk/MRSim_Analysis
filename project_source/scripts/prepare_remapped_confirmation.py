"""E047 source freeze; never modify earlier experiment artifacts."""
import json
from pathlib import Path
import tomllib
from datetime import datetime, timezone
from scripts.build_aorta_inlet_picker import sha

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/remapped_confirmation_e047'


def prepare():
    target = BASE / 'source_registration.json'
    if target.exists():
        raise FileExistsError(target)
    old = ROOT / 'outputs/aorta_inlet_replenishment/e046_legacy_snapshot_v1/registration.json'
    frozen = json.loads(old.read_text())
    for name, identity in frozen['inputs'].items():
        if sha(ROOT/name) != identity['sha256']:
            raise ValueError('Frozen legacy identity changed: '+name)
    paths = [old, BASE/'PROTOCOL.md', Path(__file__).resolve(),
             ROOT/'koma_migration_e046/export_100k_e047.jl',
             ROOT/'koma_migration_e046/run_100k_e047.jl',
             ROOT/'koma_migration_e046/Manifest.toml',
             ROOT/'scripts/build_migration_cycle_maps.py']
    for case, folder in [('aorta','aorta_inputs_v1'),('phantom','phantom_inputs_v3')]:
        infofile = ROOT/'outputs/upstream_migration_e046'/folder/'case_info.toml'
        info = tomllib.loads(infofile.read_text())
        paths.extend([infofile, Path(info['source_phantom']), Path(info['source_output'])/'simulation_report.toml'])
        paths.extend(Path(info['source_output'])/(e+'_result.jld2') for e in ['ref','x','y','z'])
        paths.extend(Path(info['sequence_directory'])/(e+'.seq') for e in ['ref','x','y','z'])
    result = dict(experiment='E-047', status='PASS_SOURCES_FROZEN_BEFORE_CANDIDATE_SIGNAL',
                  created_utc=datetime.now(timezone.utc).isoformat(), prior_legacy_identities_verified=len(frozen['inputs']),
                  inputs={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in paths},
                  physical_gpu_uuid='GPU-8be27592-e325-7b7a-f432-b84f69cb18c6', no_new_1m=True)
    temporary = target.with_suffix('.partial')
    temporary.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    temporary.rename(target)
    print(json.dumps(dict(status=result['status'],frozen_files=len(result['inputs']),legacy_checks=len(frozen['inputs']))))


if __name__=='__main__':
    prepare()
