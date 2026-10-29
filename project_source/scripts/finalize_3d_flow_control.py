"""Bind completed E051 numerical evidence and create its local handoff."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import tomllib

from scripts.prepare_3d_flow_control import ROOT, BASE, dump
from scripts.build_aorta_inlet_picker import sha


def repo_name(path):
    return str(Path(path).resolve().relative_to(ROOT))


def finalize(analysis):
    analysis = analysis.resolve()
    if (BASE/'FINAL_MANIFEST.json').exists():
        raise FileExistsError(BASE/'FINAL_MANIFEST.json')
    report = json.loads((analysis/'report.json').read_text())
    validation = json.loads((analysis/'validation.json').read_text())
    if validation['status'] != 'PASS_INDEPENDENT_E051_REPRODUCTION_CHECKS' or validation['report_sha256'] != sha(analysis/'report.json'):
        raise ValueError('Independent result binding failed')
    old = ROOT/'outputs/phantom_precision_e050'
    saved = tomllib.loads((old/'mapped_f64_v1/control_report.toml').read_text())
    unchanged = 0
    for e, digest in saved['result_sha256'].items():
        if sha(old/'mapped_f64_v1'/(e.lower()+'_result.h5')) != digest:
            raise ValueError('Preserved E050 signal changed')
        unchanged += 1
    if sha(old/'analysis_v1/report.json') != '9586b24146e7ddf04e164a3231a2111d8b667a8175ed43efa784738d699faa9c':
        raise ValueError('Preserved E050 quantitative report changed')
    lines = ['# Full 3D known-flow control — E-051', '',
        'Status: **'+report['status']+'**. This is analytic numerical verification, not independent physical validation.', '',
        '| Model | Known axial velocity (m/s) | Recovered mean (m/s) | Vector RMSE (m/s) | Fixed predefined-core coverage |',
        '|---|---:|---:|---:|---:|']
    for model, data in report['models'].items():
        m = data['metrics']
        recovered = 'Unavailable' if m is None else f"{m['mean_velocity_m_s'][0]:.9f}"
        rmse = 'Unavailable' if m is None else f"{m['vector_rmse_m_s']:.8g}"
        lines.append(f"|{model}|{data['velocity_mri_m_s'][0]:g}|{recovered}|{rmse}|{data['fixed_core_coverage_fraction']:.2%}|")
    locations = report['models']['static_landmarks']['localization']
    lines += ['', f"Exact 3D localization: {sum(r['exact_match'] for r in locations)}/{len(locations)} landmark/bin checks.",
        f"Independent source/metric reproduction: {validation['checks']} checks pass. All four E050 Float64 signals and its report remain unchanged.", '',
        '## Evidence and PNGs', '',
        '[Quantitative report](analysis_v1/report.json) · [Independent validation](analysis_v1/validation.json) · [Per-bin CSV](analysis_v1/known_velocity_bins.csv)', '',
        '[Preferred 3D sample/velocity view](figures_v2/known_flow_3d.png) · [True 2D slices](figures_v2/known_flow_slices.png) · [Static localization](analysis_v1/static_localization.png) · [Known-velocity profiles](analysis_v1/known_velocity_profiles.png)', '',
        'Raw Float64 signals, primary ComplexF32 images and checkpoints are in `acquisition_v2/`.',
        'Preflight and moment/timing evidence are in `preflight_v1/` and `inputs_v1/`.', '',
        '## Interpretation limits', '',
        'The same full 400×64×32×15 sequence/runtime/reconstruction is used. The synthetic plug flows are deterministic axial translations in a periodic readout FOV; they test positive/negative axial velocity and zero transverse cross-talk, not acceleration, shear, random-spin convergence, nonzero through-slab flow, inlet replenishment or realistic walls.', '',
        'The independently integrated gradient-phase prediction is conditional on the Koma REF signal. It is not an independent prediction of RF steady-state magnitude or the full Bloch signal. No fitted gain, time shift, axis alignment, threshold relaxation or phase filling was applied.', '',
        'A pass would show that this pipeline recovers these simple known velocities and coordinates. It does not explain or remove the stenotic-phantom throat discrepancy and does not authorize a 300k/1M run automatically.', '',
        'The first 3D layout is retained in analysis_v1. Its crowded labels were repaired in a separate figures_v2 presentation derivative; no source arrays or quantitative reports changed. PNGs were visually inspected. Physical grid-center extents, fixed velocity limits, masking and 3D aspect/stride are explicit.', '',
        'The first full attempt stopped at CUDA context creation (999) before signal generation; the registration remains in `acquisition_v1/`. A fresh tiny device check passed, and the unchanged retry completed separately. See [ATTEMPTS.md](ATTEMPTS.md).', '']
    content = '\n'.join(lines)
    readme = BASE/'README.md'
    if readme.exists():
        if readme.read_text(encoding='utf-8') != content:
            raise ValueError('Existing handoff differs; refuse overwrite')
    else:
        readme.write_text(content, encoding='utf-8')
    paths = [BASE/'PROTOCOL.md', BASE/'ATTEMPTS.md', BASE/'README.md', BASE/'VALIDATION_COMMANDS.md',
        BASE/'filesystem_compression_v1.json', BASE/'inputs_v1/registration.json',
        BASE/'inputs_v1/contract.json', BASE/'inputs_v1/controls.h5', BASE/'preflight_v1/report.toml',
        BASE/'preflight_v1/numerical_gate.toml', BASE/'preflight_v1/prefix_signals.h5',
        BASE/'acquisition_v2/registration.toml', BASE/'acquisition_v2/control_report.toml']
    paths += list(analysis.iterdir())
    paths += list((BASE/'figures_v2').iterdir())
    paths += [BASE/'visual_review.json', BASE/'SESSION_VALIDATION.json']
    paths += [ROOT/name for name in ('scripts/prepare_3d_flow_control.py', 'scripts/check_3d_flow_prefix.py',
        'scripts/analyze_3d_flow_control.py', 'scripts/validate_3d_flow_control.py', 'scripts/finalize_3d_flow_control.py',
        'scripts/plot_3d_flow_control_review.py',
        'tests/test_3d_flow_control.py', 'koma_migration_e046/known_flow_3d_e051.jl',
        'koma_migration_e046/compress_control_e051.ps1')]
    dump(BASE/'FINAL_MANIFEST.json', dict(experiment='E-051', status=report['status'],
        created_utc=datetime.now(timezone.utc).isoformat(), independent_checks=validation['checks'],
        E050_signal_identities_unchanged=unchanged, E050_quantitative_report_unchanged=True,
        new_300k=False, new_1m=False, accuracy_promoted=False,
        artifacts={repo_name(p):dict(bytes=p.stat().st_size, sha256=sha(p)) for p in paths if p.is_file()}))
    print(json.dumps(dict(status=report['status'], independent_checks=validation['checks'], E050_signals_unchanged=unchanged)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    finalize(parser.parse_args().analysis)
