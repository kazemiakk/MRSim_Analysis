"""E-046 fixed-support, three-arm migration analysis; never modify input results."""
import argparse
import csv
import json
from pathlib import Path
import tomllib
from datetime import datetime, timezone

import h5py
import numpy as np

from scripts.build_aorta_inlet_picker import sha
from scripts.aorta_hemodynamic_reference import AortaGrid
from scripts.integrate_aorta_inlet_complex import complex_velocity_samples, integrate_flow, metrics

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'outputs/upstream_migration_e046'
ENC = ('REF', 'X', 'Y', 'Z')
ARMS = ('legacy', 'updated_nomap', 'updated_map')
LABEL = dict(legacy='Legacy', updated_nomap='Updated, no map', updated_map='Updated + map')
COLORS = dict(legacy='#555c65', updated_nomap='#235f82', updated_map='#bd7519')
RUN_DIRS = {('aorta', 'updated_nomap'): 'aorta_updated_nomap_v2', ('phantom', 'updated_nomap'): 'phantom_updated_nomap_v2'}


def toml(path):
    return tomllib.loads(path.read_text(encoding='utf-8'))


def image_quantities(images, venc, threshold=.05):
    """Unchanged per-bin/encoding 5% criterion; no phase filling or gain fitting."""
    if set(images) != set(ENC) or len({a.shape for a in images.values()}) != 1:
        raise ValueError('Four identically shaped complex images required')
    if any(a.ndim != 4 or not np.isfinite(a).all() for a in images.values()):
        raise ValueError('Finite bin/z/y/x images required')
    valid = np.ones(images['REF'].shape, bool)
    for a in images.values():
        m = abs(a)
        valid &= m >= threshold*m.max(axis=(1, 2, 3), keepdims=True)
        valid &= m > 0
    velocity = np.stack([np.angle(images[e]*images['REF'].conj())*(venc/np.pi)
                         for e in ENC[1:]], axis=-1)
    velocity[~valid] = np.nan
    return abs(images['REF']), velocity, valid.all(axis=0)


def paired_delta(a, b, mask, venc):
    if not mask.any():
        return dict(voxels=0, magnitude_relative_rmse=None, velocity_wrapped_component_rmse_m_s=None)
    am, av = a; bm, bv = b
    dm = bm[:, mask].astype(float)-am[:, mask]
    # Circular velocity difference, equivalent to wrapped phase subtraction.
    dv = (bv[:, mask].astype(float)-av[:, mask]+venc) % (2*venc)-venc
    den = np.sqrt(np.mean(am[:, mask].astype(float)**2))
    return dict(voxels=int(mask.sum()), magnitude_relative_rmse=float(np.sqrt(np.mean(dm**2))/den),
                mean_magnitude_ratio=float(bm[:, mask].mean()/am[:, mask].mean()),
                velocity_wrapped_component_rmse_m_s=float(np.sqrt(np.mean(dv**2))),
                wrapped_phase_rmse_rad=float(np.sqrt(np.mean(dv**2))*np.pi/venc))


def load_arm(case, arm, case_dir, paths):
    folder = case_dir if case == 'phantom' and arm == 'legacy' else BASE/RUN_DIRS.get((case, arm), f'{case}_{arm}_v1')
    report = None if folder == case_dir else toml(folder/'control_report.toml')
    if report is not None:
        if report['status'] != 'COMPLETE_E046_CONTROL' or report['arm'] != arm or report['case'] != case:
            raise ValueError('Incomplete/misidentified control')
        if report['input_sha256'] != sha(case_dir/'case_inputs.h5'):
            raise ValueError('Unmatched neutral input')
        if not report['actual_ADC_timing_and_labels_match_legacy'] or report['case_info_sha256'] != sha(case_dir/'case_info.toml'):
            raise ValueError('Case/timing contract mismatch')
        environment = ROOT/('koma_aorta' if arm == 'legacy' else 'koma_migration_e046')/'Manifest.toml'
        worker = ROOT/'koma_migration_e046'/('run_control_legacy_frozen.jl' if arm == 'legacy' else 'run_control.jl')
        if sha(environment) != report['manifest_sha256'] or sha(worker) != report['script_sha256']:
            raise ValueError('Runtime/source identity changed')
        paths.extend([environment, worker])
        paths.append(folder/'control_report.toml')
        mapfile = BASE/f'{case}_map_v1/cycle_map.h5'
        if sha(mapfile) != report['map_sha256']:
            raise ValueError('Cycle map changed')
        paths.extend([mapfile, mapfile.parent/'map_report.json'])
        if case == 'phantom':
            sampling = toml(folder/'sampling_control.toml')
            helper = ROOT/'koma_migration_e046/sorted_event_lookup.jl'
            wrapper = ROOT/'koma_migration_e046/run_phantom_control.jl'
            preregister = folder.with_name(folder.name+'_sampling_registration.toml')
            if (sampling['status'] != 'COMPLETE_PHANTOM_EXACT_LOOKUP_CONTROL' or
                sampling['helper_sha256'] != sha(helper) or sampling['wrapper_sha256'] != sha(wrapper) or
                sampling['control_report_sha256'] != sha(folder/'control_report.toml') or
                sampling['sampling_registration_sha256'] != sha(preregister)):
                raise ValueError('Exact lookup provenance failed')
            paths.extend([folder/'sampling_control.toml', helper, wrapper, preregister,
                          BASE/'event_lookup_v1/lookup_equivalence.toml'])
    images = {}; runtimes = {}; fft_error = {}
    frozen_path = ROOT/'outputs/aorta_inlet_replenishment/e046_legacy_snapshot_v1/registration.json'
    frozen = json.loads(frozen_path.read_text(encoding='utf-8'))
    paths.append(frozen_path)
    for e in ENC:
        p = folder/(e.lower()+('_legacy_result.h5' if report is None else '_result.h5'))
        if report is not None and sha(p) != report['result_sha256'][e]:
            raise ValueError('Control result hash changed')
        with h5py.File(p) as f:
            if report is None:
                batch = Path(toml(case_dir/'case_info.toml')['source_output'])/(e.lower()+'_batches')/'batch_0001_of_0005.jld2'
                expected = frozen['inputs'][str(batch.relative_to(ROOT))]['sha256']
                stored = f.attrs['source_batch_sha256']
                stored = stored.decode('ascii') if isinstance(stored, bytes) else stored
                if sha(batch) != expected or stored != expected:
                    raise ValueError('Reused legacy batch identity changed')
                paths.append(batch)
            a = f['image'][:]; k = f['rawsort'][:]
            if not np.isfinite(k).all():
                raise ValueError('Nonfinite k-space')
            independent = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(k, axes=(1, 2, 3)), axes=(1, 2, 3)), axes=(1, 2, 3))
            error = float(abs(a-independent).max()/max(abs(a).max(), 1e-30))
            if error > 3e-6:
                raise ValueError('Independent FFT mismatch')
            images[e] = a; fft_error[e] = error
            runtimes[e] = float(f.attrs['elapsed_seconds'])
        paths.append(p)
    return images, dict(runtime_seconds=runtimes, fft_max_relative_error=fft_error,
                        runtime_scope='Acquisition plus reconstruction/packaging; warm-up and sorter cost differ, NOT a pure kernel benchmark')


def aorta_flow(images_by_arm, info, paths):
    inlet = ROOT/'outputs/aorta_inlet_flow/e031_v1/inlet_numerical_evidence.npz'
    timingp = ROOT/'outputs/aorta_inlet_replenishment/e045_native0_timing_v1.json'
    weightp = timingp.with_suffix('.npz'); paths.extend([inlet, timingp, weightp])
    if sha(inlet) != '11a76b06940052d0b13361560b8ec85581c9c0a701eaaf5e895afdea701ca5c0':
        raise ValueError('Inferred inlet identity changed')
    z = np.load(inlet, allow_pickle=False); grid = AortaGrid()
    points = ((z['triangle_centroids_source_mm']/1000-grid.center)[:, grid.order])[:, ::-1]
    velocities = {}; masks = {}
    for arm, images in images_by_arm.items():
        v, valid, _ = complex_velocity_samples(images, grid.axes()[::-1], points, info['Venc_m_s'], .05)
        velocities[arm] = v; masks[arm] = valid.all(axis=0)
    joint = np.logical_and.reduce(list(masks.values()))
    area = z['triangle_area_mm2']; normal = z['normal_source'][list(grid.order)]
    vn = np.einsum('ncf,c->nf', z['face_velocities_source_m_s'], z['normal_source'])
    native = vn[z['inferred_triangles']].mean(axis=1)
    weights = np.load(weightp, allow_pickle=False)['native_frame_weights']
    timing = json.loads(timingp.read_text())
    centers = np.array([b['source_center_s'] for b in timing['encodings']['REF']['source_bins']])
    cfd = weights@(area[joint]@native[joint])*(.72/info['period_s'])
    result = dict(method='E036 complex interpolation before phase; actual ADC-time CFD average',
        denominator='Approximate inferred inlet face, native cap topology unverified',
        own_coverage_fraction={a: float(area[m].sum()/area.sum()) for a, m in masks.items()},
        joint_coverage_fraction=float(area[joint].sum()/area.sum()), joint_area_mm2=float(area[joint].sum()),
        whole_face_area_mm2=float(area.sum()), time_s=centers.tolist(), CFD_ml_s=cfd.tolist(),
        no_MRI_measurement_at_exact_zero=True, Koma_ml_s={}, metrics={},
        native_time_s=(np.arange(72)*.01).tolist(), native_CFD_full_ml_s=(area@native).tolist())
    for arm in ARMS:
        q = integrate_flow(velocities[arm][:, joint], area[joint], normal) if joint.any() else None
        result['Koma_ml_s'][arm] = None if q is None else q.tolist()
        result['metrics'][arm] = None if q is None else metrics(cfd, q, info['period_s'])
    return result


def phantom_flow(quantities, masks, case_dir, info, paths):
    """Same MRI section/area; disclose midpoint approximation, not ADC-exact CFD."""
    reference = ROOT/'outputs/dp150_hemodynamic_comparison/e025_v1/results_v2/matched_reference.h5'
    rreport = ROOT/'outputs/dp150_hemodynamic_comparison/e025_v1/results_v2/report.json'
    manifest = reference.parent/'MANIFEST.json'
    identities = {x['file']: x['sha256'] for x in json.loads(manifest.read_text())}
    if sha(reference) != identities[reference.name] or sha(rreport) != identities[rreport.name]:
        raise ValueError('Preserved CFD reference identity changed')
    old = ROOT/'outputs/dp150_mri_pipeline/koma/simulation_motion_100k_scanner_matched_venc5_seam_safe_v3/simulation_report.toml'
    prod = ROOT/'outputs/dp150_mri_pipeline/koma/simulation_motion_1m_scanner_matched_venc5_seam_safe_v1/simulation_report.toml'
    legacy, production = toml(old), toml(prod)
    if legacy['motion_timing']['seam_alignment'] != production['motion_timing']['seam_alignment']:
        raise ValueError('CFD reference timing is not transferable')
    if any(legacy['sequence'][e]['sha256'] != production['sequence'][e]['sha256'] for e in ENC):
        raise ValueError('CFD reference protocol mismatch')
    paths.extend([reference, rreport, manifest, old, prod]); section = 70
    with h5py.File(reference) as f:
        if float(f.attrs['effective_period_s']) != info['period_s']:
            raise ValueError('CFD period mismatch')
        area = f['plane_area_fraction_2'][:, :, section]*3.125  # 1.25 x 2.5 mm2
        refs = {k: f[k][:, :, :, section, 0] for k in ('cfd_voxel_midpoint', 'cfd_voxel_three_sample_mean')}
        times = f['time_s'][:]
    own = {arm: mask[:, :, section] & (area > 0) for arm, mask in masks.items()}
    joint = np.logical_and.reduce(list(own.values()))
    result = dict(method='Fixed upstream MRI section index70, area-weighted voxel velocities',
        denominator='Subvoxel source-geometry section area; not a newly measured inlet',
        reference_limitation='Midpoint/three-sample bin references, NOT actual ADC-time averaging',
        reference_timing_identity_verified=True, section_readout_index_zero_based=section,
        own_coverage_fraction={a: float(area[m].sum()/area.sum()) for a, m in own.items()},
        joint_coverage_fraction=float(area[joint].sum()/area.sum()), joint_area_mm2=float(area[joint].sum()),
        whole_face_area_mm2=float(area.sum()), time_s=times.tolist(),
        CFD_ml_s=(refs['cfd_voxel_midpoint'][:, joint]@area[joint]).tolist(),
        CFD_three_sample_ml_s=(refs['cfd_voxel_three_sample_mean'][:, joint]@area[joint]).tolist(),
        Koma_ml_s={}, metrics={})
    for arm in ARMS:
        q = quantities[arm][1][:, :, :, section, 0][:, joint]@area[joint] if joint.any() else None
        result['Koma_ml_s'][arm] = None if q is None else q.tolist()
        result['metrics'][arm] = None if q is None else metrics(np.array(result['CFD_ml_s']), q, info['period_s'])
    return result


def figures(case, result, quantities, joint, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    flow = result['flow']; t = np.asarray(flow['time_s'])
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5))
    for arm in ARMS:
        m = quantities[arm][0]
        values = m[:, joint].mean(axis=1) if joint.any() else np.full(len(m), np.nan)
        axes[0, 0].plot(np.arange(1, len(m)+1), values, 'o-', color=COLORS[arm], ms=3, label=LABEL[arm])
    axes[0, 0].set(xlabel='Acquired bin', ylabel='Mean REF magnitude (a.u.)', title='Absolute magnitude | joint voxel support')
    axes[0, 0].legend(frameon=False, fontsize=8)
    for i, arm in enumerate(ARMS):
        axes[0, 1].bar(i, result['arms'][arm]['fixed_support_fraction_of_union']*100, color=COLORS[arm], width=.55)
    axes[0, 1].set(xticks=range(3), xticklabels=['Legacy', 'No map', 'Map'], ylim=(0, 100),
                   ylabel='Retained union support (%)', title='Fixed 5% all-bin / all-encoding coverage')
    if 'native_time_s' in flow:
        axes[1, 0].plot(flow['native_time_s'], flow['native_CFD_full_ml_s'], color='#c3c7cc', lw=1, label='CFD full inferred face')
    axes[1, 0].plot(t, flow['CFD_ml_s'], 'o-', color='#141c24', ms=3, label='CFD paired area')
    if 'CFD_three_sample_ml_s' in flow:
        axes[1, 0].plot(t, flow['CFD_three_sample_ml_s'], ':', color='#888d94', label='CFD 3-sample')
    for arm in ARMS:
        q = flow['Koma_ml_s'][arm]
        if q is not None:
            axes[1, 0].plot(t, q, '--', color=COLORS[arm], lw=1.3, label=LABEL[arm])
            axes[1, 1].plot(t, np.asarray(q)-flow['CFD_ml_s'], 'o-', color=COLORS[arm], ms=3)
    axes[1, 0].set(xlabel='Source-cycle time (s)', ylabel='Flow (mL/s)', title='Flow | identical three-arm area')
    axes[1, 0].legend(frameon=False, fontsize=7, ncol=2)
    axes[1, 1].axhline(0, color='#888d94', lw=.8)
    axes[1, 1].set(xlabel='Source-cycle time (s)', ylabel='Koma − CFD (mL/s)', title=f"Joint section coverage: {flow['joint_coverage_fraction']:.1%}")
    for ax in axes.flat:
        ax.grid(axis='y', color='#e4e6e9', lw=.6); ax.set_axisbelow(True)
    fig.suptitle(f'{case.capitalize()} | 20k-spin migration sensitivity', x=.08, ha='left')
    note = ('Same-source internal consistency, not independent validation. Fixed 5% gate; no fitted gain or time shift.\n'
            + ('Approximate inferred aortic face; actual ADC averages. No instantaneous MRI point at t = 0.' if case == 'aorta'
               else 'Scanner-derived, geometry-adapted, VENC-adjusted. CFD midpoint / 3-sample timing sensitivity.'))
    fig.text(.08, .025, note, color='#555c65', fontsize=8)
    fig.tight_layout(rect=(.02, .09, .99, .94))
    for extension in ('png', 'pdf'):
        fig.savefig(output/f'{case}_migration_comparison.{extension}', dpi=400)
    plt.close(fig)


def analyze(case, case_dir, output):
    case_dir = case_dir.resolve(); output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    info = toml(case_dir/'case_info.toml'); paths = [case_dir/'case_info.toml', case_dir/'case_inputs.h5', case_dir/'legacy_timing.h5']
    paths.extend(ROOT/p for p in ['scripts/integrate_aorta_inlet_complex.py', 'scripts/aorta_hemodynamic_reference.py',
        'koma_migration_e046/direct_sort.jl', 'koma_migration_e046/export_controls.jl',
        'koma_migration_e046/perfect_spoiling.jl', 'scripts/build_migration_cycle_maps.py',
        'outputs/aorta_inlet_replenishment/E046_MIGRATION_PROTOCOL.md'])
    if info['status'] != 'PASS_FROZEN_CASE_EXPORT' or info['case'] != case or info['physical_particles'] != 20000:
        raise ValueError('Frozen20k case required')
    if sha(case_dir/'case_inputs.h5') != info['neutral_input_sha256']:
        raise ValueError('Frozen case changed')
    quantities = {}; images_by_arm = {}; masks = {}; arms = {}
    for arm in ARMS:
        images, arms[arm] = load_arm(case, arm, case_dir, paths)
        mag, velocity, masks[arm] = image_quantities(images, info['Venc_m_s'])
        quantities[arm] = (mag, velocity)
        if case == 'aorta':
            images_by_arm[arm] = images
    union = np.logical_or.reduce(list(masks.values())); joint = np.logical_and.reduce(list(masks.values()))
    if not union.any():
        raise ValueError('No signal support')
    for arm in ARMS:
        m = quantities[arm][0]
        arms[arm].update(fixed_voxels=int(masks[arm].sum()), fixed_support_fraction_of_union=float(masks[arm].sum()/union.sum()),
            mean_REF_magnitude_by_bin_on_joint=(m[:, joint].mean(axis=1).astype(float).tolist() if joint.any() else None),
            first_last_REF_mean_ratio_on_joint=(float(m[0, joint].mean()/m[-1, joint].mean()) if joint.any() else None))
    differences = {f'{a}_to_{b}': paired_delta(quantities[a], quantities[b], joint, info['Venc_m_s'])
                   for a, b in [('legacy', 'updated_nomap'), ('updated_nomap', 'updated_map'), ('legacy', 'updated_map')]}
    flow = aorta_flow(images_by_arm, info, paths) if case == 'aorta' else phantom_flow(quantities, masks, case_dir, info, paths)
    result = dict(experiment='E-046', status='COMPLETE_CONTROLLED_MIGRATION_SENSITIVITY', case=case,
        created_utc=datetime.now(timezone.utc).isoformat(), physical_particles=20000, threshold=.05,
        evidence_class='Numerical migration sensitivity and same-source internal consistency, NOT independent validation',
        support_denominator='Union of three fixed all-bin/all-encoding signal masks, NOT whole physical lumen',
        union_voxels=int(union.sum()), joint_voxels=int(joint.sum()), joint_fraction_of_union=float(joint.sum()/union.sum()),
        arms=arms, differences=differences, flow=flow,
        cycle_edge_limitation='First/last-bin means have different physical flow; their ratio alone does not diagnose an artifact',
        limitations=['20k pilot, not particle convergence or production replacement',
            'Nearest-neighbor many-to-one map approximates ensemble state, not exact conservation',
            'Updated-no-map includes runtime/API reset changes; map comparison also removes artificial endpoint reset',
            'Map requires the full chosen spin set, no independent spin batching',
            'Sparse signal exclusions remain unavailable, not zero or imputed; wrapped phase cannot exclude aliasing'])
    output.mkdir(parents=True)
    registration = dict(code_sha256=sha(__file__), inputs={str(p.relative_to(ROOT)): dict(bytes=p.stat().st_size, sha256=sha(p)) for p in paths})
    (output/'registration.json').write_text(json.dumps(registration, indent=2)+'\n', encoding='utf-8')
    (output/'report.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    np.savez_compressed(output/'support_evidence.npz', **{f'{a}_mask': masks[a] for a in ARMS}, joint=joint, union=union)
    with (output/'flow_bins.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['time_s', 'CFD_paired_ml_s', *[f'{a}_ml_s' for a in ARMS]])
        for i, t in enumerate(flow['time_s']):
            w.writerow([t, flow['CFD_ml_s'][i], *[None if flow['Koma_ml_s'][a] is None else flow['Koma_ml_s'][a][i] for a in ARMS]])
    figures(case, result, quantities, joint, output)
    (output/'MANIFEST.json').write_text(json.dumps({p.name: dict(bytes=p.stat().st_size, sha256=sha(p)) for p in output.iterdir()}, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(case=case, status=result['status'], joint_fraction=result['joint_fraction_of_union'], differences=differences, flow=flow['metrics']), indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', choices=['aorta', 'phantom'], required=True)
    p.add_argument('--case-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); analyze(a.case, a.case_dir, a.output)
