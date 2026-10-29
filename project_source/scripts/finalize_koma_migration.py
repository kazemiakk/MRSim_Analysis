"""Create the local E046 handoff and unsent Charles draft from completed evidence."""
import argparse
import json
import re
from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image
from scripts.build_aorta_inlet_picker import sha

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'outputs/upstream_migration_e046'


def readable_copy(text):
    replacements = {
        'three-arm20k physical-spin': 'three-arm, 20,000-spin', 'new1M': 'new 1M',
        'KomaMRI0.15.0': 'KomaMRI 0.15.0', 'Base0.15.2': 'Base 0.15.2',
        'Core0.14.0': 'Core 0.14.0', 'Files0.12.0': 'Files 0.12.0',
        'old0.13-series': 'old 0.13-series', 'commit200c': 'commit 200c',
        'E04516-bin': 'E-045 16-bin', 'E01915-bin': 'E-019 15-bin',
        'first20k': 'first 20,000', 'only3 pixels': 'only 3 pixels',
        'about23.55ms': 'about 23.55 ms', 'index70': 'index 70', 'to1M': 'to 1M',
        'approved100k': 'approved 100k', 'about59%': 'about 59%',
        'multiplicity9(aorta)/8(phantom)': 'multiplicity 9 (aorta) / 8 (phantom)',
        'about1.24/1.25mm': 'about 1.24 / 1.25 mm', 'all39': 'all 39',
        'checks,8': 'checks, 8', 'checks,3': 'checks, 3', 'and5': 'and 5',
        'error0.3604%': 'error 0.3604%', 'ratio1.18910': 'ratio 1.18910',
        'broad-slab1.05857': 'broad-slab 1.05857', 'matches130': 'matches 130',
        'real120-block': 'real 120-block', 'October13,2026': 'October 13, 2026',
        'October13': 'October 13', 'matched20,000-spin': 'matched 20,000-spin',
        'the20k': 'the 20k', 'another1M': 'another 1M', 'at100k': 'at 100k', 'its1.86%': 'its 1.86%',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r'^-(?=\d)', '- ', text, flags=re.MULTILINE)
    return re.sub(r'(?<=\d)(m/s|mm²|mm|ms)(?=[\s,.;)]|$)', r' \1', text)


def finalize(aorta, phantom, validation, output):
    if output.exists():
        raise FileExistsError(output)
    reports = {name: json.loads((folder/'report.json').read_text()) for name, folder in [('aorta', aorta), ('phantom', phantom)]}
    v = json.loads(validation.read_text()); sampling = json.loads((BASE/'full_ref_sampling_equivalence_v1.json').read_text())
    if v['status'] != 'PASS_INDEPENDENT_E046_CHECKS' or sampling['status'] not in [
            'PASS_BITWISE_FULL_REF_SAMPLING_EQUIVALENCE', 'PASS_FLOAT32_FULL_REF_SAMPLING_EQUIVALENCE']:
        raise ValueError('Independent gates not complete')
    counts = {}
    for label, filename in [('local', 'local_tests.xml'), ('upstream', 'tracker_focused_tests.xml')]:
        suites = list(ET.parse(BASE/filename).getroot().iter('testsuite'))
        counts[label] = sum(int(s.attrib['tests']) for s in suites)
        if any(int(s.attrib['failures']) or int(s.attrib['errors']) for s in suites):
            raise ValueError('Failed test suite')
    figure_checks = {}
    for case, analysis in [('aorta', aorta), ('phantom', phantom)]:
        sheets = BASE/f'{case}_image_sheets_v3'
        for name, identity in json.loads((sheets/'MANIFEST.json').read_text()).items():
            if sha(sheets/name) != identity:
                raise ValueError('Image export hash mismatch')
        for folder, stem in [(analysis, f'{case}_migration_comparison'),
                             (sheets, f'{case}_magnitude_first_last'), (sheets, f'{case}_phase_RO_first_last')]:
            png, pdf = folder/(stem+'.png'), folder/(stem+'.pdf')
            with Image.open(png) as im:
                size = im.size; im.verify()
            with pdf.open('rb') as f:
                if f.read(5) != b'%PDF-':
                    raise ValueError('Invalid PDF signature')
            figure_checks[str(png.relative_to(ROOT))] = dict(pixels=list(size), sha256=sha(png),
                visual_review='Viewed in this task; scales/labels/legends/caveats checked',
                PDF_sha256=sha(pdf), PDF_rendering='Not performed')
    table = ['| Case / arm | Mean magnitude / legacy | Component delta vs legacy (m/s RMS) | Fixed voxel support / union | Inlet/section own coverage | Flow RMSE (mL/s) | Cycle-volume error |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for case, r in reports.items():
        for arm in ['legacy', 'updated_nomap', 'updated_map']:
            d = r['differences'].get('legacy_to_'+arm)
            flow = r['flow']['metrics'][arm]
            table.append(f"| {case} / {arm} | {1 if d is None else d['mean_magnitude_ratio']:.4f} | "
                f"{'reference' if d is None else format(d['velocity_wrapped_component_rmse_m_s'], '.6f')} | "
                f"{r['arms'][arm]['fixed_support_fraction_of_union']:.2%} | {r['flow']['own_coverage_fraction'][arm]:.2%} | "
                f"{flow['rmse_ml_s']:.4f} | {flow['nominal_bin_cycle_volume_ratio']-1:+.2%} |")
    text = '''# E-046 — Controlled upstream migration results

The authorized isolated upgrade, focused tests and three-arm20k physical-spin
controls are complete for both aorta and dp150. This is numerical migration
sensitivity and same-source internal consistency, not independent MRI validation.
Legacy environments, source data and production results remain unchanged.
No new1M run, publication, email send or default-environment replacement occurred.

## What changed

Pinned isolated versions: KomaMRI0.15.0, Base0.15.2, Core0.14.0 and Files0.12.0.
The old0.13-series environments remain intact. The upstream tracker was examined
at commit200c87be5541ffe36c31bad504b6cad81d9da693, not merged into the dirty local fork.

The three arms are legacy, updated without a map, and updated with an explicit
1-based destination-to-source map. The mapped condition also removes the
artificial all-spin cycle-end M0 reset while preserving real inlet-reset flags.
Therefore that contrast is **map plus reset-policy change**, not a map-only
single-factor effect. Updating the package alone does not enable the map.

Aorta uses the unchanged E04516-bin/native-time-zero sequences. Phantom uses
the unchanged E01915-bin scanner-derived, geometry-adapted, VENC-adjusted protocol.
The deterministic first20k particles and a zero-density timing sentinel preserve
the full source reset-time union. The phantom legacy first batch is reused only
after exact source, spin-range, sequence and timing-transform identity checks.

## Quantitative results

'''+ '\n'.join(table)+'\n\n'
    for case, r in reports.items():
        d = r['differences']['legacy_to_updated_map']; flow = r['flow']
        text += (f"{case.capitalize()}: three-arm joint voxel support {r['joint_voxels']:,}/{r['union_voxels']:,} "
                 f"({r['joint_fraction_of_union']:.2%}); joint inlet/section area {flow['joint_area_mm2']:.2f} mm² "
                 f"({flow['joint_coverage_fraction']:.2%} of the stated geometric denominator). "
                 f"Mapped mean magnitude changes by {d['mean_magnitude_ratio']-1:+.2%} versus legacy.\n\n")
    pf = reports['phantom']['flow']
    text += (f"**Phantom support warning:** the joint section contains only3 pixels "
             f"({pf['joint_area_mm2']:.3f}mm² of {pf['whole_face_area_mm2']:.3f}mm²; "
             f"{pf['joint_coverage_fraction']:.2%}). Its flow error describes only those pixels. "
             "It cannot certify full-section flow or production image quality. The fine scanner-derived grid "
             "was deliberately unchanged for this small control; the sparse magnitude/phase sheets make "
             "this limitation visible. No threshold was relaxed to make the result look complete.\n\n")
    text += '''Voxel coverage uses the union of the three numerical signal-support masks,
not the whole physical lumen. Flow errors use the identical joint geometric
area for all three arms. Own-area coverage is reported separately. Smaller
errors on partial support must not be compared directly with historical full-area
or higher-spin results. Relative magnitude RMSE and circular phase differences
are available in each JSON report; no fitted gain or time shift was used.

Aortic CFD flow is averaged at the actual ADC times. Its inlet topology and
normal remain inferred, and its tracking is approximate point-cloud tracking.
The first MRI window is about23.55ms, not an instantaneous acquired t=0 point.
Phantom flow is an upstream MRI section (index70), not a newly measured inlet;
its preserved CFD midpoint/three-sample references are not ADC-exact averages.
The phantom's controlled synthetic tail and shared-source limitations remain.

## Interpretation and scale-up decision

Version-only changes and the map/reset-policy changes are separated. A flatter
magnitude curve is compatible with changed saturation history; magnitude loss
alone is not evidence of poorer quality. Conversely, retained coverage and
phase/flow residuals prevent calling the mapped result universally superior.
First and last physical bins have different flows, so their ratio alone is not
an artifact diagnostic. The synthetic constant-flow tests supply that control.

Do not replace the production results or proceed directly to1M. Review the mapped
signal/support changes and run a separately approved100k mapped confirmation,
including map-distance/multiplicity and coverage sensitivity, before a production
decision. The maps are nearest-neighbor, many-to-one ensemble approximations,
not exact conservation: unique-source fractions are about59%, with maximum
multiplicity9(aorta)/8(phantom) and median distance about1.24/1.25mm.

Real-data FSI remains conditional on qualified moving-mesh/ALE inputs; the
isolated synthetic/API tests are not physical FSI validation. TKE, wall shear
and pressure are not made available by this package migration.

## Verification

'''
    text += (f"- {counts['local']} local tests passed (3 large tests deselected); {counts['upstream']} focused upstream mapping/ALE tests passed.\n"
             f"- {v['checks']} independent saved-data/source/export checks, including all39 unchanged frozen legacy identities.\n"
             "-196 manuscript-evidence checks passed. Strict MkDocs build passed using an isolated docs environment.\n"
             "-10 CPU/GPU/map/serialization checks,8 interior-reset checks,3 sorter and5 marker-equivalence checks passed.\n"
             "- Static spoiled-GRE synthetic relative error0.3604%. Thin-slice late-cycle edge ratio1.18910→0.99946; broad-slab1.05857→1.00609.\n"
             "- Sorted event lookup matches130 sampled real blocks, duplicate/closed-endpoint controls and every discrete field of a real120-block prefix.\n"
             f"- Full phantom REF acquisition lookup check: {sampling['status']}. See the bound JSON for raw-grid/image differences.\n"
             "- Six final PNGs visually reviewed; PDFs checked for file signatures/hashes, not rendered.\n\n")
    text += '''The preprocessing acceleration is local and separately provenance-bound;
installed upstream package sources are untouched. It changes search cost only,
not event/sample times. End-to-end runtimes include warm-up and differing raw-data
packaging costs, so they are not pure kernel-speed benchmarks. Encoding runtimes
are retained in the control reports. Failed/interrupted attempts are preserved
and documented in [ATTEMPTS.md](../ATTEMPTS.md).

## Open the figures

'''
    for case, folder in [('aorta', aorta), ('phantom', phantom)]:
        text += f"- [{case.capitalize()} magnitude/coverage/flow comparison](../{folder.name}/{case}_migration_comparison.png)\n"
        text += f"- [{case.capitalize()} first/last magnitude sheets](../{case}_image_sheets_v3/{case}_magnitude_first_last.png)\n"
        text += f"- [{case.capitalize()} first/last grayscale RO phase](../{case}_image_sheets_v3/{case}_phase_RO_first_last.png)\n"
    text += '''
The sheets use one identical true MRI plane and one absolute scale across arms.
Gray phase exclusions are unavailable values, not imputed phase or wall material.
Use these as pilot diagnostic illustrations, not production publication figures.

## Upstream sources

[PR935](https://github.com/JuliaHealth/KomaMRI.jl/pull/935) merged October13,2026.
[KomaMRI0.15.0](https://github.com/JuliaHealth/KomaMRI.jl/releases/tag/v0.15.0)
and [Base0.15.2](https://github.com/JuliaHealth/KomaMRI.jl/releases/tag/KomaMRIBase-v0.15.2)
are the pinned release references. The supplied map and coupled-spin restriction
are essential; installation alone is not the scientific intervention.
'''
    output.mkdir(parents=True)
    (output/'FIGURE_REVIEW.json').write_text(json.dumps(figure_checks, indent=2)+'\n', encoding='utf-8')
    (output/'README.md').write_text(readable_copy(text), encoding='utf-8')
    a, p = reports['aorta'], reports['phantom']
    ad, pd = a['differences']['legacy_to_updated_map'], p['differences']['legacy_to_updated_map']
    email = f'''Subject: KomaMRI update — controlled aorta and stenosis-phantom tests

Hi Charles and everyone,

Thank you again for your earlier guidance. I checked PR935 after its October13
merge and tested a separate pinned KomaMRI0.15.0 environment with Base0.15.2,
without changing our existing simulations or environments.

I have now completed matched20,000-spin controls for both the aorta and stenosis
phantom: the legacy setup, the updated setup without a cycle map, and the updated
setup with explicit magnetization remapping. For the mapped condition, I also
removed our artificial all-spin cycle-end M0 reset while keeping the actual inlet
resets. This distinction matters: the large magnitude effect is not a
version-only change, and I would not attribute it solely to the release fix.

On shared voxel support, the mapped mean magnitude is {ad['mean_magnitude_ratio']:.1%}
of legacy for the aorta and {pd['mean_magnitude_ratio']:.1%} for the phantom. The
corresponding RMS component-velocity changes are
{ad['velocity_wrapped_component_rmse_m_s']:.5f} and {pd['velocity_wrapped_component_rmse_m_s']:.5f}m/s.
I have kept error and retained coverage together in the attached comparisons;
these are same-source consistency tests, not independent velocity validation.

An important caveat is that the20k phantom is too sparse on the unchanged fine
grid for a full-section flow conclusion: the fixed shared gate retains only
{p['flow']['joint_coverage_fraction']:.2%} of the upstream section (3 pixels).
The aortic shared inferred-face coverage is {a['flow']['joint_coverage_fraction']:.2%}.
I am treating the phantom control as a migration/sampling test, not a successful
whole-field validation, and have not replaced the existing production results.

The figures show how the magnitude history and retained support change across
the three conditions, alongside CFD/Koma flow on exactly the same area. The
grayscale first/last-bin sheets use the same plane and absolute display scale,
so the magnitude changes are visible without rescaling each image. Excluded
phase values remain clearly unavailable rather than being filled in.

The CPU/GPU map-direction, serialization, ordinary-reset and synthetic
steady-state controls passed. I also ran the upstream tracker mapping/ALE tests
in an isolated checkout ({counts['upstream']} passed). Our current case data do not
yet qualify as an FSI validation dataset.

One practical issue I encountered was the full motion-event vector being scanned
for every sequence block during discretization. A local binary-search lookup
preserves the same event selections; I checked real block samples, a complete
discretized prefix, and the saved full REF acquisition before using it. I can
share that reproducer separately if useful.

Before moving to another1M acquisition, I plan to review the support changes and
confirm the mapped condition at100k. Do the magnitude changes look reasonable
given the removal of our forced cycle-end reset? I would also appreciate your
recommendation for checking nearest-neighbor map distances and many-to-one
multiplicity before increasing the particle count.

Best,
Amirkhosro

Suggested attachments: the aorta comparison and aorta magnitude sheet. Include
the phantom comparison as a technical pilot appendix with its1.86% coverage
caveat. The sparse phantom sheets are diagnostics, not presentation-quality
production figures; grayscale phase sheets are optional technical detail.
This file is a draft only; no email has been sent.
'''
    (output/'DRAFT_EMAIL_CHARLES.md').write_text(readable_copy(email), encoding='utf-8')
    paths = [aorta/'report.json', phantom/'report.json', validation, BASE/'full_ref_sampling_equivalence_v1.json',
             BASE/'local_tests.xml', BASE/'tracker_focused_tests.xml',
             ROOT/'koma_migration_e046/analyze_aorta_frozen_v3.py',
             ROOT/'scripts/analyze_koma_migration.py', ROOT/'scripts/plot_koma_migration_images.py']
    (output/'SOURCE_MANIFEST.json').write_text(json.dumps(dict(code_sha256=sha(__file__),
        sources={str(x.relative_to(ROOT)): sha(x) for x in paths},
        outputs={x.name: sha(x) for x in output.iterdir()}), indent=2)+'\n', encoding='utf-8')
    print('COMPLETE_E046_LOCAL_DELIVERY', output)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--aorta', type=Path, required=True); p.add_argument('--phantom', type=Path, required=True)
    p.add_argument('--validation', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); finalize(a.aorta.resolve(), a.phantom.resolve(), a.validation.resolve(), a.output.resolve())
