"""Re-layout the saved two-case Bland-Altman points with A/B case titles.

No source evaluation, new samples, filtering changes or registered metrics.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SOURCE = Path('D:/MRSim/outputs/presentation_hemodynamics_2026-10-25/bland_altman_v1')
OUT = Path('D:/MRSim/outputs/presentation_hemodynamics_2026-10-25/bland_altman_uniform_markers_v5')
COL = dict(v3='#2a78d6', v4='#1baf7a', k20='#2a78d6', k8='#eb6834')
MARKER_AREA_PT2 = 36.0  # A single 6-point circular marker, fixed for all series.


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    if OUT.exists() or OUT.with_name(OUT.name+'.building').exists():
        raise FileExistsError(OUT)
    expected = {
        SOURCE/'MANIFEST.json': '4ef2c5d0da7e53cfcb4729e19b3081a3479c35b109ff51fc46d2beb0a2e5047d',
        SOURCE/'bland_altman_koma_cfd.png': 'c6a672a7569b08dda8c2c4f471202d6ac28ae8707847052078651590ac39229a',
    }
    for p, h in expected.items():
        assert sha(p) == h, str(p)
    source = json.loads((SOURCE/'MANIFEST.json').read_text())
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 22,
        'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig = plt.figure(figsize=(22, 15), layout='constrained', facecolor='white')
    rows = fig.subfigures(2, 1)
    checks = []
    marker_checks = []
    for case, title, row, arms in [
        ('aorta', 'A  Aorta', rows[0], [('k8', '8 bins'), ('k20', '20 bins')]),
        ('phantom', 'B  Phantom', rows[1], [('v4', 'VENC 4 m/s'), ('v3', 'VENC 3 m/s')]),
    ]:
        row.suptitle(title, fontsize=28, fontweight='bold')
        axes = row.subplots(1, 2)
        for quantity, unit, ax, heading in zip(('flow', 'peak'), ('mL/s', 'm/s'), axes,
                ('Flow', 'Peak velocity')):
            for arm, label in arms:
                points = source['points'][case][arm]
                ref = np.asarray([p['cfd_'+quantity] for p in points], float)
                koma = np.asarray([p['koma_'+quantity] for p in points], float)
                finite = np.isfinite(ref) & np.isfinite(koma)
                ref, koma = ref[finite], koma[finite]
                mean, diff = (koma+ref)/2, koma-ref
                old = source['stats'][case+'_'+quantity][arm]
                assert len(diff) == old['n']
                # Validation of existing, saved statistics, not a new analysis.
                assert np.isclose(diff.mean(), old['bias'], rtol=0, atol=1e-12)
                assert np.isclose(diff.std(ddof=1), old['sd'], rtol=0, atol=1e-12)
                bias, sd = old['bias'], old['sd']
                points_artist = ax.scatter(mean, diff, s=MARKER_AREA_PT2, marker='o', color=COL[arm], alpha=.55,
                    edgecolor='none', label=label)
                assert np.array_equal(np.asarray(points_artist.get_offsets()), np.column_stack((mean, diff)))
                assert np.array_equal(points_artist.get_sizes(), [MARKER_AREA_PT2])
                marker_checks.append(dict(case=case, quantity=quantity, arm=arm,
                    marker='circle', area_pt2=float(points_artist.get_sizes()[0]), alpha=points_artist.get_alpha()))
                for y, ls in ((bias, '-'), (old['loa_high'], (0, (4,3))), (old['loa_low'], (0, (4,3)))):
                    ax.axhline(y, color=COL[arm], lw=1.4, ls=ls)
                checks.append(dict(case=case, quantity=quantity, arm=arm, n=len(diff),
                    mean_difference_sha256=hashlib.sha256(np.column_stack((mean, diff)).tobytes()).hexdigest(),
                    saved_stats_and_points_unchanged=True))
            ax.axhline(0, color='#888888', lw=.7)
            ax.set_title(heading, fontsize=22)
            ax.set_xlabel(f'Mean of Koma and CFD ({unit})', fontsize=20)
            ax.set_ylabel(f'Koma − CFD ({unit})', fontsize=20)
            ax.tick_params(labelsize=17)
            ax.grid(alpha=.25)
            legend = ax.legend(fontsize=19, frameon=False, loc='upper left', ncol=2,
                markerscale=1.0, scatterpoints=1, handletextpad=.35, columnspacing=.9)
            for handle in legend.legend_handles:
                assert np.array_equal(handle.get_sizes(), [MARKER_AREA_PT2])
                assert handle.get_alpha() == .55
    fig.suptitle('Bland–Altman agreement: Koma vs CFD (same voxels, same acquisition windows)', fontsize=24)
    fig.supxlabel('Solid: bias; dashed: 95% limits of agreement (±1.96 SD). Correlated section/phase samples; descriptive, same-source internal consistency.\n'
        'Aorta: E-062/E-027, 4 sections × 20/8 phases. Phantom: E-060/E-061, 24 sections × 30 phases. '
        'Original two-case data retained; no measured MRI included.', fontsize=14)
    stage = OUT.with_name(OUT.name+'.building')
    stage.mkdir(parents=True)
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(stage/('bland_altman_koma_cfd.'+ext), dpi=300, facecolor='white')
    fig.savefig(stage/'preview.png', dpi=75, facecolor='white')
    plt.close(fig)
    with Image.open(stage/'bland_altman_koma_cfd.png') as im:
        assert im.size == (6600, 4500)
        im.verify()
    assert (stage/'bland_altman_koma_cfd.pdf').read_bytes().startswith(b'%PDF-')
    for p, h in expected.items():
        assert sha(p) == h
    report = dict(status='PASS_SAVED_POINTS_AND_STATS_IDENTICAL',
        created_utc=datetime.now(timezone.utc).isoformat(), script_sha256=sha(__file__),
        source_manifest_sha256=expected[SOURCE/'MANIFEST.json'], source=str(SOURCE),
        case_titles=['A  Aorta', 'B  Phantom'], row_order=['aorta', 'phantom'],
        checks=checks, stats=source['stats'], new_data=False, new_metrics=False,
        legend_labels={'aorta': ['8 bins', '20 bins'], 'phantom': ['VENC 4 m/s', 'VENC 3 m/s']},
        font_sizes_pt={'legend':19,'ticks':17,'axis_labels':20,'subplot_titles':22,'case_titles':28,'main_title':24,'footer':14},
        title_edit='Removed Section from all four panel titles: Flow / Peak velocity',
        marker_checks=marker_checks, marker_area_pt2=MARKER_AREA_PT2, marker_nominal_diameter_pt=6,
        legend_markerscale=1.0, legend_marker_size_verified_equal_to_data=True,
        source_scope='Earlier two-case E060/E061 phantom and E062/E027 aorta; not latest dp122 E065')
    (stage/'REPORT.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    (stage/'README.md').write_text('# Concise legends and larger text\n\n'
        'Marker-only v5 revision of bland_altman_final_titles_v4. All eight scatter series and all legend '
        'markers use the same 36pt^2 circular marker (nominal diameter6pt), alpha0.55, no outline. '
        'Legend markerscale is1.0, not the earlier2.2. Explicit size/alpha assertions cover all data and legend artists. '
        'All other visible labels, fonts, colors, lines and plotted results unchanged. '
        'Legends shortened to protocol labels only; '
        'existing bias/LoA statistics remain in REPORT.json and are still drawn as unchanged lines. '
        'Legend19pt, ticks17pt, axes20pt, plot titles22pt, case titles28pt and footer14pt. '
        'A Aorta is first; B Phantom second. Both flow and peak-velocity panels retained per case. '
        'Original saved two-case data, arms, finite-value selection, statistics, units and colors unchanged. '
        'This re-layout uses only the previous figure manifest; no CFD/MRI evaluation is rerun. '
        'This is the earlier E060/E061 phantom and E062/E027 aorta figure, NOT the newer phantom-only dp122 E065. '
        'Points are correlated simulated sections/phases, not independent subjects. '
        'Same-source internal consistency, not independent clinical validation. Prior files preserved.\n', encoding='utf-8')
    (stage/'MANIFEST.json').write_text(json.dumps({p.name: sha(p) for p in stage.iterdir() if p.is_file()}, indent=2)+'\n')
    stage.rename(OUT)
    print(json.dumps(dict(status=report['status'], png=str(OUT/'bland_altman_koma_cfd.png')), indent=2))


if __name__ == '__main__':
    main()
