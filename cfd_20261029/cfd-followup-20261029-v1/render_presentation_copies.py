"""Remove only the two requested video annotations; retain limitations in documentation."""
import sys, subprocess
from pathlib import Path
SOURCE = Path(r'D:\FluentAgent\automation\20261029-cycle-cohorts-and-regional-metrics-v1')
sys.path.insert(0, str(SOURCE))
import render_cycle_cohorts as renderer
import normalize_videos
from common_followup import save, sha, PROJECT

OUT = PROJECT/'results/20261029-cycle-cohorts-and-regional-metrics-v1/presentation-clean-v1'
REMOVED = ['First-cycle CFD replay; not 10 solved cycles', '* Unconverged paths: exploratory']


def main():
    OUT.mkdir(exist_ok=False)
    original = renderer.pv.Plotter.add_text
    removed_calls = []

    def add_text(plotter, text, *args, **kwargs):
        if text in REMOVED:
            removed_calls.append(text)
            return None
        return original(plotter, text, *args, **kwargs)

    renderer.pv.Plotter.add_text = add_text
    # Remove the footnote marker whose on-screen footnote is being removed.
    renderer.LABELS = [s.rstrip('*') for s in renderer.LABELS]
    try:
        renderer.run(OUT/'videos-cycle-colors-v3')
    finally:
        renderer.pv.Plotter.add_text = original
    assert removed_calls.count(REMOVED[0]) == 3 and removed_calls.count(REMOVED[1]) == 1
    normalize_videos.main(OUT)
    final = OUT/'videos-final'
    movie = final/'Three-cases-release-cycle-colored.mp4'
    for t, label in [(7, 'early'), (80, 'late')]:
        subprocess.run([str(renderer.FF), '-v', 'error', '-n', '-ss', str(t), '-i', str(movie),
                        '-frames:v', '1', str(final/('decoded-'+label+'.png'))],
                       check=True, capture_output=True, timeout=60, creationflags=subprocess.CREATE_NO_WINDOW)
    save(OUT/'presentation-edit.json', {
        'status': 'completed_and_fully_decoded',
        'removed_annotations': REMOVED,
        'removed_annotation_instances': len(removed_calls),
        'case_footnote_asterisk_removed': True,
        'particle_positions_or_births_changed': False,
        'original_outputs_preserved': True,
        'new_Fluent_sessions': 0,
        'source_renderer_sha256': sha(SOURCE/'render_cycle_cohorts.py'),
        'limitations_documentation': str(OUT/'README.txt'),
    })
    with (OUT/'README.txt').open('x', encoding='utf-8') as f:
        f.write('Presentation copies: the two requested on-screen notes and their footnote marker were removed.\n'
                'Use videos-final/, not the untrimmed intermediate render. Original videos were preserved.\n'
                'The cycle colors, particle tracks, anatomy, timing, legend and simulation data are unchanged.\n\n'
                'Scientific limitations remain: ten repeats of one first-cycle CFD template, not ten solved cycles.\n'
                'CFD periodicity is unvalidated; individualized production trajectories are unconverged.\n'
                'The additional finer-step test timed out and did not establish convergence.\n'
                'These are exploratory visualizations, not validated physiological residence/washout results.\n'
                'See the parent README-reviewed.md for complete methods and limitations.\n')


if __name__ == '__main__':
    main()
