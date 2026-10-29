"""Offline-only follow-up. Never opens a Fluent session or writes source datasets."""
from pathlib import Path
import csv, hashlib, json
import numpy as np

PROJECT = Path(r'D:\FluentAgent')
RUN = PROJECT / 'runs/20261023-070441-568099-systole-first-checkpoint-recovery-v1'
TRACKS = PROJECT / 'results/20261022-systole-first-particles-10cycles-v1'
KINDS = ['patient_specific', 'individualized', 'generic']
LABELS = ['Patient-specific MRI', 'Individualized generic*', 'Population average']
PERIOD = .868
START = .005
CENTER_MM = np.array([-2., 29., -23.])
RADII_MM = np.array([4.5, 4.5, 6.5])
COLORS = ['#2563eb', '#d97706', '#d946ef', '#0d9488', '#dc2626',
          '#7c3aed', '#65a30d', '#0891b2', '#be185d', '#475569']


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, data):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(data, f, indent=2, allow_nan=False)


def csvsave(path, rows):
    assert rows
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('x', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def selected():
    selection = load(TRACKS / 'selected-tracks.json')
    return {k: TRACKS / selection[k]['directory'] for k in KINDS}


def in_ellipsoid(points_m):
    return np.sum(((np.asarray(points_m) * 1000 - CENTER_MM) / RADII_MM)**2, axis=-1) <= 1


def release_cycle(birth):
    c = np.floor((np.asarray(birth) - START + 1e-9) / PERIOD).astype(int) + 1
    assert np.all((c >= 1) & (c <= 10)), 'Birth outside the ten release cycles'
    return c


def periodic_average(values, times):
    """Explicitly assumed 5-ms closing interval; not proof of periodic CFD."""
    t = np.r_[times, times[0] + PERIOD]
    a = np.concatenate([values, values[:1]], axis=0)
    return np.trapezoid(a, t, axis=0) / PERIOD


def interval_inside(p0, p1):
    """Fractional intersection of each saved straight segment with the ellipsoid."""
    p = (np.asarray(p0) * 1000 - CENTER_MM) / RADII_MM
    d = (np.asarray(p1) - np.asarray(p0)) * 1000 / RADII_MM
    a = np.sum(d*d, axis=1)
    b = 2 * np.sum(p*d, axis=1)
    c = np.sum(p*p, axis=1) - 1
    disc = b*b - 4*a*c
    moving = a > 1e-24
    valid = moving & (disc >= 0)
    lo = np.ones(len(p)); hi = np.zeros(len(p))
    root = np.sqrt(np.maximum(disc[valid], 0))
    lo[valid] = np.maximum(0, (-b[valid]-root)/(2*a[valid]))
    hi[valid] = np.minimum(1, (-b[valid]+root)/(2*a[valid]))
    still = ~moving & (c <= 0)
    lo[still] = 0; hi[still] = 1
    return lo, hi, hi > lo + 1e-12


def km(times, event):
    """Unweighted descriptive first-exit survival with right censoring."""
    times = np.round(np.asarray(times), 9); event = np.asarray(event, bool)
    rows = [{'age_s': 0., 'survival': 1., 'at_risk': len(times), 'exits': 0, 'censored': 0}]
    s = 1.
    for t in np.unique(times):
        risk = int(np.sum(times >= t - 1e-12))
        hit = np.abs(times-t) < 1e-12
        d = int(np.sum(hit & event)); c = int(np.sum(hit & ~event))
        if risk:
            s *= 1-d/risk
        rows.append({'age_s': float(t), 'survival': float(s), 'at_risk': risk, 'exits': d, 'censored': c})
    return rows


def self_tests():
    # Unit ellipsoid, known outside/outside traversal and contained segment.
    p = (CENTER_MM + RADII_MM*np.array([[-2.,0,0],[0,0,0],[0,2,0]]))/1000
    q = (CENTER_MM + RADII_MM*np.array([[2.,0,0],[.5,0,0],[1,2,0]]))/1000
    lo, hi, ok = interval_inside(p,q)
    assert np.allclose(lo[:2], [.25,0]) and np.allclose(hi[:2], [.75,1]) and list(ok)==[True,True,False]
    s = km([1,2,3],[True,False,True])
    assert abs(s[1]['survival']-2/3)<1e-12 and s[-1]['survival']==0
    assert list(release_cycle([START, START+PERIOD, START+9*PERIOD])) == [1,2,10]
    assert abs(periodic_average(np.ones(95), np.linspace(.005,.868,95))-1)<1e-12
    return {'segment_intersections':'passed', 'right_censoring':'passed', 'cycle_boundaries':'passed', 'time_weights':'passed'}
