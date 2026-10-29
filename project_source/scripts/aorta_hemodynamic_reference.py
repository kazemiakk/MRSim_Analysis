"""E-027 source-preserving aortic reference and saved Pulseq timing audit."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree


def read_pulseq(path):
    sections = {}; name = None
    for raw in Path(path).read_text(encoding='utf-8-sig').splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('[') and line.endswith(']'):
            name = line[1:-1]; sections.setdefault(name, [])
        elif line.startswith('extension '):
            name = line; sections.setdefault(name, [])
        elif name is not None:
            sections[name].append(line.split())
    return sections


def temporal_weights(times_s, *, period_s, frame_dt_s=.01, frames=72, start_frame=18):
    """Mean coefficients of documented periodic linear source interpolation."""
    times = np.asarray(times_s, dtype=float)
    if times.size == 0 or not np.isfinite(times).all() or period_s <= 0:
        raise ValueError('Finite sample times and positive period required')
    phase = (np.mod(times, period_s) / period_s * frames + start_frame) % frames
    left = np.floor(phase).astype(int); alpha = phase - left
    weights = np.bincount(left.ravel(), weights=(1-alpha).ravel(), minlength=frames)
    weights += np.bincount(((left+1) % frames).ravel(), weights=alpha.ravel(), minlength=frames)
    return weights / times.size


def sequence_timing(path, period_s):
    """Read Pulseq1.5 blocks, linked LABELSETs and every ADC sample center.

    RF support starts at its recorded delay. Seam check covers RF onset through
    final ADC center; it does not certify the legacy simulator's reset execution.
    """
    s = read_pulseq(path)
    defs = {row[0]: row[1:] for row in s['DEFINITIONS']}
    raster = float(defs['BlockDurationRaster'][0]); bins = int(defs['CardiacBins'][0])
    extension = {int(r[0]): tuple(map(int,r[1:4])) for r in s.get('EXTENSIONS', [])}
    setters = {int(r[0]): (r[2],int(r[1])) for key,rows in s.items() if key.startswith('extension LABELSET ') for r in rows}
    adc = {int(r[0]): (int(r[1]), float(r[2])*1e-9, float(r[3])*1e-6) for r in s['ADC']}
    rf = {int(r[0]): float(r[6])*1e-6 for r in s['RF']}
    labels = {}; ticks = 0; last_rf = None; records = []
    for row in s['BLOCKS']:
        number, duration, rf_id, gx, gy, gz, adc_id, ext_id = map(int,row[:8])
        start = ticks*raster
        chain = []; visited = set()
        while ext_id:
            if ext_id in visited:
                raise ValueError('Cyclic Pulseq extension list')
            visited.add(ext_id); kind, ref, ext_id = extension[ext_id]
            if kind != 1:
                raise ValueError('Unimplemented Pulseq extension kind')
            chain.append(setters[ref])
        for key,value in reversed(chain):
            labels[key] = value
        if rf_id:
            last_rf = start + rf[rf_id]
        if adc_id:
            if last_rf is None or not all(k in labels for k in ('PHS','LIN','PAR')):
                raise ValueError('ADC missing RF or spatial/phase labels')
            n,dwell,delay = adc[adc_id]
            first = start+delay+.5*dwell; last = start+delay+(n-.5)*dwell
            crosses = int(np.floor(last_rf/period_s)) != int(np.floor(last/period_s))
            records.append(dict(block=number,bin=labels['PHS']+1,line=labels['LIN'],partition=labels['PAR'],
                                rf_start_s=last_rf,adc_first_s=first,adc_last_s=last,n=n,
                                dwell_s=dwell,seam_crossing=crosses))
        ticks += duration
    nx,ny,nz = (int(defs[k][0]) for k in ('Nx','Ny','Nz'))
    tuples = {(r['bin'],r['line'],r['partition']) for r in records}
    if len(records) != bins*ny*nz or len(tuples)!=len(records):
        raise ValueError('Incomplete/duplicate saved Pulseq k-space labels')
    weights = []; report = []
    for b in range(1,bins+1):
        rows = [r for r in records if r['bin']==b]
        times = np.concatenate([r['adc_first_s'] + np.arange(r['n'])*r['dwell_s'] for r in rows])
        phases = np.mod(times,period_s)
        weights.append(temporal_weights(times,period_s=period_s))
        angle = np.angle(np.mean(np.exp(2j*np.pi*phases/period_s))) % (2*np.pi)
        report.append(dict(bin=b,profiles=len(rows),samples=int(times.size),
                           phase_min_s=float(phases.min()),phase_max_s=float(phases.max()),
                           phase_circular_mean_s=float(angle*period_s/(2*np.pi)),
                           nominal_midpoint_s=(b-.5)*period_s/bins,
                           seam_crossing_profiles=sum(r['seam_crossing'] for r in rows)))
    return {'path': str(path), 'profiles': len(records), 'sequence_duration_s': ticks*raster,
            'RR_definition_s': float(defs['RR'][0]), 'period_used_s': period_s,
            'bin_timing': report, 'RF_to_ADC_seam_crossings': sum(r['seam_crossing'] for r in records),
            'reset_runtime_certified': False}, np.array(weights), records


@dataclass(frozen=True)
class AortaGrid:
    shape: tuple = (32,32,72)
    fov: tuple = (.28,.128,.128)
    center: tuple = (.038389891386032104,.015447176992893219,-.023723918944597244)
    order: tuple = (2,0,1)
    offset: float = 0.0

    @property
    def spacing(self):
        return np.asarray(self.fov)/np.asarray(self.shape[::-1])

    def axes(self):
        return tuple((np.arange(n)-n//2+self.offset)*f/n for n,f in zip(self.shape[::-1],self.fov))

    def points(self):
        x,y,z = self.axes()
        zz,yy,xx = np.meshgrid(z,y,x,indexing='ij')
        return np.stack((xx,yy,zz),axis=-1)

    def to_source(self, points):
        source = np.empty_like(np.asarray(points,dtype=float))
        source[...,self.order] = points
        return source + self.center


@dataclass
class PointLookup:
    grid: AortaGrid
    flat: np.ndarray
    nodes: np.ndarray
    weights: np.ndarray
    supported: np.ndarray
    sample_shape: tuple

    @property
    def fraction(self):
        out = np.zeros(np.prod(self.grid.shape),dtype=np.float32)
        out[self.flat] = self.supported.mean(axis=1)
        return out.reshape(self.grid.shape)

    def evaluate(self, node_values):
        result = np.full((len(node_values),np.prod(self.grid.shape),3),np.nan,dtype=np.float32)
        counts = self.supported.sum(axis=1)
        keep = counts>0
        for t,values in enumerate(node_values):
            for first in range(0,len(self.flat),2048):
                end = min(first+2048,len(self.flat))
                samples = np.einsum('vqk,vqkc->vqc',self.weights[first:end],values[self.nodes[first:end]],optimize=True)
                means = samples.sum(axis=1)/np.maximum(counts[first:end,None],1)
                result[t,self.flat[first:end]] = means
        result[:,self.flat[~keep]] = np.nan
        return result.reshape(len(node_values),*self.grid.shape,3)


def build_lookup(tree, grid, sample_shape=(2,2,2), support_radius=.002322211721826266):
    """Numerical tracker support, explicitly not a physical boundary estimate."""
    if len(sample_shape)!=3 or any(n<1 for n in sample_shape):
        raise ValueError('Positive sampling counts required')
    centers = grid.points().reshape(-1,3)
    distance,_ = tree.query(grid.to_source(centers),k=1,workers=-1)
    candidates = np.flatnonzero(distance<=support_radius+np.linalg.norm(grid.spacing)/2)
    offset_axes = [(np.arange(n)+.5)/n-.5 for n in sample_shape]
    offsets = np.stack(np.meshgrid(*offset_axes,indexing='ij'),axis=-1).reshape(-1,3)*grid.spacing
    points = grid.to_source(centers[candidates,None,:]+offsets)
    distances,nodes = tree.query(points.reshape(-1,3).astype(np.float32),k=8,workers=-1)
    shape = (len(candidates),len(offsets),8)
    distances,nodes = distances.reshape(shape),nodes.reshape(shape)
    supported = distances[...,0]<=support_radius
    epsilon = max(support_radius*1e-6,np.finfo(np.float32).eps)
    weights = 1/np.maximum(distances,epsilon)**2
    weights /= weights.sum(axis=-1,keepdims=True)
    weights *= supported[...,None]
    return PointLookup(grid,candidates,nodes,weights.astype(np.float32),supported,tuple(sample_shape))


def temporal_node_values(velocities, weights, *, order=(2,0,1), scale=1.0):
    if velocities.ndim!=3 or velocities.shape[1]!=3 or velocities.shape[2]!=weights.shape[1]:
        raise ValueError('Source/time weight shape mismatch')
    output = np.empty((len(weights),len(velocities),3),dtype=np.float32)
    for start in range(0,len(velocities),65536):
        block = np.asarray(velocities[start:start+65536],dtype=np.float32)
        values = np.einsum('ncf,tf->tnc',block,np.asarray(weights,dtype=np.float32),optimize=True)
        output[:,start:start+len(block)] = values[...,order]*scale
    return output
