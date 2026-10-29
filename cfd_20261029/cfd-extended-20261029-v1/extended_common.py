"""Paths, immutable sources and finite-volume geometry for exploratory offline analysis."""
from pathlib import Path
import importlib.util,sys
import numpy as np
import h5py,pyvista as pv
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
BASE=Path(r'D:\FluentAgent\automation\20261029-cycle-cohorts-and-regional-metrics-v1')
sys.path.insert(0,str(BASE))
from common_followup import *
PREVIOUS=PROJECT/'results/20261029-cycle-cohorts-and-regional-metrics-v1'


def wall_reader():
    path=PROJECT/'automation/20261023-systole-first-wss-v1/wss_compare.py'
    spec=importlib.util.spec_from_file_location('native_wall_reader',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod


class Geometry:
    def __init__(self):
        self.cfg={k:load(RUN/k/'input-configuration.json') for k in KINDS}
        self.steps={k:load(RUN/k/'progress.json')['accepted_steps'] for k in KINDS}
        self.times=np.array([x['time_s'] for x in self.steps['patient_specific']]);self.times[-1]=PERIOD
        self.grid=pv.read(PROJECT/'visualizations/20261002-measured-ica-velocity-context-v6/original-fluid-tetrahedra-mm.vtu')
        self.tet=self.grid.cells.reshape(-1,5)[:,1:];self.n=self.grid.n_cells
        self.xyz=self.grid.points/1000;self.center=self.xyz[self.tet].mean(1)
        mapping=np.load(PREVIOUS/'comparisons/fluid-ROI-mapping.npz')
        self.volume=mapping['volume_m3'];inside=mapping['ellipsoid_mask']
        assert np.array_equal(inside,in_ellipsoid(self.center))
        self.masks={'sac_ellipsoid':inside,'surrounding_fluid':~inside,'whole_fluid':np.ones(self.n,bool)}
        self.wall=wall_reader();_,self.zones=self.wall.geometry(self.cfg['patient_specific']['source_case'])
        with h5py.File(self.cfg['patient_specific']['source_case'],'r') as f:
            mesh=f['meshes/1'];self.faces=mesh['faces/nodes/1/nodes'][:].reshape(-1,3).astype(np.int64)-1
            self.c0=mesh['faces/c0/1'][:].astype(np.int64)-1;self.c1=mesh['faces/c1/1'][:].astype(np.int64)-1
            z=mesh['faces/zoneTopology'];self.face_zones={name:(int(lo)-1,int(hi)) for name,lo,hi in zip(z['name'][0].decode().split(';'),z['minId'][:],z['maxId'][:])}
        self.ni=len(self.c1);self.fc=self.xyz[self.faces].mean(1)
        self.build_gradient()
        m0=inside[self.c0[:self.ni]];m1=inside[self.c1]
        self.interface=np.flatnonzero(m0!=m1)
        self.roi_flux_sign=np.where(m0[self.interface],1.,-1.)

    def build_gradient(self):
        # Each original tetrahedron has four adjacent cells or native boundary faces.
        owners=np.r_[self.c0,self.c1];ids=np.r_[np.arange(len(self.c0)),np.arange(self.ni)]
        order=np.argsort(owners,kind='stable');assert np.all(np.bincount(owners)==4)
        self.face_for_cell=ids[order].reshape(-1,4)
        adjacent=np.r_[np.where(np.arange(len(self.c0))<self.ni,
                                  np.r_[self.c1,np.zeros(len(self.c0)-self.ni,dtype=int)],
                                  self.n+np.arange(len(self.c0))-self.ni),self.c0[:self.ni]]
        self.neighbor=adjacent[order].reshape(-1,4)
        centers=np.vstack([self.center,self.fc[self.ni:]])
        dx=centers[self.neighbor]-self.center[:,None,:]
        weight=1/np.sum(dx*dx,axis=2)
        mat=np.einsum('nki,nkj,nk->nij',dx,dx,weight)
        eig=np.linalg.eigvalsh(mat);assert np.min(eig)>1e-10
        inv=np.linalg.inv(mat)
        self.coeff=np.einsum('nij,nkj,nk->nki',inv,dx,weight)
        # Independent exact affine field including its analytical boundary-face values.
        ix=np.arange(0,self.n,977);a=np.array([[2.,3.,-1.],[-4.,1.,.5],[.3,-.4,1.7]])
        values=centers@a.T+np.array([.4,-.2,1.])
        du=values[self.neighbor[ix]]-values[ix,None,:]
        grad=np.einsum('nki,nkj->nij',du,self.coeff[ix])
        self.gradient_error=float(np.max(np.abs(grad-a)))
        assert self.gradient_error<1e-8
        del centers,values,dx,weight,mat,inv

    def gradient(self,v,boundary):
        ext=np.vstack([v,boundary]);result=np.empty((self.n,3,3))
        for a in range(0,self.n,150000):
            b=min(a+150000,self.n);du=ext[self.neighbor[a:b]]-v[a:b,None,:]
            result[a:b]=np.einsum('nki,nkj->nij',du,self.coeff[a:b])
        return result

    def fields(self,kind,j):
        step=self.steps[kind][j]
        with h5py.File(step['output_stem']+'.dat.h5','r') as f:
            p=f['results/1/phase-1'];c=p['cells'];face=p['faces']
            v=np.column_stack([c[n+'/1'][:].reshape(-1) for n in ['SV_U','SV_V','SV_W']])
            boundary=np.column_stack([face[n+'/1'][:].reshape(-1) for n in ['SV_U','SV_V','SV_W']])
            for n in ['SV_U','SV_V','SV_W']:assert int(face[n+'/1'].attrs['minId'][0])-1==self.ni
            flux=face['SV_FLUX/1'][:]
        assert v.shape==(self.n,3) and boundary.shape==(len(self.c0)-self.ni,3)
        assert np.isfinite(v).all() and np.isfinite(flux).all()
        return v,boundary,flux

    def reference_bands(self):
        alltri=np.concatenate([z['tri'] for z in self.zones.values()])
        edges=np.vstack([alltri[:,[0,1]],alltri[:,[1,2]],alltri[:,[2,0]]]);edges=np.unique(np.sort(edges,axis=1),axis=0)
        lengths=np.linalg.norm(self.xyz[edges[:,0]]-self.xyz[edges[:,1]],axis=1)*1000
        graph=coo_matrix((np.r_[lengths,lengths],(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(len(self.xyz),len(self.xyz))).tocsr()
        distances=[]
        for name in ['inlet_b1','inlet_b2']:
            lo,hi=self.face_zones[name];nodes=np.unique(self.faces[lo:hi])
            d=dijkstra(graph,directed=False,indices=nodes,min_only=True)
            distances.append(d[self.zones['wall']['tri']].mean(1))
        result={}
        for bounds in [(3,8),(5,10)]:
            masks=[(d>=bounds[0])&(d<bounds[1]) for d in distances]
            assert all(m.sum()>20 for m in masks)
            result['ICA_wall_%d_to_%d_mm'%bounds]={'right':masks[0],'left':masks[1],'pooled':masks[0]|masks[1]}
        return result


def weights(times):
    edges=np.r_[times,times[0]+PERIOD];dt=np.diff(edges)
    return (np.r_[dt[-1],dt[:-1]]+dt)/2
