"""Flux-weighted inferred-cap recycling for approximate point-cloud pilots.

Not a replacement for native connected boundary geometry. The tracker supplies
the exit count; this class controls where replacements enter, not that count.
"""
import numpy as np


class PointCloudInletReseeder:
    def __init__(self, vertices, triangles, nodal_velocity, inward_normal, *,
                 frame_dt, start_frame, dt, rng=None, inward_epsilon=1e-6):
        self.vertices=np.asarray(vertices,dtype=float)
        self.triangles=np.asarray(triangles,dtype=np.int64)
        self.velocity=np.asarray(nodal_velocity,dtype=float)
        self.normal=np.asarray(inward_normal,dtype=float)
        if self.vertices.ndim!=2 or self.vertices.shape[1]!=3 or not np.isfinite(self.vertices).all():
            raise ValueError('Finite cap vertices (N,3) required')
        if self.triangles.ndim!=2 or self.triangles.shape[1]!=3 or not len(self.triangles):
            raise ValueError('Nonempty triangles required')
        if self.triangles.min()<0 or self.triangles.max()>=len(self.vertices):
            raise ValueError('Triangle index outside vertices')
        if self.velocity.shape[:2]!=(len(self.vertices),3) or self.velocity.ndim!=3 or not np.isfinite(self.velocity).all():
            raise ValueError('Finite nodal velocity (N,3,frames) required')
        if self.normal.shape!=(3,) or not np.isclose(np.linalg.norm(self.normal),1):
            raise ValueError('Unit inward normal required')
        if frame_dt<=0 or dt<=0 or inward_epsilon<=0:
            raise ValueError('Positive timing and inward epsilon required')
        self.frame_dt=float(frame_dt);self.dt=float(dt);self.start_frame=int(start_frame)
        self.nframes=self.velocity.shape[2];self.epsilon=float(inward_epsilon)
        self.rng=np.random.default_rng() if rng is None else rng
        xyz=self.vertices[self.triangles]
        self.a=xyz[:,0];self.e1=xyz[:,1]-xyz[:,0];self.e2=xyz[:,2]-xyz[:,0]
        self.area=np.linalg.norm(np.cross(self.e1,self.e2),axis=1)/2
        if np.any(self.area<=0):raise ValueError('Degenerate cap triangle')
        self.vn=np.einsum('ncf,c->nf',self.velocity,self.normal)[self.triangles].mean(axis=1)
        self.calls=0;self.injected=0

    def inflow(self,t):
        # Midpoint of the actual integration step: avoids substituting area
        # weighting at an instantaneously zero source frame.
        phase=(float(t)+self.dt/2)/self.frame_dt+self.start_frame
        left=int(np.floor(phase))%self.nframes;alpha=phase-np.floor(phase)
        vn=np.maximum((1-alpha)*self.vn[:,left]+alpha*self.vn[:,(left+1)%self.nframes],0)
        return vn,vn*self.area

    def reseed(self,n,t):
        if n<=0:return np.empty((0,3),dtype=np.float32)
        vn,flux=self.inflow(t);total=float(flux.sum())
        if total<=0:raise ValueError('No positive inlet flux: cannot recycle exit particles without a reservoir')
        triangles=np.searchsorted(np.cumsum(flux),self.rng.random(n)*total,side='right')
        a=self.rng.random(n);b=self.rng.random(n);over=a+b>1;a[over]=1-a[over];b[over]=1-b[over]
        points=self.a[triangles]+a[:,None]*self.e1[triangles]+b[:,None]*self.e2[triangles]
        depth=self.epsilon+self.rng.random(n)*vn[triangles]*self.dt
        self.calls+=1;self.injected+=n
        return (points+depth[:,None]*self.normal).astype(np.float32)
