"""Focused regression checks for the independent reference, not result claims."""
import ast
from pathlib import Path
import numpy as np
from analyze_physics import rf_affine,phase_moments

def main():
    checks=0
    for path in Path(__file__).parent.glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'));checks+=1
    # Exact polynomial free-precession integral: constant G and linear x(t).
    template=dict(adc=np.array([.001,.002]),t=np.array([0.,.001,.002]),g=np.array([[.01,0,0]]*3))
    ordinary,moving=phase_moments(template,0.,0.,np.array([0.,.01]),np.array([0.,.01]),.01)
    np.testing.assert_allclose(ordinary[:,0],[1e-5,2e-5],rtol=0,atol=1e-18);checks+=1
    np.testing.assert_allclose(moving,[5e-9,2e-8],rtol=0,atol=1e-19);checks+=1
    # A periodic seam is explicitly split; the path here is triangular.
    template=dict(adc=np.array([.002]),t=np.array([0.,.002]),g=np.array([[.01,0,0]]*2))
    _,moving=phase_moments(template,.009,0.,np.array([0.,.005,.01]),np.array([0.,.005,0.]),.01)
    np.testing.assert_allclose(moving,[1e-8],rtol=0,atol=1e-19);checks+=1
    # Finite RF: rect pulse with zero gradient, negligible relaxation.
    gamma=42.577478518e6;duration=.0002;angle=np.deg2rad(8)
    b=angle/(2*np.pi*gamma*duration)
    template=dict(t=np.array([0.,duration]),b1=np.array([b,b],complex),g=np.zeros((2,3)),frequency=np.zeros(2))
    result=rf_affine(template,np.array([0.]),1.,1e15,1e15,gamma)
    np.testing.assert_allclose(result['az'],np.cos(angle),rtol=0,atol=2e-12);checks+=1
    np.testing.assert_allclose(result['axy'],-1j*np.sin(angle),rtol=0,atol=2e-12);checks+=1
    print(dict(status='PASS_INDEPENDENT_REFERENCE_SOFTWARE_TESTS',checks=checks),flush=True)

if __name__=='__main__':main()
