import h5py
import numpy as np
import pytest
from scripts.diagnose_phantom_signal import read_complex,signal_decomposition,voxel_indices
from scripts.audit_phantom_reference import catmull_weights,CenteredFFTGrid,fft_controls,averaged_complex
from mrsimtracks.io import _interp_time
from scripts.diagnose_phantom_encoding_kernel import central_reset_exclusion,phase_kernel,quadrature
from pathlib import Path


def test_compound_complex_roundtrip(tmp_path):
    p=tmp_path/'values.h5';a=np.zeros(4,dtype=[('re','f4'),('im','f4')]);a['re']=2;a['im']=-3
    with h5py.File(p,'w') as f:f['a']=a
    with h5py.File(p) as f:np.testing.assert_array_equal(read_complex(f['a']),np.full(4,2-3j))


def test_nonfinite_complex_rejected(tmp_path):
    with h5py.File(tmp_path/'bad.h5','w') as f:
        f['a']=np.array([1+1j,np.nan+0j])
        with pytest.raises(ValueError):read_complex(f['a'])


def test_signal_decomposition_is_mutually_exclusive():
    a=np.ones((2,1,1,72),complex);images={e:a.copy() for e in ('REF','X','Y','Z')}
    images['REF'][0,0,0,0]=.01;images['X'][1,0,0,1]=.01
    fraction=np.ones((1,1,72));area=np.ones((1,1))
    report,mask,_,_=signal_decomposition(images,fraction,area)
    assert mask.sum()==70
    assert report['fixed_all_bin_all_encoding_volume_fraction']==pytest.approx(70/72)
    assert sum(r['volume_fraction'] for r in report['failure_patterns'])==pytest.approx(1)
    assert report['REF_failure_volume_fraction']==pytest.approx(1/72)
    assert report['REF_pass_but_encoding_failure_fraction']==pytest.approx(1/72)


def test_nearest_center_indices_distinguish_conventions():
    p=np.array([[0.,0.,0.],[.004,0,0]])
    fft,valid=voxel_indices(p,(8,8,8),(.08,.08,.08),offset=0)
    half,_=voxel_indices(p,(8,8,8),(.08,.08,.08),offset=.5)
    assert valid.all()
    assert np.unravel_index(fft[0],(8,8,8))==(4,4,4)
    assert fft[1]==fft[0] and half[1]==half[0]
    p=np.array([[-.004,0,0]])
    assert voxel_indices(p,(8,8,8),(.08,.08,.08),offset=0)[0][0]!=voxel_indices(p,(8,8,8),(.08,.08,.08),offset=.5)[0][0]


def test_outside_grid_is_not_clipped_to_boundary():
    flat,valid=voxel_indices(np.array([[1.,0,0],[0,0,0]]),(8,8,8),(.08,.08,.08))
    assert valid.tolist()==[False,True]
    assert len(flat)==1


@pytest.mark.parametrize('time',[0.,.000001,.013,.211,.719999])
def test_catmull_weights_match_original_sampler(time):
    values=np.random.default_rng(8).normal(size=(73,3));values[-1]=values[0]
    reference=_interp_time(np.arange(73)*.01,.72,72,lambda i:values[i],time,'cubic')
    weights=catmull_weights(np.array([time/.72]))
    np.testing.assert_allclose(weights@values[:72],reference,atol=1e-12)
    assert weights.sum()==pytest.approx(1)


def test_centered_fft_controls_have_no_half_voxel_shift():
    grid=CenteredFFTGrid((8,8,16),(.16,.08,.08),(0,0,0))
    rows=fft_controls(grid)
    assert len(rows)==9
    for r in rows:assert r['coordinate_m']==pytest.approx(r['FFT_coordinate_m'])
    assert grid.axis_centers_m()[0][8]==0


def test_instantaneous_complex_proxy_constant_velocity():
    class Lookup:
        flat_indices=np.array([0]);barycentric=np.array([[1.,0],[0,1.]])
        vertex_indices=np.array([[0,1],[0,1]]);sample_voxel=np.array([0,0]);counts=np.array([2])
        @staticmethod
        def scatter(a):return a.reshape(1,1,1)
    coherence,velocity=averaged_complex(Lookup(),np.array([[2.,-1,0],[2.,-1,0]]))
    np.testing.assert_allclose(coherence,1,atol=1e-7)
    np.testing.assert_allclose(velocity[0,0,0],[2,-1,0],atol=1e-6)


def test_central_velocity_excludes_resets_on_either_side():
    result=central_reset_exclusion([False,False,False,True],[True,False,False,False],[False,True,False,False])
    assert result.tolist()==[True,True,False,True]


def test_gradient_quadrature_integrates_quadratic_exactly():
    times,weights=quadrature(np.array([-.003,-.001,0]))
    assert np.sum(weights*times**2)==pytest.approx(.003**3/3,rel=1e-12)


def test_actual_encoding_gain_and_static_cancel():
    root=Path(__file__).resolve().parents[1]
    _,_,_,moments=phase_kernel(root/'koma_dp150/pc_3d_scanner_matched_venc5')
    assert abs(moments[0])<1e-7
    assert -10*moments[1]==pytest.approx(1,abs=2e-6)
    assert -.003<moments[2]/(2*moments[1])<-.001
