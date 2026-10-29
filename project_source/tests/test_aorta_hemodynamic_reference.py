import numpy as np
import pytest
from scipy.spatial import cKDTree
from scripts.aorta_hemodynamic_reference import (AortaGrid, build_lookup,
    sequence_timing, temporal_node_values, temporal_weights)


def test_frame18_origin_and_linear_periodic_wrap():
    period=.7261599898338318
    assert temporal_weights([0],period_s=period)[18] == 1
    assert temporal_weights([period],period_s=period)[18] == 1
    weights=temporal_weights([.5*period/72],period_s=period)
    assert weights[18] == pytest.approx(.5)
    assert weights[19] == pytest.approx(.5)
    assert weights.sum() == pytest.approx(1)


def test_constant_velocity_retiming_and_axis_permutation():
    source=np.broadcast_to(np.array([1,2,3],dtype=np.float32)[None,:,None],(4,3,72))
    weights=np.array([temporal_weights([0,.3],period_s=.72)])
    actual=temporal_node_values(source,weights,scale=.5)
    np.testing.assert_allclose(actual[0],np.tile([1.5,.5,1.],(4,1)))


def test_no_data_support_is_nan_and_constant_field_preserved():
    coordinates=np.array(np.meshgrid([-1,0,1],[-1,0,1],[-1,0,1])).reshape(3,-1).T.astype(float)
    tree=cKDTree(coordinates)
    grid=AortaGrid(shape=(1,1,1),fov=(1,1,1),center=(0,0,0),order=(0,1,2))
    lookup=build_lookup(tree,grid,support_radius=1)
    result=lookup.evaluate(np.tile([[[2.,3.,4.]]],(1,len(coordinates),1)))
    np.testing.assert_allclose(result[0,0,0,0],[2,3,4])
    assert lookup.fraction[0,0,0] == 1
    distant=AortaGrid(shape=(1,1,1),fov=(1,1,1),center=(30,0,0),order=(0,1,2))
    missing=build_lookup(tree,distant,support_radius=.1)
    assert not missing.fraction.any()
    assert np.isnan(missing.evaluate(np.ones((1,len(coordinates),3)))).all()


def test_fft_grid_origin_and_inverse_transform():
    grid=AortaGrid(shape=(8,8,8),fov=(.08,.08,.08),center=(0,0,0),order=(0,1,2))
    x=grid.axes()[0]
    assert x[4] == 0
    k=np.fft.fftshift(np.fft.fftfreq(8,d=.01))
    signal=np.exp(-2j*np.pi*k*.02)
    image=np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(signal)))
    assert x[np.argmax(abs(image))] == pytest.approx(.02)
    np.testing.assert_allclose(grid.to_source([[1,2,3]]),[[1,2,3]])


def test_saved_pulseq_label_chain_and_seam_detection(tmp_path):
    path=tmp_path/'test.seq'
    path.write_text('''[DEFINITIONS]
BlockDurationRaster 0.001
CardiacBins 2
Nx 1
Ny 1
Nz 1
RR 0.01
[BLOCKS]
1 6 1 0 0 0 0 0
2 6 0 0 0 0 1 3
3 6 1 0 0 0 0 0
4 6 0 0 0 0 1 4
[RF]
1 1 1 1 1 0 0 0 0 0 0 e
[ADC]
1 1 1000 0 0 0 0 0 0
[EXTENSIONS]
1 1 1 0
2 1 2 1
3 1 3 2
4 1 4 2
extension LABELSET 1
1 0 PAR
2 0 LIN
3 0 PHS
4 1 PHS
''',encoding='utf-8')
    audit,weights,records=sequence_timing(path,.01)
    assert [r['bin'] for r in records] == [1,2]
    assert audit['RF_to_ADC_seam_crossings'] == 0
    np.testing.assert_allclose(weights.sum(1),1)
    audit,_,_=sequence_timing(path,.015)
    assert audit['RF_to_ADC_seam_crossings'] == 1


def test_invalid_weights_and_samples_rejected():
    with pytest.raises(ValueError):
        temporal_weights([],period_s=.72)
    with pytest.raises(ValueError):
        temporal_node_values(np.ones((1,3,2)),np.ones((1,3)))
