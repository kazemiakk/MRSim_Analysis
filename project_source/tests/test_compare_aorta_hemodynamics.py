import numpy as np
from scripts.aorta_hemodynamic_reference import AortaGrid
from scripts.compare_aorta_hemodynamics import masks_for,section_components,derivative_set


def test_fixed_mask_does_not_treat_zeroed_missing_velocity_as_measured():
    mag=np.ones((2,4,4,4)); stored=np.ones_like(mag,dtype=bool)
    stored[1,1,1,1]=False; mag[1,2,2,2]=.1
    masks=masks_for(mag,stored,np.ones((4,4,4)))
    assert not masks['core05'][1,1,1]
    assert masks['core05'][2,2,2]
    assert not masks['core15'][2,2,2]


def test_sections_keep_disconnected_opposing_limbs_separate():
    fraction=np.zeros((12,12,5)); fraction[1:4,1:4,:]=1; fraction[7:10,7:10,:]=1
    grid=AortaGrid(shape=(12,12,5),fov=(.1,.12,.12))
    sections=section_components(fraction,grid,targets=(0,))
    assert len(sections)==2
    assert all(s['roi'].sum()==9 for s in sections)
    assert not np.any(sections[0]['roi'] & sections[1]['roi'])


def test_constant_vector_field_has_zero_interior_derivative_error():
    field=np.ones((2,5,5,5,3))
    result=derivative_set(field,field,np.ones((5,5,5),bool),[.01]*3)
    assert result['stencil_voxels_per_bin']==27
    assert all(m['rmse']==0 for m in result['pooled'].values())
