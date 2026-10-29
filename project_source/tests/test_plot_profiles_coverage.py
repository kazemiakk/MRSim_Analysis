import numpy as np
import pytest
import json
import hashlib
import scripts.plot_profiles_coverage as plotting
from scripts.plot_profiles_coverage import (ecdf, phase_bins, retained_coverage,
    vector_error, curl_magnitude, main)


def test_weighted_ecdf_ties_missing_and_empty():
    x,y=ecdf([2,1,1,np.nan,99],[1,1,2,5,0])
    np.testing.assert_array_equal(x,[1,2])
    np.testing.assert_allclose(y,[.75,1])
    assert len(ecdf([np.nan],[1])[0])==0
    with pytest.raises(ValueError):ecdf([1,2],[1])


def test_phase_selection_uses_absolute_cfd_flow_and_first_tie():
    rows=[dict(bin=3,net_m3_s=2),dict(bin=1,net_m3_s=-8),dict(bin=2,net_m3_s=2)]
    assert phase_bins(rows)==dict(peak=1,low=2)
    with pytest.raises(ValueError):phase_bins(rows+[rows[0]])


def test_retained_coverage_uses_volume_not_voxel_count():
    assert retained_coverage(np.array([True,False]),np.array([1.,3.]),4)==25
    assert retained_coverage(np.zeros(2,bool),np.array([1.,3.]),4)==0
    with pytest.raises(ValueError):retained_coverage(np.ones(2,bool),np.ones(2),0)


def test_vector_error_and_empty_are_not_zero_fill():
    a=np.ones((2,2,2,2,3));c=a.copy();c[...,0]+=2
    mask=np.ones((2,2,2),bool);v=np.ones(mask.shape)
    rmse,nrmse,actual=vector_error(a,c,mask,v)
    assert rmse==2
    assert nrmse==pytest.approx(200/np.sqrt(3))
    assert actual.all()
    assert vector_error(a,c,np.zeros_like(mask),v)[:2]==(None,None)


def test_curl_solid_rotation_anisotropic_and_stencil():
    spacing=np.array([.1,.2,.3]);z,y,x=np.indices((5,6,7))*spacing[::-1,None,None,None]
    u=np.stack((-3*y,3*x,np.zeros_like(x)),axis=-1)
    values,mask=curl_magnitude(u,spacing,np.ones(u.shape[:-1],bool))
    assert mask.sum()==3*4*5
    np.testing.assert_allclose(values,6,atol=1e-12)
    sparse=np.zeros_like(mask);sparse[2,2,2]=True
    values,mask=curl_magnitude(u,spacing,sparse)
    assert not mask.any() and not len(values)


def test_completed_directory_is_never_overwritten(tmp_path):
    with pytest.raises(FileExistsError):main(tmp_path)


def test_original_list_manifest_and_profile_gaps(tmp_path,monkeypatch):
    path=tmp_path/'profiles.csv'
    path.write_text('bin,section,dataset,profile_axis,coordinate_mm,axial_velocity_m_s\n1,throat,cfd,MRI_Y,0,\n',encoding='utf-8')
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path/'MANIFEST.json').write_text(json.dumps([dict(file='profiles.csv',sha256=digest)]),encoding='utf-8')
    monkeypatch.setattr(plotting,'DP',tmp_path)
    rows=plotting.read_profiles({},dict(profile_series=[]),dict(aorta_pilot_sections=dict(profiles=[])))
    assert rows['phantom_1m'][0]['velocity'] is None
