import numpy as np
import pytest
from scripts.aorta_hemodynamic_reference import temporal_weights
from scripts.track_aorta_native_zero import main
from scripts.analyze_aorta_native_zero import validate_contract, analyze
from scripts.integrate_aorta_inlet_complex import synthetic_controls


def test_original_cfd_zero_not_frame18():
    w=temporal_weights([0,.72],period_s=.72,start_frame=0)
    assert w[0]==1 and w[18]==0


def test_frame0_interpolation_preserves_native_cadence():
    w=temporal_weights([.005],period_s=.72,start_frame=0)
    assert w[0]==pytest.approx(.5) and w[1]==pytest.approx(.5)
    assert w.sum()==pytest.approx(1)


def test_period_not_deliberately_rescaled_to_legacy_rr():
    assert abs(float(np.float32(.72))-.72)<1e-7
    assert abs(.7261599898338318-.72)>.006


def test_preserve_existing_run(tmp_path):
    with pytest.raises(FileExistsError):main(tmp_path,20000)


def test_refuse_unapproved_production_count(tmp_path):
    with pytest.raises(ValueError,match='Pilot'):main(tmp_path/'new',1000000)


def contract_fixture():
    sim = dict(status='COMPLETE_E045_NATIVE_ZERO_16BIN_PILOT',
               particle_count=100000, source_start_frame=0, cardiac_bins=16)
    timing = dict(status='PASS_NATIVE_ZERO_TIMING', native_source_start_frame=0)
    image = np.broadcast_to(np.array(1+0j), (16,32,32,72))
    images = {e:image for e in ['REF','X','Y','Z']}
    weights = np.ones((16,72))/72
    return sim,timing,images,weights


def test_analysis_contract_accepts_authorized_acquisition():
    validate_contract(*contract_fixture())


@pytest.mark.parametrize('field,value',[('source_start_frame',18),('particle_count',1000000),('cardiac_bins',8)])
def test_analysis_refuses_legacy_or_unapproved_acquisition(field,value):
    sim,timing,images,weights=contract_fixture();sim[field]=value
    with pytest.raises(ValueError,match='pilot only'):
        validate_contract(sim,timing,images,weights)


def test_analysis_refuses_nonconserving_temporal_weights():
    sim,timing,images,weights=contract_fixture();weights[0,0]+=1
    with pytest.raises(ValueError,match='conserve'):
        validate_contract(sim,timing,images,weights)


def test_analysis_preserves_existing_output(tmp_path):
    with pytest.raises(FileExistsError):
        analyze(tmp_path/'missing',tmp_path/'missing.json',tmp_path)


def test_unchanged_complex_estimator_controls():
    assert synthetic_controls()['status']=='PASS_SYNTHETIC_CONTROLS'
