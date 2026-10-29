from pathlib import Path
import numpy as np
import pytest

from scripts.prepare_phantom_encoding import shot_contract, phase_expectation
from scripts.analyze_phantom_encoding import centered_ifft, rhs
from scripts.diagnose_phantom_encoding_kernel import first_shot

ROOT=Path(__file__).resolve().parents[1]
SEQ=ROOT/'koma_dp150/pc_3d_scanner_matched_venc5'


def test_actual_adc_has_half_dwell_offset_from_gate_delay():
    parsed=shot_contract(SEQ/'ref.seq')
    historical=first_shot(SEQ/'ref.seq')
    old_echo=historical['echo']-parsed['rf_block_start_absolute']
    assert parsed['echo']-old_echo==pytest.approx(3.5715e-6,abs=1e-12)
    np.testing.assert_allclose(np.diff(parsed['adc_times']),7.143e-6,atol=1e-12)
    assert len(parsed['adc_times'])==400


@pytest.mark.parametrize('axis',[0,1,2])
def test_known_constant_velocity_phase_expectation(axis):
    ref=shot_contract(SEQ/'ref.seq');enc=shot_contract(SEQ/('xyz'[axis]+'.seq'))
    time=np.linspace(0,ref['duration'],1001)
    position=np.zeros((len(time),3));position[:,axis]=.75*(time-ref['echo'])
    value=phase_expectation(ref,enc,time,position,ref['echo'])
    assert value==pytest.approx(.75,abs=1e-8)


def test_acceleration_response_is_not_echo_velocity():
    ref=shot_contract(SEQ/'ref.seq');enc=shot_contract(SEQ/'x.seq')
    time=np.linspace(0,ref['duration'],20001);tau=time-ref['echo']
    position=np.zeros((len(time),3));position[:,0]=1.5*tau+.5*200*tau**2
    value=phase_expectation(ref,enc,time,position,ref['echo'])
    assert value==pytest.approx(1.5-200*.0021784995,abs=1e-7)


@pytest.mark.parametrize('n,shift',[(400,0),(400,1),(64,-2),(32,1)])
def test_centered_fft_physical_localization(n,shift):
    k=np.arange(n)-n//2
    image=centered_ifft(np.exp(-2j*np.pi*k*shift/n))
    assert np.argmax(abs(image))==n//2+shift
    assert abs(image).max()==pytest.approx(1,abs=1e-12)


def test_independent_bloch_rhs_rotation_and_relaxation():
    gamma=42.577e6
    m=np.array([[0.,0.,1.],[1.,0.,0.]])
    field=np.array([[1e-6,0,0],[0,0,1e-6]])
    value=rhs(m,field,np.array([1.,1.]),np.array([.1,.1]),gamma)
    assert value[0,1]==pytest.approx(2*np.pi*gamma*1e-6)
    assert value[1,1]==pytest.approx(-2*np.pi*gamma*1e-6)
    assert value[1,0]==pytest.approx(-10)
    assert value[1,2]==pytest.approx(1)
