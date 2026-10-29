import h5py
import numpy as np
import pytest

from scripts.inventory_additional_phantom_cfd import frame_metadata


def test_metadata_cadence_comes_from_solver_not_filename(tmp_path):
    path = tmp_path/'arbitrary-name.dat.h5'
    with h5py.File(path, 'w') as f:
        f.create_dataset('settings/Data Variables', data=[np.bytes_('(flow-time 0.01) (time-step 1)')])
        f.create_dataset('results/1/phase-1/cells/SV_U/1', shape=(19,), dtype='f4')
    r = frame_metadata(path)
    assert r['native_time_s'] == .01
    assert r['native_step'] == 1
    assert r['velocity_component_shape'] == [19]
    assert r['full_file_hash_recomputed'] is False
    assert len(r['settings_payload_sha256']) == 64


def test_missing_timing_fails_closed(tmp_path):
    path = tmp_path/'bad.dat.h5'
    with h5py.File(path, 'w') as f:
        f.create_dataset('settings/Data Variables', data=[np.bytes_('(flow-time 0.01)')])
    with pytest.raises(ValueError, match='Missing native timing'):
        frame_metadata(path)
