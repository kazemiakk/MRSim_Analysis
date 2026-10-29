from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.reconstruct_dp150_seam_safe_support_phase import (
    METHOD_SPECIFICATION,
    validate_gate_report_binding,
    verify_registration,
)


def test_gate_binding_rejects_nonpass_before_hashing(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="not PASS"):
        validate_gate_report_binding(
            {"status": "FAIL"}, tmp_path / "gate.json", tmp_path
        )


def test_registration_rejects_method_drift(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    registration = {
        "candidate_run": str(run_dir),
        "method_specification": {**METHOD_SPECIFICATION, "support": "changed"},
        "implementation": {"sha256": "unused"},
    }
    path = tmp_path / "registration.json"
    path.write_text(json.dumps(registration), encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not match"):
        verify_registration(path, run_dir)


def test_registration_rejects_wrong_run_before_script_hash(tmp_path: Path) -> None:
    registration = {
        "candidate_run": str(tmp_path / "first"),
        "method_specification": METHOD_SPECIFICATION,
        "implementation": {"sha256": "unused"},
    }
    path = tmp_path / "registration.json"
    path.write_text(json.dumps(registration), encoding="utf-8")
    with pytest.raises(RuntimeError, match="candidate run"):
        verify_registration(path, tmp_path / "second")
