"""Regression tests for the S-001 shareable evidence bundle builder.

These cover the two ways the builder can silently do harm: publishing a file
that still carries operator or subject identifiers, and corrupting a structured
report while trying to sanitize it.
"""

from __future__ import annotations

import json

import pytest

from scripts.build_evidence_release import (
    ABSOLUTE_PATH_PATTERN,
    REVIEWED_IDENTIFIER_EXEMPTIONS,
    WORKSPACE_TOKEN,
    review_text,
)


SCANNER_README = "reference_inputs/scanner_metadata/README.md"


def test_redacting_json_escaped_windows_paths_keeps_the_json_parseable() -> None:
    """The pattern must consume a doubled backslash as one separator.

    A pattern that stops inside the escaped pair replaces only ``C:\\`` and
    leaves ``\\Users`` behind, which is an invalid JSON escape.
    """
    original = json.dumps(
        {
            "run_dir": r"C:\Users\operator\Desktop\MRSim\run",
            "ffmpeg": r"C:\Users\operator\AppData\Local\ffmpeg.EXE",
            "keep": "bin 14 peak flow",
        },
        indent=2,
    )
    assert r"C:\\Users" in original

    reviewed, redactions = review_text("outputs/example_report.json", "x.json", original)

    restored = json.loads(reviewed)
    assert restored["keep"] == "bin 14 peak flow"
    assert restored["run_dir"] == WORKSPACE_TOKEN
    assert restored["ffmpeg"] == WORKSPACE_TOKEN
    assert len(redactions) == 2
    assert not ABSOLUTE_PATH_PATTERN.search(reviewed)


def test_redacting_forward_slash_paths_in_markdown() -> None:
    original = "Run directory: C:/Users/operator/MRSim/outputs and nothing else.\n"
    reviewed, redactions = review_text("outputs/NOTES.md", "x.md", original)

    assert reviewed == f"Run directory: {WORKSPACE_TOKEN} and nothing else.\n"
    assert len(redactions) == 1


def test_corrupted_structured_output_fails_closed(monkeypatch) -> None:
    """If the sanitizer ever breaks a report, refuse to publish it."""
    import scripts.build_evidence_release as builder

    # Simulate a regressed pattern that severs an escape sequence.
    monkeypatch.setattr(builder, "ABSOLUTE_PATH_PATTERN", builder.re.compile(r"C:\\"))
    payload = json.dumps({"run_dir": r"C:\Users\operator\run"})

    with pytest.raises(SystemExit, match="Redaction corrupted"):
        builder.review_text("outputs/example_report.json", "x.json", payload)


def test_identifying_field_names_block_publication() -> None:
    payload = json.dumps({"note": "PatientName was stripped upstream"})

    with pytest.raises(SystemExit, match="identifying field"):
        review_text("outputs/leaky_report.json", "outputs/leaky_report.json", payload)


def test_reviewed_scanner_readme_is_exempt_from_the_identifier_scan() -> None:
    """The de-identified summary must be able to quote the flag it warns about."""
    assert SCANNER_README in REVIEWED_IDENTIFIER_EXEMPTIONS
    text = "The source says `Patient Identity Removed: NO` and is restricted.\n"

    reviewed, redactions = review_text(
        "../reference_inputs/scanner_metadata/README.md", SCANNER_README, text
    )

    assert reviewed == text
    assert redactions == []


def test_unreviewed_contact_addresses_block_publication() -> None:
    payload = json.dumps({"contact": "someone@example.org"})

    with pytest.raises(SystemExit, match="contact addresses"):
        review_text("outputs/report.json", "outputs/report.json", payload)


def test_allowlisted_maintainer_address_is_permitted() -> None:
    payload = json.dumps({"maintainer": "cmmcg@stanford.edu"})

    reviewed, _ = review_text("outputs/report.json", "outputs/report.json", payload)

    assert "cmmcg@stanford.edu" in reviewed
