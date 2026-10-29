r"""Freeze a reviewed, shareable evidence bundle for the manuscript claims.

This implements governance step S-001. The repository's blanket ``*.json``
ignore rule keeps generated analysis reports out of version control because
they are large, machine-specific, or derived from non-public inputs. A reviewer
with only the source tree therefore cannot run
``paper/verify_manuscript_claims.py``. This script selects the small subset of
reports that the verifier actually reads, reviews each one for local absolute
paths and identifying fields, copies them into a versioned bundle, and records
byte counts and SHA-256 digests in a manifest.

The bundle contents are chosen by asking the verifier itself
(``--list-inputs``), so the bundle cannot drift away from the checks. Files the
verifier only needs a size or digest for -- the multi-gigabyte HDF5 artifacts
and the production video -- are recorded as declared artifacts rather than
copied. The identifying scanner source is recorded as withheld and never
copied.

Usage::

    uv run python scripts/build_evidence_release.py
    uv run python scripts/build_evidence_release.py --version 1.1.0 --force
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tomllib
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
VERIFIER = ROOT / "paper" / "verify_manuscript_claims.py"
DEFAULT_OUTPUT = ROOT / "paper" / "evidence_release"
DEFAULT_VERSION = "1.0.0"

# A copied evidence file is meant to be a small machine-readable report. Anything
# larger is either the wrong file or a sign that the claim map grew a binary, so
# fail closed rather than silently publishing it.
MAX_COPIED_BYTES = 2 * 1024 * 1024

# Local absolute paths leak the operator's machine layout. Replace them with a
# stable token and record every substitution in the manifest.
# Paths appear both raw (in Markdown) and JSON-escaped (``C:\\Users\\...``). The
# separator alternation must consume a doubled backslash as a single unit; a
# pattern that stops inside the pair leaves a dangling escape and corrupts the
# JSON it was meant to sanitize.
ABSOLUTE_PATH_PATTERN = re.compile(
    r"[A-Za-z]:(?:\\\\|[\\/])(?:[^\"'\s\\/]+(?:\\\\|[\\/]))*[^\"'\s\\/]*"
)
WORKSPACE_TOKEN = "<WORKSPACE>"

# Fail closed if any DICOM-style identifying field reaches a shareable file.
IDENTIFIER_PATTERN = re.compile(
    r"patient\s*name|patient\s*id|\bMRN\b|birth\s*date|accession\s*number"
    r"|study\s*instance\s*uid|institution\s*name|patient\s*identity",
    re.IGNORECASE,
)
EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

# The one contact address already published in packaging metadata and on PyPI.
ALLOWED_CONTACTS = {"cmmcg@stanford.edu"}

# Files reviewed by hand and cleared despite matching IDENTIFIER_PATTERN. The
# scanner README is the de-identified derivative itself: it has to name the
# field it is warning about, and weakening that warning to satisfy this lint
# would lose provenance. Keyed by the verifier's normalized path.
REVIEWED_IDENTIFIER_EXEMPTIONS = {
    "reference_inputs/scanner_metadata/README.md": (
        "De-identified fields-only summary. It quotes the source's "
        "'Patient Identity Removed: NO' flag as a warning and carries no "
        "identifier values."
    ),
}

WITHHELD_REASONS = {
    # Keyed by the verifier's normalized path (leading ../ removed).
    "reference_inputs/scanner_metadata/phantom_8_28_2025_metadeta.rtf": (
        "Identifying scanner metadata source. The file records "
        "'Patient Identity Removed: NO' and contains subject and institution "
        "fields. Rules.md section 4 forbids publishing it. The de-identified "
        "fields-only derivative at reference_inputs/scanner_metadata/README.md "
        "is included instead."
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild over an existing bundle version. Bundles are immutable by "
        "default so a published manifest cannot change under a reviewer.",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Do not re-run the verifier against the finished bundle.",
    )
    return parser.parse_args()


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_inputs() -> list[dict]:
    """Ask the verifier which evidence files it reads."""
    result = subprocess.run(
        [sys.executable, str(VERIFIER), "--list-inputs"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise SystemExit(
            "The manuscript verifier failed against the working repository, so "
            "there is nothing trustworthy to freeze.\n"
            f"{result.stdout}\n{result.stderr}"
        )
    start = result.stdout.index("{")
    return json.loads(result.stdout[start:])["inputs"]


def review_text(relative: str, normalized: str, text: str) -> tuple[str, list[str]]:
    """Redact local absolute paths and reject identifying content."""
    identifiers = IDENTIFIER_PATTERN.findall(text)
    if identifiers and normalized not in REVIEWED_IDENTIFIER_EXEMPTIONS:
        raise SystemExit(
            f"Refusing to publish {relative}: it contains identifying field "
            f"names {sorted(set(identifiers))}. Review it by hand before "
            f"including it in a shareable bundle."
        )
    contacts = {
        address
        for address in EMAIL_PATTERN.findall(text)
        if address not in ALLOWED_CONTACTS
    }
    if contacts:
        raise SystemExit(
            f"Refusing to publish {relative}: it contains contact addresses "
            f"{sorted(contacts)} that are not on the reviewed allow list."
        )

    redactions: list[str] = []

    def _replace(match: re.Match[str]) -> str:
        redactions.append(match.group(0))
        return WORKSPACE_TOKEN

    reviewed = ABSOLUTE_PATH_PATTERN.sub(_replace, text)

    # Redaction rewrites bytes inside structured files. Re-parse the result and
    # fail closed if it no longer decodes, so a sanitizer bug can never publish
    # a corrupted report under a manifest digest that looks authoritative.
    suffix = PurePosixPath(relative).suffix.lower()
    try:
        if suffix == ".json":
            json.loads(reviewed)
        elif suffix == ".toml":
            tomllib.loads(reviewed)
    except (json.JSONDecodeError, tomllib.TOMLDecodeError) as error:
        raise SystemExit(
            f"Redaction corrupted {relative}: the reviewed copy no longer "
            f"parses as {suffix.lstrip('.').upper()} ({error}). Fix "
            f"ABSOLUTE_PATH_PATTERN rather than shipping the file."
        ) from error

    return reviewed, redactions


def main() -> None:
    args = parse_args()
    bundle = (args.output / f"v{args.version}").resolve()
    if bundle.exists():
        if not args.force:
            raise SystemExit(
                f"{bundle} already exists. Evidence bundles are immutable; pass "
                f"--force to rebuild or choose a new --version."
            )
        shutil.rmtree(bundle)
    bundle.mkdir(parents=True)

    inputs = collect_inputs()
    files: list[dict] = []
    declared: list[dict] = []
    withheld: list[dict] = []

    evidence_dir = bundle / "evidence"
    evidence_dir.mkdir()
    index = 0

    for entry in inputs:
        relative = entry["path"]
        kinds = set(entry["kinds"])
        source = (ROOT / relative).resolve()
        if not source.is_file():
            raise SystemExit(f"Verifier input is missing: {relative}")

        if entry["restricted"]:
            withheld.append(
                {
                    "source_path": relative,
                    "bytes": source.stat().st_size,
                    "sha256": sha256_of(source),
                    "reason": WITHHELD_REASONS.get(
                        entry["normalized"], "Restricted source evidence."
                    ),
                }
            )
            continue

        if kinds <= {"size", "sha256"}:
            # The verifier only needs this artifact's size and digest, never its
            # bytes. Declaring it keeps the bundle small and honest.
            declared.append(
                {
                    "source_path": relative,
                    "bytes": source.stat().st_size,
                    "sha256": sha256_of(source),
                    "included": False,
                    "reason": (
                        "Production binary artifact preserved in the private "
                        "workspace. The bundle carries its recorded size and "
                        "digest so the manuscript stays bound to it, but a "
                        "bundle-only reviewer cannot recompute the digest."
                    ),
                }
            )
            continue

        size = source.stat().st_size
        if size > MAX_COPIED_BYTES:
            raise SystemExit(
                f"Refusing to copy {relative} ({size:,} bytes) into a shareable "
                f"bundle: it exceeds the {MAX_COPIED_BYTES:,}-byte report limit."
            )

        original = source.read_text(encoding="utf-8")
        reviewed, redactions = review_text(
            relative, entry["normalized"], original
        )
        # Store copies under short flat names. Mirroring the production output
        # tree would push several paths past the Windows 260-character limit on
        # a reviewer's machine; MANIFEST.json carries the full source path so
        # nothing about the provenance is lost.
        index += 1
        bundle_path = f"evidence/{index:03d}_{source.name}"
        destination = bundle / bundle_path
        destination.write_text(reviewed, encoding="utf-8", newline="")
        files.append(
            {
                "source_path": relative,
                "bundle_path": bundle_path,
                "bytes": destination.stat().st_size,
                "sha256": sha256_of(destination),
                "source_bytes": size,
                "source_sha256": sha256_of(source),
                "redacted": bool(redactions),
                # Only the count. Listing the redacted strings would reprint the
                # operator's machine layout into the very file meant to be
                # shared, undoing the redaction it is reporting.
                "redacted_absolute_path_count": len(redactions),
                "distinct_redacted_absolute_paths": len(set(redactions)),
                "access_kinds": sorted(kinds),
                "reviewed_identifier_exemption": (
                    REVIEWED_IDENTIFIER_EXEMPTIONS.get(entry["normalized"])
                ),
            }
        )

    files.sort(key=lambda item: item["source_path"])
    declared.sort(key=lambda item: item["source_path"])
    withheld.sort(key=lambda item: item["source_path"])

    manifest = {
        "bundle_version": args.version,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "governance_step": "S-001",
        "purpose": (
            "Reviewed, immutable subset of machine-readable evidence that lets a "
            "source-only reviewer run paper/verify_manuscript_claims.py without "
            "the private workspace or the identifying scanner source."
        ),
        "verify_command": (
            "uv run python paper/verify_manuscript_claims.py "
            "--evidence-root paper/evidence_release/v" + args.version
        ),
        "evidence_class": (
            "Internal integrity and internal consistency evidence. This bundle "
            "does not contain, and its checks do not establish, independent "
            "PC-MRI velocity accuracy, scanner-protocol equivalence, or "
            "boundary-resolved aortic physics."
        ),
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "files": files,
        "declared_artifacts": declared,
        "withheld": withheld,
    }
    manifest_path = bundle / "MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline=""
    )

    redacted_files = [item for item in files if item["redacted"]]
    lines = [
        f"# MRSim evidence bundle v{args.version}",
        "",
        "Frozen for governance step S-001. This directory is the reviewed,",
        "shareable subset of the machine-readable evidence behind",
        "`paper/MANUSCRIPT.md`.",
        "",
        "## Verify",
        "",
        "```powershell",
        "uv run python paper/verify_manuscript_claims.py \\",
        f"    --evidence-root paper/evidence_release/v{args.version}",
        "```",
        "",
        "## Contents",
        "",
        f"- {len(files)} reviewed report files "
        f"({manifest['total_bytes']:,} bytes), with byte counts and SHA-256",
        "  digests for both the bundle copy and the preserved source recorded",
        "  in `MANIFEST.json`.",
        f"- {len(declared)} declared binary artifacts too large to ship. The",
        "  manifest records each one's size and digest so the manuscript stays",
        "  bound to it, but a bundle-only reviewer cannot recompute those",
        "  digests.",
        f"- {len(withheld)} withheld file{'' if len(withheld) == 1 else 's'}, "
        "excluded on privacy grounds and listed below.",
        f"- {len(redacted_files)} file{'' if len(redacted_files) == 1 else 's'} "
        "required local absolute-path redaction.",
        "",
        "## Scientific hold on the dp150 phase evidence",
        "",
        "Experiment E-018 rejected the preserved dp150 one-million-spin phase",
        "for scientific use: every saved PHS8 readout in all four encodings was",
        "acquired across an artificial periodic coordinate seam. The E-010,",
        "E-015, E-016, and E-017 reports carried here remain valid records of",
        "what was computed and are retained as historical and presentation",
        "evidence, but their phase must not be read as a corrected result. A",
        "corrected acquisition is tracked as governance step S-000.",
        "",
        "## What this bundle does not establish",
        "",
        "Passing the verifier against this bundle demonstrates that the",
        "manuscript's numbers agree with the recorded reports and that the",
        "reports are internally consistent. It does not demonstrate independent",
        "PC-MRI velocity accuracy, equivalence to the intended balanced-Hadamard",
        "scanner acquisition, boundary-resolved aortic wall physics, or native",
        "fine-time CFD convergence. Those remain blocked on the external inputs",
        "listed in `EXTERNAL_INPUTS_REQUIRED.md`.",
        "",
        "## Withheld evidence",
        "",
    ]
    for item in withheld:
        lines.extend(
            [
                f"- `{item['source_path']}` ({item['bytes']:,} bytes, SHA-256",
                f"  `{item['sha256']}`)",
                f"  {item['reason']}",
                "",
            ]
        )
    lines.extend(["## Declared artifacts", ""])
    for item in declared:
        lines.append(
            f"- `{item['source_path']}` -- {item['bytes']:,} bytes, SHA-256 "
            f"`{item['sha256']}`"
        )
    lines.append("")
    (bundle / "README.md").write_text(
        "\n".join(lines), encoding="utf-8", newline=""
    )

    # Final privacy sweep over everything the bundle actually ships, the
    # manifest and README included. Those two are what a reviewer opens first,
    # so a leak there is worse than a leak in a report.
    leaks: list[str] = []
    for path in sorted(bundle.rglob("*")):
        if not path.is_file():
            continue
        for match in ABSOLUTE_PATH_PATTERN.finditer(
            path.read_text(encoding="utf-8")
        ):
            leaks.append(
                f"{path.relative_to(bundle).as_posix()}: {match.group(0)[:60]}"
            )
    if leaks:
        raise SystemExit(
            "Refusing to publish the bundle: absolute local paths survived in "
            + f"{len(leaks)} place(s):\n  " + "\n  ".join(leaks[:20])
        )

    print(f"Bundle: {bundle}")
    print(
        f"  {len(files)} reviewed files, {manifest['total_bytes']:,} bytes; "
        f"{len(declared)} declared artifacts; {len(withheld)} withheld; "
        f"{len(redacted_files)} redacted"
    )

    if args.skip_verify:
        return
    result = subprocess.run(
        [sys.executable, str(VERIFIER), "--evidence-root", str(bundle)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    print(result.stdout.strip())
    if result.returncode != 0:
        print(result.stderr.strip())
        raise SystemExit("The frozen bundle failed verification.")


if __name__ == "__main__":
    main()
