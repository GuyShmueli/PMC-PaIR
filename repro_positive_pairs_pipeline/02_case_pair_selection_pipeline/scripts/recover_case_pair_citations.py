#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from case_pair_selection_pipeline.citation_recovery import recover_case_pair_citations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recover missing PMC/JATS citation contexts and populate only "
            "case_pair_citations.json.extracted_citation."
        )
    )
    parser.add_argument("--citation-records", required=True)
    parser.add_argument("--case-pair-citations", required=True)
    parser.add_argument(
        "--audit-output",
        default=None,
        help="Optional audit JSON; required with --apply.",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Atomically replace the unified file after all pre-write checks pass.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.apply and not args.audit_output:
        raise SystemExit("--audit-output is required with --apply")
    audit = recover_case_pair_citations(
        args.citation_records,
        args.case_pair_citations,
        audit_path=args.audit_output,
        apply=args.apply,
        workers=args.workers,
    )
    summary = {
        "applied": audit["applied"],
        "counts": audit["counts"],
        "invariants": audit["invariants"],
        "input_sha256": audit["inputs"]["unified_before_sha256"],
        "output_sha256": audit["output"]["unified_after_sha256"],
        "audit_output": str(Path(args.audit_output).resolve()) if args.audit_output else None,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
