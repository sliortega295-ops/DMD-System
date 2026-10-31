#!/usr/bin/env python3
"""Validate a small reviewable reuse decision, without applying a patch."""
import argparse
import json
from pathlib import Path
import sys
from dmd_perf import load, require, text, check_evidence


def validate(record, base):
    require(record.get("schema_version") == 1, "reuse schema_version must be 1")
    for key in ("target_project", "candidate_id", "source_identity", "implementation_id", "existing_equivalent",
                "integration", "expected_path", "rollback"):
        text(record.get(key), "reuse." + key)
    require(record.get("mode") in {"direct", "minimal_adaptation", "new_implementation", "already_enabled"}, "invalid reuse mode")
    for key in ("dependencies", "conflicts"):
        require(isinstance(record.get(key), list), key + " must be an explicit list")
    require(record.get("status") in {"PLANNED", "VERIFIED", "ROLLED_BACK"}, "invalid reuse status")
    if record["status"] != "PLANNED":
        for key in ("path_evidence", "numerical_evidence", "rollback_evidence"):
            errors = check_evidence(record.get(key), base, key)
            require(not errors, "; ".join(errors))
    return {"status": "VALID", "reuse_decision": record, "runtime_claim": "PROJECT_EVIDENCE_REQUIRED",
            "automatic_application": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(validate(load(args.input), args.input.resolve().parent), indent=2))
        return 0
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(json.dumps({"status": "INVALID", "error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
