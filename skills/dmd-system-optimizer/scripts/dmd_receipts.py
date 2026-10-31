#!/usr/bin/env python3
"""Parse frozen checker/runner outputs; never infer PASS from a caller's summary.

This is a local adapter, not an execution sandbox or signature authority. The
project runner owns invocation identity, checker independence and raw outputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

from dmd_perf import check_evidence, load, number, require, text

VERSION = "dmd-receipts-v1"
BINDING = ("run_id", "candidate_id", "source_identity", "config_identity", "input_identity",
           "comparator_id", "comparator_source_identity", "validation_scope")


def checked_ref(ref, base):
    errors = check_evidence([ref], base, "adapter source")
    require(not errors, "; ".join(errors))
    return {**ref, "path": str((base / ref["path"]).resolve())}


def adapt(request, base):
    """Reparse raw bytes every time, including when verifying an existing receipt."""
    fmt = request.get("format")
    require(fmt in {"junit", "aba-json", "attempt-json"}, "unsupported receipt format")
    binding = request["binding"]
    for key in BINDING:
        text(binding.get(key), "receipt binding." + key)
    checker = checked_ref(request["checker"], base)
    text(checker.get("version"), "checker.version")
    raw = checked_ref(request["raw"], base)
    out = {"schema_version": 1, "adapter_version": VERSION, "format": fmt,
           "binding": binding, "checker": checker, "raw": raw}
    if fmt == "junit":
        selection = request["selection"]
        require(isinstance(selection, dict) and selection, "frozen check selection required")
        try:
            root = ET.parse(raw["path"]).getroot()
        except ET.ParseError as exc:
            raise ValueError("invalid JUnit checker output") from exc
        require(root.tag in {"testsuite", "testsuites"}, "expected JUnit testsuite(s)")
        cases = list(root.iter("testcase"))
        require(cases, "checker output has no test cases")
        # Separate named checks (capacity is not numerical correctness). Unknown
        # failures or inconsistent suite summaries fail closed for every label.
        names_in_policy = {name for spec in selection.values() for name in spec["tests"]}
        failures = [e for e in root.iter() if e.tag in {"failure", "error"}]
        case_failures = [e for c in cases for e in c if e.tag in {"failure", "error"}]
        unexplained = len(failures) != len(case_failures) or any(
            c.get("name") not in names_in_policy and any(c.find(k) is not None for k in ("failure", "error")) for c in cases)
        unexplained |= any(int(e.get(k, "0")) > sum(f.tag == tag for f in failures)
                           for e in root.iter() if e.tag in {"testsuite", "testsuites"}
                           for k, tag in (("failures", "failure"), ("errors", "error")))
        checks = {}
        for level, spec in selection.items():
            text(spec.get("scope"), "check scope")
            names = spec.get("tests")
            require(isinstance(names, list) and names and all(isinstance(n, str) and n for n in names),
                    "each check needs explicit test names")
            selected = [c for c in cases if c.get("name") in names]
            complete = all(sum(c.get("name") == name for c in cases) == 1 for name in names)
            failed = unexplained or any(c.find(k) is not None for c in selected for k in ("failure", "error"))
            checks[level] = "FAIL" if failed else "PASS" if complete and all(
                c.find("skipped") is None for c in selected) else "NOT_RUN"
        out.update(selection=selection, checks=checks,
                   validation_scopes={k: v["scope"] for k, v in selection.items()})
    else:
        body = load(Path(raw["path"]))
        require(body.get("binding") == binding, "raw runner binding differs from receipt")
        if fmt == "aba-json":
            require(body.get("status") == "PASS", "timing runner did not PASS")
            require(body.get("unprofiled") is True and body.get("repeat_unit") == "independent_processes",
                    "timing runner must report independent unprofiled processes")
            require(body.get("measurement_kind") in {"observed", "synthetic"}, "declare timing measurement_kind")
            times = body["timing"]
            require(times.get("run_id") == binding["run_id"], "timing run identity differs")
            for key in ("control_before", "candidate", "control_after"):
                number(times.get(key), key, positive=True)
            out.update(timing=times, measurement_kind=body["measurement_kind"], unprofiled=True,
                       repeat_unit="independent_processes")
        else:
            require(body.get("status") in {"error", "interrupted"}, "only failed attempts may be invalidated")
            sample = body["sample"]
            require(sample.get("status") == "invalid" and sample.get("valid_window") is False,
                    "a valid measurement cannot be excluded")
            require(sample.get("failure_origin") == "environment", "candidate failures cannot be replaced")
            require(sample.get("reason") in {"launch_failure", "environment_interruption", "invalid_timer_window"},
                    "unrecognized environmental failure")
            out.update(outcome=body["status"], sample=sample)
    return out


def verify(receipt, policy, binding, base):
    require(receipt.get("adapter_version") == VERSION, "unknown receipt adapter version")
    require(receipt.get("binding") == binding, "receipt does not match reserved run/source/config/input/comparator/scope")
    expected = policy["checkers"].get(receipt.get("format"))
    require(expected is not None, "checker format not frozen in contract")
    require(all(receipt["checker"].get(k) == expected.get(k) for k in ("sha256", "version")),
            "checker differs from frozen version")
    if receipt["format"] == "junit":
        require(receipt.get("selection") == policy["checks"], "checker selection differs from frozen validation scope")
    parsed = adapt(receipt, base)
    require(parsed == receipt, "receipt declaration conflicts with parsed checker/runner body")
    return parsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(adapt(load(args.input), args.input.resolve().parent), indent=2, allow_nan=False))
        return 0
    except (ValueError, KeyError, TypeError, OSError, ET.ParseError) as exc:
        print(json.dumps({"status": "INVALID", "error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
