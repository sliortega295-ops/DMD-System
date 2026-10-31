"""Actual CPU numerical/lifecycle checker, emitting ordinary JUnit for the adapter.

The scalar protocol stand-in tests integration only, not real FSDP or gradients.
"""
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import workload

inputs = json.loads(Path("input.json").read_text())
config = json.loads(Path("config.json").read_text())
suite = ET.Element("testsuite")
observed = {}


def check(name, fn):
    case = ET.SubElement(suite, "testcase", name=name)
    try:
        assert fn()
    except Exception as error:
        ET.SubElement(case, "failure").text = str(error) or "assertion failed"


def run(enabled):
    engine = workload.Engine()
    values, receipt = workload.execute(engine, inputs, enabled=enabled)
    return values, receipt, engine


reference, _, ref_engine = run(False)
candidate, path, engine = run(True)
if config.get("inject_error"):
    candidate[-1] += 1
observed.update(reference=reference, candidate=candidate, path=path,
                baseline_copies=ref_engine.module.materializations, candidate_copies=engine.module.materializations)
check("source_guard", lambda: callable(workload.execute) and len(inputs) == 4)
check("numerical", lambda: candidate == reference)
check("resume", lambda: run(True)[0] == reference and path["restored"] and not engine.module.group.is_unsharded)
check("fallback", lambda: run(False)[0] == reference and ref_engine.module.group._reshard_after_forward)
check("path", lambda: path["enabled"] and path["engine_forward_count"] == len(inputs)
      and engine.module.materializations == 1 and ref_engine.module.materializations == len(inputs))
check("composition", lambda: sum(x * x for x in candidate) == sum(x * x for x in reference))
Path("path.json").write_text(json.dumps(observed, indent=2))
ET.ElementTree(suite).write("checker.xml", encoding="unicode")
sys.exit(1 if suite.findall(".//failure") else 0)
