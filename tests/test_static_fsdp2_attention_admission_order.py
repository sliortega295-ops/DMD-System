"""CPU-only regression for the pinned local FA2 admission import order."""

from __future__ import annotations

import ast
import importlib
import os
from pathlib import Path
import unittest
from types import SimpleNamespace
from typing import Any


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "scripts/distillschedule/probe_static_fsdp2_p0_world16.py"
)


def _load_admission_helper():
    tree = ast.parse(SOURCE.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_attention_backend_admission"
    )
    namespace = {
        "os": os,
        "importlib": SimpleNamespace(import_module=importlib.import_module),
        "Any": Any,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace[function.name], namespace


class AttentionAdmissionOrderTests(unittest.TestCase):
    def setUp(self):
        self.old_env = {key: os.environ.get(key) for key in (
            "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "LOCAL_KERNELS"
        )}

    def tearDown(self):
        for key, value in self.old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_local_fa2_dispatch_precedes_sealing_remote_kernel_lookup(self):
        admit, namespace = _load_admission_helper()
        events = []
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["LOCAL_KERNELS"] = "kernels-community/flash-attn2=/opt/pinned/flash-attn2"
        namespace["_pinned_local_fa2_identity"] = lambda value: events.append(("pin-checked", value))
        namespace["importlib"].import_module = lambda value: events.append(("dispatch-imported", value))
        namespace["_attention_backend_receipt"] = lambda expected: (
            events.append(("receipt", expected)) or {"status": "PASS"}
        )

        result = admit("local-fa2", lambda: events.append(("kernel-lookup-sealed", None)))

        self.assertEqual(result, {"status": "PASS"})
        self.assertEqual([event[0] for event in events], [
            "pin-checked", "dispatch-imported", "kernel-lookup-sealed", "receipt"
        ])
        self.assertEqual(events[1][1], "helios.modules.helios_kernels.attention_dispatch")

    def test_sdpa_keeps_offline_policy_before_backend_receipt(self):
        admit, namespace = _load_admission_helper()
        events = []
        namespace["importlib"].import_module = lambda value: events.append(("dispatch-imported", value))
        namespace["_attention_backend_receipt"] = lambda expected: (
            events.append(("receipt", expected)) or {"runtime": "sdpa"}
        )

        result = admit("sdpa", lambda: events.append(("kernel-lookup-sealed", None)))

        self.assertEqual(result, {"runtime": "sdpa"})
        self.assertEqual(events, [("kernel-lookup-sealed", None), ("receipt", "sdpa")])

    def test_online_capable_environment_is_rejected_before_import(self):
        admit, namespace = _load_admission_helper()
        events = []
        os.environ["HF_HUB_OFFLINE"] = "0"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["LOCAL_KERNELS"] = "kernels-community/flash-attn2=/opt/pinned/flash-attn2"
        namespace["_pinned_local_fa2_identity"] = lambda value: events.append(("pin-checked", value))
        namespace["importlib"].import_module = lambda value: events.append(("dispatch-imported", value))

        with self.assertRaisesRegex(RuntimeError, "explicit offline hub settings"):
            admit("local-fa2", lambda: events.append(("kernel-lookup-sealed", None)))

        self.assertEqual(events, [])


if __name__ == "__main__":
    unittest.main()
