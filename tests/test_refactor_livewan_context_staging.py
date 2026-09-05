import inspect
from pathlib import Path
import json
import subprocess
import sys

import pytest

from distillschedule.runtime.livewan_context_staging import PinnedScoreContextStager


@pytest.mark.parametrize("indices", [set(), {-1}, {"0"}])
def test_requires_registered_trace_indices(indices):
    with pytest.raises(ValueError):
        PinnedScoreContextStager(None, None, allowed_indices=indices)


def test_context_has_no_device_sync_value_read_or_rng():
    source = inspect.getsource(PinnedScoreContextStager.context_from_trace)
    assert "non_blocking=True" in source and ".pin_memory()" in source
    assert all(s not in source for s in (".cpu(", ".item(", ".tolist(", "synchronize(", "randn(", "manual_seed("))


def test_plan_only_does_not_load_torch_assets_or_create_output(tmp_path):
    root = Path(__file__).resolve().parents[1]
    p = subprocess.run([sys.executable, str(root/'scripts/distillschedule/canary_livewan_context_staging.py'),
                        '--plan-only', '--trace', '/nonexistent/trace', '--output', str(tmp_path/'output.json')],
                       check=True, capture_output=True, text=True)
    assert json.loads(p.stdout)['status'] == 'NOT_RUN'
    assert not (tmp_path/'output.json').exists()
