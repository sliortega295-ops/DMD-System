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
    p = subprocess.run([sys.executable, str(root/'benchmarks/kernels/canary_livewan_context_staging.py'),
                        '--plan-only', '--trace', '/nonexistent/trace', '--output', str(tmp_path/'output.json')],
                       check=True, capture_output=True, text=True)
    assert json.loads(p.stdout)['status'] == 'NOT_RUN'
    assert not (tmp_path/'output.json').exists()


def test_canary_uses_complete_original_operation_rows():
    import importlib.util
    from types import SimpleNamespace
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('context_canary', root/'benchmarks/kernels/canary_livewan_context_staging.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = [SimpleNamespace(dataset_index=i, operation_key=f'op{k}') for k, i in enumerate([2, 2, 7, 8, 9])]
    selected = module.select_prompt_rows(rows)
    assert selected == [rows[0], rows[2], rows[3]]
    assert all(a is b for a, b in zip(selected, [rows[0], rows[2], rows[3]]))


def test_c27_is_only_cpu_prompt_staging_on_c25(monkeypatch):
    import copy
    from distillschedule.core.config import load_plan
    from distillschedule.runtime.livewan_context_staging import PLAN_CONTRACT
    from distillschedule.runtime.livewan_crossscale_worker import _score_prompt_staging_enabled

    root = Path(__file__).resolve().parents[1] / 'configs/distillschedule/plans'
    parent = root / 'livewan_wan13b_14b_sf_dmd_c25_continuous_b16_score_rmsnorm.json'
    candidate = root / 'livewan_wan13b_14b_sf_dmd_c27_continuous_b16_pinned_prompt_staging.json'
    a, b = json.loads(parent.read_text()), json.loads(candidate.read_text())
    assert b['metadata'].pop('score_prompt_staging') == PLAN_CONTRACT
    assert a == b
    for key in ('LIVEWAN_STUDENT_CHECKPOINT', 'LIVEWAN_CRITIC_CHECKPOINT',
                'LIVEWAN_TEACHER_CHECKPOINT', 'LIVEWAN_OPERATION_TRACE',
                'LIVEWAN_SYSTEMS_BOOTSTRAP', 'DISTILLSCHEDULE_OUTPUT_DIR'):
        monkeypatch.setenv(key, '/unread/' + key)
    plan = load_plan(candidate).plan.to_dict()
    assert _score_prompt_staging_enabled(plan, mode='pipeline')
    assert not _score_prompt_staging_enabled(load_plan(parent).plan.to_dict(), mode='pipeline')
    with pytest.raises(RuntimeError):
        _score_prompt_staging_enabled(plan, mode='baseline')
    for key, value in [('h2d_per_request', False), ('gpu_input_cache', True),
                       ('cpu_cache_dtype', 'bfloat16'), ('scope', 'all_ranks')]:
        changed = copy.deepcopy(plan)
        changed['metadata']['score_prompt_staging'][key] = value
        with pytest.raises(RuntimeError):
            _score_prompt_staging_enabled(changed, mode='pipeline')
    for key, value in [('critic_attention_checkpoint', {}), ('score_context_prefetch', True)]:
        changed = copy.deepcopy(plan)
        changed['metadata'][key] = value
        with pytest.raises(RuntimeError):
            _score_prompt_staging_enabled(changed, mode='pipeline')


def test_original_input_source_identity_is_exact_and_mismatch_fails():
    import hashlib
    from types import SimpleNamespace
    from distillschedule.runtime.livewan_context_staging import SOURCE_SHA256, install
    from distillschedule.runtime.livewan_crossscale_worker import InputLoader

    assert hashlib.sha256(inspect.getsource(InputLoader.context_from_trace).encode()).hexdigest() == SOURCE_SHA256
    with pytest.raises(RuntimeError, match='original context source changed'):
        install(None, SimpleNamespace(context_from_trace=install), allowed_indices={1})


def test_production_receipt_requires_actual_h2d_and_pin_census():
    from distillschedule.runtime.livewan_context_staging import (
        NAME, SOURCE_SHA256, BYTES_PER_REQUEST, runtime_receipt_errors,
    )
    r = dict(name=NAME, source_context_method_sha256=SOURCE_SHA256,
             execution_status='PASS', requests=20, h2d_bytes=20 * BYTES_PER_REQUEST,
             cache_entries=10, cache_hits=10, pinned_cpu_bytes=10 * BYTES_PER_REQUEST,
             contract_failures=0, gpu_input_cache=False, h2d_per_request=True)
    assert not runtime_receipt_errors(r, rank=8, cycles=2)
    for key, value in [('requests', 19), ('cache_entries', 0), ('h2d_bytes', 0),
                       ('contract_failures', 1), ('gpu_input_cache', True),
                       ('source_context_method_sha256', 'wrong')]:
        assert runtime_receipt_errors(dict(r, **{key: value}), rank=8, cycles=2)
    assert runtime_receipt_errors(r, rank=0, cycles=2)
    assert not runtime_receipt_errors({'execution_status': 'NOT_APPLICABLE'}, rank=0, cycles=2)


def test_every_request_copies_but_pins_only_first_consumption():
    from types import SimpleNamespace

    class Tensor:
        shape = (512, 4096)
        device = SimpleNamespace(type='cpu')
        requires_grad = False
        dtype = 'fp32'
        pins = 0
        copies = 0

        def float(self):
            return self

        def pin_memory(self):
            self.pins += 1
            return self

        def is_pinned(self):
            return self.pins > 0

        def to(self, device, *, non_blocking):
            assert non_blocking is True
            self.copies += 1
            return self

        def unsqueeze(self, dim):
            assert dim == 0
            return self

    tensor = Tensor()
    torch = SimpleNamespace(float32='fp32', device=lambda _: SimpleNamespace(type='cuda'))
    helper = PinnedScoreContextStager(torch, {'pos': {21: tensor}}, allowed_indices={21})
    row = SimpleNamespace(dataset_index=21, operation_key='original-identity')
    assert helper.context_from_trace(row, device='cuda:0') is tensor
    assert helper.context_from_trace(row, device='cuda:0') is tensor
    assert tensor.pins == 1 and tensor.copies == 2
    assert helper.receipt()['requests'] == 2
    assert helper.receipt()['cache_hits'] == 1
    with pytest.raises(RuntimeError, match='unregistered sample'):
        helper.context_from_trace(SimpleNamespace(dataset_index=22), device='cuda:0')
    assert helper.receipt()['contract_failures'] == 1
