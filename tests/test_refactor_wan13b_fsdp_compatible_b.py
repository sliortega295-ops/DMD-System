"""CPU contracts for the explicit compatible subset, not complete B or GPU proof."""
from __future__ import annotations

import copy

import pytest

from distillschedule.runtime import wan13b_factorial_r2 as r2


def configuration():
    return r2.make_configuration('only_B_fsdp_compatible', expected_head='2' * 40,
                                source_hashes={p: 'a' * 64 for p in r2.SOURCE_PATHS})


def test_compatible_subset_does_not_change_A_or_enable_AxB():
    cfg = configuration()
    r2.validate_configuration(cfg, expected_head='2' * 40)
    assert cfg['placement'] == r2.FSDP
    assert cfg['operator_flags'] == {
        name: name in {'generator_nograd_lora_fused_epilogue',
                      'grad_lora_fused_epilogue', 'modulated_norm_fusion'} for name in r2.FLAGS}
    assert all(cfg['runtime_policy'].values())
    assert cfg['matched_contract'] == r2.make_configuration('baseline',
        expected_head='2' * 40, source_hashes=cfg['source']['files_sha256'])['matched_contract']
    for key in ('generator_rollout_cuda_graph', 'critic_grad_cuda_graph',
                'generator_nograd_lora_cast_cache'):
        bad = copy.deepcopy(cfg)
        bad['operator_flags'][key] = True
        with pytest.raises(ValueError):
            r2.validate_configuration(bad, expected_head='2' * 40)
    with pytest.raises(ValueError, match='BLOCKED'):
        r2.make_configuration('only_B', expected_head='2' * 40,
                              source_hashes=cfg['source']['files_sha256'])


def test_real_checkpoint_scope_no_tensor_retention_and_stale_param_rejected():
    import torch
    from torch.utils.checkpoint import checkpoint
    from distillschedule.runtime.wan13b_fsdp_compatible_b import install

    class TinyLoRA(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16).requires_grad_(False)
            self.a = torch.nn.Linear(4, 2, bias=False)
            self.b = torch.nn.Linear(2, 4, bias=False)

        def forward(self, x):
            return (self.base(x) + self.b(self.a(x.float())) * 0.3).to(x.dtype)

    torch.manual_seed(20260903)
    reference = TinyLoRA()
    candidate = copy.deepcopy(reference)
    receipt = install(candidate, role='generator', placement=r2.FSDP, linear_type=TinyLoRA)
    x = torch.randn(3, 4, dtype=torch.bfloat16, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    before = torch.get_rng_state().clone()
    ref = checkpoint(reference, x, use_reentrant=False)
    ref.float().square().sum().backward()
    rng = torch.get_rng_state().clone()
    torch.set_rng_state(before)
    out = checkpoint(candidate, y, use_reentrant=False)
    out.float().square().sum().backward()
    assert torch.equal(ref, out) and torch.equal(x.grad, y.grad)
    for a, b in zip(reference.parameters(), candidate.parameters(), strict=True):
        assert (a.grad is None and b.grad is None) or torch.equal(a.grad, b.grad)
    assert torch.equal(torch.get_rng_state(), rng)
    assert receipt['grad_enabled_calls'] == 2  # original + real recompute
    with torch.no_grad():
        candidate(y)
    assert receipt['no_grad_calls'] == 1 and receipt['cached_parameter_bytes'] == 0
    candidate.a.weight.device_mesh = 'SYNTHETIC_SHARDED_SENTINEL'
    with pytest.raises(RuntimeError, match='materialization/dtype'):
        candidate(y)
    assert receipt['contract_failures'] == 1


def test_configuration_to_real_worker_cycles_and_collector(monkeypatch):
    import json
    from test_refactor_wan13b_factorial_r2 import cpu_chain, load_script
    rows = []
    for rank in range(16):
        with monkeypatch.context() as patch:
            cfg, row, events, _ = cpu_chain(patch, 'only_B_fsdp_compatible', rank)
            assert [e[0] for e in events] == (['G'] + ['R'] * 5 + ['C'] * 5) * 4
            assert row['runtime_cache_created'] and row['runtime_cache_keys']
            assert row['mechanism_receipts']['graph']['enabled'] is False
            assert row['mechanism_receipts']['cast']['enabled'] is False
            assert 'fsdp_compatible_scope' in row['mechanism_receipts']
            rows.append(row)
    collector = load_script('only_b_collector', 'scripts/distillschedule/summarize_wan13b_factorial_r2_world16.py')
    rows = json.loads(json.dumps(rows))
    summary = collector.summarize_rows(rows, cfg, run_id='SYNTHETIC_TEST_ONLY', allow_synthetic_test_only=True)
    assert summary['errors'] == [], summary['errors']
    assert summary['numerical_status'] == 'NOT_RUN'
    assert collector.summarize_rows(rows, cfg, run_id='SYNTHETIC_TEST_ONLY')['status'] == 'FAIL'
    for edit in ('chunk', 'scope', 'graph'):
        bad = copy.deepcopy(rows)
        meta = bad[0]['build_evidence']['generator']['metadata']
        if edit == 'chunk':
            meta['modulation_chunk_runtime']['effective_tokens'] = [2048]
        elif edit == 'scope':
            meta['fsdp_compatible_b']['contract_failures'] = 1
        else:
            meta['generator_rollout_cuda_graph']['enabled'] = True
        assert collector.summarize_rows(bad, cfg, run_id='SYNTHETIC_TEST_ONLY',
                                       allow_synthetic_test_only=True)['status'] == 'FAIL'


def test_only_new_source_bound_config_unlocks_controller_and_worker(tmp_path, monkeypatch):
    import json
    import hashlib
    from test_refactor_wan13b_factorial_r2 import load_script, ROOT
    cfg = configuration()
    p = tmp_path / 'compatible.json'
    p.write_text(json.dumps(cfg))
    ctrl = load_script('only_b_controller', 'scripts/distillschedule/run_wan13b_fair_world16.py')
    args = ctrl.parser().parse_args([
        '--run-id', 'SYNTHETIC_TEST_ONLY', '--mode', 'four-cycle-factorial-r2',
        '--placement', r2.FSDP, '--expected-head', '2' * 40, '--model-root', '/model',
        '--bootstrap', '/bootstrap', '--bootstrap-hash', r2.MATCHED['bootstrap_hash'],
        '--factorial-config', str(p), '--factorial-config-sha256', hashlib.sha256(p.read_bytes()).hexdigest(),
        '--master-addr', '192.0.2.1'])
    command = ctrl._command(args, 0, args.master_addr)
    text = ' '.join(command)
    assert 'MODULATED_NORM_FUSION=1' in text and 'GENERATOR_ROLLOUT_CUDA_GRAPH=0' in text
    worker = load_script('only_b_worker', 'scripts/distillschedule/probe_wan13b_factorial_r2_world16.py')
    legacy = worker.load_legacy()
    for key, value in r2.lower_environment(cfg).items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv('NCCL_ALGO', raising=False)
    assert legacy._validate_arm_environment(r2.FSDP, factorial_configuration=cfg)['modulated_norm_fusion']
    with pytest.raises(ValueError, match='FSDP control'):
        legacy._validate_arm_environment(r2.FSDP)
    monkeypatch.setenv('NCCL_ALGO', 'Tree')
    with pytest.raises(ValueError, match='NCCL algorithm'):
        legacy._validate_arm_environment(r2.FSDP, factorial_configuration=cfg)


def test_norm_registration_uses_512_and_rejects_wrong_shared_scope(monkeypatch):
    import sys
    from types import SimpleNamespace
    from distillschedule.runtime import wan13b_norm_policy as policy
    from distillschedule.runtime import wan13b_modulated_norm_fusion as impl
    transformer = SimpleNamespace()
    monkeypatch.setitem(sys.modules, 'helios.modules', SimpleNamespace(transformer_helios=transformer))
    monkeypatch.setenv(policy.ENV, '1')
    def install(**kwargs):
        receipt = dict(enabled=True, requested=True)
        transformer._distillschedule_modulated_norm_receipt = receipt
        return receipt
    def warmup(receipt, *, device, chunk_tokens):
        assert device == 'SYNTHETIC_TEST_ONLY' and chunk_tokens == 512
        receipt['test_warmup_tokens'] = chunk_tokens
    monkeypatch.setattr(impl, 'install_exact_modulated_norm_epilogue', install)
    monkeypatch.setattr(impl, 'warmup_exact_modulated_norm_epilogue', warmup)
    with pytest.raises(ValueError):
        policy.install(role='generator', placement=r2.FSDP, device=None)
    g = policy.install(role='generator', placement=r2.FSDP, device='SYNTHETIC_TEST_ONLY', fsdp_compatible=True)
    c = policy.install(role='critic', placement=r2.FSDP, device=None, fsdp_compatible=True)
    assert g is c and g['native_chunk_tokens'] == g['test_warmup_tokens'] == 512
    with pytest.raises(RuntimeError, match='scope differs'):
        policy.install(role='critic', placement=r2.RESIDENT, device=None)
