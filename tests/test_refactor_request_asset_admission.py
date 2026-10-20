"""Disjoint controller/asset-host CPU metadata, never GPU admission evidence."""
import json
from pathlib import Path

import pytest

from distillschedule.search import execution, native
from distillschedule.search.helios_request import OriginalHeliosRequest
from distillschedule.search.livewan_implementation import LiveWanImplementation, LIVEWAN
from distillschedule.search.livewan_request import LiveWanRequest
from distillschedule.workloads.best_known import dispatch
from distillschedule.workloads.best_known.registry import file_sha256, fingerprint, git_head, repository_root


def request_assets(tmp_path, family):
    """Real request trace/bootstrap/source files; no weights or model imports."""
    if family == "livewan":
        from test_refactor_livewan_request import request_trace
        from distillschedule.runtime.livewan_crossscale_worker import LIVEWAN_COMMIT, WAN21_COMMIT
        impl = LiveWanImplementation("pipeline", request=LiveWanRequest(frames=69))
        trace, value = request_trace(tmp_path)
        source_file = tmp_path / "prepared.py"
        source_file.write_text("# CPU identity-only prepared-source fixture\n")
        source = tmp_path / "source-receipt.json"
        source_value = dict(status="PASS", source_root=str(tmp_path), livewan_commit=LIVEWAN_COMMIT,
            wan21_commit=WAN21_COMMIT, files={"prepared": dict(path=str(source_file), sha256=file_sha256(source_file))})
        source_value["source_receipt_sha256"] = fingerprint(source_value)
        source.write_text(json.dumps(source_value))
        boot = tmp_path / "bootstrap.json"
        boot_value = dict(kind="systems_only_livewan_crossscale_bootstrap",
            trace=dict(path=str(trace), sha256=value["trace_sha256"]),
            source_receipt=dict(path=str(source), sha256=source_value["source_receipt_sha256"]),
            critic_lora=dict(seed=2026083001))
        boot_value["bootstrap_sha256"] = fingerprint(boot_value)
        boot.write_text(json.dumps(boot_value))
        physical = [trace, tmp_path / "prompts.bin", boot, source, source_file]
        specs = {k: dict(path=str(p), sha256=file_sha256(p))
                 for k, p in (("trace", trace), ("bootstrap", boot), ("source_receipt", source))}
    else:
        from test_helios_parameter_request import implementation, write_bundle
        impl = implementation(OriginalHeliosRequest(variant=family, video_frames=17, stage2_inference_steps=(1,)*3))
        trace = tmp_path / "trace"
        write_bundle(impl.request, trace)
        specs = {"trace": dict(path=str(trace), identity_file="bundle.json", sha256=file_sha256(trace / "bundle.json"))}
        physical = [trace]
    root = repository_root()
    assets = json.loads((root / "configs/distillschedule/best_known/assets_examples" / f"{impl.candidate().workload}.json").read_text())
    assets.update(run_root=str(tmp_path / "runs"))
    for i, node in enumerate(assets["nodes"]):
        node.update(host=f"cpu-asset-host-{i}", repo_root=str(root), python_env="/cpu/venv")
    assets["assets"].update(specs)
    return impl, assets, physical


def clean_checkout(monkeypatch):
    actual = dispatch.subprocess.check_output
    def output(argv, **kwargs):
        return "" if "status" in argv else actual(argv, **kwargs)
    monkeypatch.setattr(dispatch.subprocess, "check_output", output)


@pytest.mark.parametrize("family", ["no-gan", "gan", "livewan"])
def test_asset_host_rpc_resolve_then_disjoint_controller_dispatch(tmp_path, monkeypatch, capsys, family):
    impl, assets, physical = request_assets(tmp_path, family)
    clean_checkout(monkeypatch)
    # Real asset-host RPC entry point and resolver; only the intentionally dirty
    # checkout is a CPU test seam. No transport, model, or physical checks stubbed.
    receipt = native.handle_request(dict(action="resolve", source_head=git_head(repository_root()),
        workload=impl.candidate().workload, candidate_id=impl.id, assets=assets, run_id="disjoint", mode="correctness"))
    for path in physical:
        path.rename(path.with_name(path.name + ".asset-host-only"))
    assert execution.run_plan(receipt, dry_run=True) == 0
    assert json.loads(capsys.readouterr().out)["search_candidate"]["id"] == impl.id
    # The same portable plan is NOT physical admission on the executing host.
    with pytest.raises((ValueError, RuntimeError, FileNotFoundError)):
        dispatch.validate_local(receipt, root=repository_root(), node_rank=0)


@pytest.mark.parametrize("family", ["no-gan", "gan", "livewan"])
@pytest.mark.parametrize("mutation", ["asset_sha", "request", "environment", "native_plan", "command"])
def test_portable_request_binding_never_trusts_tampered_metadata(tmp_path, family, mutation):
    impl, assets, _ = request_assets(tmp_path, family)
    config = tmp_path / "assets.json"; config.write_text(json.dumps(assets))
    receipt = execution.resolve_candidate(impl.candidate().workload, impl.id, assets_path=config, run_id="tamper")
    if mutation == "asset_sha":
        receipt["asset_config"]["assets"]["trace"]["sha256"] = "d"*64
        receipt["asset_fingerprint"] = fingerprint(receipt["asset_config"]["assets"])
    elif mutation == "request":
        key = "livewan_request_trace_binding" if family == "livewan" else "original_helios_trace_binding"
        receipt[key]["request_sha256"] = "d"*64
    elif mutation == "environment":
        receipt["nodes"][0]["environment"]["HF_HUB_OFFLINE"] = "0"
    elif mutation == "command":
        receipt["nodes"][0]["command"].append("--unregistered")
    elif family == "livewan":
        receipt["nodes"][0]["resolved_native_plan"]["workload"]["shape"]["frames"] = 21
    else:
        receipt["nodes"][0]["environment"][impl.trace_hash_environment] = "d"*64
    with pytest.raises(ValueError):
        execution.validated_profile(receipt, root=repository_root())


@pytest.mark.parametrize("family", ["no-gan", "gan", "livewan"])
def test_physical_request_readmission_rejects_asset_corruption(tmp_path, monkeypatch, family):
    impl, assets, _ = request_assets(tmp_path, family)
    config = tmp_path / "assets.json"; config.write_text(json.dumps(assets))
    receipt = execution.resolve_candidate(impl.candidate().workload, impl.id, assets_path=config)
    clean_checkout(monkeypatch)
    path = Path(assets["assets"]["trace"]["path"])
    if family != "livewan": path = path / "rank_15" / "trace.json"
    path.write_text("{}")
    with pytest.raises((ValueError, RuntimeError)):
        dispatch.validate_local(receipt, root=repository_root(), node_rank=0)


@pytest.mark.parametrize('mutation', ['missing_derivation', 'trace_sha', 'request_sha', 'derivation_sha',
    'short_steps', 'short_slots', 'boolean_world', 'changed_cycle', 'bootstrap_path', 'source_sha', 'source_root',
    'native_template_digest', 'node_plan_digest', 'launch_plan_environment'])
def test_livewan_portable_metadata_schema_and_all_plan_digests(tmp_path, mutation):
    impl, assets, physical = request_assets(tmp_path, 'livewan')
    config = tmp_path/'assets.json'; config.write_text(json.dumps(assets))
    receipt = execution.resolve_candidate(LIVEWAN, impl.id, assets_path=config)
    binding = receipt['livewan_request_trace_binding']
    routing = binding['routing_metadata']
    if mutation == 'missing_derivation': binding.pop('routing_metadata')
    if mutation == 'trace_sha': binding['trace_sha256']='c'*64
    if mutation == 'request_sha': routing['request_sha256']='c'*64
    if mutation == 'derivation_sha': routing['sha256']='c'*64
    if mutation == 'short_steps': routing['world_frames_by_step'].pop()
    if mutation == 'short_slots': routing['world_frames_by_step'][0].pop()
    if mutation == 'boolean_world': routing['world_frames_by_step'][0][0]=True
    if mutation == 'changed_cycle': routing['world_frames_by_step'][5][0]=6 if routing['world_frames_by_step'][0][0]!=6 else 9
    if mutation in {'short_steps','short_slots','boolean_world','changed_cycle','request_sha'}:
        routing['sha256']=fingerprint({k:v for k,v in routing.items() if k!='sha256'})
    if mutation == 'bootstrap_path': binding['bootstrap']['trace']['path']='/asset-host/another-trace.json'
    if mutation == 'source_sha': binding['source_receipt']['file_sha256']='c'*64
    if mutation == 'source_root': binding['source_receipt']['source_root']='relative/path'
    if mutation == 'native_template_digest': receipt['native_plan']['digest']='c'*64
    if mutation == 'node_plan_digest': receipt['nodes'][1]['resolved_native_plan_sha256']='c'*64
    if mutation == 'launch_plan_environment': receipt['nodes'][1]['environment']['DISTILLSCHEDULE_LIVEWAN_LAUNCH_PLAN_SHA256']='c'*64
    for path in physical:
        path.rename(path.with_name(path.name+'.asset-host-only'))
    with pytest.raises(ValueError): execution.validated_profile(receipt, root=repository_root())


@pytest.mark.parametrize('asset,match', [('trace','trace file SHA'), ('prompts','prompt bundle SHA'),
    ('bootstrap','bootstrap file SHA'), ('source_receipt','source file SHA'), ('prepared','source receipt file changed')])
def test_livewan_asset_host_rechecks_every_physical_binding(tmp_path, monkeypatch, asset, match):
    impl, assets, _ = request_assets(tmp_path, 'livewan')
    config=tmp_path/'assets.json'; config.write_text(json.dumps(assets))
    receipt=execution.resolve_candidate(LIVEWAN,impl.id,assets_path=config)
    clean_checkout(monkeypatch)
    path = (tmp_path/'prompts.bin' if asset=='prompts' else tmp_path/'prepared.py' if asset=='prepared'
            else Path(assets['assets'][asset]['path']))
    path.write_text('CORRUPTED CPU METADATA')
    with pytest.raises((ValueError,RuntimeError), match=match):
        dispatch.validate_local(receipt,root=repository_root(),node_rank=0)
    # Public asset-host resolution must run the same full physical checks.
    with pytest.raises((ValueError,RuntimeError), match=match):
        execution.resolve_candidate(LIVEWAN,impl.id,assets_path=config)


@pytest.mark.parametrize('family', ['no-gan', 'gan', 'livewan'])
def test_cpu_physical_metadata_readmission_happy_path(tmp_path, monkeypatch, family):
    impl, assets, _=request_assets(tmp_path,family)
    config=tmp_path/'assets.json'; config.write_text(json.dumps(assets))
    receipt=execution.resolve_candidate(impl.candidate().workload,impl.id,assets_path=config)
    clean_checkout(monkeypatch)
    # Heavyweight external payloads and initialized submodules are deliberately
    # absent, not waived in production. All new request checks are real here.
    monkeypatch.setattr(dispatch,'source_layout',lambda *a,**k:{})
    monkeypatch.setattr(dispatch,'verify_assets',lambda *a,**k:[])
    result=dispatch.validate_local(receipt,root=repository_root(),node_rank=0)
    assert result['scope']=='CPU_SOURCE_AND_ASSET_ONLY'
