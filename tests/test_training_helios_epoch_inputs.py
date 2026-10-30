"""CPU/SYNTHETIC_TEST_ONLY record/operation protocol, not Helios quality/GPU proof."""
from copy import deepcopy
import json
from pathlib import Path
import random
import os
import subprocess
import sys

import pytest
import torch

from distillschedule.training import helios_inputs as nogan, helios_gan_inputs as gan
from distillschedule.workloads.best_known.registry import file_sha256, fingerprint

POLICY = "distillschedule.helios_record_epochs.v1"


@pytest.fixture(scope="module")
def finite_pool(tmp_path_factory):
    root = tmp_path_factory.mktemp("helios-epoch-CPU-SYNTHETIC_TEST_ONLY")
    records = []
    for index in range(80):
        path = root / f"record-{index:03d}.pt"
        # Native-format encoded sections, no trace/noise/rollout aliases. Views
        # keep physical fixture files tiny, not the admitted tensor geometry.
        torch.save({"vae_latent": torch.tensor(.01 * (index + 1)).expand(2,16,9,48,80),
                    "prompt_embed": torch.tensor(.02 * (index + 1)).expand(2,4096),
                    "attention_mask": torch.ones(2, dtype=torch.bool)}, path)
        records.append(dict(id=f"source-{index:03d}", path=str(path), sha256=file_sha256(path)))
    return dict(schema_version="distillschedule.helios_record_pool.v1", seed=31415, cycles=100,
                evidence_kind="CPU_SYNTHETIC_TEST_ONLY", records=records)


def test_v1_default_still_refuses_pool_reuse_without_loading_or_capturing(finite_pool, tmp_path):
    paths = [r["path"] for r in finite_pool["records"]]
    source = nogan.HeliosCohortSources(31415, tuple(tuple(paths[i:i+16]) for i in range(0,80,16)))
    with pytest.raises(ValueError, match="globally distinct feature paths"):
        nogan.prepare_stream(nogan.HeliosCohortTask(0).request, [source]*100,
                             tmp_path/"absent.yaml", tmp_path/"refused")
    assert not (tmp_path/"refused").exists()


@pytest.mark.parametrize("task_type", [nogan.HeliosCohortTask, gan.GANCohortTask])
def test_finite_80_record_pool_has_100_fresh_logical_cohorts(finite_pool, task_type):
    from distillschedule.data.helios_epoch_inputs import build_cohorts, validate_identity
    random.seed(271); torch.manual_seed(272)
    py_rng, th_rng = random.getstate(), torch.get_rng_state().clone()
    sources, inventories, policy = build_cohorts(finite_pool, task_type=task_type, input_policy=POLICY)
    identity = dict(sources=[s.to_dict() for s in sources], inventory=inventories, input_policy=policy)
    validate_identity(identity, task_type=task_type)
    rows = [r for cohort in inventories for r in cohort]
    assert len(sources) == len(inventories) == 100
    assert len({r["path"] for r in rows}) == len({r["record_id"] for r in rows}) == 80
    assert len({r["draw_id"] for r in rows}) == 8000
    assert [r["draw_index"] for r in rows] == list(range(8000))
    assert policy["schema_version"] == POLICY and policy["epoch_count"] == 100
    for cycle, cohort in enumerate(inventories):
        assert len({r["record_id"] for r in cohort}) == 80
        assert [(r["step"],r["rank"]) for r in cohort] == [(s,r) for s in task_type(cycle).source_steps for r in range(16)]
        assert all(r["epoch_index"] == cycle for r in cohort)
    assert random.getstate() == py_rng and torch.equal(torch.get_rng_state(), th_rng)
    assert not torch.cuda.is_initialized()


@pytest.fixture(scope="module")
def epoch_bindings(finite_pool):
    from distillschedule.data.helios_epoch_inputs import build_cohorts
    return {task_type: build_cohorts(finite_pool, task_type=task_type, input_policy=POLICY)
            for task_type in (nogan.HeliosCohortTask, gan.GANCohortTask)}


def test_real_native_loader_reuses_only_original_records_and_operation_rng_advances(epoch_bindings):
    from distillschedule.strategies.wan_inputs import load_feature
    from distillschedule.strategies.wan_capture import input_stream
    for task_type, (sources, inventories, _) in epoch_bindings.items():
        a, b = inventories[0][0], inventories[1][0]
        assert a["record_id"] == b["record_id"] and a["draw_id"] != b["draw_id"]
        tensors = [load_feature(task_type(i), row) for i,row in enumerate((a,b))]
        assert all(torch.equal(x,y) for x,y in zip(*tensors))
        random.seed(401); torch.manual_seed(402)
        py_rng, th_rng = random.getstate(), torch.get_rng_state().clone()
        probes = []
        # Actual existing operation RNG owner; tiny draws, NOT native math.
        for index,row in enumerate((a,b)):
            with input_stream(sources[index].seed,index,row["rank"],row["step"],"critic","entry_noise"):
                probes.append((torch.randn(8), random.random()))
        assert not torch.equal(probes[0][0],probes[1][0]) and probes[0][1] != probes[1][1]
        assert py_rng == random.getstate() and torch.equal(th_rng,torch.get_rng_state())


@pytest.mark.parametrize("mutation", ["insufficient","duplicate_id","duplicate_path","copy","reserialized",
                                    "hash","shape","mask","seed","cycles","evidence","unknown_field"])
def test_invalid_pool_fails_closed_before_native_capture(finite_pool,tmp_path,mutation):
    from distillschedule.data.helios_epoch_inputs import build_cohorts
    value = deepcopy(finite_pool)
    if mutation == "insufficient": value["records"].pop()
    elif mutation == "duplicate_id": value["records"][1]["id"] = value["records"][0]["id"]
    elif mutation == "duplicate_path": value["records"][1]["path"] = value["records"][0]["path"]
    elif mutation in ("copy","reserialized","shape","mask"):
        source = Path(value["records"][0]["path"])
        replacement = tmp_path/"replacement.pt"
        if mutation == "copy": replacement.write_bytes(source.read_bytes())
        else:
            payload = torch.load(source,weights_only=True)
            if mutation == "reserialized": payload["vae_latent"] = payload["vae_latent"].clone()
            elif mutation == "shape": payload["vae_latent"] = torch.zeros(1,16,8,48,80)
            else: payload["attention_mask"][0] = 2
            # bool masks cannot represent 2; use an explicit illegal mask.
            if mutation == "mask": payload["attention_mask"] = torch.tensor([2,1])
            torch.save(payload,replacement)
        value["records"][1].update(path=str(replacement),sha256=file_sha256(replacement))
    elif mutation == "hash": value["records"][0]["sha256"] = "f"*64
    elif mutation == "seed": value["seed"] = True
    elif mutation == "cycles": value["cycles"] = 101
    elif mutation == "evidence": value["evidence_kind"] = "PRETRAINED_QUALITY_PASS"
    else: value["ignored"] = True
    with pytest.raises(ValueError):
        build_cohorts(value,task_type=nogan.HeliosCohortTask,input_policy=POLICY)


@pytest.mark.parametrize("policy", [None,"epochs","distillschedule.helios_record_epochs.v2",True])
def test_unknown_or_implicit_policy_cannot_enable_record_reuse(finite_pool,tmp_path,policy):
    with pytest.raises(ValueError):
        nogan.prepare_stream(nogan.HeliosCohortTask(0).request,finite_pool,tmp_path/"no-config",tmp_path/"output",
                             input_policy=policy)
    assert not (tmp_path/"output").exists()


@pytest.mark.parametrize("mutation", ["order","draw_id","rank","epoch","source_seed","pool_hash","policy","stripped"])
def test_rehashed_identity_cannot_change_epoch_or_source_semantics(epoch_bindings,mutation):
    from distillschedule.data.helios_epoch_inputs import validate_identity
    sources,inventory,policy = epoch_bindings[nogan.HeliosCohortTask]
    identity = deepcopy(dict(sources=[s.to_dict() for s in sources],inventory=inventory,input_policy=policy))
    if mutation == "order": identity["input_policy"]["record_order"].reverse()
    elif mutation == "draw_id": identity["inventory"][50][0]["draw_id"] = "0"*64
    elif mutation == "rank": identity["inventory"][50][0]["rank"] = 9
    elif mutation == "epoch": identity["inventory"][50][0]["epoch_index"] = 0
    elif mutation == "source_seed": identity["sources"][50]["seed"] += 1
    elif mutation == "pool_hash": identity["input_policy"]["records"][0]["sha256"] = "0"*64
    elif mutation == "policy": identity["input_policy"]["schema_version"] = "unknown"
    else: identity["input_policy"] = None
    with pytest.raises(ValueError): validate_identity(identity,task_type=nogan.HeliosCohortTask)


def test_non_multiple_pool_never_drops_or_repeats_within_a_cohort(finite_pool,tmp_path):
    from distillschedule.data.helios_epoch_inputs import build_cohorts
    value = deepcopy(finite_pool); value["cycles"] = 3
    path = tmp_path/"record-80.pt"
    torch.save({"vae_latent":torch.tensor(9.).expand(2,16,9,48,80),
                "prompt_embed":torch.tensor(8.).expand(2,4096),"attention_mask":torch.ones(2)},path)
    value["records"].append(dict(id="record-80",path=str(path),sha256=file_sha256(path)))
    _,inventory,policy = build_cohorts(value,task_type=nogan.HeliosCohortTask,input_policy=POLICY)
    rows = sum(inventory,[])
    assert len({r["record_id"] for r in rows[:81]}) == 81
    assert all(len({r["record_id"] for r in cohort}) == 80 for cohort in inventory)
    assert rows[80]["epoch_index"] == 0 and rows[81]["epoch_index"] == 1
    assert policy["epoch_count"] == 3 and len(rows) == 240


def _metadata_capture(task,sources,inventory,config,*,rank,output,provenance):
    """Bounded substitute for capture ONLY: real metadata loader, no tensors.

    Deliberately does not prove actual capture, native dependencies, history,
    scheduler, training, noise content or rank payload numerical admission.
    """
    from runtime.operation_trace import OperationTrace, _manifest_hash
    is_gan = task.request.variant == "gan"
    inputs = gan if is_gan else nogan
    selected_keys = gan.keys if is_gan else nogan._keys
    manifest = dict(schema_version="distillschedule.operation_trace.v1",
        classification="CPU_SYNTHETIC_TEST_ONLY_METADATA_NO_PAYLOAD",
        source=dict(**provenance,rank=rank,samples=[r for r in inventory if r["rank"] == rank],
                    seed=sources.seed,cohort=task.index,rng_scope="new_native_CPU_streams_not_prior_training_rng_continuation"),
        contract={inputs.TRACE_KEY:task.to_dict(),"resolution":[384,640],"full_latent_shape":[1,16,9,48,80],
                  "stage_spatial_shapes":[[12,20],[24,40],[48,80]],"dtype":"bfloat16","dfake_gen_update_ratio":5},
        tensors={},operations=[])
    def tensor(key,shape,dtype="bfloat16"):
        manifest["tensors"][key] = dict(shape=shape,dtype=dtype,sha256=fingerprint([key,rank]))
    def operation(key,control): manifest["operations"].append(dict(key=key,control=control))
    for step in task.source_steps:
        operation(f"step/{step}/branch",dict(global_step=step,dfake_gen_update_ratio=5,
            train_generator=step == task.first_step,train_critic=True,gan=is_gan,reward=False,vae_decode=False))
        tensor(f"step/{step}/sample/text_embedding",[2,4096]); tensor(f"step/{step}/sample/attention_mask",[2],"bool")
        if is_gan:
            for source in ("primary","secondary"):
                for role,frames in (("x0",1),("history",19),("target",9)):
                    tensor(f"step/{step}/sample/{source}/{role}",[16,frames,48,80])
        for phase in (("generator","critic") if step == task.first_step else ("critic",)):
            prefix = f"step/{step}/{phase}"
            for role,key in selected_keys(step,phase).items():
                if role in ("text_embedding","attention_mask"): continue
                shape,dtype = [1,16,9,48,80],"bfloat16"
                if role.startswith("indices_"):
                    shape,dtype = [1,{"indices_hidden_states":9,"indices_history_short":2,
                                     "indices_history_mid":2,"indices_history_long":16}[role]],"int64"
                elif role.endswith("_timestep"): shape,dtype = [1],"int64"
                elif role.endswith("_crop"): shape,dtype = [1,2],"int64"
                elif role == "block_noise_stage1": shape[-2:] = [24,40]
                elif role in ("history_short","history_mid","history_long","continuous_history"):
                    shape[2] = {"history_short":2,"history_mid":2,"history_long":16,"continuous_history":19}[role]
                tensor(key,shape,dtype)
            schedule = [dict(stage=s,call_count=1,denoising_exit=0,denoising_step_budget=2,
                             latent_shape=[1,16,9,*hw],timestep=[float(900-s*300)]) for s,hw in enumerate(task.stage_spatial_shapes)]
            operation(prefix+"/rollout",dict(pyramid_stage_call_sequence=schedule,
                initial_synchronized_pyramid_stage_exit=[0,0,0],call_count=3,stage_schedule_scope="global_rank0_broadcast_equivalent"))
            operation(prefix+("/dmd" if phase == "generator" else "/train"),dict(global_step=step))
            if is_gan: operation(prefix+("/gan_g" if phase == "generator" else "/gan_d"),dict(global_step=step))
    manifest["trace_hash"] = _manifest_hash(manifest)
    output = Path(output); output.mkdir(parents=True)
    (output/"trace.json").write_text(json.dumps(manifest))
    return OperationTrace(root=output,manifest=manifest,tensors={})


@pytest.fixture(scope="module")
def metadata_streams(finite_pool,tmp_path_factory):
    """Original public producer + real stream/trace metadata admission for 100.

    Only capture is replaced with clearly labeled metadata, never claimed to
    exercise payload/history generation. The immutable pool loader is real.
    """
    import yaml
    from distillschedule.workloads.best_known.registry import repository_root
    root = tmp_path_factory.mktemp("helios-100-metadata-only")
    result = {}
    with pytest.MonkeyPatch.context() as mp:
        for inputs,task_type in ((nogan,nogan.HeliosCohortTask),(gan,gan.GANCohortTask)):
            variant = "gan" if inputs is gan else "nogan"
            config = root/f"{variant}.yaml"
            if inputs is gan: value = gan.portable_config()
            else: value = yaml.safe_load((repository_root()/"recipes/helios_nogan/inputs.reference.yaml").read_text())
            config.write_text(yaml.safe_dump(value))
            mp.setattr(inputs,"capture_rank",_metadata_capture)
            output = root/variant
            if inputs is gan: doc = inputs.prepare_stream(finite_pool,config,output,input_policy=POLICY)
            else: doc = inputs.prepare_stream(task_type(0).request,finite_pool,config,output,input_policy=POLICY)
            result[inputs.KIND] = output/"stream.json",doc
    return result


@pytest.mark.parametrize("inputs",[nogan,gan])
def test_public_producer_and_consumer_bind_100_cohorts_without_8000_source_aliases(metadata_streams,inputs):
    path,doc = metadata_streams[inputs.KIND]
    assert inputs.load_stream(path,cycles=100) == doc
    assert len(doc["cohorts"]) == 100 and len(doc["identity"]["input_policy"]["records"]) == 80
    assert len({c["input_policy"]["cohort_id"] for c in doc["cohorts"]}) == 100
    # No payload materialization or GPU claim can leak from metadata admission.
    with pytest.raises((FileNotFoundError,ValueError)):
        inputs.load_stream(path,cycles=100,verify_rank=0)
    assert doc["gpu_execution"] == "NOT_RUN" and not doc["claimable_speedup"]


@pytest.mark.parametrize("inputs",[nogan,gan])
@pytest.mark.parametrize("mutation",["policy_removed","cohort","draw","rank","trace_source"])
def test_real_consumer_refuses_rehashed_policy_draw_and_rank_metadata(metadata_streams,inputs,mutation):
    path,doc = metadata_streams[inputs.KIND]
    changed = deepcopy(doc)
    if mutation == "policy_removed": changed["identity"].pop("input_policy")
    elif mutation == "cohort": changed["cohorts"][50]["input_policy"]["cohort_id"] = "0"*64
    elif mutation == "draw": changed["identity"]["inventory"][50][0]["draw_id"] = "0"*64
    elif mutation == "rank": changed["identity"]["inventory"][50][0]["rank"] = 1
    else:
        # Correctly re-hash trace and bundle: fail must be source provenance.
        from runtime.operation_trace import _manifest_hash, write_operation_trace_bundle_manifest
        trace = path.parent/"cohort-0050/rank_00/trace.json"
        original = trace.read_text(); manifest = json.loads(original)
        bundle_files = {p:p.read_bytes() for p in trace.parent.parent.glob("*.json")}
        manifest["source"]["input_policy"]["first_draw"] = 0
        manifest["trace_hash"] = _manifest_hash(manifest); trace.write_text(json.dumps(manifest))
        ranks = [dict(rank=r,path=f"rank_{r:02d}",trace_hash=json.loads((trace.parent.parent/f"rank_{r:02d}/trace.json").read_text())["trace_hash"])
                 for r in range(16)]
        try:
            bundle = write_operation_trace_bundle_manifest(trace.parent.parent,ranks)
            changed["cohorts"][50]["trace_hash"] = bundle["bundle_hash"]
            changed.pop("stream_sha256"); changed["stream_sha256"] = fingerprint(changed)
            altered = path.with_name("changed.json"); altered.write_text(json.dumps(changed))
            with pytest.raises(ValueError,match="provenance"):
                inputs.load_stream(altered,cycles=100)
        finally:
            trace.write_text(original)
            for p,content in bundle_files.items(): p.write_bytes(content)
        return
    changed.pop("stream_sha256"); changed["stream_sha256"] = fingerprint(changed)
    altered = path.with_name("changed.json"); altered.write_text(json.dumps(changed))
    with pytest.raises(ValueError): inputs.load_stream(altered,cycles=100)


def _checkpoint_protocol_run(spec,output,*,resume=None,end=100):
    """Real checkpoint/data/RNG protocol, not a substitute Helios trainer.

    No neural model, loss, optimizer, EMA or native capture is executed. Those
    unchanged owners have separate R30/R64 proofs; this tests only new inputs.
    """
    from runtime.operation_trace import tensor_sha256
    from distillschedule.data.helios_epoch_inputs import build_cohorts,identity_binding,receipt
    from distillschedule.strategies.wan_capture import input_stream
    from distillschedule.training import checkpoint as io,helios_lifecycle,helios_gan_training
    import numpy as np
    value = json.loads(Path(spec).read_text())
    task_type = gan.GANCohortTask if value.pop("variant") == "gan" else nogan.HeliosCohortTask
    owner = helios_gan_training if task_type is gan.GANCohortTask else helios_lifecycle
    random.seed(501); torch.manual_seed(502); np.random.seed(503)
    before = io.rng_state()
    sources,inventory,policy = build_cohorts(value,task_type=task_type,input_policy=POLICY)
    assert fingerprint(before["python"]) == fingerprint(io.rng_state()["python"])
    assert torch.equal(before["torch"],torch.get_rng_state())
    assert before["numpy"] == io.rng_state()["numpy"]
    stream = dict(identity=dict(input_policy=policy),cohorts=list(range(100)),stream_sha256=fingerprint(inventory))
    identity = dict(schema_version="CPU_SYNTHETIC_TEST_ONLY_HELIOS_EPOCH_RESTART",
                    stream_sha256=stream["stream_sha256"],**identity_binding(stream))
    checkpoint_policy = (owner._checkpoint_policy(identity) if task_type is gan.GANCohortTask
                         else owner._checkpoint_policy(identity,optimized=False))
    binding = dict(stream=stream,identity=identity,checkpoint_policy=checkpoint_policy)
    state = dict(draws=[],probes=[],training_draws=[],counter=torch.tensor([0],dtype=torch.int64))
    start = 0
    if resume:
        header = io.read_manifest(resume,identity=identity)
        owner.validate_progress(header,binding)
        restored = io.restore_checkpoint(resume,state,identity=identity)
        start = restored["cursor"]
        assert receipt(stream,cycle=start)["input_draw_cursor"] == 4000
    for cycle in range(start,end):
        rows = inventory[cycle]
        state["draws"].extend(r["draw_id"] for r in rows)
        # Use every rank and both native phase namespaces, with actual existing
        # input_stream contexts; tiny RNG leaves do not claim full capture math.
        probes = []
        for row in rows:
            phases = ("generator","critic") if row["step"] == task_type(cycle).first_step else ("critic",)
            for phase in phases:
                with input_stream(sources[cycle].seed,cycle,row["rank"],row["step"],phase,"entry_noise"):
                    probes.append([tensor_sha256(torch.randn(4)),random.random()])
        state["probes"].append(fingerprint(probes))
        state["training_draws"].append([random.random(),torch.rand(1).item(),float(np.random.rand())])
        state["counter"] += 80
        if cycle+1 == end and end == 50:
            checkpoint = Path(output).parent/"checkpoint50"
            io.save_checkpoint(checkpoint,state,identity=identity,cycle=50,cursor=50,
                               versions=owner.expected_versions(50),policy=checkpoint_policy)
    rng = io.rng_state()
    result = dict(counter=state["counter"].item(),draws_sha256=fingerprint(state["draws"]),
        probes=state["probes"],training_draws=state["training_draws"],
        python_rng_sha256=fingerprint(rng["python"]),torch_rng_sha256=tensor_sha256(rng["torch"]),
        numpy_rng_sha256=fingerprint(rng["numpy"]),
        input_receipt=receipt(stream,cycle=end),checkpoint_identity=identity,scope="CPU_SYNTHETIC_TEST_ONLY_PROTOCOL")
    Path(output).write_text(json.dumps(result))


@pytest.mark.parametrize("variant",["nogan","gan"])
def test_fresh_process_horizon100_checkpoint50_preserves_data_and_rng(finite_pool,tmp_path,variant):
    spec = tmp_path/"spec.json"; spec.write_text(json.dumps(dict(**finite_pool,variant=variant)))
    full,partial,resumed = [tmp_path/f"{label}.json" for label in ("full","partial","resumed")]
    _checkpoint_protocol_run(spec,full)
    _checkpoint_protocol_run(spec,partial,end=50)
    paths = [str(Path(__file__).resolve().parent)]
    if os.environ.get('PYTHONPATH'):
        paths.append(os.environ['PYTHONPATH'])
    child = subprocess.run([sys.executable,"-c",
        "from test_training_helios_epoch_inputs import _checkpoint_protocol_run; "
        "import sys; _checkpoint_protocol_run(sys.argv[1],sys.argv[2],resume=sys.argv[3])",
        str(spec),str(resumed),str(tmp_path/"checkpoint50")],
        env=dict(os.environ,PYTHONPATH=os.pathsep.join(paths)),text=True,capture_output=True,timeout=120)
    assert child.returncode == 0, child.stdout+child.stderr
    result = json.loads(full.read_text()); restored = json.loads(resumed.read_text())
    assert result == restored
    assert result["counter"] == result["input_receipt"]["input_draw_cursor"] == 8000
    assert len(set(result["probes"])) == 100
    assert json.loads(partial.read_text())["input_receipt"]["input_draw_cursor"] == 4000


@pytest.mark.parametrize("variant",["nogan","gan"])
def test_wrong_cursor_or_rng_or_policy_cannot_resume(finite_pool,tmp_path,variant):
    from distillschedule.training import checkpoint as io,helios_lifecycle,helios_gan_training
    spec = tmp_path/"spec.json"; spec.write_text(json.dumps(dict(**finite_pool,variant=variant)))
    output = tmp_path/"partial.json"; _checkpoint_protocol_run(spec,output,end=50)
    result = json.loads(output.read_text()); path = tmp_path/"checkpoint50"
    identity = result["checkpoint_identity"]
    header = io.read_manifest(path,identity=identity)
    owner = helios_gan_training if variant == "gan" else helios_lifecycle
    binding = dict(stream=dict(cohorts=list(range(100)),identity=dict(input_policy=dict(cycles=100))),
                   checkpoint_policy=header["policy"])
    for cursor in (0,49,51,True,50.):
        changed = deepcopy(header); changed["cursor"] = cursor
        with pytest.raises(ValueError): owner.validate_progress(changed,binding)
    changed = deepcopy(identity); changed["input_policy"]["draws_sha256"] = "0"*64
    with pytest.raises(ValueError,match="identity changed"): io.read_manifest(path,identity=changed)
    payload = torch.load(path/"state.pt",weights_only=True)
    # Adversarial re-signing does not authorize a malformed saved training RNG.
    payload["rng"]["torch"] = torch.ones(2,dtype=torch.float32)
    torch.save(payload,path/"state.pt")
    header["files"]["state.pt"] = file_sha256(path/"state.pt")
    (path/"manifest.json").write_text(json.dumps(header))
    (path/"_SUCCESS").write_text(file_sha256(path/"manifest.json"))
    random.seed(601); torch.manual_seed(602)
    before = (random.getstate(),torch.get_rng_state().clone())
    with pytest.raises((ValueError,TypeError),match="RNG|rng"):
        io.restore_checkpoint(path,dict(draws=[],probes=[],training_draws=[],counter=torch.tensor([0])),identity=identity)
    assert before[0] == random.getstate() and torch.equal(before[1],torch.get_rng_state())


@pytest.mark.parametrize("command",["prepare-helios-stream","prepare-helios-gan-stream"])
def test_existing_public_data_command_requires_explicit_policy(finite_pool,tmp_path,capsys,command):
    from distillschedule.data.cli import main
    path = tmp_path/"pool.json"; path.write_text(json.dumps(finite_pool))
    args = [command,"--sources",str(path),"--training-config",str(tmp_path/"absent.yaml"),"--output",str(tmp_path/"out")]
    assert main(args) == 2
    assert "explicit source cohort list required" in capsys.readouterr().err
    # Recognized opt-in reaches normal exact native-config admission, not a
    # private preparer/second trainer. No config means no publication.
    assert main(args+["--input-policy",POLICY]) == 2
    assert "absent.yaml" in capsys.readouterr().err
    assert not (tmp_path/"out").exists()


@pytest.mark.parametrize("function,sha",[
    (nogan._require_fresh_cohorts,"463342446fc2cfe30bd428e7d5e61751d37d0db13987ab6f97f873b4ea181ef2"),
])
def test_native_capture_math_and_frozen_v1_freshness_are_byte_identical(function,sha):
    import hashlib
    import inspect
    assert hashlib.sha256(inspect.getsource(function).rstrip("\n").encode()).hexdigest() == sha


def test_nogan_capture_has_only_the_versioned_reviewed_native_history_opt_in():
    import inspect
    from helios_capture_source_review import assert_nogan_capture_lineage
    # Actual source is reviewed as a bounded extension, never called byte-identical.
    # The helper retains the original golden and compares with its actual parent.
    assert_nogan_capture_lineage(inspect.getsource(nogan.capture_rank))


def test_gan_capture_has_only_the_versioned_reviewed_native_history_opt_in():
    import inspect
    from helios_capture_source_review import assert_gan_capture_lineage
    assert_gan_capture_lineage(inspect.getsource(gan.capture_rank))


@pytest.mark.parametrize("inputs,command",[(nogan,"prepare-helios-stream"),(gan,"prepare-helios-gan-stream")])
def test_public_opt_in_command_publishes_only_new_metadata_not_source_aliases(
        finite_pool,tmp_path,capsys,monkeypatch,inputs,command):
    import yaml
    from distillschedule.data.cli import main
    from distillschedule.workloads.best_known.registry import repository_root
    value = deepcopy(finite_pool); value["cycles"] = 2
    path = tmp_path/"pool.json"; path.write_text(json.dumps(value))
    config = tmp_path/"native.yaml"
    config.write_text(yaml.safe_dump(gan.portable_config() if inputs is gan else
        yaml.safe_load((repository_root()/"recipes/helios_nogan/inputs.reference.yaml").read_text())))
    output = tmp_path/"prepared"
    monkeypatch.setattr(inputs,"capture_rank",_metadata_capture)
    assert main([command,"--sources",str(path),"--training-config",str(config),"--output",str(output),
                 "--input-policy",POLICY]) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert inputs.load_stream(output/"stream.json",cycles=2) == receipt
    assert receipt["identity"]["input_policy"]["draw_count"] == 160
    assert not list(output.rglob("*.pt"))
    assert len(list(Path(value["records"][0]["path"]).parent.glob("*.pt"))) == 80


@pytest.mark.parametrize("task_type",[nogan.HeliosCohortTask,gan.GANCohortTask])
def test_repeated_pool_keeps_native_rank0_stage_control_seeds_fresh(epoch_bindings,task_type):
    from distillschedule.strategies.wan_capture import input_stream
    sources,_,_ = epoch_bindings[task_type]
    random.seed(701); torch.manual_seed(702)
    before = (random.getstate(),torch.get_rng_state().clone())
    seeds = []
    for cycle,source in enumerate(sources):
        for offset,step in enumerate(task_type(cycle).source_steps):
            for phase in (("generator","critic") if offset == 0 else ("critic",)):
                # Exact existing stage_control namespace: rank0 is global,
                # unlike rank-qualified history/noise. This tests seed scope,
                # not native generate_and_sync_flag/scheduler execution.
                with input_stream(source.seed,cycle,step,phase,"rank0-stage-exits"):
                    seeds.append(torch.initial_seed())
    assert len(seeds) == len(set(seeds)) == 600
    assert before[0] == random.getstate() and torch.equal(before[1],torch.get_rng_state())


def test_new_pool_uses_existing_restricted_loader_without_unsafe_pickle(finite_pool,tmp_path):
    import pickle
    from distillschedule.data.helios_epoch_inputs import build_cohorts
    marker = tmp_path/"UNSAFE_PICKLE_EXECUTED"
    class Unsupported:
        def __reduce__(self):
            return eval,(f"__import__('pathlib').Path({str(marker)!r}).write_text('unsafe')",)
    path = tmp_path/"unsafe.pt"; torch.save(dict(vae_latent=Unsupported()),path)
    value = deepcopy(finite_pool)
    value["records"][0].update(path=str(path),sha256=file_sha256(path))
    with pytest.raises(pickle.UnpicklingError):
        build_cohorts(value,task_type=nogan.HeliosCohortTask,input_policy=POLICY)
    assert not marker.exists()


@pytest.mark.parametrize("task_type", [nogan.HeliosCohortTask, gan.GANCohortTask])
@pytest.mark.parametrize("mask_dtype", [torch.int64, torch.float32])
def test_binary_mask_dtype_does_not_manufacture_a_distinct_record(finite_pool, tmp_path, task_type, mask_dtype):
    from distillschedule.data.helios_epoch_inputs import build_cohorts
    value = deepcopy(finite_pool)
    original = torch.load(value["records"][0]["path"], weights_only=True)
    duplicate = {**original, "attention_mask": original["attention_mask"].to(mask_dtype)}
    assert all(torch.equal(original[key], duplicate[key]) for key in original)
    path = tmp_path / "same-native-content-different-mask-dtype.pt"
    torch.save(duplicate, path)
    value["records"][1].update(path=str(path), sha256=file_sha256(path))
    with pytest.raises(ValueError, match="identical native features"):
        build_cohorts(value, task_type=task_type, input_policy=POLICY)
