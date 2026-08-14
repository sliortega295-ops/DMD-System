import datetime
import importlib.util
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import pytest


def _load():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "distillschedule"
        / "probe_routeb_full5_2g6c_continuous.py"
    )
    spec = importlib.util.spec_from_file_location(
        "routeb_full5_2g6c_continuous", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_continuous_layout_and_switch_sets_are_exact() -> None:
    module = _load()
    assert module.STEP10A_G_RANKS == tuple(range(8))
    assert module.STEP10A_STUDENT_RANKS == (8, 9, 10, 11)
    assert module.STEP10A_TEACHER_RANKS == (12, 13, 14, 15)
    assert module.CONTRACT.generator_ranks == (0, 1, 8, 9)
    assert module.CONTRACT.critic_ranks == (
        2,
        3,
        4,
        5,
        6,
        7,
        10,
        11,
        12,
        13,
        14,
        15,
    )
    assert module.SWITCH_TO_CRITIC_RANKS == (2, 3, 4, 5, 6, 7)
    assert module.SWITCH_TO_GENERATOR_RANKS == (8, 9)
    assert module.PRECONSTRUCT_TARGET_RANKS == tuple(range(10))


def test_phase_b_groups_are_world_ordered_and_have_no_ack_group(monkeypatch) -> None:
    module = _load()
    calls = []
    barriers = []

    def fake_new_group(**kwargs):
        group = object()
        calls.append((kwargs, group))
        return group

    monkeypatch.setattr(module.dist, "new_group", fake_new_group)
    monkeypatch.setattr(
        module.dist, "barrier", lambda **kwargs: barriers.append(kwargs)
    )
    module._create_phase_b_groups(module.TOPOLOGY)

    assert calls[0][0]["ranks"] == list(range(16))
    assert calls[0][0]["backend"] == "gloo"
    assert calls[0][0]["timeout"].total_seconds() == 2 * 60 * 60
    assert calls[1][0] == {"ranks": [0, 1, 8, 9], "backend": "nccl"}
    assert calls[2][0] == {
        "ranks": [2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15],
        "backend": "nccl",
    }
    pairs = [call[0] for call in calls[3:15]]
    assert len(pairs) == 12
    assert all(call["backend"] == "nccl" for call in pairs)
    assert all(call["ranks"][0] // 8 == call["ranks"][1] // 8 for call in pairs)
    singletons = [call[0] for call in calls[15:]]
    assert singletons == [
        {"ranks": [rank], "backend": "nccl"} for rank in range(16)
    ]
    control = calls[0][1]
    assert barriers == [{"group": control}] * 31


def test_phase_a_initializes_only_the_local_torch_world(monkeypatch) -> None:
    module = _load()
    calls = []
    initialized = False

    def fake_is_initialized():
        return initialized

    def fake_init_process_group(**kwargs):
        nonlocal initialized
        initialized = True
        calls.append(("torch", kwargs))

    monkeypatch.setattr(module.dist, "is_initialized", fake_is_initialized)
    monkeypatch.setattr(module.dist, "init_process_group", fake_init_process_group)
    monkeypatch.setattr(module.dist, "get_world_size", lambda: 8)
    module._initialize_phase_a_distributed()
    assert calls == [
        (
            "torch",
            {"backend": "nccl", "timeout": datetime.timedelta(hours=2)},
        ),
    ]


def test_phase_a_global_rank_mapping_is_exact() -> None:
    module = _load()
    assert [
        module._phase_a_global_rank(node_rank=node, local_rank=local)
        for node in (0, 1)
        for local in range(8)
    ] == list(range(16))


def test_process_group_gate_checks_rank_membership_not_object_identity(
    monkeypatch,
) -> None:
    module = _load()
    clone = object()
    monkeypatch.setattr(
        module.dist, "get_process_group_ranks", lambda group: list(range(8))
    )
    assert module._require_process_group_ranks(
        clone, expected=tuple(range(8)), label="clone"
    ) == tuple(range(8))

    monkeypatch.setattr(
        module.dist, "get_process_group_ranks", lambda group: list(range(4))
    )
    with pytest.raises(AssertionError, match="ranks changed"):
        module._require_process_group_ranks(
            clone, expected=tuple(range(8)), label="clone"
        )


def test_transition_phase_records_host_wall_without_adding_sync(monkeypatch) -> None:
    module = _load()
    values = iter((10.0, 12.5))
    timeline = {}
    monkeypatch.setattr(module.time, "perf_counter", lambda: next(values))

    with module._transition_phase(timeline, "world_init"):
        pass

    assert timeline == {"world_init": 2.5}
    with pytest.raises(AssertionError, match="recorded twice"):
        with module._transition_phase(timeline, "world_init"):
            pass


def test_exchange_directory_wait_is_type_correct_and_fail_closed(tmp_path) -> None:
    module = _load()
    exchange = tmp_path / "exchange"
    exchange.mkdir()
    assert module._wait_for_directory(exchange, 0.01) >= 0.0

    payload = tmp_path / "payload.json"
    payload.write_text("{}")
    assert module._wait_for_path(payload, 0.01) >= 0.0
    with pytest.raises(TimeoutError):
        module._wait_for_directory(payload, 0.001)
    with pytest.raises(TimeoutError):
        module._wait_for_path(exchange, 0.001)

    source = inspect.getsource(module.main)
    assert "_wait_for_directory(exchange, args.timeout_seconds)" in source
    assert "_wait_for_path(exchange, args.timeout_seconds)" not in source


def test_step10a_restores_two_graphs_first_schedule() -> None:
    module = _load()
    source = inspect.getsource(module._run_step10a_generator)
    first_loop = source.index("for production_round in range(2):")
    second_loop = source.index("for state in states:")
    wait = source.index("_wait_for_path", second_loop)
    assert first_loop < second_loop < wait
    assert source[:second_loop].count("run_generator_update_trace_rollout") == 1
    assert "two_selected_graphs_first_then_two_backwards" in source
    assert "Consume this detached gradient before creating the next" not in source


def test_ack_free_cycle_and_in_memory_transition_are_fail_closed() -> None:
    module = _load()
    source = Path(module.__file__).read_text()
    cycle = inspect.getsource(module._run_generator_cycle)
    student = inspect.getsource(module._run_student_cycle)
    main = inspect.getsource(module.main)

    assert "CriticCommitAck" not in source
    assert "receive_ack" not in source
    assert "send_ack" not in source
    assert "ack_group" not in source
    assert "NCCL_send_completion_no_commit_ack" in cycle
    assert cycle.count("_control_barrier(control_group)") == 2
    assert student.count("_control_barrier(control_group)") == 2
    assert "produced_batches - committed_batches > 2" in cycle
    assert "for update in range(CONTRACT.logical_critic_updates)" in cycle
    assert "ema.step(model.parameters())" in cycle

    assert "_broadcast_generator_g3" in main
    assert "_materialize_zero3_g3_into_target" in main
    assert "_destroy_obsolete_zero3_engine" in main
    assert "restore_zero1_optimizer_partition_from_logical" in source
    assert "load_checkpoint(" not in main
    assert "save_checkpoint(" not in main
    assert "transition-package" not in main


def test_role_relinquish_destroys_zero3_before_materializing_target() -> None:
    module = _load()
    release = inspect.getsource(module._destroy_obsolete_zero3_engine)
    verify = inspect.getsource(module._verify_obsolete_zero3_release)
    main = inspect.getsource(module.main)
    assert "engine.destroy()" in release
    assert '"module_forward_pre_hook"' in release
    assert '"module_forward_post_hook"' in release
    assert 'engine.__dict__["module"] = None' in release
    assert 'modules["module"] = None' in release
    assert 'for shard_name in ("ds_tensor", "ds_secondary_tensor")' in release
    assert 'torch.empty(0, dtype=shard.dtype, device="cpu")' in release
    assert '"drained_partition_bytes"' in release
    assert '"DESTROY_OBSOLETE_DP8_ZERO3_BEFORE_TARGET_MATERIALIZATION"' in release
    assert "after_zero3_source_destroy" not in release
    assert "after_zero3_source_destroy" in verify
    assert "_destroy_obsolete_zero3_engine(" in main
    destroy = main.index("_destroy_obsolete_zero3_engine(")
    drop_alias = main.index("step10a_engine = None", destroy)
    verify_release = main.index("_verify_obsolete_zero3_release(", drop_alias)
    assert destroy < drop_alias < verify_release
    assert main.index("_destroy_obsolete_zero3_engine(") < main.index(
        "_materialize_transition_target("
    )


def test_obsolete_zero1_cleanup_destroys_hooks_then_views_then_owners() -> None:
    module = _load()
    calls = []
    optimizer = SimpleNamespace(
        bit16_groups_flat=[torch.ones(8)],
        parallel_partitioned_bit16_groups=[[torch.ones(4), torch.ones(4)]],
        averaged_gradients={0: [torch.ones(2)]},
        all_grad_tensors={0: [torch.ones(2)]},
        destroy=lambda: calls.append("destroy"),
    )
    result = module._destroy_obsolete_zero1_gpu_flat_buffers(optimizer)
    assert calls == ["destroy"]
    assert optimizer.parallel_partitioned_bit16_groups == []
    assert optimizer.bit16_groups_flat == []
    assert optimizer.averaged_gradients == {}
    assert optimizer.all_grad_tensors == {}
    assert result["flat_buffer_count"] == 1
    assert result["partition_view_count"] == 2


def test_deepspeed_batch_clip_and_zero_stage_are_frozen() -> None:
    module = _load()
    step10a = module._step10a_generator_config()
    critic = module._student_config(12)
    assert (
        step10a["train_micro_batch_size_per_gpu"],
        step10a["gradient_accumulation_steps"],
        step10a["train_batch_size"],
        step10a["gradient_clipping"],
        step10a["zero_optimization"]["stage"],
    ) == (1, 2, 16, 1.0, 3)
    assert (
        critic["train_micro_batch_size_per_gpu"],
        critic["gradient_accumulation_steps"],
        critic["train_batch_size"],
        critic["gradient_clipping"],
        critic["zero_optimization"]["stage"],
    ) == (1, 1, 12, 1.0, 1)
    assert step10a["zero_optimization"]["offload_optimizer"]["device"] == "cpu"
    assert critic["zero_optimization"]["offload_optimizer"]["device"] == "cpu"
    assert step10a["zero_optimization"]["stage3_param_persistence_threshold"] == 0
    candidate = module._step10a_generator_config(
        param_persistence_threshold=2_000_000,
        module_granularity_threshold=8_000_000,
        prefetch_bucket_elements=402_653_184,
        max_live_parameters=805_306_368,
        overlap_comm=True,
    )
    assert (
        candidate["zero_optimization"]["stage3_param_persistence_threshold"]
        == 2_000_000
    )
    assert (
        candidate["zero_optimization"]["stage3_module_granularity_threshold"]
        == 8_000_000
    )
    assert candidate["zero_optimization"]["stage3_prefetch_bucket_size"] == 402_653_184
    assert candidate["zero_optimization"]["stage3_max_live_parameters"] == 805_306_368
    assert candidate["zero_optimization"]["stage3_max_reuse_distance"] == 268_435_456
    assert candidate["zero_optimization"]["overlap_comm"] is True
    assert candidate["train_batch_size"] == step10a["train_batch_size"]
    assert candidate["gradient_clipping"] == step10a["gradient_clipping"]



def test_step10a_explicit_leaf_policy_is_block_only_and_fail_closed() -> None:
    module = _load()
    source = inspect.getsource(module._initialize_step10a_generator)
    assert "set_z3_leaf_modules(" in source
    assert 'model, ["HeliosTransformerBlock"]' in source
    assert "len(configured_leaf_layers) != 40" in source
    assert 'record["module_type"] != "HeliosTransformerBlock"' in source
    main = inspect.getsource(module.main)
    assert "explicit block leaves and module-granularity threshold are mutually exclusive" in main

def test_step10a_zero3_initialization_requires_default_world8() -> None:
    module = _load()
    source = inspect.getsource(module._initialize_step10a_generator)
    assert "group is not dist.group.WORLD" in source
    assert "engine.data_parallel_group" in source
    assert "engine.optimizer.dp_process_group" in source
    assert " is not group" not in source
    assert '"NODE0_DEFAULT_WORLD8_NO_SUBGROUP"' in source
    assert "deepspeed_subgroup_default_barriers" not in source
    assert "mpu=" not in source


def test_resident_generator_cast_changes_only_trainable_parameters() -> None:
    module = _load()
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 3, bias=False),
        torch.nn.Linear(3, 2, bias=False),
    )
    model[0].weight.requires_grad_(False)
    frozen_before = model[0].weight.detach().clone()
    result = module._cast_resident_generator_trainables_to_bf16(model)
    assert model[0].weight.dtype == torch.float32
    assert torch.equal(model[0].weight, frozen_before)
    assert model[1].weight.dtype == torch.bfloat16
    assert result == {
        "policy": "TRAINABLE_LORA_ONLY_BF16",
        "trainable_tensor_count": 1,
        "trainable_numel": 6,
        "dtype_before": ["torch.float32"],
        "dtype_after": ["torch.bfloat16"],
        "frozen_tensors_untouched": True,
    }


def test_g3_broadcast_buckets_preserve_order_and_values(monkeypatch) -> None:
    module = _load()

    class TinyGenerator(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.values = torch.nn.ParameterList(
                [
                    torch.nn.Parameter(torch.randn(300_000, dtype=torch.bfloat16))
                    for _ in range(3)
                ]
            )

    class Logical:
        def component(self, name):
            assert name == "generator"
            return {"parameter_names": [name for name, _ in model.named_parameters()]}

    model = TinyGenerator()
    before = [parameter.detach().clone() for parameter in model.parameters()]
    collectives = []
    monkeypatch.setattr(
        module.dist,
        "broadcast",
        lambda tensor, **kwargs: collectives.append((tensor.numel(), kwargs)),
    )
    monkeypatch.setattr(module.torch.cuda, "synchronize", lambda _device: None)

    result = module._broadcast_generator_g3(
        model=model,
        logical=Logical(),
        rank=0,
        group=object(),
        timing_mode=True,
        bucket_mib=1,
    )

    assert result["tensor_count"] == 3
    assert result["collective_count"] == 3
    assert len(collectives) == 3
    assert max(result["bucket_bytes"]) <= 2**20
    assert all(
        torch.equal(observed, expected)
        for observed, expected in zip(model.parameters(), before)
    )


def test_active_source_construction_has_no_historical_wave_control() -> None:
    module = _load()
    source = inspect.getsource(module._construct_step10a_source)
    main = inspect.getsource(module.main)
    assert "construction_wave" not in source
    assert "for begin in range" not in source
    assert source.count("_control_barrier(control_group)") == 1
    assert "--construction-wave" not in main


def test_preconstructed_target_generator_casts_before_ema_restore() -> None:
    module = _load()
    preconstruct = inspect.getsource(module._preconstruct_switch_target)
    main = inspect.getsource(module.main)
    assert "_cast_resident_generator_trainables_to_bf16(target_model)" in preconstruct
    assert "for selected_local_rank in range(8)" not in preconstruct
    assert preconstruct.count("_control_barrier(control_group)") == 1
    assert main.index("SparseShardEMA.load_pretrained") < main.index(
        "_run_step10a_generator"
    )
    assert main.index("pin_module_cpu_registered_state(target_model)") < main.index(
        "_run_step10a_generator"
    )
    assert '"timing_scope": "excluded_cold_preparation"' in main
    target_ready = main.index('"target_ready_fence_wall_seconds"')
    critic_timer = main.index("critic_phase_started = time.perf_counter()")
    late_cleanup = main.index("start_deferred_host_allocator_cleanup()")
    assert target_ready < critic_timer < late_cleanup
    post_full5_marker = main.index(
        '"after_step15a_boundary_outside_full5_timing_main_thread"'
    )
    full5_timer = main.index(
        'result["full5_execution_seconds"] = time.perf_counter() - full5_started',
        critic_timer,
    )
    assert critic_timer < full5_timer < post_full5_marker
    post_full5_branch = main[
        main.index('if args.transition_pinned_cleanup_policy == "post_full5":'):
        main.index("else:", main.index('if args.transition_pinned_cleanup_policy == "post_full5":'))
    ]
    assert "trim_host_allocator()" in post_full5_branch
    assert "start_deferred_host_allocator_cleanup()" not in post_full5_branch
    assert '"execution_context": "main_thread"' in post_full5_branch
    assert "install_saved_tensor_census_hooks(" in main
    assert 'result["step10a_saved_tensor_census"] = step10a_census.remove()' in main
    assert "STEP10A_BOUNDARY_OFFLOAD_SIGNATURES" in main
    assert "--step10a-boundary-pinned-sync" in main
    assert "SELECTIVE_EXACT_SIGNATURE_PINNED_SYNC" in main
    assert "live_cpu_bytes" in main


def test_launcher_is_bound_to_continuous_executor_and_no_transition_package() -> None:
    root = Path(__file__).resolve().parents[1]
    launcher = (
        root
        / "scripts"
        / "distillschedule"
        / "launch_routeb_full5_2g6c_continuous_node.sh"
    ).read_text()
    assert "probe_routeb_full5_2g6c_continuous.py" in launcher
    assert "validate_routeb_full5_inputs.py" in launcher
    assert "--native-world8-root" in launcher
    assert "--negative-prompt-asset" in launcher
    assert "--mmap-cache-root" in launcher
    assert "transition-package" not in launcher
    assert "reserve_gate=disabled_by_user_actual_oom_only" in launcher
    assert "syn_feature_trace_rank16_v4_strict_sync_steps20" in launcher
    assert "654d49015921b278ab82e2fc84ea1896692da3fd38ce82fbc500cb1ebfec46d3" in launcher
    assert "syn_feature_trace_rank16_v3_sync_steps20" not in launcher
    assert "9e282f4bc24eb589d259e5e5603cae02fa5cd0c6cdd0f06f7c5416461e4564d0" not in launcher
    assert "construction_policy=concurrent_all_local_ranks" in launcher
    assert "construction_wave=" not in launcher
    assert "--construction-wave" not in launcher
    assert "persistent_pinned" in launcher
    assert "DISTILLSCHEDULE_TRANSITION_PINNED_CLEANUP_POLICY" in launcher
    assert "--transition-pinned-cleanup-policy" in launcher
    assert "late_deferred" in launcher
    assert "post_full5" in launcher
    assert "DISTILLSCHEDULE_STEP10A_MODULE_GRANULARITY_THRESHOLD" in launcher
    assert "--step10a-module-granularity-threshold" in launcher
    assert "DISTILLSCHEDULE_STEP10A_BLOCK_LEAF_FETCH" in launcher
    assert "--step10a-block-leaf-fetch" in launcher
    assert "DISTILLSCHEDULE_STEP10A_SAVED_TENSOR_CENSUS" in launcher
    assert "--step10a-saved-tensor-census" in launcher
    assert "DISTILLSCHEDULE_STEP10A_BOUNDARY_PINNED_SYNC" in launcher
    assert "--step10a-boundary-pinned-sync" in launcher
    worker = launcher[launcher.index("WORKER=(") :]
    assert '"${TORCHRUN}" --standalone --nproc-per-node=8' in worker
    assert '--node-rank "${NODE_RANK}"' in worker
    assert '--phase-b-master-addr "${MASTER_ADDR}"' in worker
    assert '--phase-b-master-port "${ACTUAL_PORT}"' in worker
    assert "--nnodes=2" not in worker


def test_phase_transition_gathers_then_destroys_then_reinitializes() -> None:
    module = _load()
    main = inspect.getsource(module.main)
    gather = main.index("_materialize_zero3_g3_into_target(")
    destroy_engine = main.index("_destroy_obsolete_zero3_engine(", gather)
    destroy_world = main.index("dist.destroy_process_group()", destroy_engine)
    init_phase_b = main.index("_initialize_phase_b_distributed(", destroy_world)
    create_groups = main.index("_create_phase_b_groups(", init_phase_b)
    broadcast = main.index("_broadcast_generator_g3(", create_groups)
    assert gather < destroy_engine < destroy_world < init_phase_b < create_groups < broadcast
    assert 'os.environ.pop("TORCHELASTIC_USE_AGENT_STORE", None)' in inspect.getsource(
        module._initialize_phase_b_distributed
    )


def test_continuous_executor_records_each_timed_full5_phase() -> None:
    module = _load()
    source = inspect.getsource(module.main)
    assert 'result["step10a_wall_seconds"]' in source
    assert 'result["transition_wall_seconds"]' in source
    assert 'result["critic_phase_wall_seconds"]' in source
    assert 'result["full5_execution_seconds"]' in source
    assert 'result["transition"]["subphase_seconds"]' in source
    assert '"phase_b_world_init_wall_seconds"' in source
    assert '"phase_b_group_create_wall_seconds"' in source
    assert '"target_model_h2d_wall_seconds"' in source
    assert '"student_initialize_total_host_wall_seconds"' in source
    assert '"target_ready_fence_wall_seconds"' in source

    student = inspect.getsource(module._initialize_student)
    assert '"deepspeed_initialize_host_wall_seconds"' in student
    assert '"logical_optimizer_restore_host_wall_seconds"' in student
    assert "torch.cuda.synchronize" not in student


def test_step10a_only_diagnostic_stops_before_transition_and_is_timing_only() -> None:
    module = _load()
    main = inspect.getsource(module.main)
    run = inspect.getsource(module._run_step10a_generator)
    stop = main.index("if args.stop_after_step10a:")
    transition = main.index("transition_started = time.perf_counter()")
    assert stop < transition
    assert 'parser.error("--stop-after-step10a is a timing diagnostic only")' in main
    assert '"critic": 10' in main
    assert '"ema": 10' in main
    assert "split_diagnostic = bool(args.stop_after_step10a and args.timing_mode)" in run
    assert '"diagnostic_split_synchronize": split_diagnostic' in run


def test_launcher_can_enable_bounded_step10a_only_timing() -> None:
    root = Path(__file__).resolve().parents[1]
    launcher = (
        root
        / "scripts"
        / "distillschedule"
        / "launch_routeb_full5_2g6c_continuous_node.sh"
    ).read_text()
    assert 'STOP_AFTER_STEP10A="${DISTILLSCHEDULE_STOP_AFTER_STEP10A:-0}"' in launcher
    assert "stop_after_step10a_requires_timing_mode" in launcher
    assert 'WORKER+=(--stop-after-step10a)' in launcher
    assert (
        'STEP10A_PARAM_PERSISTENCE_THRESHOLD="${DISTILLSCHEDULE_STEP10A_PARAM_PERSISTENCE_THRESHOLD:-0}"'
        in launcher
    )
    assert '--step10a-param-persistence-threshold' in launcher
    assert (
        'STEP10A_PREFETCH_BUCKET_ELEMENTS="${DISTILLSCHEDULE_STEP10A_PREFETCH_BUCKET_ELEMENTS:-16777216}"'
        in launcher
    )
    assert (
        'STEP10A_MAX_LIVE_PARAMETERS="${DISTILLSCHEDULE_STEP10A_MAX_LIVE_PARAMETERS:-268435456}"'
        in launcher
    )
    assert (
        'STEP10A_MAX_REUSE_DISTANCE="${DISTILLSCHEDULE_STEP10A_MAX_REUSE_DISTANCE:-268435456}"'
        in launcher
    )
    assert 'STEP10A_OVERLAP_COMM="${DISTILLSCHEDULE_STEP10A_OVERLAP_COMM:-0}"' in launcher
    assert '--step10a-prefetch-bucket-elements' in launcher
    assert '--step10a-max-live-parameters' in launcher
    assert '--step10a-max-reuse-distance' in launcher
    assert 'WORKER+=(--step10a-overlap-comm)' in launcher
    assert 'TRANSITION_H2D_POLICY="${DISTILLSCHEDULE_TRANSITION_H2D_POLICY:-module_to}"' in launcher
    assert '--transition-h2d-policy' in launcher
    assert '--transition-h2d-chunk-mib' in launcher
    assert '--transition-h2d-slot-count' in launcher
    assert 'G3_BROADCAST_BUCKET_MIB="${DISTILLSCHEDULE_G3_BROADCAST_BUCKET_MIB:-0}"' in launcher
    assert '--g3-broadcast-bucket-mib' in launcher


def test_step10a_nsys_capture_is_bounded_to_node0_and_cuda_profiler_range() -> None:
    module = _load()
    main = inspect.getsource(module.main)
    run = inspect.getsource(module._run_step10a_generator)
    root = Path(__file__).resolve().parents[1]
    launcher = (
        root
        / "scripts"
        / "distillschedule"
        / "launch_routeb_full5_2g6c_continuous_node.sh"
    ).read_text()
    assert "DISTILLSCHEDULE_STEP10A_CUDA_PROFILER" in main
    assert "torch.cuda.profiler.start()" in main
    assert "torch.cuda.profiler.stop()" in main
    assert "phase=rollout" in run
    assert "phase=backward" in run
    assert "phase=engine_step" in run
    assert 'STEP10A_NSYS="${DISTILLSCHEDULE_STEP10A_NSYS:-0}"' in launcher
    assert "step10a_nsys_requires_node0_bounded_diagnostic" in launcher
    assert "--capture-range=cudaProfilerApi" in launcher
    assert "--trace=cuda,nvtx,osrt" in launcher


def test_continuous_numerical_contract_was_frozen_before_gpu_run() -> None:
    root = Path(__file__).resolve().parents[1]
    path = (
        root
        / "artifacts"
        / "manifests"
        / "iteration280_full5_2g6c_continuous_numerical_contract_v1.json"
    )
    value = json.loads(path.read_text())
    assert value["status"] == "ACTIVE"
    assert value["frozen_before_first_gpu_run"] is True
    assert value["experiment_id"] == "RB16-F5-CONT-8G4S4T-TO-2G6C"
    assert value["numerical_limits"]["critic"]["exp_avg"]["relative_l2"] == 0.007623718564191509
    assert "thresholds may not be relaxed" in value["failure_policy"]


def test_zero3_step10a_correction_is_preregistered_and_old_run_superseded() -> None:
    root = Path(__file__).resolve().parents[1]
    prereg = json.loads(
        (
            root
            / "artifacts/manifests/iteration281_full5_2g6c_zero3_step10a_prereg.json"
        ).read_text()
    )
    supersession = json.loads(
        (
            root
            / "artifacts/manifests/iteration281_attempt005_backend_mismatch.json"
        ).read_text()
    )
    assert prereg["status"] == "ACTIVE"
    assert prereg["frozen_contract"]["step10a"] == {
        "layout": "8G+4S+4T",
        "generator_backend": "ZeRO-3",
        "generator_dp": 8,
        "micro_batch_per_rank": 1,
        "gradient_accumulation_steps": 2,
        "global_batch": 16,
        "parameter_offload": False,
        "optimizer_offload": "cpu",
        "checkpoint_policy": "zero3_metadata_tolerant_block_local_non_reentrant",
        "frozen_block_streaming": False,
        "activation_offload": False,
        "versions": "G2/C10/EMA10 -> G3/C10/EMA10",
    }
    assert supersession["status"] == "SUPERSEDED_BACKEND_MISMATCH"
    assert supersession["superseded_run"].endswith("20260810_005")


def test_step10a_collective_plan_requires_strict_dp8_parity() -> None:
    from runtime.routeb_full5_trace import RouteBFull5Trace

    class _ControlTrace:
        def __init__(self, exits):
            self.exits = exits

        def operation(self, key):
            assert key == "step/10/generator/rollout"
            return {
                "control": {
                    "call_count": sum(value + 1 for value in self.exits),
                    "pyramid_stage_call_sequence": [
                        {"denoising_exit": value} for value in self.exits
                    ],
                }
            }

    trace = object.__new__(RouteBFull5Trace)
    trace._control_traces = {
        rank: _ControlTrace((1, 0, 0)) for rank in range(16)
    }

    first = trace.step10a_collective_plan(production_round=0)
    second = trace.step10a_collective_plan(production_round=1)
    for plan in (first, second):
        assert plan.synchronized_stage_exits == (1, 0, 0)
        assert plan.physical_stage_call_counts == (2, 1, 1)
        assert plan.physical_call_count == 4
        assert plan.logical_call_counts == (4,) * 8
        assert plan.dummy_call_counts == (0,) * 8


def test_step10a_collective_plan_rejects_one_divergent_rank() -> None:
    from runtime.routeb_full5_trace import RouteBFull5Trace

    class _ControlTrace:
        def __init__(self, exits):
            self.exits = exits

        def operation(self, key):
            return {
                "control": {
                    "call_count": sum(value + 1 for value in self.exits),
                    "pyramid_stage_call_sequence": [
                        {"denoising_exit": value} for value in self.exits
                    ],
                }
            }

    trace = object.__new__(RouteBFull5Trace)
    trace._control_traces = {
        rank: _ControlTrace((0, 0, 0) if rank == 7 else (1, 0, 0))
        for rank in range(16)
    }
    with pytest.raises(ValueError, match="identical stage exits"):
        trace.step10a_collective_plan(production_round=0)


def test_generator_wrapper_has_no_collective_padding_seam() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "runtime/gst_pool_helios.py").read_text()
    assert "collective_padded_exit" not in source
    assert "collective_padded_denoising_slots" not in source


def test_step10a_strict_executor_records_fail_closed_boundaries() -> None:
    module = _load()
    source = inspect.getsource(module._run_step10a_generator)
    assert "trace.step10a_collective_plan" in source
    assert "collective_padded_exit_flag_list=" not in source
    for stage in (
        "step10a_generator_checksum_begin",
        "step10a_generator_checksum_complete",
        "trace_load_begin",
        "rollout_begin",
    ):
        assert stage in source
