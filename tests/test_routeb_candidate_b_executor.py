import datetime
import importlib.util
from pathlib import Path
from types import SimpleNamespace


def _load():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "distillschedule"
        / "probe_routeb_full5_candidate_b.py"
    )
    spec = importlib.util.spec_from_file_location("routeb_candidate_b_executor", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_candidate_b_creates_node_local_pairs_and_dp12_group(monkeypatch) -> None:
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
    module._create_groups(module.TOPOLOGY)
    assert calls[0][0]["ranks"] == list(range(16))
    assert calls[0][0]["backend"] == "gloo"
    assert calls[0][0]["timeout"].total_seconds() == 2 * 60 * 60
    assert calls[1][0] == {"ranks": [0, 1, 8, 9], "backend": "nccl"}
    assert calls[2][0] == {
        "ranks": [2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15],
        "backend": "nccl",
    }
    pair_and_ack = calls[3:27]
    p2p = [pair_and_ack[index][0] for index in range(0, 24, 2)]
    acks = [pair_and_ack[index][0] for index in range(1, 24, 2)]
    assert len(p2p) == 12
    assert all(call["backend"] == "nccl" for call in p2p)
    assert all(call["ranks"][0] // 8 == call["ranks"][1] // 8 for call in p2p)
    assert len(acks) == 12
    assert all(call["backend"] == "gloo" for call in acks)
    assert [call["ranks"] for call in p2p] == [call["ranks"] for call in acks]
    assert len(calls[27:]) == 12
    assert all(call[0]["backend"] == "nccl" for call in calls[27:])
    control = calls[0][1]
    assert len(barriers) == 39
    assert barriers == [{"group": control}] * 39


def test_candidate_b_initializes_torch_before_attaching_deepspeed(monkeypatch) -> None:
    module = _load()
    calls = []
    initialized = False

    def fake_is_initialized():
        return initialized

    def fake_init_process_group(**kwargs):
        nonlocal initialized
        calls.append(("torch", kwargs))
        initialized = True

    deepspeed = SimpleNamespace(
        init_distributed=lambda **kwargs: calls.append(("deepspeed", kwargs))
    )
    monkeypatch.setattr(module.dist, "is_initialized", fake_is_initialized)
    monkeypatch.setattr(module.dist, "init_process_group", fake_init_process_group)

    module._initialize_distributed_runtime(deepspeed)

    assert calls == [
        (
            "torch",
            {
                "backend": "nccl",
                "timeout": datetime.timedelta(hours=2),
            },
        ),
        (
            "deepspeed",
            {"dist_backend": "nccl", "dist_init_required": False},
        ),
    ]


def test_candidate_b_deepspeed_batch_and_clip_are_frozen() -> None:
    module = _load()
    value = module._student_config(12)
    assert value["train_micro_batch_size_per_gpu"] == 1
    assert value["gradient_accumulation_steps"] == 1
    assert value["train_batch_size"] == 12
    assert value["gradient_clipping"] == 1.0
    assert value["zero_optimization"]["stage"] == 1
    assert value["zero_optimization"]["offload_optimizer"]["device"] == "cpu"
    assert "offload_param" not in value["zero_optimization"]
