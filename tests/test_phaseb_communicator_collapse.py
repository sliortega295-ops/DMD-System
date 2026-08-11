import importlib.util
import inspect
from pathlib import Path


def _load():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "distillschedule"
        / "probe_phaseb_communicator_collapse.py"
    )
    spec = importlib.util.spec_from_file_location("phaseb_communicator_collapse", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gate_covers_all_critics_with_disjoint_routes() -> None:
    module = _load()
    targets = tuple(target for values in module.ROUTES.values() for target in values)
    assert module.GENERATOR_RANKS == (0, 1)
    assert module.CRITIC_RANKS == (2, 3, 4, 5, 6, 7)
    assert tuple(sorted(targets)) == module.CRITIC_RANKS
    assert len(set(targets)) == len(targets)


def test_gate_has_no_pair_or_singleton_group_creation() -> None:
    module = _load()
    source = inspect.getsource(module.main)
    assert source.count("dist.new_group(") == 3  # control, Generator and Critic only
    assert 'singleton_model_parallel_group=None' in source
    assert 'group=dist.group.WORLD' in source
    assert 'dist.barrier(group=dist.group.WORLD)' in source
    assert '"pair_groups_created": 0' in source
    assert '"singleton_groups_created": 0' in source
