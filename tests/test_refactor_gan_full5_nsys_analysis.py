import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'tools/profiling/analyze_gan_full5_nsys.py'
SPEC = importlib.util.spec_from_file_location("gan_full5_nsys_analysis", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_merge_interval_nanoseconds_deduplicates_overlap():
    assert MODULE.merge_interval_nanoseconds([(0, 10), (5, 12), (20, 25)]) == 17


def test_receipt_wall_uses_max_stop_minus_min_start(tmp_path: Path):
    for rank in range(8):
        value = {
            "status": "PASS",
            "rank": rank,
            "logical_window": [1000, 1004],
            "start_unix_ns": 1_000_000_000 + rank * 1_000,
            "stop_unix_ns": 3_000_000_000 + rank * 2_000,
            # Deliberately inconsistent with the global-range definition.
            "elapsed_ms": 1.0,
        }
        (tmp_path / f"rank_{rank:03d}.json").write_text(json.dumps(value))
    seconds, ranks = MODULE._receipt_wall_seconds(tmp_path, 0)
    assert ranks == list(range(8))
    assert seconds == (2_000_000_000 + 7 * 2_000) / 1e9
