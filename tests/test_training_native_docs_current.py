"""Public documentation reflects shipped source support, not new GPU evidence."""
from pathlib import Path
import re
import subprocess
import sys

import pytest

from distillschedule.training.config import load_training_config
from distillschedule.training.native import describe_native

ROOT = Path(__file__).resolve().parents[1]
STREAMS = [
    ("W1", "wan_ar/training.incumbent-stream.json", "INCUMBENT_STREAM.md", 100),
    ("W2", "helios_nogan/training.safe-stream.json", "SAFE_STREAM.md", 100),
    ("W3", "helios_gan/training.stream.json", "NATIVE_STREAM.md", None),
    ("W4", "self_forcing/training.c51-stream.json", "C51_CONTINUATION.md", 100),
    ("W5", "h3_fullseq/training.original-reference.json", "h3-grid-original-training.md", 64),
    ("W6", "h3_ar/training.native-stream.json", "NATIVE_CONTINUATION.md", 100),
]


@pytest.mark.parametrize("workload,recipe,guide,maximum", STREAMS)
def test_native_stream_documentation_matches_current_descriptors(workload, recipe, guide, maximum):
    report = describe_native(load_training_config(ROOT / "recipes" / recipe))
    assert report["workload"] == workload
    assert report["capabilities"]["resume"]["supported"] is True
    assert report["capabilities"]["cycles"].get("max") == maximum
    doc = (ROOT / "docs/guides/native_workloads.md").read_text()
    section = doc.split("## Continuous native training", 1)[1].split("\n## ", 1)[0]
    row = next(line for line in section.splitlines() if line.startswith("| " + workload + " |"))
    assert "../../recipes/" + recipe in row and guide in row
    assert ("2.." + str(maximum) if maximum else "finite prepared stream") in row
    assert report["native_api"].split(":")[0].split(".")[-1] in row


def test_native_output_docs_link_real_current_generator_media_paths():
    matrix = (ROOT / "docs/guides/support_matrix.md").read_text()
    for family in ("wan_ar", "helios_nogan", "helios_gan"):
        assert "../../recipes/" + family + "/NATIVE_OUTPUT.md" in matrix
    native = (ROOT / "docs/guides/native_workloads.md").read_text()
    notes = (ROOT / "docs/guides/release_notes.md").read_text()
    for guide in ("livewan_native_output.md", "h3_grid_inference.md",
                  "h3_ar_native_inference.md", "h3_audio_video_encode.md"):
        assert guide in matrix and guide in native and guide in notes, guide
    for workload, recipe, _, _ in STREAMS:
        report = describe_native(load_training_config(ROOT / "recipes" / recipe))
        assert report["capabilities"]["export"]["supported"] is True, workload
    for text in (matrix, native, notes):
        assert "W5/W6 native AV export/codec integration remains pending" not in text
        assert "W5/W6 native audio-video output remains implementation work" not in text
    grid = (ROOT / "recipes/h3-grid-original-training.md").read_text()
    assert "Generation/export are unsupported" not in grid
    assert "h3_grid_inference.md" in grid
    assert "C51's physical drain/save/resume seam remains unsupported" not in (ROOT / "docs/guides/native_workloads.md").read_text()
    assert "W3/W4/W6 full optimized native long\n  lifecycles are **not complete**" not in (ROOT / "docs/guides/release_notes.md").read_text()


def test_current_docs_preserve_frozen_benchmark_and_actual_gpu_boundaries():
    docs = [ROOT / "docs/guides" / name for name in ("support_matrix.md", "native_workloads.md", "release_notes.md")]
    for path in docs:
        text = path.read_text()
        assert "two-cycle" in text and "Grid1000" in text and "100-cycle" in text
        for link in re.findall(r"\]\(([^ )]+)", text):
            if not link.startswith(("https://", "http://", "#", "mailto:")):
                assert (path.parent / link.split("#")[0]).exists(), (path.name, link)
    # The source metadata query remains lightweight even for the new stream API.
    script = """import sys
from pathlib import Path
from distillschedule.training.config import load_training_config
from distillschedule.training.native import describe_native
for p in sys.argv[1:]: describe_native(load_training_config(Path(p)))
assert 'torch' not in sys.modules
print('METADATA_ONLY_NO_TORCH_OR_MODEL_IMPORT')
"""
    result = subprocess.run([sys.executable, "-B", "-c", script, *[str(ROOT / "recipes" / r) for _, r, _, _ in STREAMS]],
                            cwd=ROOT, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "METADATA_ONLY_NO_TORCH_OR_MODEL_IMPORT" in result.stdout
