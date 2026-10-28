"""Public documentation reflects shipped source support, not new GPU evidence."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

from distillschedule.training.config import load_training_config
from distillschedule.training.native import describe_native
from distillschedule.training.inference import read_export
from distillschedule.workloads.best_known.registry import file_sha256

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


PROJECTED_REQUESTS = [
    ("dmd2_sd15/training.json", "sd"),
    ("dmd2_sdxl/training.json", "sd"),
    ("self_forcing_official/training.fsdp2-reference.json", "self_forcing"),
    ("flux_dmd/training.template.json", "flux"),
    ("qwen_image_dmd/training.template.json", "qwen_image"),
    (None, "sd"),
]


def _projected_request_fixture(tmp_path, recipe, module):
    """SYNTHETIC_TEST_ONLY: export admission metadata, not model/weight proof."""
    training_dir = tmp_path / "old training request"
    training_dir.mkdir()
    if recipe is not None:
        request = json.loads((ROOT / "recipes" / recipe).read_text())
    else:
        # Local factories resolve relative to the original request, without
        # opening them, the old inputs, a saved plan or a restart checkpoint.
        request = dict(schema_version="distillschedule.training.v1",
            model=dict(factory="training_model.py:build_model", parameters={}),
            algorithm=dict(factory="../plugins/workflow.py:build_algorithm", parameters={"keep": 3}),
            data=dict(streams={"old": dict(prepared="missing/manifest.json", samples_per_cycle=1)}),
            execution=dict(adapter="program", mode="plan", plan="missing/plan.json"),
            trainer=dict(cycles=4, seed=17, device="cpu", distributed=dict(backend="gloo"),
                         output_dir="missing/run", resume_from="missing/checkpoint"))
    training_path = training_dir / "train.json"
    training_path.write_text(json.dumps(request))
    output = tmp_path / "inference configs" / "inference.json"
    output.parent.mkdir()
    prepared = tmp_path / "new-prepared" / "manifest.json"
    prepared.parent.mkdir()
    prepared.write_text(json.dumps({"fixture_scope": "SYNTHETIC_TEST_ONLY"}))
    weights = tmp_path / "exported"
    weights.mkdir()
    payload = weights / "adapter-state.json"
    payload.write_text(json.dumps({"fixture_scope": "SYNTHETIC_TEST_ONLY"}))
    factory = f"distillschedule.training.{module}:build_inference_model"
    source = dict(reference=factory, sha256=file_sha256(ROOT / "src/distillschedule/training" / (module + ".py")))
    metadata = dict(fixture_scope="SYNTHETIC_TEST_ONLY")
    model = dict(factory=factory, source=source, metadata=metadata, modules={},
                 parameters=dict(seed=17, source_root=str(tmp_path / "external source retained"),
                                 architecture={"fixture_scope": "SYNTHETIC_TEST_ONLY"}))
    manifest = dict(schema_version="distillschedule.inference_export.v1", status="EXPORTED",
        kind="INFERENCE_WEIGHTS_NOT_RESUMABLE_CHECKPOINT", checkpoint_cycle=4, world_size=1,
        identity=dict(model=source, model_metadata=metadata), model=model, quality_validation="NOT_RUN",
        export=dict(format="adapter", metadata=metadata), files={payload.name: file_sha256(payload)})
    (weights / "export.json").write_text(json.dumps(manifest))
    (weights / "_SUCCESS").write_text(file_sha256(weights / "export.json") + "\n")
    return training_path, weights, prepared, output


def _common_generate_example():
    section = (ROOT / "docs/guides/training.md").read_text().split("## Generate and export", 1)[1]
    request = re.search(r"^distillctl generate (\S+) --weights exported\b", section, re.MULTILINE)
    assert request, "common guide needs its copyable portable-generate command"
    builder = re.search(
        r"^python - train\.json exported new-prepared/manifest\.json inference\.json <<'PY'\n(.*?)\nPY$",
        section, re.MULTILINE | re.DOTALL)
    return request.group(1), builder.group(1) if builder else None


def _documented_example_env():
    # The example runs outside the checkout. Do not rely on an editable install
    # or relative PYTHONPATH entries changing meaning with its working directory.
    return {**os.environ, "PYTHONPATH": os.pathsep.join(map(str, (ROOT / "src", ROOT)))}


@pytest.mark.parametrize("recipe,module", PROJECTED_REQUESTS)
def test_common_generate_example_uses_verified_export_projection(tmp_path, recipe, module):
    training, weights, prepared, output = _projected_request_fixture(tmp_path, recipe, module)
    request_name, builder = _common_generate_example()
    if builder is not None:
        script = builder + "\nassert 'torch' not in sys.modules\nprint('METADATA_ONLY_NO_TORCH_OR_MODEL_IMPORT')\n"
        result = subprocess.run([sys.executable, "-B", "-c", script,
            *[str(path.relative_to(tmp_path)) for path in (training, weights, prepared, output)]],
            cwd=tmp_path, env=_documented_example_env(), text=True, capture_output=True, timeout=30)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "METADATA_ONLY_NO_TORCH_OR_MODEL_IMPORT" in result.stdout
    paths = {"train.json": training, "inference.json": output}
    inference = load_training_config(paths[request_name])
    original = load_training_config(training)
    _, manifest = read_export(weights)
    assert inference.model["factory"] == manifest["model"]["factory"], "generate must use the exported inference factory"
    assert inference.model == {key: manifest["model"][key] for key in ("factory", "parameters")}
    assert inference.algorithm == original.algorithm
    assert inference.data == {"prepared": str(prepared)}
    assert inference.execution == {"adapter": "program", "mode": "reference"}
    assert set(inference.trainer) == {"cycles", "seed", "device", "output_dir"}
    assert inference.trainer["cycles"] == 1 and inference.trainer["seed"] == 909
    assert inference.trainer["device"] == "cpu"
    assert Path(inference.trainer["output_dir"]).parent == output.parent
    assert not Path(inference.trainer["output_dir"]).exists()
    assert inference.model["parameters"]["seed"] == 17
    if recipe is None:
        assert inference.algorithm["factory"] == str(tmp_path / "plugins/workflow.py") + ":build_algorithm"


@pytest.mark.parametrize("fault,error", [("incomplete", "not complete"), ("payload_hash", "file hash changed")])
def test_common_inference_builder_refuses_unverified_export(tmp_path, fault, error):
    training, weights, prepared, output = _projected_request_fixture(tmp_path, None, "sd")
    _, builder = _common_generate_example()
    assert builder is not None
    if fault == "incomplete":
        (weights / "_SUCCESS").unlink()
    else:
        (weights / "adapter-state.json").write_text("SYNTHETIC_TEST_ONLY corrupted payload")
    result = subprocess.run([sys.executable, "-B", "-c", builder,
        *[str(path.relative_to(tmp_path)) for path in (training, weights, prepared, output)]],
        cwd=tmp_path, env=_documented_example_env(), text=True, capture_output=True, timeout=30)
    assert result.returncode != 0 and error in result.stderr, result.stdout + result.stderr
    assert not output.exists()
    assert not (output.parent / "unused-inference-run").exists()


def test_projected_model_recipes_share_common_inference_request_steps():
    for recipe in ("dmd2_sd15", "self_forcing_official", "flow_image"):
        text = (ROOT / "recipes" / recipe / "README.md").read_text()
        assert "../../docs/guides/training.md#generate-and-export" in text, recipe
    sd = (ROOT / "recipes/dmd2_sd15/README.md").read_text()
    assert "../dmd2_sdxl/README.md" in sd
