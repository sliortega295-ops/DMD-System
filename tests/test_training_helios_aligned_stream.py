"""Synchronized lifecycle inputs: actual original CPU producer, no GPU proof."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("gan", [False, True], ids=["W2", "W3"])
def test_real_aligned_capture_keeps_zero_history_grid_and_synchronized_calls(tmp_path, gan):
    from distillschedule.training import helios_inputs as w2, helios_gan_inputs as w3
    from distillschedule.search.helios_request import OriginalHeliosRequest
    from distillschedule.strategies.wan_capture import native_primitives, read_training_config, stage_control
    from distillschedule.workloads.best_known.registry import repository_root, file_sha256
    from runtime.operation_trace import OperationTrace
    module = w3 if gan else w2
    task_cls = w3.GANCohortTask if gan else w2.HeliosCohortTask
    request = OriginalHeliosRequest(variant="gan" if gan else "no-gan",
                                   stage_schedule_policy="benchmark_v4_strict_global")
    task = task_cls(0, request)
    native = native_primitives(source_only=True)
    config_path = repository_root()/"configs/distillschedule/templates"/(
        "helios_stage3_gan_384x640_systems.yaml" if gan else "helios_stage3_384x640.yaml")
    config = w3.read_config(config_path) if gan else read_training_config(config_path)
    paths, rows = [], []
    for offset, step in enumerate(task.source_steps):
        sample = tmp_path/f"native-feature-{offset}.pt"
        torch.save(dict(vae_latent=torch.zeros(1,dtype=torch.bfloat16).expand(3,16,9,48,80),
            prompt_embed=torch.full((2,4096),float(offset+1),dtype=torch.bfloat16),
            attention_mask=torch.ones(2,dtype=torch.bool)),sample)
        paths.append([str(sample)] + [str(tmp_path/f"unused-fixture-{offset}-{rank}.pt") for rank in range(1,16)])
        rows.append(dict(rank=0,step=step,path=str(sample),sha256=file_sha256(sample)))
    source = w2.HeliosCohortSources(731,tuple(tuple(row) for row in paths))
    # Only rank0 is produced here; the other names are not opened. This is an
    # explicit source fixture, not a world16 qualification or real training.
    before = torch.random.get_rng_state().clone()
    captured = module.capture_rank(task,source,rows,config,rank=0,output=tmp_path/"captured",
        provenance={"evidence_kind":"CPU_ACTUAL_PRODUCER_SYNTHETIC_INPUT_ONLY"})
    assert torch.equal(before,torch.random.get_rng_state())
    assert not torch.cuda.is_initialized()
    trace = SimpleNamespace(task=task,logical_steps=task.source_steps,_load=lambda rank:captured)
    module.validate_rank_payload(trace,0,source)
    different_grid = False
    for step in task.source_steps:
        for phase in (("generator","critic") if step==task.first_step else ("critic",)):
            control = captured.operation(f"step/{step}/{phase}/rollout")["control"]
            assert control["all_history_zero"] is True
            assert all(s["denoising_step_budget"]==4 for s in control["pyramid_stage_call_sequence"])
            normal = stage_control(native,task,source.seed,task.index,step,phase,all_history_zero=False,rank=1)
            assert [s["call_count"] for s in normal["pyramid_stage_call_sequence"]] == [
                s["call_count"] for s in control["pyramid_stage_call_sequence"]]
            assert all(s["denoising_step_budget"]==2 for s in normal["pyramid_stage_call_sequence"])
            different_grid |= any(a["timestep"] != b["timestep"] for a,b in zip(
                normal["pyramid_stage_call_sequence"],control["pyramid_stage_call_sequence"],strict=True))
    assert different_grid
    # Consumer sees real producer controls and tensor descriptors. Metadata
    # copies stand in for 15 ranks only to exercise the cross-rank validator.
    controls = {rank:OperationTrace(root=captured.root,manifest=deepcopy(captured.manifest),tensors={}) for rank in range(16)}
    module.validate_controls(task,controls)
    prefix=f"step/{task.first_step}/generator/rollout"
    control=controls[1].operation(prefix)["control"]
    control["pyramid_stage_call_sequence"][0]["call_count"] = 3
    control["pyramid_stage_call_sequence"][0]["denoising_exit"] = 2
    with pytest.raises(ValueError,match="stage scope"):
        module.validate_controls(task,controls)


@pytest.mark.parametrize("variant", ["no-gan", "gan"])
def test_unsupported_workload_dimensions_stay_rejected(variant):
    from distillschedule.search.helios_request import OriginalHeliosRequest
    from distillschedule.training.helios_inputs import HeliosCohortTask
    from distillschedule.training.helios_gan_inputs import GANCohortTask
    task = GANCohortTask if variant=="gan" else HeliosCohortTask
    with pytest.raises(ValueError,match="reference"):
        task(0,OriginalHeliosRequest(variant=variant,video_frames=17,
            stage_schedule_policy="benchmark_v4_strict_global"))
