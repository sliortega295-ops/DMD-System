# BAAI runbook: LightX2V MiniMax-H3 DMD world16

This runbook launches the pinned LightX2V MiniMax-H3 DMD baseline on the same
two A100 nodes used by the Helios experiments.

## 1. Required assets

The run requires:

1. a DMD-System checkout at the exact benchmark branch/commit, including the
   `third_party/lightx2v` submodule;
2. a Python environment that can import PyTorch FSDP2 and Diffusers'
   `MiniMaxH3Transformer3DModel`;
3. the converted MiniMax-H3 Diffusers model root containing
   `transformer/config.json` plus the text-encoder/tokenizer/processor assets;
4. a LightX2V `latent_dataset` prompt cache.

The benchmark does not download model weights during the timed run.

## 2. Prepare a prompt cache

LightX2V ships the cache builder in the pinned submodule:

```bash
export DMD_SYSTEM_ROOT=/share/project/shuchenweng/yongyan_liu/DMD-System_fsdp2_p0_20260815
export LIGHTX2V_PYTHON=/path/to/lightx2v/python
export MINIMAX_MODEL=/path/to/converted/MiniMax-H3
export PROMPTS=/path/to/prompts.txt
export PROMPT_CACHE=/share/project/.../minimax_h3_prompt_cache

PYTHONPATH="$DMD_SYSTEM_ROOT/third_party/lightx2v/lightx2v_train" \
"$LIGHTX2V_PYTHON" \
"$DMD_SYSTEM_ROOT/third_party/lightx2v/lightx2v_train/data_process/minimax_h3/build_minimax_h3_prompt_latents.py" \
"$PROMPTS" \
--model-path "$MINIMAX_MODEL" \
--output-dir "$PROMPT_CACHE" \
--device cuda:0 \
--dtype bf16
```

At least 16 usable samples are needed for a world16 B1 update; use a larger
cache for the six sequential update stages so each rank does not immediately
recycle the same item.

## 3. Prepare the pinned benchmark source

```bash
export DMD_SYSTEM_ROOT=/share/project/shuchenweng/yongyan_liu/DMD-System_fsdp2_p0_20260815
export LIGHTX2V_PYTHON=/path/to/lightx2v/python
bash "$DMD_SYSTEM_ROOT/scripts/distillschedule/prepare_lightx2v_minimax.sh"
```

This verifies the LightX2V submodule commit and patch SHA, creates an immutable
patched source copy, and runs the MiniMax import gate. Reusing a mismatched or
dirty source root fails closed.

## 4. One world16 acceptance/timing run

From the controller host:

```bash
python scripts/distillschedule/run_lightx2v_minimax_world16.py \
  --run-id minimax_h3_g1f5_r1 \
  --expected-head <FULL_DMD_SYSTEM_HEAD> \
  --model-path "$MINIMAX_MODEL" \
  --prompt-cache "$PROMPT_CACHE" \
  --python "$LIGHTX2V_PYTHON"
```

Defaults:

```text
hosts              baai-helios-node0,baai-helios-node1
world size         16
sequence parallel  1
FSDP2 mesh         world16
Student/Fake       LoRA rank 128
Student updates    1
Fake updates       5
end_step_idx       3 (four model evaluations)
benchmark seed     20260817
```

The controller owns both SSH transports, uses the existing DMD-System remote
process wrapper, cancels the peer on failure, and only accepts the result after
the 16-rank summarizer passes.

If `_flash_3_hub` is unavailable on the A100 software environment, pass a
Diffusers attention backend known to work in that exact environment with
`--attention-backend`. The chosen backend is recorded in node receipts and must
remain fixed across compared MiniMax runs.

## 5. Formal timing

Run at least three independent unprofiled cycles with different run IDs and the
same code/config/assets:

```bash
# r1, r2, r3 ...
python scripts/distillschedule/compare_lightx2v_minimax_helios.py \
  --minimax-summary /path/r1/summary.json \
  --minimax-summary /path/r2/summary.json \
  --minimax-summary /path/r3/summary.json \
  --output /path/minimax_vs_helios_context.json
```

The comparison script reports MiniMax median/CV/MAD and the existing Helios
PR16 wall as **cross-workload absolute context only**. It deliberately refuses
to call the ratio a speedup.

A claimable DistillSchedule speedup requires a second MiniMax implementation
that preserves this exact MiniMax contract and changes only the systems
execution strategy.
