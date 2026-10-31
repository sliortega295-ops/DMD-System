# H3 native output: explicit campaign reproduction

`distillctl native-output` is a narrow **opt-in reproduction tool**, not an
alternative trainer, scheduler, generic sampler or fallback for `generate`.
It accepts only the reviewed R375 source228/C50/AP40-history/new232-text
composition and v2 nonretaining extension pair. This functional condition
reuses AP40 history: it is **not held-out quality evidence** or support for
arbitrary user media. Ordinary train/export/portable generate/decode stay as
documented in [native inference](../../docs/guides/h3_ar_native_inference.md).

## Explicit identities, no mixed-source restore

Start with [the request template](native-output.reproduce.template.json).
Every path is absolute and every file reference has a SHA256. Fill fields from
the **same** operator-admitted original-owner launch, without rewriting it:

| Public field | Existing packet source |
| --- | --- |
| `launch` | Original owner's versioned, admitted native-output launch file and byte hash |
| `runtime` | Launch `python`, `source_root`, `source_head` (historical source228, not current package HEAD) |
| `owner.entry` | Pinned `native_output_node_owner.py`, never `rank_native_output.py` |
| `owner.original_launch`, `owner.admission` | `w6_native_output.original_launch`, `scope_v2_gate` |
| `owner.nodes` | Both launch nodes' `node_rank`, `hostname`, all eight ordered `gpu_uuids`, plus matching `target_cpu_gates` reference |
| `owner.run_root` | Launch `run_root` (the original owner owns namespace freshness) |
| `checkpoint_manifest` | Original C50 `manifest.json` path and fixed historical byte hash |
| `extensions` | `w6_native_output.scope_v2_pins`: separately reviewed scope and generation source files |
| `conditions`, `input_witness` | `w6_native_output.conditions`, then that binding's actual completed `conversion_witness` |

Template nulls are deliberately invalid. Missing owner/peer/CPU gates/input
witness, old forward-zero export kind, source drift and unknown schemas refuse.
No shape, precision, capacity, training seed, epoch, C0 or reexport option is
provided. The unchanged native runner validates the original parser, five
lossless inverse-BF16 tensors, untouched FP64 positions and real geometry; this
metadata front door does not replace those physical input checks.

## Inspect first; execution remains original-owner scheduled

Use the **current** reviewed repository entry script explicitly so an inherited
historical `PYTHONPATH` cannot select the old CLI instead:

```sh
PUBLIC=/absolute/path/to/reviewed/current/DMD-System
PUBLIC_PY=/absolute/path/to/public/python
REQUEST=/absolute/path/to/operator-native-output-request.json
"$PUBLIC_PY" -B "$PUBLIC/scripts/distillctl.py" native-output "$REQUEST"
```

The default only reads bounded metadata/source/executable bytes. It imports no
Torch/native runtime, constructs no model, acquires no GPU and returns
`INSPECTED_REPRODUCTION_NOT_OWNER_ADMISSION`, with two **owner** argument lists.
This is not a fresh lock/FD/resource admission or permission to occupy a node.

Only the existing operator may schedule execution on both bound host roles,
outside `torchrun`, after all other owners release the resources. On node 0
(use `--node 1` on the separately scheduled peer):

```sh
"$PUBLIC_PY" -B "$PUBLIC/scripts/distillctl.py" native-output "$REQUEST" --node 0 --execute
```

Execution rechecks actual hostname, clean historical Git HEAD and pinned bytes,
then **execs** the bound old Python and unchanged original node owner. It does
not import old modules into the current process or execute a rank directly.
Only the original launch environment crosses the process boundary (including
CPU threads=4); caller `PYTHONPATH`, preload and distributed-rank state cannot
leak. The original owner still admits both peers, acquires its original lock,
checks full C50/state/source/capacity/FD bindings, launches its existing world16
entry, monitors it and performs cleanup. A GO file/CPU receipt is not current
ownership. Existing stages/outputs refuse rather than retrying an old run.

`NATIVE_OUTPUT_OWNER_HANDOFF` is only an argument/identity handoff, **not owner
success**. Require both actual owner cleanup receipts and the original newkind
collector's full 16 ranks × 8 no-grad predictions, zero updates and unchanged
state/Adam/RNG before using generated output. The independent media bridge
then verifies that collector and makes a public `decode` request; neither
inspection nor handoff proves generation or strict real-codec decoding. Do not
retag this kind as the old forward-zero weights export or repeat AP50/export.
