# Frozen experiment names

`world16` describes only a topology. It is not an experiment name and must not
be used by itself in a report, manifest, run ID, plot legend or verbal result.
The experiment ID must also identify the execution route, logical scope and
lifecycle policy.

| Experiment ID | Frozen meaning | Current evidence |
|---|---|---|
| `RA16-F5-FS40` | Route-A, 16 ranks, continuous Full5 steps 10--14; Generator and Score both use all-40 frozen-block streaming; five Critic and five EMA updates | One-run functional PASS; formal repeated timing Not Run |
| `RA16-F5-PR4` | Route-A, 16 ranks, continuous Full5 steps 10--14; Step10-A Generator uses `g4=[0,24,1,5]`; Score has no block streamer; Critic-rollout Generator keeps those four blocks resident for the rollout phase; five Critic and five EMA updates | FAIL: 16-rank Step10-A backward CUDA OOM; no updates completed |
| `RA16-F5-PR4-EC` | Same PR4 contract plus one bounded allocator-cache release immediately before backward | Bounded negative: reproduced the exact PR4 Step10-A OOM |
| `RA16-F5-PR8` | Route-A continuous Full5 with the PR4 phase policy, but Step10-A streams `g8=[0,24,1,5,8,25,10,32]`; Score remains unstreamed | FAIL: progressed farther into Step10 backward, then reproduced the 216-MiB CUDA OOM; no update completed |
| `RA16-F5-PR12` | Route-A continuous Full5 with the PR8 phase policy, but Step10-A streams `g12=[0,24,1,5,8,25,10,32,36,15,16,26]`; Score remains unstreamed | Bounded negative: g12 completed the Step10 Generator update, but all ranks OOMed in resident Score/Critic backward; timing Not Run |
| `RA16-F5-GLP-OFFLOAD` | PR12 plus typed phase-scoped offload of the inactive Generator ZeRO-1 BF16 LP flat state after Critic rollout; Score remains unstreamed | Storage mechanism verified exactly (-2.330994 GiB/rank at Critic entry), but all ranks still OOMed in Critic backward; bounded negative; timing Not Run |
| `GST16-A-G4` | Isolated Step10-A capacity/correctness gate with `8G+4S+4T`; Generator uses `g4=[0,24,1,5]`, Student and Teacher are resident | Isolated gate PASS; not Full5 |
| `CP8-C4-SEQ-RES` | One-node `4G+4S` resident Critic-only four-wave sequential execution | Isolated comparison PASS |
| `CP8-C4-PIPEQ1-RES` | Same Critic-only workload with queue-depth-1 producer--consumer pipeline | Isolated comparison PASS; EMA deliberately skipped by that contract |
| `RB16-F5-PIPE-MATCH16` | Route-B Candidate A: continuous Full5 with Step10-A `8G+4S+4T`, then matched `8G+8C`; Critic DP8/MB1/GAS2/batch16, resident pools and official five-step EMA | Implementation/preflight only; real transition/Sequential/Pipeline Not Run |
| `RB16-F5-PIPE-2G6C` | Route-B Candidate B: continuous Step10-A `8G+4S+4T` with DP8 ZeRO-3/MB1/GAS2/batch16, then per-node `2G+6C` (global `4G+12C`); resident Critic DP12/MB1/GAS1/batch12; ACK-free Pipeline; five Critic and five official early-bootstrap EMA updates | One real-14B strict-trace Pipeline functional PASS at `G2/C10/EMA10 -> G3/C15/EMA15`; Sequential pair and formal timing Not Run; not a direct batch16 speedup denominator |

`PR4` means **phase-resident Critic rollout plus four streamed Generator blocks
for Step10-A**. It never means that every model call or every logical step
streams four blocks.

`PR8` is a distinct successor, not a repaired or renamed PR4 result. It keeps
the same official workload and phase policy while increasing only the Step10-A
Generator streamed set to eight blocks.

`PR12` is a separately user-authorized successor to the bounded PR8 failure.
It adds only the next four frozen Generator blocks and does not authorize an
automatic g16 search.

`RA16-F5-GLP-OFFLOAD` is the one-variable successor that tests the inactive
Generator ZeRO-1 BF16 LP flat state. It does not add Generator blocks or change
the Score policy. The exact state was released, but it was insufficient for
resident Score/Critic backward; unchanged retries and automatic g16 remain
closed.

## Run-ID rule

New runs use the lower-case experiment ID followed by date and ordinal, for
example:

```text
ra16_f5_pr4_20260808_001
```

Legacy run IDs and artifacts are immutable. Their status pages may add the
corresponding experiment ID as an alias, but files must not be renamed or
silently reclassified.

Every new manifest must include at least:

- `experiment_id`;
- `run_id`;
- topology and rank mapping;
- logical step range and update counts;
- lifecycle policy by component and phase;
- EMA execution contract;
- claim scope and evidence label.
