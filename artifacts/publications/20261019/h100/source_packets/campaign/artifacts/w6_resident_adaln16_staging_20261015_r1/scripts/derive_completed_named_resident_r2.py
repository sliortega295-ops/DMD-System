"""Derive bounded, small W6 evidence from original completed receipts; never load tensors."""
import csv,datetime,hashlib,json,pathlib
P=pathlib.Path("/home/jiangzhikun/yongyan_liu/Helios-H100/research_artifacts/five_workload_h100_20260930/h100_resident_baselines_20261014_r1/artifacts/w6_resident_adaln16_staging_20261015_r1")
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):
    with path.open("x") as f:json.dump(value,f,indent=2);f.write("\n")
def main():
    raw=P/"capacity_and_single_timing_r1/result.json"
    r=json.load(raw.open())
    assert r["status"]=="PASS_RESIDENT_ADALN_CPU_NAMED_CAPACITY_FALLBACK_SINGLE_DIAGNOSTIC"
    out=P/"reports/completed_named_resident_r2";out.mkdir(exist_ok=False,parents=True)
    expected={"student":18,"fake_score":4,"fake_score_grad":2,"critic_backward":2,"teacher":2,"student_grad":2,"student_backward":2}
    raw_hashes={str(raw):sha(raw)}
    compact=[];phase_rows=[];counter_rows=[]
    for arm in r["arms"]:
        if arm["name"] not in ("resident_correctness","resident_timing"):continue
        path=pathlib.Path(arm["path"]);rows=[]
        for rank in range(8):
            f=path/f"rank_receipts/rank_{rank:02d}.json";row=json.load(f.open());raw_hashes[str(f)]=sha(f)
            assert row["status"]=="PASS_REAL_NATIVE_AR8_G1F1_LOGICAL_B16" and row["physical_world"]==8 and row["logical_global_batch"]==16
            assert row["final_versions"]=={"student":2,"fake":2,"ema":0}
            assert row["parameter_offload"] and row["activation_offload"] and not row["model_swap"]
            assert row["strict_cross_arm_numerical_parity"]=="NOT_RUN"
            assert len(row["cycles"])==2
            for cycle in row["cycles"]:
                assert cycle["actual_census"]==expected
                b=cycle["checkpoint_boundary"];a=cycle["resident_adaln_CPU"]
                assert b["counters"]["checkpoint_calls"]==b["counters"]["saved_input_count"]==b["counters"]["D2H_count"]==b["counters"]["H2D_count"]==200
                assert b["counters"]["D2H_bytes"]==b["counters"]["H2D_bytes"]>0
                assert sum(int(size)*count for size,count in b["copy_size_histogram"].items())==b["counters"]["D2H_bytes"]
                assert sum(b["copy_size_histogram"].values())==200
                assert a["selected_blocks"]==16 and a["counters"]["H2D_bytes"]==233147400192
                counter_rows.append(dict(arm=arm["name"],rank=rank,cycle=cycle["cycle"],
                    boundary_checkpoint_calls=200,boundary_D2H_bytes=b["counters"]["D2H_bytes"],boundary_H2D_bytes=b["counters"]["H2D_bytes"],
                    staged_AdaLN_H2D_bytes=233147400192,canonical_AdaLN_CPU_bytes=a["canonical_CPU_bytes"],
                    byte_scope="application_requested_tensor_bytes_NOT_Nsight_measured_PCIe"))
            points=[*row["memory"],*[dict(phase=f"cycle_{c['cycle']}_after_timer",**c["memory_after_timer"]) for c in row["cycles"]]]
            for point in points:
                phase_rows.append(dict(arm=arm["name"],rank=rank,**{k:point.get(k) for k in
                    ("phase","allocated","reserved","peak_allocated","peak_reserved","driver_free","host_rss","cgroup_current")}))
            rows.append(row)
        c2=max(x["cycles"][1]["timer"]["stop"] for x in rows)-min(x["cycles"][1]["timer"]["start"] for x in rows)
        assert abs(c2-arm["timed_C2_seconds"])<1e-9
        assert not arm["owned_alive_after_cleanup"] and arm["GPU_contexts_after"]==""
        assert arm["cgroup_oom_delta"]=={"oom":0,"oom_kill":0}
        memory=[p for p in phase_rows if p["arm"]==arm["name"]]
        compact.append(dict(name=arm["name"],status=arm["status"],head=r["head"],physical_ranks=8,
            logical_global_batch=16,independent_slots=16,cycles=2,final_versions=rows[0]["final_versions"],
            native_C2_seconds=c2 if arm["name"]=="resident_timing" else None,
            correctness_mode_C2_diagnostic_seconds=c2 if arm["name"]=="resident_correctness" else None,
            excluded_C1_seconds=arm["warmup_C1_seconds"],peak_allocated_bytes=arm["peak_allocated_bytes"],
            peak_reserved_bytes=arm["peak_reserved_bytes"],minimum_boundary_sampled_driver_free_bytes=min(p["driver_free"] for p in memory),
            sampled_host_cgroup_peak=arm["observed_host_cgroup_peak"],
            cgroup_peak_api="NOT_AVAILABLE_ON_THIS_KERNEL",host_oom_delta=arm["cgroup_oom_delta"],
            owned_cleanup="PASS_EMPTY_CONTEXTS_AND_PROCESSES",strict_full_model_parity="NOT_RUN"))
    for name,rows in (("per_rank_phase_memory.csv",phase_rows),("per_rank_application_copy_census.csv",counter_rows)):
        with (out/name).open("x",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    timed=next(a for a in compact if a["name"]=="resident_timing")
    summary=dict(status="PASS_SINGLE_NATIVE_NAMED_RESIDENT_CAPACITY_FALLBACK_LOW_HEADROOM",
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),head=r["head"],raw_result=str(raw),
        native_C2_seconds=r["single_C2_seconds"],arms=compact,
        policy=dict(physical_backbone="one shared native AR H3 backbone",attention="unchanged aten FA2",
            sigma="native continuous shifted logit-normal",LoRA="FP32 rank64 native Student/Fake",
            activation="native checkpoint boundary pinned CPU",parameter="16 frozen AdaLN units pinned CPU",
            no_offload_resident=False),
        prior_pure_resident="CUDA_OOM preserved, no new attempt",
        low_headroom=True,stability="NOT_LONG_RUN_PROVEN",
        full_model_strict_cross_arm_parity="NOT_RUN",formal_speedup_eligibility=False,
        comparisons="Await fresh same0419 FSDP/candidate. Do not inflate primary denominator using this offload fallback.",
        bytes_scope="Application requested bytes only; hardware memcpy telemetry NOT_RUN")
    write(out/"result.json",summary);write(out/"source_receipts_sha256.json",raw_hashes)
    text=f"""# W6 native named Resident capacity fallback — completed, LOW_HEADROOM

- Exact source: {r['head']}.
- Single native C2: **{r['single_C2_seconds']:.9f} s**; separate C1 warmup **{timed['excluded_C1_seconds']:.9f} s** excluded.
- Eight physical H100 workers, sixteen independent logical slots, two cycles, final Student2/Fake2/EMA0. All runtime/census/version checks PASS. Pure no-offload Resident OOM retained.
- Peak allocated: **{timed['peak_allocated_bytes']/2**30:.6f} GiB**; peak reserved **{timed['peak_reserved_bytes']/2**30:.6f} GiB**.
- Minimum boundary-sampled driver free: **{timed['minimum_boundary_sampled_driver_free_bytes']/2**30:.6f} GiB**. Not a continuous minimum. This is LOW_HEADROOM, not long-run stability proof.
- Sampled node cgroup peak: **{timed['sampled_host_cgroup_peak']/2**30:.6f} GiB**. Kernel peak API unavailable. New cgroup OOM/kill zero; task-owned processes and contexts empty at each arm teardown.
- CPU checkpoint-boundary staging + selected16 frozen AdaLN parameter staging are both enabled. **Not a pure GPU Resident/no-offload baseline.**
- Per physical rank per cycle: boundary D2H/H2D each75,005,952,000–75,212,390,400 B (actual prompt-dependent shapes retained per rank in CSV), AdaLN parameter H2D233,147,400,192 B. These are requested tensor bytes, **not measured PCIe traffic**.
- Real full520MB AdaLN primitive28 bitwise comparisons and8rank tiny1536 parameter/Adam checks PASS. Full-model strict cross-arm parity NOT_RUN. No formal speedup eligibility.
- Correctness-mode C2 {compact[0]['correctness_mode_C2_diagnostic_seconds']:.9f}s is not the native performance result.
- Next: same0419 default-off source-bound FSDP/candidate native comparison; retain original strict C1 at1abd with explicit CPU AST equivalence, never relabel it fresh0419 numerical proof.
"""
    (out/"report.md").write_text(text)
    write(out/"artifact_sha256.json",{f.name:sha(f) for f in out.iterdir() if f.is_file()})
    print(json.dumps(dict(status=summary["status"],out=str(out),bytes=sum(f.stat().st_size for f in out.iterdir()),native=r["single_C2_seconds"])))
if __name__=="__main__":main()
