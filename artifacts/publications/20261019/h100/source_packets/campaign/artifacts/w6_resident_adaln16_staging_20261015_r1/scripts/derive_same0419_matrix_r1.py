"""Small auditable same-0419 W6 native matrix, never load or transfer tensors."""
import pathlib,json,csv,hashlib,datetime
P=pathlib.Path("/home/jiangzhikun/yongyan_liu/Helios-H100/research_artifacts/five_workload_h100_20260930/h100_resident_baselines_20261014_r1/artifacts/w6_resident_adaln16_staging_20261015_r1")
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
    with p.open("x") as f:json.dump(x,f,indent=2)
def main():
    f=P/"same_revision_native_matrix_r1/result.json";x=json.loads(f.read_text())
    assert x["status"]=="PASS_SAME_REVISION_W6_THREE_ARM_SINGLE_DIAGNOSTIC_MATRIX"
    prior_path=pathlib.Path(x["Resident_prerequisite"]["path"]);prior=json.loads(prior_path.read_text())
    assert sha(prior_path)==x["Resident_prerequisite"]["sha256"]
    report=P/"reports/completed_same0419_matrix_r1";report.mkdir(exist_ok=False)
    values=[];hashes={str(f):sha(f),str(prior_path):sha(prior_path)}
    arms=[dict(placement="resident_named_CPU_fallback",**prior["arms"][-1]),*x["arms"]]
    for arm in arms:
        assert arm["status"]=="PASS_REAL_8_RANK_16_LOGICAL_SLOT_NATIVE_GATE"
        assert arm["owned_alive_after_cleanup"]==[] and arm["GPU_contexts_after"]==""
        assert arm["cgroup_oom_delta"]=={"oom":0,"oom_kill":0}
        rows=[]
        for rank in range(8):
            f=pathlib.Path(arm["path"])/f"rank_receipts/rank_{rank:02d}.json";r=json.loads(f.read_text());hashes[str(f)]=sha(f)
            assert r["status"]=="PASS_REAL_NATIVE_AR8_G1F1_LOGICAL_B16" and r["final_versions"]=={"student":2,"fake":2,"ema":0}
            assert all(c["actual_census"]=={"student":18,"fake_score":4,"fake_score_grad":2,"critic_backward":2,"teacher":2,"student_grad":2,"student_backward":2} for c in r["cycles"])
            rows.append(r)
        wall=max(r["cycles"][1]["timer"]["stop"] for r in rows)-min(r["cycles"][1]["timer"]["start"] for r in rows)
        assert wall==arm["timed_C2_seconds"]
        values.append(dict(placement=arm["placement"],head=x["head"],native_C2_seconds=wall,C1_warmup_excluded_seconds=arm["warmup_C1_seconds"],
            peak_allocated_bytes=arm["peak_allocated_bytes"],peak_reserved_bytes=arm["peak_reserved_bytes"],
            sampled_host_cgroup_peak_bytes=arm["observed_host_cgroup_peak"],physical_ranks=8,independent_logical_slots=16,
            final_versions="Student2/Fake2/EMA0",new_host_OOM=0,owned_cleanup="EMPTY",
            full_model_strict_native_C2_state="NOT_RUN",formal_speedup_eligible=False))
    with (report/"arm_table.csv").open("x",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
    summary=dict(status=x["status"],source_head=x["head"],single_native_trials=values,
        fsdp_to_candidate_ratio=x["fsdp_to_candidate_ratio"],
        inherited_numerical="Actual27008 full C1 bitwise comparisons at1abd, explicit default-off entireworker AST equivalence to0419; not a fresh0419 strict proof.",
        native_C2_full_state_parity="NOT_RUN",formal_speedup_eligibility=False,
        Resident_boundary="CPU checkpoint boundaries+16 frozenAdaLN CPU parameters, LOW_HEADROOM, not pure no-offload Resident and not placement-only comparator.",
        optimized_attention_cache="existing exact-source FA2 result CPU reuse remains; generic activation_offload=False is not a zero-copy claim",
        source_identity_AST_proof="manifests/default_off_nonresident_source_equivalence_r1.json",
        one_cycle="private native AR8 continuous sigma G1/F1 LoRA64; not W5 Grid Student1/Fake5",
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    write(report/"result.json",summary);write(report/"source_receipt_sha256.json",hashes)
    lines=["# W6 same0419 native single-trial matrix","","|Arm|NativeC2(s)|ExcludedC1(s)|Peakallocated(GiB)|Peakreserved(GiB)|Samplednodecgrouppeak(GiB)|","|---|---:|---:|---:|---:|---:|"]
    for r in values:lines.append("|"+r["placement"]+"|"+f"{r['native_C2_seconds']:.9f}|{r['C1_warmup_excluded_seconds']:.9f}|{r['peak_allocated_bytes']/2**30:.6f}|{r['peak_reserved_bytes']/2**30:.6f}|{r['sampled_host_cgroup_peak_bytes']/2**30:.6f}|")
    lines.extend(["",f"Source {x['head']}. FSDP→candidate descriptive ratio{x['fsdp_to_candidate_ratio']:.9f}x; FA2<=280s target reached,1.5x not reached.",
        "All8physical/16independent logicalslots,2cyclesfinalStudent2/Fake2/EMA0,census/runtime/versions/leasePASS,zero newOOM andowncleanupempty.",
        "Resident CPUparameter+boundary staging is namedcapacityfallbackLOW_HEADROOM (~0.458GiBminimumsampledfree), NOTpureGPUResident.",
        "Sampledcgroupmemoryincludesfilecache/kernel/anon, not summedmodelRSS; kernelmemory.peakAPIunavailable.",
        "Oldactual27008strictbitwiseC1at1abd plusdefault-offentireworkerASTproof preserved. No fresh0419fullmodelcrossarmstateparity andnoC2fullstateproof. No formal numerical equivalence/speedup claim orCVfromsingletrial.",
        "No Nsight in these measurements; profilingtimes excluded. No A100or external190sMFU claim."])
    (report/"report.md").write_text("\n".join(lines)+"\n")
    write(report/"artifact_sha256.json",{f.name:sha(f) for f in report.iterdir() if f.is_file()})
    print(json.dumps(dict(status=summary["status"],out=str(report),bytes=sum(f.stat().st_size for f in report.iterdir()))))
if __name__=="__main__":main()
