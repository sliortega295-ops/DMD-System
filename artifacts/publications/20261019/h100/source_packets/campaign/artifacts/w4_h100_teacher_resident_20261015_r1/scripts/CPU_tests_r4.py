import os,json,datetime,pathlib,subprocess,sys,xml.etree.ElementTree as ET
p=pathlib.Path("/home/jiangzhikun/yongyan_liu/Helios-H100/research_artifacts/five_workload_h100_20260930/h100_resident_baselines_20261014_r1/artifacts/w4_h100_teacher_resident_20261015_r1");repo=pathlib.Path("/home/jiangzhikun/yongyan_liu/Helios-H100/repos/DMD-System-h100-livewan-teacher-resident-20261015-r1");head="1726d151efeeee6205229922f5f7a46da1a53103"
assert os.environ.get("CUDA_VISIBLE_DEVICES")==""
assert subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()==head
assert not subprocess.check_output(["git","-C",str(repo),"status","--porcelain"],text=True).strip()
os.chdir(repo)
import torch,pytest
assert not torch.cuda.is_initialized()
rc=pytest.main(["-q","tests/test_refactor_livewan_pipeline_teacher_placement.py","tests/test_refactor_livewan_baseline_placement.py","tests/test_refactor_livewan_h100_hardware.py","tests/test_refactor_livewan_validation_trace.py","tests/test_refactor_livewan_trajectory.py","--junitxml="+str(p/"reports/CPU_tests_r4.xml")])
t=ET.parse(p/"reports/CPU_tests_r4.xml").getroot();counts={k:sum(int(x.attrib.get(k,0)) for x in t.iter("testsuite")) for k in ("tests","failures","errors","skipped")}
r={"utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"head":head,"status":"PASS" if rc==0 and not torch.cuda.is_initialized() and not any(counts[k] for k in ("failures","errors","skipped")) else "FAIL","counts":counts,"CUDA_initialized":torch.cuda.is_initialized(),"torch":torch.__version__,"CUDA":torch.version.cuda,"pytest":pytest.__version__,"GPU":"NOT_RUN","workload":"W4 existingC34;Teacherplacementonly; physical8/global16unchanged"}
with (p/"manifests/CPU_tests_r4.json").open("x") as h:json.dump(r,h,indent=2)
print(json.dumps(r));raise SystemExit(0 if r["status"]=="PASS" else 1)
