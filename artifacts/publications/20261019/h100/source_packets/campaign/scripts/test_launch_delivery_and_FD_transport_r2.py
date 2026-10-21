from pathlib import Path
import os,subprocess,tempfile,ast,json
O=Path(__file__).resolve().parents[1]
r=(O/'scripts/run_h3_resident_boundary_fallback_r2.py').read_text()
assert "text.replace('export H3_ACTIVATION_POLICY=none', 'export H3_ACTIVATION_POLICY=checkpoint_boundary_cpu')" in r
observer=(O/'scripts/h3_h100_boundary_recording_entry_r2.py').read_text();tree=ast.parse(observer)
assert "self.activation_policy != 'checkpoint_boundary_cpu'" in observer
assert "self.activation_checkpoint_segment_size != 1" in observer
# Actual OS inheritance, scoped test FD only. No GPU or real project lock.
p=O/'artifacts/fd_transport_CPU_r2';p.mkdir(exist_ok=False)
f=open(p/'test_only.lock','w');fd=f.fileno()
code='import os,sys;os.fstat(int(sys.argv[1]));print("FD_PRESENT")'
a=subprocess.run([__import__('sys').executable,'-c',code,str(fd)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
b=subprocess.run([__import__('sys').executable,'-c',code,str(fd)],pass_fds=(fd,),stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
assert a.returncode!=0 and 'Bad file descriptor' in a.stderr
assert b.returncode==0 and b.stdout.strip()=='FD_PRESENT'
f.close()
s=(O/'scripts/run_w6_native_three_arm_r2.py').read_text()
assert "'--asset-fd','7'],check=True,pass_fds=(7,8,9)" in s
print(json.dumps({'status':'PASS_CPU_NAMED_POLICY_RENDER_AND_REAL_FD_INHERITANCE','GPU':'NOT_RUN','shared_locks':'NOT_TOUCHED','default_close_fds_EBADF':'REPRODUCED','explicit_pass_fds':'PASS'}))
