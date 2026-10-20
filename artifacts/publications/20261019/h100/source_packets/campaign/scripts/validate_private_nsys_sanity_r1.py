from pathlib import Path
import json
import re
import sqlite3
import sys

p=Path(sys.argv[1]);log=(p/'profiler.log').read_text(errors='replace')+(p/'export.log').read_text(errors='replace')
errors=re.findall(r'ProcessEventsError|Unknown runtime API|Cannot find bucket|Cannot find string|Errors occurred while processing',log)
assert (p/'sanity.nsys-rep').stat().st_size>0 and (p/'sanity.sqlite').read_bytes()[:16]==b'SQLite format 3\x00'
c=sqlite3.connect('file:'+str(p/'sanity.sqlite')+'?mode=ro',uri=True)
tables={r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
required={'CUPTI_ACTIVITY_KIND_KERNEL','CUPTI_ACTIVITY_KIND_MEMCPY','NVTX_EVENTS','StringIds'}
assert required<=tables, sorted(tables)
cols={r[1] for r in c.execute('PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)')}
assert {'deviceId','start','end','demangledName'}<=cols, cols
device_rows=[dict(zip(('device','events','resolved_events','unresolved_duration_ns','kernel_sum_ns'),r)) for r in c.execute('''
SELECT k.deviceId,COUNT(*),SUM(CASE WHEN s.value IS NOT NULL AND length(s.value)>0 THEN 1 ELSE 0 END),
SUM(CASE WHEN s.value IS NULL OR length(s.value)=0 THEN k.end-k.start ELSE 0 END),SUM(k.end-k.start)
FROM CUPTI_ACTIVITY_KIND_KERNEL k LEFT JOIN StringIds s ON k.demangledName=s.id GROUP BY k.deviceId ORDER BY k.deviceId''')]
nvtx=c.execute('SELECT COUNT(*) FROM NVTX_EVENTS').fetchone()[0]
copy=c.execute('SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_MEMCPY').fetchone()[0]
rows=[json.load(open(f)) for f in sorted(p.glob('rank_??.json'))]
def events(name):return {k:int(v) for k,v in map(str.split,(p/name).read_text().splitlines())}
a,b=events('cgroup_before.txt'),events('cgroup_after.txt');delta={k:b[k]-a[k] for k in ('oom','oom_kill')}
passed=not errors and [r['device'] for r in device_rows]==list(range(8)) and all(r['events']>0 and r['unresolved_duration_ns']/max(r['kernel_sum_ns'],1)<0.01 for r in device_rows) and nvtx>0 and copy>0 and len(rows)==8 and all(r['status'].startswith('PASS_') for r in rows) and not any(delta.values()) and not (p/'gpu_contexts_after.txt').read_text().strip()
result={'status':'PASS_PARSER_COMPATIBILITY_ONLY' if passed else 'FAIL_PROFILE_PARTIAL_PRESERVED',
        'devices':device_rows,'NVTX_events':nvtx,'copy_events':copy,'parser_errors':errors,
        'cgroup_oom_delta':delta,'rank_count':len(rows),'DMD_workload':'NOT_RUN','DMD_utilization_or_speedup':'NOT_CLAIMABLE_FROM_CANARY'}
with (p/'result.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result));assert passed
