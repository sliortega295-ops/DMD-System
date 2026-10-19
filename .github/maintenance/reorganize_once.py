#!/usr/bin/env python3
"""One-shot, source-preserving repository organization. See cleanup manifest."""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

ROOT=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path.cwd()
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from distillschedule.core.config import _load_with_includes
from distillschedule.workloads.best_known.registry import fingerprint

MARKER=ROOT/'docs/architecture/repository_cleanup.json'
if MARKER.exists():
    raise SystemExit('Refusing to apply this one-time migration twice')
files={str(p.relative_to(ROOT)):p.read_bytes() for p in ROOT.rglob('*')
       if p.is_file() and '.git' not in p.parts and '__pycache__' not in p.parts
       and '.pytest_cache' not in p.parts and not any(x.endswith('.egg-info') for x in p.parts)}
sha=lambda b:hashlib.sha256(b).hexdigest()
profiles={p:b for p,b in files.items() if p.startswith('configs/distillschedule/best_known/')}
profile_dir=ROOT/'configs/distillschedule/best_known'
canonical=[json.loads(p.read_text()) for p in profile_dir.glob('*_best.json')]
final_plan=next(v['runtime']['typed_plan'] for v in canonical if v['id']=='livewan_crossscale_pipeline_best')
final_raw=_load_with_includes(ROOT/final_plan,stack=(),sources=[])
assert fingerprint(final_raw)==next(v['runtime']['typed_plan_template_sha256'] for v in canonical if v['id']=='livewan_crossscale_pipeline_best')

# Retain canonical launchers and their implementation closure. Never classify a
# script used by a current package as an unused experiment based on its name.
script_paths={p for p in files if p.startswith('scripts/distillschedule/')}
keep={v['runtime']['launcher'] for v in canonical}
keep|={'scripts/distillschedule/run_best_known_workload.py','scripts/distillschedule/check_best_known_integration.py'}
source_text='\n'.join(b.decode('utf-8','replace') for p,b in files.items()
                      if p.startswith('src/') or p.startswith('configs/distillschedule/templates/'))
keep|={p for p in script_paths if Path(p).name in source_text}
while True:
    text='\n'.join(files[p].decode('utf-8','replace') for p in keep if p in files)
    nxt=keep|{p for p in script_paths if Path(p).name in text}
    if nxt==keep:break
    keep=nxt

moves={}
for p in files:
    if p.startswith('runtime/'):
        moves[p]='src/'+p
    elif p.startswith('configs/distillschedule/best_known/'):
        moves[p]=p.replace('configs/distillschedule/best_known/','configs/workloads/',1)
    elif p.startswith('configs/distillschedule/plans/livewan') and p!=final_plan:
        moves[p]='benchmarks/ablations/livewan/'+Path(p).name
    elif p.startswith('artifacts/experiments/'):
        moves[p]=p.replace('artifacts/experiments/','benchmarks/fixtures/historical/',1)
    elif p.startswith('docs/experiments/'):
        moves[p]=p.replace('docs/experiments/','docs/history/experiments/',1)
    elif p in script_paths and p not in keep:
        n=Path(p).name
        if n.startswith('canary_'):folder='benchmarks/kernels'
        elif n.startswith(('analyze_','plot_','profile_','bench_')):folder='tools/profiling'
        elif n.startswith(('build_','prepare_','derive_','assemble_','calibrate_')):folder='tools/assets'
        elif n.startswith(('check_','compare_','summarize_','validate_','evaluate_','export_')):folder='tools/validation'
        else:folder='benchmarks/baselines'
        moves[p]=folder+'/'+n

# Original incremental LiveWan plans are retained only as ablation fixtures;
# the canonical runtime plan becomes self-contained with identical resolved JSON.
historical_final='benchmarks/ablations/livewan/'+Path(final_plan).name
(ROOT/historical_final).parent.mkdir(parents=True,exist_ok=True)
(ROOT/historical_final).write_bytes(files[final_plan])
for old,new in sorted(moves.items()):
    target=ROOT/new
    if target.exists():raise RuntimeError(f'path collision: {old} -> {new}')
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.move(str(ROOT/old),target)
(ROOT/final_plan).write_text(json.dumps(final_raw,indent=2,ensure_ascii=False)+'\n')

# Rewrite source/document/test references, not immutable vendor or evidence data.
# Import names under the legacy `runtime` namespace remain unchanged.
substitutions={o:n for o,n in moves.items() if not o.startswith(('artifacts/','docs/experiments/'))}
substitutions={o:n for o,n in substitutions.items() if not o.startswith('runtime/')}
for old,new in list(moves.items()):
    if old.startswith('runtime/'):
        substitutions[old]=new
module_substitutions={o[:-3].replace('/','.'):n[:-3].replace('/','.')
                      for o,n in moves.items() if o.startswith('scripts/') and o.endswith('.py')}

changed=[]
for p in sorted(ROOT.rglob('*')):
    if not p.is_file() or '.git' in p.parts or '__pycache__' in p.parts:continue
    rel=str(p.relative_to(ROOT))
    if rel.startswith(('third_party/','benchmarks/fixtures/','artifacts/','configs/workloads/','.github/maintenance/')):continue
    if p.suffix not in {'.py','.sh','.md','.json','.yaml','.yml','.toml'}:continue
    if rel in {final_plan,historical_final,'docs/workloads/source_import_map.json'}:continue
    text=p.read_text(); original=text
    for old,new in sorted(substitutions.items(),key=lambda kv:-len(kv[0])):
        text=text.replace(old,new)
    for old,new in sorted(module_substitutions.items(),key=lambda kv:-len(kv[0])):
        text=text.replace(old,new)
    text=text.replace('configs/distillschedule/best_known','configs/workloads')
    text=text.replace('artifacts/experiments/','benchmarks/fixtures/historical/')
    text=text.replace('docs/experiments/','docs/history/experiments/')
    # Relative links between historical LiveWan plans stay local in their new directory.
    if rel.startswith('benchmarks/ablations/livewan/'):
        # Existing includes were same-directory basenames, so remain intact.
        pass
    # Old per-candidate test files intentionally compare historical raw configs.
    # Canonical registry tests continue to resolve the new self-contained winner.
    if rel.startswith('tests/') and 'livewan' in p.name and 'best_known' not in p.name:
        text=text.replace('configs/distillschedule/plans','benchmarks/ablations/livewan')
    for quote in ('"',"'"):
        text=text.replace('ROOT / '+quote+'runtime'+quote,'ROOT / '+quote+'src/runtime'+quote)
        text=text.replace('ROOT/'+quote+'runtime'+quote,'ROOT/'+quote+'src/runtime'+quote)
    text=text.replace('compileall -q src runtime','compileall -q src')
    text=text.replace('for folder in ("src","runtime","scripts/distillschedule"):','for folder in ("src","scripts","tools","benchmarks"):')
    # The import-map checker now resolves both namespaces from src/.
    text=text.replace('ROOT/("src" if name.startswith("distillschedule.") else "")/rel','ROOT/"src"/rel')
    if text!=original:p.write_text(text);changed.append(rel)

# The final manifest itself remains unchanged; its digest is over the resolved
# typed plan, not over the obsolete include-file layout.
assert _load_with_includes(ROOT/final_plan,stack=(),sources=[])==final_raw
for old,content in profiles.items():
    assert (ROOT/moves[old]).read_bytes()==content,old

project=ROOT/'pyproject.toml'
s=project.read_text().replace('include = ["distillschedule*"]','include = ["distillschedule*", "runtime*"]')
project.write_text(s)

# Refresh only layout-affected entries while preserving upstream provenance.
mp=ROOT/'docs/workloads/source_import_map.json'
m=json.loads(mp.read_text());mapped=[]
for row in m['files']:
    old=row['target'];new=moves.get(old,old)
    actual=sha((ROOT/new).read_bytes())
    if old!=new or row['integrated_sha256']!=actual:
        row['layout_origin']={'path':old,'integrated_sha256':row['integrated_sha256'],
                              'source_commit':'7b2c5a0d3b372cb1ae18b1d862783ff095b8c85e'}
        row['target']=new;row['integrated_sha256']=actual
    mapped.append(row)
m['files']=mapped
mp.write_text(json.dumps(m,indent=2,sort_keys=True)+'\n')

# Remove empty previous directories, never remove nonempty user content.
for p in sorted(ROOT.rglob('*'),key=lambda p:len(p.parts),reverse=True):
    if p.is_dir() and '.git' not in p.parts and p not in {ROOT}:
        try:p.rmdir()
        except OSError:pass

record={
 'schema_version':1,'base_commit':'7b2c5a0d3b372cb1ae18b1d862783ff095b8c85e',
 'runtime_semantics':'UNCHANGED_MODEL_LOSS_OPTIMIZER_AND_VERSION_CONTRACTS',
 'gpu_validation':'NOT_RUN',
 'canonical_profile_sha256':{Path(p).name:sha(b) for p,b in profiles.items()},
 'livewan_resolved_template_sha256':fingerprint(final_raw),
 'canonical_launch_closure':sorted(keep),
 'relocations':[{'from':o,'to':n,'original_sha256':sha(files[o])} for o,n in sorted(moves.items())],
 'livewan_original_final_fixture':historical_final,
 'edited_reference_files':sorted(changed),
}
MARKER.parent.mkdir(parents=True,exist_ok=True)
MARKER.write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
print(json.dumps({'moved_files':len(moves),'canonical_scripts':len(keep),'moved_scripts':sum(p.startswith('scripts/') for p in moves),
                  'final_livewan_digest':fingerprint(final_raw),'profile_count':len(canonical)},indent=2))

# Repair references constructed from separate path components and import forms.
R = ROOT
p = R / "scripts/distillschedule/verify_helios_compat_source.py"
p.write_text(p.read_text().replace('runtime_root=repo_root, file_root=source_root', 'runtime_root=repo_root / "src", file_root=source_root'))
p = R / "src/distillschedule/search/registry.py"
p.write_text(p.read_text().replace('configs/distillschedule/plans', 'benchmarks/ablations/livewan'))
p = R / "tests/test_refactor_livewan_calibration_summary.py"
p.write_text(p.read_text().replace('path = ROOT / "scripts/distillschedule" / f"{name}.py"', 'folder = "tools/assets" if name == "build_livewan_continuous_trace" else "scripts/distillschedule"\n    path = ROOT / folder / f"{name}.py"'))
from pathlib import Path
import re,json,hashlib
R=ROOT; rec=json.loads((R/'docs/architecture/repository_cleanup.json').read_text())
moves={v['from']:v['to'] for v in rec['relocations'] if v['from'].startswith('scripts/distillschedule/')}
for p in R.rglob('*'):
    if not p.is_file() or p.suffix not in {'.py','.sh','.md','.yml','.yaml','.json'}:continue
    rel=p.relative_to(R).as_posix()
    if rel.startswith(('.git/','.github/maintenance/','third_party/','artifacts/','benchmarks/fixtures/','configs/workloads/')) or p.name in {'source_import_map.json','repository_cleanup.json'}:continue
    s=p.read_text();orig=s
    for old,new in moves.items():
        filename=Path(old).name
        s=re.sub(r'''["']scripts["']\s*/\s*["']distillschedule["']\s*/\s*["']'''+re.escape(filename)+r'''["']''',repr(new),s)
        module=Path(old).stem
        s=re.sub(r'from scripts\.distillschedule import '+re.escape(module)+r'\b',f'from {str(Path(new).parent).replace("/",".")} import {module}',s)
    if s!=orig:p.write_text(s)

import runpy
runpy.run_path(str(ROOT / ".github/maintenance/cleanup_docs.py"), init_globals={"ROOT": ROOT})
runpy.run_path(str(ROOT / ".github/maintenance/cleanup_checks.py"), init_globals={"ROOT": ROOT, "profiles": profiles})

p = ROOT / "tests/test_refactor_helios_compat_source.py"
p.write_text(p.read_text().replace('shutil.copytree(ROOT / "src/runtime", self.repo / "runtime")', 'shutil.copytree(ROOT / "src/runtime", self.repo / "src/runtime")'))
from pathlib import Path
import json,hashlib
R=ROOT;p=R/'docs/workloads/source_import_map.json';m=json.loads(p.read_text())
for row in m['files']:
    value=hashlib.sha256((R/row['target']).read_bytes()).hexdigest()
    if value!=row['integrated_sha256']:
        row.setdefault('layout_origin',{'path':row['target'],'integrated_sha256':row['integrated_sha256'],'source_commit':'7b2c5a0d3b372cb1ae18b1d862783ff095b8c85e'})
        row['integrated_sha256']=value
p.write_text(json.dumps(m,indent=2,sort_keys=True)+'\n')

# Keep gitlinks present: empty worktree directories are not disposable packages.
for name in ("lightx2v", "livewan", "wan21"):
    (ROOT / "third_party" / name).mkdir(exist_ok=True)
# Retain this one-shot in commit history, not in the final source tree.
for rel in (".github/maintenance/reorganize_once.py", ".github/maintenance/cleanup_docs.py", ".github/maintenance/cleanup_checks.py", ".github/workflows/repository-cleanup-once.yml"):
    (ROOT / rel).unlink(missing_ok=True)
print("Repository cleanup applied. GPU and quality validation: NOT_RUN.")
