"""Apply already-reviewed incremental source patches on the isolated branch.

Transport only: does not edit training algorithms, suppress failed tests, use
external credentials, or force-push. Publishing is a separate CI step after
validation. Every patch remains recoverable if a conversation is interrupted.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import lzma
from pathlib import Path
import subprocess

BRANCH = "agent/strategy-native-integration-20261012"
LEDGER = Path("docs/status/wan-checkpoint-ledger.json")
PENDING = Path(".git/wan-checkpoint-pending.json")


def git(*args, data=None):
    return subprocess.check_output(["git",*args],input=data).decode().strip()


def apply():
    if git("status","--porcelain"):
        raise SystemExit("Checkpoint application requires a clean checkout")
    ledger=json.loads(LEDGER.read_text()) if LEDGER.exists() else {"applied":{}}
    applied,targets={},set()
    for file in sorted(Path("checkpoints/wan").glob("*.patch.xz")):
        sha=hashlib.sha256(file.read_bytes()).hexdigest()
        if file.name in ledger["applied"]:
            if ledger["applied"][file.name]!=sha:
                raise SystemExit("Already-applied checkpoint changed")
            continue
        raw=lzma.decompress(file.read_bytes())
        expected=file.with_suffix(".sha256").read_text().strip()
        if len(raw)>2_000_000 or hashlib.sha256(raw).hexdigest()!=expected:
            raise SystemExit("Invalid checkpoint size or hash")
        rows=git("apply","--numstat",data=raw).splitlines()
        for row in rows:
            plus,minus,name=row.split("\t")
            path=Path(name)
            if (not plus.isdigit() or not minus.isdigit() or path.is_absolute()
                or any(part in {"..",".git",".github"} for part in path.parts)
                or path.is_symlink() or any(parent.is_symlink() for parent in path.parents)
                or not name.startswith(("src/distillschedule/","runtime/","tests/","docs/","configs/","scripts/distillschedule/"))):
                raise SystemExit(f"Unapproved source target: {name}")
            targets.add(name)
        git("apply","--check",data=raw)
        git("apply","--whitespace=error",data=raw)
        applied[file.name]=sha
    PENDING.write_text(json.dumps(dict(base=git('rev-parse','HEAD'),applied=applied,targets=sorted(targets))))
    print(json.dumps({"pending":applied,"targets":sorted(targets)}))


def publish():
    pending=json.loads(PENDING.read_text())
    if git('rev-parse','HEAD')!=pending['base']:
        raise SystemExit('HEAD changed during checkpoint validation')
    if not pending['applied']:
        print('No pending checkpoint'); return
    ledger=json.loads(LEDGER.read_text()) if LEDGER.exists() else {"applied":{}}
    ledger['applied'].update(pending['applied'])
    ledger.update(gpu_validation='NOT_RUN',scope='CPU source/dispatch tests; no speedup claim')
    LEDGER.parent.mkdir(parents=True,exist_ok=True)
    LEDGER.write_text(json.dumps(ledger,indent=2,sort_keys=True)+'\n')
    git('config','user.name','github-actions[bot]')
    git('config','user.email','41898282+github-actions[bot]@users.noreply.github.com')
    git('add','--',*pending['targets'],str(LEDGER))
    git('diff','--cached','--check')
    git('commit','-m','feat(strategies): apply validated Wan checkpoint '+','.join(pending['applied']))
    git('push','origin','HEAD:'+BRANCH)
    print('Published checkpoint:',git('rev-parse','HEAD'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['apply','publish'])
    args=parser.parse_args()
    apply() if args.command=='apply' else publish()
