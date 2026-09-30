"""Explicit immutable-source path reuse for relocated experiment checkouts."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import subprocess


def source_prepare_script(*, repo_root: Path, receipt_path: Path,
                          canonical_root: Path, current_head: str,
                          canonical_head: str) -> Path:
    """Return only a byte-identical canonical preparation script, not runtime."""
    def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def git(root,*args):
        return subprocess.check_output(['git','-c',f'safe.directory={root}',
            '-C',str(root),*args],text=True).strip()
    v=json.loads(receipt_path.read_text())
    if v.get('status')!='PASS': raise RuntimeError('source receipt is not PASS')
    upstream=Path(v['pythonpath'][0])
    if (upstream.name!='livewan' or upstream.parent.name!='third_party'
        or upstream.parent.parent.resolve()!=canonical_root.resolve()):
        raise RuntimeError('canonical root differs from immutable source receipt')
    for root,head in ((repo_root,current_head),(canonical_root,canonical_head)):
        if (len(head)!=40 or git(root,'rev-parse','HEAD')!=head
            or git(root,'status','--porcelain')):
            raise RuntimeError('source preparation checkout revision/cleanliness changed')
        for name,revision in [('livewan','1eabf43c777bc5b2816926ca998cd4e4c2dfa3d5'),
                              ('wan21','9737cba9c1c3c4d04b33fcad41c111989865d315')]:
            if git(root/'third_party'/name,'rev-parse','HEAD')!=revision:
                raise RuntimeError('source preparation upstream revision changed')
    relative=Path('scripts/distillschedule/prepare_livewan_crossscale_source.sh')
    canonical=canonical_root/relative
    if sha(repo_root/relative)!=sha(canonical):
        raise RuntimeError('source preparation script is not byte-identical')
    for row in v['files'].values():
        original=Path(row['path'])
        if sha(original)!=row['sha256']:
            raise RuntimeError('immutable registered source file changed')
        if original.is_relative_to(upstream):
            current=repo_root/'third_party/livewan'/original.relative_to(upstream)
            if sha(current)!=row['sha256']:
                raise RuntimeError('relocated upstream source differs')
    return canonical
