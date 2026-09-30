import hashlib
import json
from pathlib import Path
import pytest
from distillschedule.runtime.livewan_source_relocation import source_prepare_script


def test_exact_canonical_script_and_files_only(tmp_path,monkeypatch):
    current,native=tmp_path/'current',tmp_path/'native'
    rel=Path('scripts/distillschedule/prepare_livewan_crossscale_source.sh')
    up=native/'third_party/livewan'
    for root in (current,native):
        (root/rel).parent.mkdir(parents=True)
        (root/rel).write_text('#!/bin/bash\ntrue\n')
        (root/'third_party/livewan').mkdir(parents=True)
        (root/'third_party/livewan/block.py').write_text('unchanged\n')
    receipt=tmp_path/'source.json'
    receipt.write_text(json.dumps(dict(status='PASS',pythonpath=[str(up)],files={
        'block':dict(path=str(up/'block.py'),sha256=hashlib.sha256((up/'block.py').read_bytes()).hexdigest())})))
    def git(argv,text):
        root=Path(argv[argv.index('-C')+1]);command=argv[argv.index('-C')+2:]
        if command==['status','--porcelain']: return ''
        if root.name=='livewan':return '1eabf43c777bc5b2816926ca998cd4e4c2dfa3d5\n'
        if root.name=='wan21':return '9737cba9c1c3c4d04b33fcad41c111989865d315\n'
        return ('a' if root==current else 'b')*40+'\n'
    monkeypatch.setattr('subprocess.check_output',git)
    kwargs=dict(repo_root=current,canonical_root=native,receipt_path=receipt,
        current_head='a'*40,canonical_head='b'*40)
    original=receipt.read_bytes()
    assert source_prepare_script(**kwargs)==native/rel
    assert receipt.read_bytes()==original
    with pytest.raises(RuntimeError):source_prepare_script(**dict(kwargs,current_head='c'*40))
    (current/rel).write_text('changed')
    with pytest.raises(RuntimeError,match='not byte-identical'):source_prepare_script(**kwargs)
    (current/rel).write_bytes((native/rel).read_bytes())
    (current/'third_party/livewan/block.py').write_text('changed')
    with pytest.raises(RuntimeError,match='differs'):source_prepare_script(**kwargs)
