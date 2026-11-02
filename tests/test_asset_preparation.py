"""Source-byte preparation only: no real model payloads, imports or downloads."""
from __future__ import annotations

import errno
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from distillschedule.workloads.best_known.registry import fingerprint


@pytest.fixture
def local_source(tmp_path):
    root = tmp_path / 'source'
    (root / 'transformer').mkdir(parents=True)
    (root / 'transformer/config.json').write_text('{"fixture": true}\n')
    (root / 'transformer/model.safetensors').write_bytes(b'NOT_MODEL_WEIGHTS')
    (root / 'untrusted.py').write_text('raise AssertionError("source code executed")\n')
    return root


def cli(*argv, env=None, allow_hub=False):
    # A fresh process makes an import assertion meaningful even when other tests
    # in the caller have already imported optional training dependencies.
    forbidden = {'torch', 'diffusers', 'transformers', 'safetensors'}
    if not allow_hub:
        forbidden.add('huggingface_hub')
    script = '''import importlib.abc
import sys
class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, name, *args):
        if name.split('.')[0] in FORBIDDEN:
            raise AssertionError('unexpected dependency import: ' + name)
sys.meta_path.insert(0, NoModels())
from distillschedule.cli import main
code = main(sys.argv[1:])
assert not any(name in sys.modules for name in FORBIDDEN)
raise SystemExit(code)
'''.replace('FORBIDDEN', repr(forbidden))
    return subprocess.run([sys.executable, '-c', script, *map(str, argv)],
                          env=env, text=True, capture_output=True)


def test_local_prepare_hashes_in_place_and_verifies(local_source, tmp_path):
    from distillschedule import assets
    before = {str(p.relative_to(local_source)): p.stat().st_ino
              for p in local_source.rglob('*') if p.is_file()}
    output = tmp_path / 'receipt.json'
    value = assets.prepare(local_dir=local_source, output=output)
    assert value == json.loads(output.read_text()) == assets.verify(output)
    assert value['schema_version'] == 'distillschedule.assets.source.v1'
    assert value['status'] == 'complete'
    assert value['scope'] == 'SOURCE_BYTES_ONLY_NOT_NATIVE_MODEL_ADMISSION'
    assert value['root'] == str(local_source)
    assert value['source'] == {'kind': 'local', 'path': str(local_source)}
    assert value['sha256'] == fingerprint({k: v for k, v in value.items() if k != 'sha256'})
    assert set(value['files']) == set(before)
    for name, pin in value['files'].items():
        path = local_source / name
        assert pin == {'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        assert path.stat().st_ino == before[name]
    assert not any(p.name.startswith('.receipt.json.') for p in tmp_path.iterdir())


def test_local_frontdoor_prepare_and_fresh_verify_have_no_optional_imports(local_source, tmp_path):
    output = tmp_path / 'receipt.json'
    prepared = cli('assets', 'prepare', '--local-dir', local_source, '--output', output)
    assert prepared.returncode == 0, prepared.stderr
    checked = cli('assets', 'verify', output)
    assert checked.returncode == 0, checked.stderr
    assert json.loads(prepared.stdout) == json.loads(checked.stdout) == json.loads(output.read_text())
    assert prepared.stderr == checked.stderr == ''


def test_receipt_cannot_mutate_inventoried_root(local_source):
    from distillschedule import assets
    output = local_source / 'receipt.json'
    with pytest.raises(ValueError, match='outside'):
        assets.prepare(local_dir=local_source, output=output)
    assert not output.exists()


@pytest.mark.parametrize('change', ['corrupt', 'missing', 'extra'])
def test_verify_rejects_changed_file_set_or_bytes(local_source, tmp_path, change):
    from distillschedule import assets
    output = tmp_path / 'receipt.json'
    assets.prepare(local_dir=local_source, output=output)
    path = local_source / 'transformer/model.safetensors'
    if change == 'corrupt':
        path.write_bytes(b'X' * path.stat().st_size)  # Same size; hashes must catch this.
    elif change == 'missing':
        path.unlink()
    else:
        (local_source / 'extra').write_bytes(b'new file')
    with pytest.raises(ValueError, match='inventory|hash|bytes'):
        assets.verify(output)


def rewrite_receipt(path, change):
    value = json.loads(path.read_text())
    change(value)
    value['sha256'] = fingerprint({k: v for k, v in value.items() if k != 'sha256'})
    path.write_text(json.dumps(value))


@pytest.mark.parametrize('field,value', [
    ('status', 'partial'), ('schema_version', 'unknown'), ('scope', 'MODEL_ADMITTED'),
    ('root', 'relative'), ('source', {'kind': 'local', 'path': '/not-the-root'}),
    ('files', {}), ('unknown', True),
])
def test_verify_rejects_incomplete_or_foreign_receipt(local_source, tmp_path, field, value):
    from distillschedule import assets
    output = tmp_path / 'receipt.json'
    assets.prepare(local_dir=local_source, output=output)
    rewrite_receipt(output, lambda row: row.__setitem__(field, value))
    with pytest.raises(ValueError):
        assets.verify(output)


@pytest.mark.parametrize('name', ['../outside', '/absolute', './model', 'a//b', 'a/../b', 'a\\b', ''])
def test_verify_refuses_escaping_or_noncanonical_paths(local_source, tmp_path, name):
    from distillschedule import assets
    output = tmp_path / 'receipt.json'
    assets.prepare(local_dir=local_source, output=output)
    def change(value):
        value['files'][name] = value['files'].pop('untrusted.py')
    rewrite_receipt(output, change)
    with pytest.raises(ValueError, match='path'):
        assets.verify(output)


@pytest.mark.parametrize('pin', [
    {'bytes': True, 'sha256': '0' * 64}, {'bytes': -1, 'sha256': '0' * 64},
    {'bytes': 1, 'sha256': 'not-a-digest'}, {'bytes': 1},
])
def test_verify_checks_exact_file_pin_schema(local_source, tmp_path, pin):
    from distillschedule import assets
    output = tmp_path / 'receipt.json'
    assets.prepare(local_dir=local_source, output=output)
    rewrite_receipt(output, lambda row: row['files'].__setitem__('untrusted.py', pin))
    with pytest.raises(ValueError):
        assets.verify(output)


@pytest.mark.parametrize('kind', ['duplicate', 'nested-duplicate', 'digest', 'truncated', 'missing-field'])
def test_strict_json_and_receipt_digest(local_source, tmp_path, kind):
    from distillschedule import assets
    output = tmp_path / 'receipt.json'
    assets.prepare(local_dir=local_source, output=output)
    text = output.read_text()
    if kind == 'duplicate':
        text = text.replace('{', '{"status":"complete",', 1)
    elif kind == 'nested-duplicate':
        text = text.replace('"bytes":', '"bytes": 0, "bytes":', 1)
    elif kind == 'digest':
        text = text.replace('"status": "complete"', '"status": "partial"')
    elif kind == 'truncated':
        text = text[:len(text) // 2]
    else:
        value = json.loads(text); del value['files']; text = json.dumps(value)
    output.write_text(text)
    with pytest.raises(ValueError):
        assets.verify(output)


@pytest.mark.parametrize('kind', ['internal-file', 'outside-file', 'directory', 'broken', 'root'])
def test_local_symlinks_are_explicitly_refused(local_source, tmp_path, kind):
    from distillschedule import assets
    source = local_source
    link = local_source / 'link'
    if kind == 'root':
        source = tmp_path / 'linked-root'; source.symlink_to(local_source, target_is_directory=True)
    elif kind == 'internal-file':
        link.symlink_to(local_source / 'untrusted.py')
    elif kind == 'directory':
        link.symlink_to(local_source / 'transformer', target_is_directory=True)
    else:
        outside = tmp_path / 'outside'
        if kind != 'broken': outside.write_bytes(b'outside')
        link.symlink_to(outside)
    output = tmp_path / 'receipt.json'
    with pytest.raises(ValueError, match='symlink'):
        assets.prepare(local_dir=source, output=output)
    assert not output.exists()


@pytest.mark.parametrize('kind', ['empty', 'fifo', 'file'])
def test_non_inventory_sources_refuse_without_receipt(tmp_path, kind):
    from distillschedule import assets
    root = tmp_path / 'source'
    if kind == 'file': root.write_bytes(b'not a directory')
    else:
        root.mkdir()
        if kind == 'fifo': os.mkfifo(root / 'pipe')
    output = tmp_path / 'receipt.json'
    with pytest.raises(ValueError):
        assets.prepare(local_dir=root, output=output)
    assert not output.exists()


@pytest.mark.parametrize('kind', ['existing', 'dangling-symlink', 'racing-writer'])
def test_publication_never_overwrites_an_output(local_source, tmp_path, monkeypatch, kind):
    from distillschedule import assets
    output = tmp_path / 'receipt.json'
    if kind == 'existing': output.write_bytes(b'prior receipt')
    elif kind == 'dangling-symlink': output.symlink_to(tmp_path / 'missing')
    else:
        link = assets.os.link
        def racing(source, target):
            Path(target).write_bytes(b'prior receipt')
            link(source, target)
        monkeypatch.setattr(assets.os, 'link', racing)
    with pytest.raises(FileExistsError):
        assets.prepare(local_dir=local_source, output=output)
    assert output.is_symlink() if kind == 'dangling-symlink' else output.read_bytes() == b'prior receipt'
    assert not list(tmp_path.glob('.receipt.json.*'))


@pytest.mark.parametrize('failure', ['read', 'changed-during-read', 'fsync', 'link'])
def test_failed_prepare_cleans_staging_and_leaves_no_valid_receipt(local_source, tmp_path, monkeypatch, failure):
    from distillschedule import assets
    real_hash = assets.file_sha256
    def failed(*args): raise PermissionError('denied')
    def changed(path):
        value = real_hash(path)
        path.write_bytes(path.read_bytes() + b'changed')
        return value
    if failure in ('read', 'changed-during-read'):
        monkeypatch.setattr(assets, 'file_sha256', failed if failure == 'read' else changed)
    else:
        monkeypatch.setattr(assets.os, failure, failed)
    output = tmp_path / 'receipt.json'
    with pytest.raises((OSError, ValueError)):
        assets.prepare(local_dir=local_source, output=output)
    assert not output.exists() and not list(tmp_path.glob('.receipt.json.*'))


def test_cleanup_error_after_atomic_publication_does_not_report_failed_preparation(local_source, tmp_path, monkeypatch):
    from distillschedule import assets
    unlink = Path.unlink
    def refused(path, *args, **kwargs):
        if path.name.startswith('.receipt.json.'):
            raise PermissionError('simulated staging cleanup denial')
        return unlink(path, *args, **kwargs)
    output = tmp_path / 'receipt.json'
    with monkeypatch.context() as patch:
        patch.setattr(Path, 'unlink', refused)
        value = assets.prepare(local_dir=local_source, output=output)
    assert assets.verify(output) == value
    # A post-commit cleanup failure can leave a non-authoritative staging link;
    # the already completed output must not be misreported as failed.
    for path in tmp_path.glob('.receipt.json.*'):
        path.unlink()


def test_receipt_parent_symlink_cannot_place_output_inside_root(local_source, tmp_path):
    from distillschedule import assets
    alias = tmp_path / 'alias'; alias.symlink_to(local_source, target_is_directory=True)
    with pytest.raises(ValueError, match='outside'):
        assets.prepare(local_dir=local_source, output=alias / 'receipt.json')
    assert not (local_source / 'receipt.json').exists()


def test_missing_receipt_is_normal_cli_error(tmp_path):
    result = cli('assets', 'verify', tmp_path / 'missing.json')
    assert result.returncode == 2 and result.stdout == ''
    assert 'ERROR:' in result.stderr and 'Traceback' not in result.stderr


@pytest.mark.parametrize('argv', [('assets', '--help'), ('assets', 'prepare', '--help'), ('assets', 'verify', '--help')])
def test_asset_help_is_dependency_free(argv):
    result = cli(*argv)
    assert result.returncode == 0, result.stderr
    assert 'usage: distillctl assets' in result.stdout


HF_REPO = 'fixture/model'
HF_REVISION = 'a' * 40


def git_blob(payload):
    return hashlib.sha1(b'blob ' + str(len(payload)).encode() + b'\0' + payload).hexdigest()


@pytest.fixture
def hub_cache(tmp_path):
    hub = pytest.importorskip('huggingface_hub')
    from huggingface_hub.hf_api import RepoFile, RepoFolder
    cache = tmp_path / 'hub'
    repo = cache / 'models--fixture--model'
    root = repo / 'snapshots' / HF_REVISION
    (repo / 'blobs').mkdir(parents=True)
    (root / 'transformer').mkdir(parents=True)
    payloads = {'config.json': b'{"tiny":true}\n', 'transformer/weights.safetensors': b'NOT_MODEL_WEIGHTS'}
    tree = [RepoFolder(path='transformer', oid='0' * 40)]
    for name, payload in payloads.items():
        lfs = name.endswith('.safetensors')
        digest = hashlib.sha256(payload).hexdigest() if lfs else git_blob(payload)
        blob = repo / 'blobs' / digest; blob.write_bytes(payload)
        path = root / name
        path.symlink_to(os.path.relpath(blob, path.parent))
        metadata = {'path': name, 'size': len(payload), 'oid': git_blob(payload)}
        if lfs: metadata['lfs'] = {'size': len(payload), 'oid': digest, 'pointerSize': 128}
        tree.append(RepoFile(**metadata))
    return hub, cache, root, tree


def bind_hub(monkeypatch, hub_cache):
    from distillschedule import assets
    hub, cache, root, tree = hub_cache
    calls = []
    def listing(self, repo_id, **kwargs):
        calls.append(('tree', repo_id, kwargs)); return iter(tree)
    def download(repo_id, **kwargs):
        calls.append(('file', repo_id, kwargs)); return str(root / kwargs['filename'])
    monkeypatch.setattr(hub.HfApi, 'list_repo_tree', listing)
    monkeypatch.setattr(hub, 'hf_hub_download', download)
    # In-process fakes cover byte/path admission. Installed-Hub loopback tests
    # separately exercise the real bounded subprocess and SDK transport.
    monkeypatch.setattr(assets, '_bounded_hub_files', lambda repo, revision, timeout: assets._hub_files(repo, revision))
    return calls


def test_hf_prepare_binds_complete_tree_commit_cache_and_remote_file_hashes(hub_cache, tmp_path, monkeypatch):
    from distillschedule import assets
    _, cache, root, tree = hub_cache
    calls = bind_hub(monkeypatch, hub_cache)
    output = tmp_path / 'receipt.json'
    value = assets.prepare(hf_repo=HF_REPO, revision=HF_REVISION, cache_dir=cache, output=output)
    assert value == assets.verify(output)
    assert value['root'] == str(root)
    assert value['source'] == {'kind': 'huggingface', 'repo_id': HF_REPO,
                               'revision': HF_REVISION, 'cache_dir': str(cache)}
    assert calls[0] == ('tree', HF_REPO, dict(revision=HF_REVISION, repo_type='model', recursive=True))
    assert sorted(calls[1:], key=lambda call: call[2]['filename']) == [
        ('file', HF_REPO, dict(filename=name, revision=HF_REVISION, repo_type='model', cache_dir=str(cache),
                              local_files_only=False, etag_timeout=10.0)) for name in sorted(value['files'])]
    assert set(value['files']) == {'config.json', 'transformer/weights.safetensors'}
    assert value['files']['config.json']['hub_digest'] == {'algorithm': 'git_blob_sha1', 'value': tree[1].blob_id}
    assert value['files']['transformer/weights.safetensors']['hub_digest'] == {
        'algorithm': 'sha256', 'value': tree[2].lfs.sha256}
    checked = cli('assets', 'verify', output)
    assert checked.returncode == 0, checked.stderr
    assert json.loads(checked.stdout) == value


@pytest.mark.parametrize('revision', [None, 'main', 'v1', 'a' * 39, 'a' * 41, 'A' * 40, '../' + 'a' * 40])
def test_hf_requires_immutable_full_commit_before_hub_calls(hub_cache, tmp_path, monkeypatch, revision):
    from distillschedule import assets
    calls = bind_hub(monkeypatch, hub_cache)
    output = tmp_path / 'receipt.json'
    with pytest.raises(ValueError, match='commit|revision'):
        assets.prepare(hf_repo=HF_REPO, revision=revision, cache_dir=hub_cache[1], output=output)
    assert calls == [] and not output.exists()


@pytest.mark.parametrize('timeout', [0, -1, float('nan'), float('inf'), -float('inf'), True, '1', 10 ** 400])
def test_hf_tree_timeout_requires_finite_positive_seconds_before_hub_import(tmp_path, timeout):
    from distillschedule import assets
    output, target = tmp_path / 'receipt.json', tmp_path / 'bundle'
    with pytest.raises(ValueError, match='finite positive'):
        assets.prepare(hf_repo=HF_REPO, revision=HF_REVISION, tree_timeout=timeout,
                       materialize_dir=target, output=output)
    assert not output.exists() and not target.exists()
    assert not list(tmp_path.glob('.receipt.json.*')) and not list(tmp_path.glob('.bundle.*'))


@pytest.mark.parametrize('timeout', ['0', '-1', 'nan', 'inf'])
def test_hf_cli_refuses_nonpositive_or_nonfinite_tree_timeout_without_hub_import(tmp_path, timeout):
    result = cli('assets', 'prepare', '--hf-repo', HF_REPO, '--revision', HF_REVISION,
                 '--tree-timeout=' + timeout, '--output', tmp_path / 'receipt.json')
    assert result.returncode == 2 and 'finite positive' in result.stderr
    assert 'Traceback' not in result.stderr and not (tmp_path / 'receipt.json').exists()


@pytest.mark.parametrize('kind', ['partial', 'extra', 'lfs-hash', 'git-hash', 'size', 'wrong-revision',
                                 'foreign-cache', 'symlink-escape', 'symlink-directory', 'missing-metadata',
                                 'duplicate-file', 'escaping-metadata', 'incomplete-list'])
def test_hf_rejects_partial_cache_or_foreign_pins_without_receipt(hub_cache, tmp_path, monkeypatch, kind):
    from distillschedule import assets
    hub, cache, root, tree = hub_cache
    calls = bind_hub(monkeypatch, hub_cache)
    if kind == 'partial': (root / 'transformer/weights.safetensors').unlink()
    elif kind == 'extra': (root / 'extra').write_bytes(b'unlisted')
    elif kind in ('lfs-hash', 'git-hash', 'size'):
        entry = tree[1 if kind == 'git-hash' else 2]
        if kind == 'lfs-hash': entry.lfs.sha256 = '0' * 64
        elif kind == 'git-hash': entry.blob_id = '0' * 40
        else: entry.size += 1
    elif kind in ('wrong-revision', 'foreign-cache'):
        other = root.parent / ('b' * 40) if kind == 'wrong-revision' else tmp_path / 'foreign' / HF_REVISION
        other.mkdir(parents=True)
        monkeypatch.setattr(hub, 'hf_hub_download', lambda *a, **kw: str(other / kw['filename']))
    elif kind == 'symlink-escape':
        (tmp_path / 'outside').write_bytes((root / 'config.json').read_bytes())
        (root / 'config.json').unlink(); (root / 'config.json').symlink_to(tmp_path / 'outside')
    elif kind == 'symlink-directory':
        import shutil
        shutil.rmtree(root / 'transformer')
        (root / 'transformer').symlink_to(tmp_path, target_is_directory=True)
    elif kind == 'missing-metadata': tree.clear()
    elif kind == 'duplicate-file': tree.append(tree[-1])
    elif kind == 'escaping-metadata': tree[-1].path = '../weights.safetensors'
    else:
        def interrupted(self, *args, **kwargs):
            yield tree[1]
            raise OSError('metadata pagination failed')
        monkeypatch.setattr(hub.HfApi, 'list_repo_tree', interrupted)
    output = tmp_path / 'receipt.json'
    with pytest.raises((ValueError, OSError, RuntimeError)):
        assets.prepare(hf_repo=HF_REPO, revision=HF_REVISION, cache_dir=cache, output=output)
    assert not output.exists() and not list(tmp_path.glob('.receipt.json.*'))
    if kind in ('missing-metadata', 'duplicate-file', 'escaping-metadata', 'incomplete-list'):
        assert not any(call[0] == 'file' for call in calls)


def test_hf_output_inside_snapshot_refused_before_any_hub_call(hub_cache, monkeypatch):
    from distillschedule import assets
    _, cache, root, _ = hub_cache
    calls = bind_hub(monkeypatch, hub_cache)
    with pytest.raises(ValueError, match='outside'):
        assets.prepare(hf_repo=HF_REPO, revision=HF_REVISION, cache_dir=cache, output=root / 'receipt.json')
    assert calls == [] and not (root / 'receipt.json').exists()


def test_hf_rejects_symlinked_snapshot_ancestors_before_hub_calls(hub_cache, monkeypatch, tmp_path):
    from distillschedule import assets
    _, cache, root, _ = hub_cache
    calls = bind_hub(monkeypatch, hub_cache)
    snapshots = root.parent
    snapshots.rename(snapshots.with_name('elsewhere'))
    snapshots.symlink_to(snapshots.with_name('elsewhere'), target_is_directory=True)
    with pytest.raises(ValueError, match='symlink|snapshot'):
        assets.prepare(hf_repo=HF_REPO, revision=HF_REVISION, cache_dir=cache, output=tmp_path / 'receipt.json')
    assert calls == [] and not (tmp_path / 'receipt.json').exists()


def test_hf_default_cache_uses_the_installed_hub_setting(hub_cache, monkeypatch, tmp_path):
    from distillschedule import assets
    from huggingface_hub import constants
    _, cache, root, _ = hub_cache
    calls = bind_hub(monkeypatch, hub_cache)
    monkeypatch.setattr(constants, 'HF_HUB_CACHE', str(cache))
    value = assets.prepare(hf_repo=HF_REPO, revision=HF_REVISION, output=tmp_path / 'receipt.json')
    assert value['root'] == str(root) and calls[-1][2]['cache_dir'] == str(cache)


@pytest.mark.parametrize('copy_fallback', [False, True])
def test_explicit_materialization_reuses_verified_cache_bytes_as_regular_files(hub_cache, monkeypatch, tmp_path, copy_fallback):
    from distillschedule import assets
    _, cache, snapshot, _ = hub_cache
    bind_hub(monkeypatch, hub_cache)
    link = assets.os.link
    def cross_device(source, target):
        if Path(source).parent.name == 'blobs': raise OSError(errno.EXDEV, 'cross device fixture')
        return link(source, target)
    if copy_fallback: monkeypatch.setattr(assets.os, 'link', cross_device)
    target = tmp_path / 'materialized'
    output = tmp_path / 'receipt.json'
    value = assets.prepare(hf_repo=HF_REPO, revision=HF_REVISION, cache_dir=cache,
                           materialize_dir=target, output=output)
    assert value == assets.verify(output)
    assert value['root'] == str(target) and value['source']['materialized'] is True
    assert value['source']['cache_dir'] == str(cache) and value['source']['revision'] == HF_REVISION
    for name in value['files']:
        original, actual = snapshot / name, target / name
        assert actual.is_file() and not actual.is_symlink()
        assert actual.read_bytes() == original.read_bytes()
        assert (actual.stat().st_ino == original.stat().st_ino) is not copy_fallback
        assert value['files'][name]['sha256'] == hashlib.sha256(actual.read_bytes()).hexdigest()
    assert set(p.relative_to(target).as_posix() for p in target.rglob('*') if p.is_file()) == set(value['files'])
    assert cli('assets', 'verify', output).returncode == 0
    # The materialized bytes do not need their cache paths to remain present.
    import shutil
    shutil.rmtree(cache)
    assert assets.verify(output) == value


@pytest.mark.parametrize('collision', ['target-directory', 'target-file', 'target-symlink', 'lock', 'receipt',
                                      'receipt-in-target', 'target-in-receipt', 'target-in-cache'])
def test_materialization_collisions_refuse_before_hub_calls(hub_cache, monkeypatch, tmp_path, collision):
    from distillschedule import assets
    _, cache, _, _ = hub_cache
    calls = bind_hub(monkeypatch, hub_cache)
    target, output = tmp_path / 'bundle', tmp_path / 'receipt.json'
    preserved = None
    if collision == 'target-directory':
        target.mkdir(); preserved = target / 'keep'; preserved.write_bytes(b'preserve target')
    elif collision == 'target-file':
        preserved = target; preserved.write_bytes(b'preserve target')
    elif collision == 'target-symlink':
        preserved = tmp_path / 'foreign'; preserved.write_bytes(b'preserve target'); target.symlink_to(preserved)
    elif collision == 'lock':
        preserved = tmp_path / '.bundle.lock'; preserved.write_bytes(b'preserve target')
    elif collision == 'receipt':
        preserved = output; preserved.write_bytes(b'preserve target')
    elif collision == 'receipt-in-target': output = target / 'receipt.json'
    elif collision == 'target-in-receipt': target = output / 'bundle'
    else: target = cache / 'bundle'
    with pytest.raises((ValueError, FileExistsError)):
        assets.prepare(hf_repo=HF_REPO, revision=HF_REVISION, cache_dir=cache,
                       materialize_dir=target, output=output)
    assert calls == []
    if preserved is not None: assert preserved.read_bytes() == b'preserve target'
    if collision != 'receipt': assert not output.exists()


@pytest.mark.parametrize('failure', ['link', 'copy', 'inventory', 'rename', 'rename-after-commit', 'receipt'])
def test_failed_materialization_cleans_owned_stage_target_and_receipt(hub_cache, monkeypatch, tmp_path, failure):
    from distillschedule import assets
    _, cache, snapshot, _ = hub_cache
    bind_hub(monkeypatch, hub_cache)
    target, output = tmp_path / 'bundle', tmp_path / 'receipt.json'
    before = {str(p): p.read_bytes() for p in snapshot.rglob('*') if p.is_file()}
    link = assets.os.link
    def failing_link(source, destination):
        if Path(source).parent.name == 'blobs':
            raise OSError(errno.EXDEV if failure == 'copy' else errno.EACCES, 'materialization fixture')
        if failure == 'receipt': raise PermissionError('receipt publication denied')
        return link(source, destination)
    if failure in ('link', 'copy', 'receipt'): monkeypatch.setattr(assets.os, 'link', failing_link)
    if failure == 'copy':
        def denied_copy(*args, **kwargs): raise PermissionError('streaming copy denied')
        monkeypatch.setattr(assets.shutil, 'copyfileobj', denied_copy)
    elif failure == 'inventory':
        inventory = assets._inventory
        def changed(root, **kwargs):
            if root.name.startswith('.bundle.'):
                (root / 'config.json').unlink(); (root / 'config.json').write_bytes(b'corrupt copy')
            return inventory(root, **kwargs)
        monkeypatch.setattr(assets, '_inventory', changed)
    elif failure in ('rename', 'rename-after-commit'):
        rename = assets.os.rename
        def denied_rename(*args):
            if failure == 'rename-after-commit':
                rename(*args)
                raise KeyboardInterrupt('interrupted immediately after directory commit')
            raise PermissionError('directory publication denied')
        monkeypatch.setattr(assets.os, 'rename', denied_rename)
    with pytest.raises((OSError, ValueError, KeyboardInterrupt)):
        assets.prepare(hf_repo=HF_REPO, revision=HF_REVISION, cache_dir=cache,
                       materialize_dir=target, output=output)
    assert not target.exists() and not output.exists()
    assert not list(tmp_path.glob('.bundle.*')) and not list(tmp_path.glob('.receipt.json.*'))
    assert {str(p): p.read_bytes() for p in snapshot.rglob('*') if p.is_file()} == before


@pytest.mark.parametrize('race', ['target', 'receipt'])
def test_materialization_preserves_targets_or_receipts_that_appear_during_work(hub_cache, monkeypatch, tmp_path, race):
    from distillschedule import assets
    _, cache, _, _ = hub_cache
    bind_hub(monkeypatch, hub_cache)
    target, output = tmp_path / 'bundle', tmp_path / 'receipt.json'
    if race == 'receipt':
        publish = assets._publish
        def racing(path, value):
            path.write_bytes(b'foreign receipt')
            return publish(path, value)
        monkeypatch.setattr(assets, '_publish', racing)
    else:
        inventory = assets._inventory
        def racing(root, **kwargs):
            value = inventory(root, **kwargs)
            if root.name.startswith('.bundle.'):
                target.mkdir(); (target / 'keep').write_bytes(b'foreign target')
            return value
        monkeypatch.setattr(assets, '_inventory', racing)
    with pytest.raises(FileExistsError):
        assets.prepare(hf_repo=HF_REPO, revision=HF_REVISION, cache_dir=cache,
                       materialize_dir=target, output=output)
    if race == 'receipt': assert output.read_bytes() == b'foreign receipt' and not target.exists()
    else: assert not output.exists() and (target / 'keep').read_bytes() == b'foreign target'


def test_materialized_cache_passes_actual_native_file_admission_and_unchanged_locker(hub_cache, monkeypatch, tmp_path):
    from distillschedule import assets
    from distillschedule.training.plugins import load_callable
    from huggingface_hub.hf_api import RepoFile
    _, cache, snapshot, tree = hub_cache
    for name in ('vae', 'text_encoder', 'tokenizer'):
        payload = b'{"fixture":"not model weights"}'
        relative = name + '/fixture.json'
        blob = snapshot.parent.parent / 'blobs' / git_blob(payload)
        blob.write_bytes(payload); (snapshot / name).mkdir()
        (snapshot / relative).symlink_to(os.path.relpath(blob, (snapshot / name)))
        tree.append(RepoFile(path=relative, size=len(payload), oid=git_blob(payload)))
    bind_hub(monkeypatch, hub_cache)
    target, output = tmp_path / 'native-bundle', tmp_path / 'receipt.json'
    value = assets.prepare(hf_repo=HF_REPO, revision=HF_REVISION, cache_dir=cache,
                           materialize_dir=target, output=output)
    assert assets.verify(output) == value
    locker, _ = load_callable(str(Path(__file__).resolve().parents[1] / 'recipes/flow_image/lock_local_assets.py') + ':lock_bundle')
    with pytest.raises(ValueError, match='budget'):
        locker(target, 'qwen-image', max_parameters=True, text_max_length=8)
    native = json.loads(locker(target, 'qwen-image', max_parameters=100, text_max_length=8).read_text())
    assert not (snapshot / 'assets.json').exists()  # Native locking never mutates the Hub snapshot.
    # This one explicit integration imports the existing CPU helper only. No
    # model is constructed and no safetensors/model payload loader is invoked.
    from distillschedule.training.flow_image import asset_files
    import torch
    with pytest.raises(ValueError, match='SHA256/path'):
        asset_files(snapshot, native['roles']['student'])
    for spec in {**native['roles'], **native['media']}.values():
        _, paths = asset_files(target, spec)
        assert all(not path.is_symlink() for path in paths)
        assert spec['max_parameters'] == 100
    assert not torch.cuda.is_initialized()
    with pytest.raises(ValueError, match='inventory'):
        assets.verify(output)  # Native assets.json is a new, intentional file.


def test_missing_optional_hub_dependency_is_actionable_without_publication(monkeypatch, tmp_path):
    import builtins
    from distillschedule import assets
    original = builtins.__import__
    def missing(name, *args, **kwargs):
        if name == 'huggingface_hub': raise ImportError('not installed')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', missing)
    with pytest.raises(ImportError, match=r'distillschedule\[assets\]'):
        assets.prepare(hf_repo=HF_REPO, revision=HF_REVISION, output=tmp_path / 'receipt.json')
    assert not (tmp_path / 'receipt.json').exists()


@pytest.mark.parametrize('options', [
    {}, {'hf_repo': HF_REPO}, {'hf_repo': 'https://user:secret@host/repo', 'revision': HF_REVISION},
    {'hf_repo': '../escape', 'revision': HF_REVISION},
])
def test_explicit_source_selection_is_validated_without_optional_dependencies(tmp_path, options):
    from distillschedule import assets
    with pytest.raises(ValueError):
        assets.prepare(output=tmp_path / 'receipt.json', **options)
    assert not (tmp_path / 'receipt.json').exists()


@pytest.mark.parametrize('options', [
    {'hf_repo': HF_REPO, 'revision': HF_REVISION}, {'revision': HF_REVISION}, {'cache_dir': '/cache'},
    {'materialize_dir': '/bundle'}, {'tree_timeout': 1},
])
def test_local_source_refuses_hub_options(local_source, tmp_path, options):
    from distillschedule import assets
    with pytest.raises(ValueError):
        assets.prepare(local_dir=local_source, output=tmp_path / 'receipt.json', **options)


@pytest.mark.parametrize('phase', ['tree', 'file'])
def test_hub_errors_do_not_echo_auth_material_or_publish(hub_cache, monkeypatch, tmp_path, capsys, phase):
    from distillschedule import assets
    hub, cache, _, _ = hub_cache
    bind_hub(monkeypatch, hub_cache)
    def denied(*args, **kwargs):
        raise RuntimeError('private hub failure: hf_DO_NOT_SERIALIZE_THIS_TOKEN')
    if phase == 'tree': monkeypatch.setattr(hub.HfApi, 'list_repo_tree', denied)
    else: monkeypatch.setattr(hub, 'hf_hub_download', denied)
    output = tmp_path / 'receipt.json'
    assert assets.main(['prepare', '--hf-repo', HF_REPO, '--revision', HF_REVISION,
                        '--cache-dir', str(cache), '--output', str(output)]) == 2
    captured = capsys.readouterr()
    assert captured.out == '' and 'ERROR:' in captured.err
    assert 'hf_DO_NOT_SERIALIZE' not in captured.err and not output.exists()


@pytest.mark.parametrize('stalled_page', ['first', 'next'])
def test_real_installed_hub_stalled_tree_has_complete_enumeration_deadline(hub_cache, tmp_path, stalled_page):
    """Real loopback HTTP stalls, with an outer watchdog to keep a broken fix safe."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import signal
    from threading import Event, Thread
    from time import monotonic
    _, cache, _, _ = hub_cache
    requested, release = Event(), Event()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_): pass
        def do_GET(self):
            if stalled_page == 'first' or 'cursor=' in self.path:
                requested.set()
                release.wait(10)
                return
            body = json.dumps([dict(type='file', path='config.json', size=1, oid='0' * 40)]).encode()
            self.send_response(200)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Link', f'<http://127.0.0.1:{self.server.server_port}{self.path}&cursor=next>; rel="next"')
            self.end_headers(); self.wfile.write(body)
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True); thread.start()
    env = {k: v for k, v in os.environ.items() if k not in ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN')}
    env.update(HF_ENDPOINT=f'http://127.0.0.1:{server.server_port}', HF_HUB_OFFLINE='0',
               HF_HUB_DISABLE_IMPLICIT_TOKEN='1', HF_HUB_DISABLE_TELEMETRY='1',
               NO_PROXY='127.0.0.1,localhost', HF_HOME=str(tmp_path / 'isolated-hf-home'))
    output, target = tmp_path / 'receipt.json', tmp_path / 'bundle'
    # Shorten only the default tree deadline, not any Hub/network setting. The
    # tracking hook confirms every worker is reaped before returning an error.
    script = '''import subprocess, sys
from distillschedule import assets
assets.DEFAULT_TREE_TIMEOUT = 1.0
children, real_popen = [], subprocess.Popen
def tracked(*args, **kwargs):
    process = real_popen(*args, **kwargs)
    children.append(process)
    return process
subprocess.Popen = tracked
code = assets.main(sys.argv[1:])
assert children and all(child.poll() is not None for child in children), 'metadata worker leaked'
raise SystemExit(code)
'''
    start = monotonic()
    process = subprocess.Popen([sys.executable, '-c', script, 'prepare', '--hf-repo', HF_REPO,
        '--revision', HF_REVISION, '--cache-dir', str(cache), '--materialize-dir', str(target),
        '--output', str(output)], env=env, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, start_new_session=True)
    try:
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            assert requested.is_set(), 'watchdog expired before a real metadata request'
            pytest.fail('stalled complete Hub file-tree enumeration ignored its 1s deadline')
    finally:
        release.set(); server.shutdown(); server.server_close(); thread.join()
    assert requested.is_set() and monotonic() - start < 5
    assert process.returncode == 2 and stdout == ''
    assert 'file-tree enumeration exceeded' in stderr and 'Traceback' not in stderr
    assert not output.exists() and not target.exists()
    assert not list(tmp_path.glob('.receipt.json.*')) and not list(tmp_path.glob('.bundle.*'))


@pytest.mark.parametrize('materialize,uncached', [(False, False), (True, False), (False, True)])
def test_real_installed_hub_paginates_metadata_and_reuses_existing_cache(hub_cache, tmp_path, materialize, uncached,
                                                                      record_property):
    """Loopback only: real Hub APIs reuse blobs and fetch one optional tiny file."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread
    from urllib.parse import parse_qs, urlsplit
    hub, cache, root, tree = hub_cache
    requests = []
    file_payload = b'{"tiny":true}\n'
    file_url = f'/{HF_REPO}/resolve/{HF_REVISION}/config.json'
    if uncached:
        blob = (root / 'config.json').resolve()
        (root / 'config.json').unlink(); blob.unlink()
    rows = [dict(type='file', path=row.path, size=row.size, oid=row.blob_id,
                 **({'lfs': dict(size=row.lfs.size, oid=row.lfs.sha256, pointerSize=128)} if row.lfs else {}))
            for row in tree[1:]]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_): pass
        def do_GET(self):
            request = urlsplit(self.path)
            requests.append(('GET', request.path, parse_qs(request.query)))
            if uncached and request.path == file_url:
                self.send_file_headers(); self.wfile.write(file_payload); return
            if request.path == f'/api/models/{HF_REPO}/tree/{HF_REVISION}':
                second = 'cursor' in parse_qs(request.query)
                payload = rows[1:] if second else [dict(type='directory', path='transformer', oid='0' * 40), rows[0]]
                link = None if second else f'<http://127.0.0.1:{self.server.server_port}{request.path}?cursor=next>; rel="next"'
            else:
                self.send_error(500, 'unexpected endpoint'); return
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            if link: self.send_header('Link', link)
            self.end_headers(); self.wfile.write(body)
        def do_HEAD(self):
            requests.append(('HEAD', self.path, {}))
            if uncached and self.path == file_url:
                self.send_file_headers()
            else:
                self.send_error(500, 'all other fixture files are already cached')
        def send_file_headers(self):
            self.send_response(200)
            self.send_header('Content-Length', str(len(file_payload)))
            self.send_header('ETag', '"' + git_blob(file_payload) + '"')
            self.send_header('X-Repo-Commit', HF_REVISION)
            self.end_headers()
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True); thread.start()
    env = {k: v for k, v in os.environ.items() if k not in ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN')}
    env.update(HF_ENDPOINT=f'http://127.0.0.1:{server.server_port}', HF_HUB_OFFLINE='0',
               HF_HUB_DISABLE_IMPLICIT_TOKEN='1', HF_HUB_DISABLE_TELEMETRY='1',
               HF_HUB_DISABLE_PROGRESS_BARS='1', HF_HUB_DISABLE_XET='1',
               NO_PROXY='127.0.0.1,localhost', HF_HOME=str(tmp_path / 'isolated-hf-home'))
    output = tmp_path / 'receipt.json'
    before = {str(p): p.stat().st_ino for p in root.rglob('*') if p.is_file()}
    try:
        result = cli('assets', 'prepare', '--hf-repo', HF_REPO, '--revision', HF_REVISION,
                     '--cache-dir', cache, '--tree-timeout', '10', '--output', output,
                     *(['--materialize-dir', tmp_path / 'bundle'] if materialize else []), env=env, allow_hub=True)
    finally:
        server.shutdown(); server.server_close(); thread.join()
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(result.stdout)
    assert len(value['files']) == 2 and value['root'] == str(tmp_path / 'bundle' if materialize else root)
    after = {str(p): p.stat().st_ino for p in root.rglob('*') if p.is_file()}
    assert {path: after[path] for path in before} == before
    assert set(after) == set(before) | ({str(root / 'config.json')} if uncached else set())
    assert (root / 'config.json').read_bytes() == file_payload
    assert [(request[0], request[1]) for request in requests] == [
        ('GET', f'/api/models/{HF_REPO}/tree/{HF_REVISION}'),
        ('GET', f'/api/models/{HF_REPO}/tree/{HF_REVISION}'),
    ] + ([('HEAD', file_url), ('GET', file_url)] if uncached else [])
    record_property('hub_version', hub.__version__)
    record_property('loopback_requests', json.dumps(requests))
    # Hub 0.36 uses Requests' True/False spelling; Hub 1 uses HTTPX's
    # lowercase spelling. Both must request the same complete recursive tree.
    assert {key: [item.casefold() for item in items] for key, items in requests[0][2].items()} == {
        'recursive': ['true'], 'expand': ['false']}
    checked = cli('assets', 'verify', output)
    assert checked.returncode == 0, checked.stderr
    assert json.loads(checked.stdout) == value


def test_prepared_root_reuses_unmodified_native_locker_not_a_capacity_grant(tmp_path):
    from distillschedule import assets
    from distillschedule.training.catalog import resolve
    from distillschedule.training.plugins import load_callable
    root = tmp_path / 'bundle'
    for name in ('transformer', 'vae', 'text_encoder', 'tokenizer'):
        (root / name).mkdir(parents=True)
        (root / name / 'fixture.json').write_text('{}')
    value = assets.prepare(local_dir=root, output=tmp_path / 'receipt.json')
    locker, _ = load_callable(str(Path(__file__).resolve().parents[1] / 'recipes/flow_image/lock_local_assets.py') + ':lock_bundle')
    with pytest.raises(ValueError, match='budget'):
        locker(value['root'], 'qwen-image', max_parameters=True, text_max_length=8)
    outside = tmp_path / 'outside'; outside.mkdir(); (outside / 'file').write_text('outside')
    with pytest.raises(ValueError, match='inside'):
        locker(value['root'], 'qwen-image', max_parameters=100, text_max_length=8, teacher=str(outside))
    native_path = locker(value['root'], 'qwen-image', max_parameters=100, text_max_length=8)
    native = json.loads(native_path.read_text())
    source = resolve('model', 'qwen-image')['source']
    assert native['source'] == {key: source[key] for key in ('model_id', 'revision')}
    assert native['evidence_kind'] == 'LOCAL_PRETRAINED_UNVERIFIED'
    assert all(spec['max_parameters'] == 100 for spec in {**native['roles'], **native['media']}.values())
    # The existing locker adds assets.json to this writable local bundle. That
    # mutation must invalidate the old source inventory, not be silently ignored.
    with pytest.raises(ValueError, match='inventory'):
        assets.verify(tmp_path / 'receipt.json')


@pytest.mark.skipif(os.environ.get('DMD_MODERN_FLOW_CPU') != '1', reason='actual native models use the modern-flow lane')
@pytest.mark.parametrize('family,recipe', [('flux2-klein-base', 'flow_image'), ('wan22-ti2v5b-t2v', 'wan22_dmd')])
def test_prepared_native_image_video_source_trains_without_modifying_assets(tmp_path, family, recipe):
    from distillschedule.training.plugins import load_callable
    create, _ = load_callable(str(Path(__file__).resolve().parents[1] /
        f'recipes/{recipe}/create_cpu_fixture.py') + ':build_fixture')
    root = tmp_path / 'native-source'
    fixture = (create(root, family, cycles=1) if recipe == 'flow_image' else create(root, cycles=1))
    receipt = tmp_path / 'source-receipt.json'
    result = cli('assets', 'prepare', '--local-dir', root, '--output', receipt)
    assert result.returncode == 0, result.stderr
    original = json.loads(result.stdout)
    trained = subprocess.run([sys.executable, '-m', 'distillschedule', 'train', str(fixture['config']),
        '--cycles', '1', '--output', str(tmp_path / 'training')], text=True, capture_output=True, timeout=120)
    assert trained.returncode == 0, trained.stdout + trained.stderr
    summary = json.loads(trained.stdout)
    assert summary['status'] == 'COMPLETE' and summary['completed_cycles'] == 1 and summary['finite']
    assert summary['final_versions']['student'] == 1 and summary['final_versions']['fake'] == 5
    assert Path(summary['checkpoint']).is_dir()
    checked = cli('assets', 'verify', receipt)
    assert checked.returncode == 0, checked.stderr
    assert json.loads(checked.stdout) == original
