"""Prepare local or pinned Hugging Face model files and verify their hashes.

Local inventory and verification use the standard library. Explicit Hub
preparation reuses its download cache and can materialize regular model files
for the selected recipe's native asset loader.
"""
from __future__ import annotations

import argparse
import errno
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tempfile

from distillschedule._cli_streams import redirect_diagnostics
from distillschedule.core.contracts import require
from distillschedule.workloads.best_known.registry import SHA256_RE, file_sha256, fingerprint

SCHEMA = 'distillschedule.assets.source.v1'
SCOPE = 'SOURCE_BYTES_ONLY_NOT_NATIVE_MODEL_ADMISSION'
COMMIT = re.compile(r'[0-9a-f]{40}')
REPO_PART = re.compile(r'[A-Za-z0-9_][A-Za-z0-9_.-]{0,95}')
DEFAULT_TREE_TIMEOUT = 60.0


def _require(condition, message):
    require(condition, message, path='assets')


def _relative(name):
    _require(isinstance(name, str) and name and '\\' not in name and '\0' not in name
             and not PurePosixPath(name).is_absolute()
             and all(part not in ('', '.', '..') for part in name.split('/')),
             'canonical relative file path required')


def _repo_id(value):
    _require(isinstance(value, str) and 1 <= len(value.split('/')) <= 2
             and all(REPO_PART.fullmatch(part) and not part.endswith(('.', '-')) for part in value.split('/'))
             and '..' not in value and '--' not in value and not value.endswith('.git'),
             'Hugging Face source must be a repository ID, not a URL or credential')


def _absolute(value):
    _require(isinstance(value, str) and Path(value).is_absolute()
             and str(Path(value).resolve()) == value, 'canonical absolute source path required')
    return Path(value)


def _hub_root(source):
    cache = _absolute(source['cache_dir'])
    return cache / ('models--' + source['repo_id'].replace('/', '--')) / 'snapshots' / source['revision']


def _output(path, root, *, new=True):
    path = Path(path).expanduser()
    if new and (path.exists() or path.is_symlink()):
        raise FileExistsError(f'asset receipt already exists: {path}')
    _require(not path.is_symlink(), 'asset receipt cannot be a symlink')
    path = path.resolve()
    _require(not path.is_relative_to(root), 'receipt output must be outside the inventoried source root')
    return path


def _entries(root, blobs):
    _require(not root.is_symlink() and root.is_dir() and root.resolve() == root,
             'source root must be an existing directory, not a symlink')
    result, pending = {}, [root]
    while pending:
        for path in sorted(pending.pop().iterdir()):
            name = path.relative_to(root).as_posix()
            _relative(name)
            if path.is_symlink():
                _require(blobs is not None and not blobs.is_symlink()
                         and path.resolve().parent == blobs and not path.resolve().is_symlink()
                         and path.resolve().is_file(),
                         'symlink is not a file in the declared Hub repository blob cache: ' + name)
            elif path.is_dir():
                pending.append(path)
                continue
            _require(path.is_file(), 'source entries must be regular files: ' + name)
            result[name] = path
    _require(bool(result), 'source file inventory is empty')
    return result


def _state(path):
    def fields(stat):
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns
    return str(path.resolve(strict=True)), fields(path.lstat()), fields(path.stat())


def _git_blob_sha1(path, size):
    digest = hashlib.sha1(b'blob ' + str(size).encode() + b'\0')
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _inventory(root, *, blobs=None, expected=None):
    entries = _entries(root, blobs)
    if expected is not None:
        _require(set(entries) == set(expected), 'source file inventory differs from the complete expected file list')
    states = {name: _state(path) for name, path in entries.items()}
    files = {}
    for name, path in sorted(entries.items()):
        size = path.stat().st_size
        row = {'bytes': size, 'sha256': file_sha256(path)}
        if expected is not None and 'hub_digest' in expected[name]:
            pin = expected[name]
            _require(size == pin['bytes'], 'Hub file bytes differ: ' + name)
            origin = pin['hub_digest']
            actual = row['sha256'] if origin['algorithm'] == 'sha256' else _git_blob_sha1(path, size)
            _require(actual == origin['value'], 'Hub file hash differs: ' + name)
            row['hub_digest'] = dict(origin)
        _require(states[name] == _state(path), 'source changed while hashing: ' + name)
        files[name] = row
    _require(states == {name: _state(path) for name, path in _entries(root, blobs).items()},
             'source inventory changed while hashing')
    return files


def _hub_files(repo_id, revision):
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import RepoFile, RepoFolder
    # The paginated recursive tree is authoritative for completeness. The Hub's
    # repo_info.siblings can be truncated for large repositories.
    try:
        tree = list(HfApi().list_repo_tree(repo_id, revision=revision, repo_type='model', recursive=True))
    except Exception as exc:
        # HTTP exceptions can embed credential-bearing URLs/headers. Do not echo
        # them into diagnostics or serialize any authentication material.
        raise RuntimeError(f'Hugging Face file-list request failed ({type(exc).__name__}); check configured access/network') from exc
    files, seen = {}, set()
    for item in tree:
        _require(isinstance(item, (RepoFile, RepoFolder)), 'unknown Hub tree entry')
        _relative(item.path)
        _require(item.path not in seen, 'duplicate Hub file/directory path')
        seen.add(item.path)
        if isinstance(item, RepoFolder):
            continue
        _require(type(item.size) is int and item.size >= 0, 'Hub file needs an exact byte size')
        if item.lfs is not None:
            _require(type(item.lfs.size) is int and item.lfs.size == item.size
                     and isinstance(item.lfs.sha256, str) and SHA256_RE.fullmatch(item.lfs.sha256),
                     'Hub LFS file needs exact size/SHA256 metadata')
            origin = {'algorithm': 'sha256', 'value': item.lfs.sha256}
        else:
            _require(isinstance(item.blob_id, str) and COMMIT.fullmatch(item.blob_id),
                     'Hub file needs an exact Git blob SHA1')
            origin = {'algorithm': 'git_blob_sha1', 'value': item.blob_id}
        files[item.path] = {'bytes': item.size, 'hub_digest': origin}
    _require(bool(files), 'complete Hub file inventory is empty or unavailable')
    return files


def _bounded_hub_files(repo_id, revision, timeout):
    """Bound the complete tree, including every page; never time model payloads."""
    import subprocess
    # A fresh process is killable even when a supported Hub version explicitly
    # disables HTTP timeouts. communicate() drains large inventories and run()
    # kills/reaps the worker on expiry, without any temporary metadata files.
    script = (f'import json, sys\nsys.path[:] = {sys.path!r}\n'
              'from distillschedule.assets import _hub_files, redirect_diagnostics\n'
              'with redirect_diagnostics():\n    files = _hub_files(*sys.argv[1:])\n'
              'print(json.dumps(files))\n')
    try:
        result = subprocess.run([sys.executable, '-c', script, repo_id, revision],
                                capture_output=True, text=True, check=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f'Hugging Face complete file-tree enumeration exceeded {timeout:g}s; '
                           'check access/network or increase --tree-timeout') from None
    except subprocess.CalledProcessError:
        # Worker diagnostics can contain credential-bearing HTTP exceptions.
        raise RuntimeError('Hugging Face file-list request failed; check configured access/network') from None
    return json.loads(result.stdout)


def _publish(path, value):
    """Use the existing fsync + same-directory no-replace hard-link convention."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    temporary = Path(name)
    try:
        stream = os.fdopen(descriptor, 'w', encoding='utf-8')
        descriptor = None
        with stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            # Never turn a successful atomic commit into a reported failure,
            # or mask a pre-commit error with non-authoritative stage cleanup.
            pass


def _receipt(root, source, files):
    value = {'schema_version': SCHEMA, 'status': 'complete', 'scope': SCOPE,
             'source': source, 'root': str(root), 'files': files}
    return {**value, 'sha256': fingerprint(value)}


def _materialization_target(target, cache, output):
    target = Path(target).expanduser()
    if target.exists() or target.is_symlink():
        raise FileExistsError(f'asset materialization target already exists: {target}')
    target = target.resolve()
    _require(not target.is_relative_to(cache) and not cache.is_relative_to(target),
             'materialization directory must be outside and separate from the Hub cache')
    _output(output, target)
    _require(not target.is_relative_to(output), 'materialization directory cannot be inside the receipt path')
    lock = target.parent / ('.' + target.name + '.lock')
    if lock.exists() or lock.is_symlink():
        raise FileExistsError(f'asset materialization already locked: {lock}')
    return target


def _publish_materialized(snapshot, target, source, files, output):
    """Explicit regular-file bundle; the existing scoped lock/stage convention."""
    target.parent.mkdir(parents=True, exist_ok=True)
    lock = target.parent / ('.' + target.name + '.lock')
    # All publishers of this target cooperate through one exclusive sibling
    # lock, as in data preparation. Never adopt or remove another writer's path.
    handle = lock.open('x', encoding='utf-8')
    stage, installed, complete = None, None, False
    try:
        with handle:
            if target.exists() or target.is_symlink():
                raise FileExistsError(target)
            stage = Path(tempfile.mkdtemp(prefix='.' + target.name + '.', dir=target.parent))
            for name in sorted(files):
                original, prepared = (snapshot / name).resolve(strict=True), stage / name
                prepared.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(original, prepared)
                except OSError as exc:
                    if exc.errno != errno.EXDEV:
                        raise
                    with original.open('rb') as reader, prepared.open('xb') as writer:
                        shutil.copyfileobj(reader, writer, length=1024 * 1024)
                        writer.flush(); os.fsync(writer.fileno())
            _require(_inventory(stage, expected=files) == files, 'materialized file bytes/hash differ')
            value = _receipt(target, {**source, 'materialized': True}, files)
            if target.exists() or target.is_symlink():
                raise FileExistsError(target)
            owned = stage.stat()
            installed = (owned.st_dev, owned.st_ino)
            os.rename(stage, target)
            stage = None
            _publish(output, value)
            complete = True
            return value
    finally:
        if stage is not None and stage.exists():
            shutil.rmtree(stage)
        if installed is not None and not complete and not target.is_symlink() and target.exists():
            current = target.stat()
            if (current.st_dev, current.st_ino) == installed:
                shutil.rmtree(target)
        try:
            lock.unlink(missing_ok=True)
        except OSError:
            pass  # Do not reject an already committed receipt over lock cleanup.


def prepare(*, output, local_dir=None, hf_repo=None, revision=None, cache_dir=None, materialize_dir=None,
            tree_timeout=None):
    """Hash a local directory, or explicitly fetch one full pinned Hub snapshot.

    The returned root is the actual source directory. Native lockers/loaders
    that forbid cache symlinks need the explicit materialize_dir opt-in: verified
    cache bytes become regular hardlinks, or streamed copies across filesystems.
    Local directories are always inventoried in place without duplication.
    There is deliberately no offline-Hub initial preparation: without a full
    pinned remote file list a partial cache cannot prove completeness. Existing
    receipts can always be verified offline without importing the Hub package.
    """
    _require((local_dir is not None) != (hf_repo is not None), 'select exactly one explicit local directory or Hugging Face ID')
    if local_dir is not None:
        _require(revision is None and cache_dir is None and materialize_dir is None and tree_timeout is None,
                 'Hub revision/cache/materialization/timeout options require a Hugging Face source')
        local = Path(local_dir).expanduser()
        _require(not local.is_symlink(), 'local source root cannot be a symlink')
        root = local.resolve(strict=True)
        source = {'kind': 'local', 'path': str(root)}
        output = _output(output, root)
        files = _inventory(root)
    else:
        _repo_id(hf_repo)
        _require(isinstance(revision, str) and COMMIT.fullmatch(revision), 'Hub revision must be an immutable full 40-character lowercase commit SHA')
        tree_timeout = DEFAULT_TREE_TIMEOUT if tree_timeout is None else tree_timeout
        _require(type(tree_timeout) in (int, float) and 0 < tree_timeout <= sys.float_info.max
                 and math.isfinite(tree_timeout), 'Hub file-tree timeout must be finite positive seconds')
        try:
            from huggingface_hub import constants, hf_hub_download
        except ImportError as exc:
            raise ImportError('explicit Hub preparation requires the optional distillschedule[assets] dependency') from exc
        cache = Path(cache_dir if cache_dir is not None else constants.HF_HUB_CACHE).expanduser().resolve()
        source = {'kind': 'huggingface', 'repo_id': hf_repo, 'revision': revision, 'cache_dir': str(cache)}
        root = _hub_root(source)
        _require(root.resolve() == root, 'Hub cache snapshot path cannot traverse symlink directories')
        output = _output(output, root)
        target = _materialization_target(materialize_dir, cache, output) if materialize_dir is not None else None
        expected = _bounded_hub_files(hf_repo, revision, tree_timeout)
        # Fetch only the authoritative tree: snapshot_download would repeat an
        # unbounded repo_info/tree discovery on supported Hub versions. Per-file
        # metadata uses the SDK's finite HEAD timeout/retries, while payloads
        # retain its normal streaming timeout, not a total transfer deadline.
        def download(name):
            try:
                actual = hf_hub_download(hf_repo, filename=name, revision=revision, repo_type='model',
                                         cache_dir=str(cache), local_files_only=False, etag_timeout=10.0)
            except Exception as exc:
                raise RuntimeError(f'Hugging Face file preparation failed ({type(exc).__name__}); check configured access/network') from None
            _require(isinstance(actual, str) and Path(actual) == root / name and root.resolve() == root,
                     'Hub returned a foreign cache or unpinned snapshot file')
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=8) as workers:
            list(workers.map(download, sorted(expected)))
        files = _inventory(root, blobs=root.parent.parent / 'blobs', expected=expected)
        if target is not None:
            return _publish_materialized(root, target, source, files, output)
    value = _receipt(root, source, files)
    _publish(output, value)
    return value


def verify(receipt):
    """Verify every inventoried byte offline; never load model code or weights."""
    receipt = Path(receipt).expanduser()
    _require(not receipt.is_symlink(), 'asset receipt cannot be a symlink')
    def unique(pairs):
        value = {}
        for key, item in pairs:
            _require(key not in value, 'duplicate asset receipt key: ' + key)
            value[key] = item
        return value
    value = json.loads(receipt.read_bytes(), object_pairs_hook=unique)
    _require(type(value) is dict and set(value) == {'schema_version', 'status', 'scope', 'source', 'root', 'files', 'sha256'},
             'exact complete asset source receipt required')
    _require(value['schema_version'] == SCHEMA and value['status'] == 'complete' and value['scope'] == SCOPE,
             'unsupported/incomplete asset receipt or admission scope')
    _require(isinstance(value['sha256'], str) and SHA256_RE.fullmatch(value['sha256'])
             and fingerprint({k: v for k, v in value.items() if k != 'sha256'}) == value['sha256'],
             'asset receipt digest differs')
    root, source = _absolute(value['root']), value['source']
    _output(receipt, root, new=False)
    _require(type(source) is dict, 'explicit asset source required')
    hub = source.get('kind') == 'huggingface'
    materialized = hub and source.get('materialized') is True
    if hub:
        fields = {'kind', 'repo_id', 'revision', 'cache_dir'}
        _require(set(source) == fields | ({'materialized'} if materialized else set()), 'exact Hub source fields required')
        _repo_id(source['repo_id'])
        _require(isinstance(source['revision'], str) and COMMIT.fullmatch(source['revision']), 'immutable Hub commit revision required')
        snapshot = _hub_root(source)
        if materialized:
            cache = _absolute(source['cache_dir'])
            _require(not root.is_relative_to(cache) and not cache.is_relative_to(root), 'materialized root must be outside the Hub cache')
            _output(receipt, snapshot, new=False)
        else:
            _require(root == snapshot, 'Hub cache/revision source root differs')
    else:
        _require(source == {'kind': 'local', 'path': str(root)}, 'local source path/root differs')
    files = value['files']
    _require(type(files) is dict and bool(files), 'complete file inventory required')
    for name, pin in files.items():
        _relative(name)
        _require(type(pin) is dict and set(pin) == ({'bytes', 'sha256', 'hub_digest'} if hub else {'bytes', 'sha256'})
                 and type(pin['bytes']) is int and pin['bytes'] >= 0
                 and isinstance(pin['sha256'], str) and SHA256_RE.fullmatch(pin['sha256']), 'exact file bytes/SHA256 pin required')
        if hub:
            origin = pin['hub_digest']
            _require(type(origin) is dict and set(origin) == {'algorithm', 'value'}
                     and origin['algorithm'] in ('sha256', 'git_blob_sha1') and isinstance(origin['value'], str)
                     and (SHA256_RE if origin['algorithm'] == 'sha256' else COMMIT).fullmatch(origin['value']),
                     'exact Hub file digest required')
    actual = _inventory(root, blobs=root.parent.parent / 'blobs' if hub and not materialized else None, expected=files)
    _require(actual == files, 'source file bytes/hash differ from receipt')
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(prog='distillctl assets', description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare_parser = commands.add_parser('prepare', help='explicit local inventory or full pinned Hub snapshot; no model load')
    source = prepare_parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--local-dir', type=Path)
    source.add_argument('--hf-repo', help='explicit Hugging Face model repository ID')
    prepare_parser.add_argument('--revision', help='required for Hub: exact immutable 40-character commit SHA')
    prepare_parser.add_argument('--cache-dir', type=Path, help='existing Hub cache convention; never a copied local model bundle')
    prepare_parser.add_argument('--tree-timeout', type=float,
                                help='complete paginated Hub file-tree deadline in finite positive seconds (default: 60); not a payload deadline')
    prepare_parser.add_argument('--materialize-dir', type=Path,
                                help='explicit new regular-file bundle outside the Hub cache for native lockers; hardlink or cross-filesystem copy')
    prepare_parser.add_argument('--output', required=True, type=Path, help='new receipt outside the source root; never overwrite')
    commands.add_parser('verify', help='offline full-file/hash verification; not native model admission').add_argument('receipt', type=Path)
    args = parser.parse_args(argv)
    try:
        with redirect_diagnostics():
            result = verify(args.receipt) if args.command == 'verify' else prepare(
                output=args.output, local_dir=args.local_dir, hf_repo=args.hf_repo,
                revision=args.revision, cache_dir=args.cache_dir, materialize_dir=args.materialize_dir,
                tree_timeout=args.tree_timeout)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (ValueError, TypeError, KeyError, OSError, RuntimeError, ImportError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
