"""Read-only source/assets admission before importing DiffSynth or loading H3."""
from dataclasses import asdict, dataclass
import importlib
from pathlib import Path
import socket
import sys

from distillschedule.core.contracts import require
from distillschedule.workloads.best_known.registry import file_sha256, fingerprint, write_new_json
from .h3_ar_task import SOURCE_PINS, SOURCE_RECEIPT

# Imported small-source provenance from AP's accepted immutable manifest and
# its two-cycle epoch extension. Binary payloads remain remote external assets.
HELPER_PINS = {
    'native_accelerate_gpu_info_cache_r1': '2089453c01cd201c382f2256093553f0deb30cade0f048a47be99dd915454780',
    'native_ptxas_fd_isolation_r1': '1d98e23d29218dfb975e90667fada167f334d7bcd17a7cfc493958aa643cd19e',
    'native_selected_cycle_scope_r1': '6988483b0cf8b6cdf276f6a51a860ac5e1f972a5e1270a118923f6e6d6385af1',
    'native_selected_h2d_prefetch_r1': '53a6e6962005981d22ed35b4ef6c33ff819f1bdb22fd7c1c9ae4007de39bc68a',
    'selected_attention_pool_r1': '20bed542097baf40a97cd80e45c11af26fd25d5b18f76dcbb0183c9a09c2e728',
    'native_sdpa_cpu_replay_r1': '6d097fb99c4c0cba4ed2455ab49db3f087ca4c0e29205f8a9c98efbbc94122b2',
    'zero3_block_leaf_r3': 'fe3c36b005d35aaa706aad929b47ab55e91f95b13832b8badaebeef3d5e6ec2d',
    'zero3_block_leaf_r1': 'd8f4f5bef72060996aec644502f76eedc07406aff7426050788690baf548f8f7',
    'native_selected_block_fifo_r1': '8919853e7a7d6403c3e9ace0cb3cdf1f9c03f6bf780585ca175fae30b3b2ccb2',
    'native_selected_rollout_reuse_r1': 'c117141f8f5d46f2087a79d996bfd4e6313adc619180ff0aacd1dfcd6426b782',
    'zero3_unit_prefetch_audit_r1': 'eba4aaf991aa83dbe6b00ab5a623aed54a17f83eb887d3ab6931eee2665f4522',
    'native_cutlass_lora_r1': 'c8c0788b3bf6adaed561e2c33928ef777e7238f5a6880e2454ba92701bcfef4d',
    'native_lora_epilogue_r1': '346abe954feea85e670555d6673dcc9b3a49c508c8ff79dae20ba7ca980385e6',
    'native_rms_fusion_r3': '7c58a5247e97b4f86ce7af9d09c3b875fb636b01487268ff57097b6769ad9618',
    'native_pointwise_fusion_r1': 'e89a2ad1194e4488dddfb20c6ce59450a958741b7d757b7a3f54ccbb5a0905d9',
    'native_swiglu_fusion_r1': 'f8c2b36e7cee0a748afe543a3e65f5984f5891c5b75e0c8899f85b97d27bfef6',
    'native_mechanisms_r3': 'fd0531c6fe11e30f6b41eaa1896737f238c62f8b171c54cc3fa87315cafb7c7f',
    'zero3_checkpoint_tensor_identity_r2': '00662561103b2a5b13b01a838303ff00267c927921a2ee376e9ba5c6ce348b0c',
}
LIBRARY_PINS = dict(pinned='7fee2a6e146038d198baeb3cbc3a5ada409767d2f022d85ecc7e97803d73b5da',
    rms='59e2b251d174b49d2a718aea4e53354f5471ef8e85946bd63348891492f1bcc7',
    cutlass='30f3ac172a82c1aeb407dcc4e4ae1747853c44b0da30dbf2a01b4425fc2f21a4')


def checked(ref):
    require(isinstance(ref, dict) and set(ref) == {'path', 'sha256'}, 'exact file reference required', path='h3_ar.assets')
    path = Path(ref['path']).resolve()
    require(path.is_file() and file_sha256(path) == ref['sha256'], 'source/asset file changed: '+str(path), path='h3_ar.assets')
    return path


def validate_source_tree(root, receipt_ref, source_files):
    import json
    require(receipt_ref.get('sha256') == SOURCE_RECEIPT,
            'wrong exact native source receipt', path='h3_ar.assets')
    receipt = json.loads(checked(receipt_ref).read_text())
    require(receipt.get('status') == 'PASS_EXACT_SOURCE_SNAPSHOT_ONLY',
            'native source receipt is incomplete', path='h3_ar.assets')
    objects = receipt.get('objects', [])
    names = [r['relative_path'] for r in objects]
    require(objects and len(set(names)) == len(names), 'duplicate/empty native source pins', path='h3_ar.assets')
    expected = {r['relative_path']: r['sha256'] for r in objects if r['relative_path'].endswith('.py')}
    require(expected == source_files and all(source_files.get(p) == sha for p, sha in SOURCE_PINS.items()),
            'native math/full source inventory differs from original receipt', path='h3_ar.assets')
    for name in expected:
        require((root/name).resolve().is_relative_to(root), 'native source path escapes root', path='h3_ar.assets')
    observed = {str(p.relative_to(root)): file_sha256(p) for p in root.rglob('*.py')}
    require(observed == expected, 'native source tree changed or incomplete inventory', path='h3_ar.assets')


def checkpoint_inventory(root):
    return [root/'config.json', *sorted(root.glob('*.safetensors')), *sorted(root.glob('*.index.json'))]


def payload_stat(path):
    stat = path.stat()
    return dict(device=stat.st_dev, inode=stat.st_ino, size=stat.st_size,
                mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns)


def validate_checkpoint_audit(root, reference, expected_files):
    """Small receipt + complete file stat inventory, NOT per-rank weight scans."""
    import json
    audit = json.loads(checked(reference).read_text())
    require(audit.get('schema_version') == 'distillschedule.h3_ar_checkpoint_admission.v1'
            and audit.get('status') == 'PASS_ONCE_HASHED_CHECKPOINT_ADMISSION'
            and audit.get('checkpoint_root') == str(root), 'invalid checkpoint admission receipt', path='h3_ar.assets')
    files = audit.get('files', {})
    require({k: v['sha256'] for k, v in files.items()} == expected_files
            and set(files) == {p.name for p in checkpoint_inventory(root)},
            'checkpoint inventory/manifest changed', path='h3_ar.assets')
    observed = {}
    same_host = audit.get('origin_host') == socket.gethostname()
    for name, item in files.items():
        path = root/name
        now = observed[name] = payload_stat(path)
        keys = tuple(now) if same_host else ('size', 'mtime_ns', 'ctime_ns')
        require(all(now[key] == item['stat'][key] for key in keys),
                'checkpoint payload stat changed; owner must re-audit '+name, path='h3_ar.assets')
        if not name.endswith('.safetensors'):
            require(file_sha256(path) == item['sha256'], 'small checkpoint config/index changed', path='h3_ar.assets')
    return {**audit, 'worker_witness': dict(local_host=socket.gethostname(), origin_host=audit.get('origin_host'),
        audit=reference, stat_scope='same_host_dev_inode_size_mtime_ctime' if same_host else 'cross_host_portable_size_mtime_ctime',
        observed_stat=observed, full_payload_rehashed=False,
        payload_identity='same SHA256-bound owner admission; stat recheck is not a fresh cryptographic payload read')}


def admit_checkpoint_once(root, output):
    """Owner CPU/I/O preflight. Must finish BEFORE reserving training GPUs."""
    root = Path(root).resolve(); output = Path(output).resolve()
    if output.exists():
        import json
        audit = json.loads(output.read_text())
        expected = {k: v['sha256'] for k, v in audit['files'].items()}
        validate_checkpoint_audit(root, dict(path=str(output), sha256=file_sha256(output)), expected)
        return expected
    files = {}
    for path in checkpoint_inventory(root):
        before = payload_stat(path); sha = file_sha256(path)
        require(payload_stat(path) == before, 'checkpoint changed while hashing', path='h3_ar.assets')
        files[path.name] = dict(sha256=sha, stat=before)
    require('config.json' in files and any(p.endswith('.safetensors') for p in files),
            'real H3 checkpoint required', path='h3_ar.assets')
    write_new_json(output, dict(schema_version='distillschedule.h3_ar_checkpoint_admission.v1',
        status='PASS_ONCE_HASHED_CHECKPOINT_ADMISSION', checkpoint_root=str(root), files=files, origin_host=socket.gethostname(),
        scope='FULL_PAYLOAD_SHA256_ONCE_BEFORE_GPU; WORKER_STAT_RECHECK_NOT_REHASHED'))
    return {name: row['sha256'] for name, row in files.items()}


@dataclass(frozen=True)
class H3ARAssets:
    source_root: str
    source_files: dict
    source_receipt: dict
    helper_files: dict
    checkpoint_root: str
    checkpoint_files: dict
    checkpoint_audit: dict
    cache_manifest: dict
    libraries: dict
    startup_gpu_info: dict
    schema_version: str = 'distillschedule.original_h3_ar_assets.v1'

    @classmethod
    def from_dict(cls, value):
        require(isinstance(value, dict) and set(value) == set(cls.__dataclass_fields__),
                'exact H3 AR asset fields required', path='h3_ar.assets')
        result = cls(**value)
        require(result.schema_version == cls.__dataclass_fields__['schema_version'].default,
                'unsupported H3 AR assets version', path='h3_ar.assets')
        return result

    def to_dict(self):
        return asdict(self)

    @property
    def digest(self):
        return fingerprint(self.to_dict())

    def validate(self):
        root = Path(self.source_root).resolve()
        validate_source_tree(root, self.source_receipt, self.source_files)
        require({name: ref.get('sha256') for name, ref in self.helper_files.items()} == HELPER_PINS,
                'accepted native helper inventory differs', path='h3_ar.assets')
        for name, ref in self.helper_files.items():
            require(checked(ref).stem == name, 'helper module name/path mismatch', path='h3_ar.assets')
        checkpoint = Path(self.checkpoint_root).resolve()
        require('config.json' in self.checkpoint_files and any(p.endswith('.safetensors') for p in self.checkpoint_files),
                'real H3 checkpoint identity required', path='h3_ar.assets')
        checkpoint_validation = validate_checkpoint_audit(checkpoint, self.checkpoint_audit, self.checkpoint_files)
        checked(self.cache_manifest)
        require({name: ref.get('sha256') for name, ref in self.libraries.items()} == LIBRARY_PINS,
                'explicit accepted native library identities required', path='h3_ar.assets')
        for ref in self.libraries.values():
            checked(ref)
        require(set(self.startup_gpu_info) == {'0', '1'}, 'fresh node-qualified native startup GPU metadata required', path='h3_ar.assets')
        for ref in self.startup_gpu_info.values():
            checked(ref)
        return dict(checkpoint_admission=checkpoint_validation['worker_witness'], small_source_full_sha256=True)

    def install_import_paths(self):
        # No arbitrary worker command or launch callback is an asset. Only the
        # byte-bound ORIGINAL source/helper modules are admitted.
        root = Path(self.source_root).resolve()
        paths = [root/'examples/minimax_h3/model_training', root]
        paths += sorted({checked(ref).parent for ref in self.helper_files.values()})
        sys.path[:0] = [str(p) for p in paths]

    def helper(self, name):
        require(name in self.helper_files, 'unregistered native helper '+name, path='h3_ar.assets')
        expected = checked(self.helper_files[name])
        module = importlib.import_module(name)
        require(Path(module.__file__).resolve() == expected,
                'shadowed native helper '+name, path='h3_ar.assets')
        return module

    def source_module(self, name):
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        root = Path(self.source_root).resolve()
        require(path.is_relative_to(root) and self.source_files.get(str(path.relative_to(root))) == file_sha256(path),
                'shadowed/unbound native module '+name, path='h3_ar.assets')
        return module

    def audit_imports(self):
        root = Path(self.source_root).resolve()
        helpers = {checked(ref) for ref in self.helper_files.values()}
        helper_roots = {p.parent for p in helpers}
        for name, module in tuple(sys.modules.items()):
            value = getattr(module, '__file__', None)
            if not isinstance(value, (str, Path)) or not value:
                continue
            path = Path(value).resolve()
            if path.suffix != '.py':
                continue
            if path.is_relative_to(root):
                require(self.source_files.get(str(path.relative_to(root))) == file_sha256(path),
                        'unbound transitive native source '+name, path='h3_ar.assets')
            elif path.parent in helper_roots:
                require(path in helpers, 'unbound transitive helper '+name, path='h3_ar.assets')


def prepare_from_native_manifest(path, *, checkpoint_audit_path):
    """Read existing accepted small-source manifest; do not launch its worker.

    Model/cache payloads remain at their original remote paths. Hashing them is
    CPU/I/O admission, never calibration or a performance observation.
    """
    import json
    manifest = json.loads(Path(path).read_text())
    require(manifest.get('source_receipt_sha256') == SOURCE_RECEIPT and manifest.get('cycles') == 2,
            'accepted original two-cycle manifest required', path='h3_ar.assets')
    root = Path(manifest['source_root']).resolve()
    def ref(p):
        p = Path(p).resolve()
        return dict(path=str(p), sha256=file_sha256(p))
    helpers = {}
    for name, sha in HELPER_PINS.items():
        matches = [Path(p) for p, digest in manifest['file_sha256'].items()
                   if Path(p).stem == name and digest == sha]
        require(matches, 'accepted manifest missing native helper '+name, path='h3_ar.assets')
        helpers[name] = dict(path=str(matches[0].resolve()), sha256=sha)
        checked(helpers[name])
    flags = manifest['worker_flags']
    def flag(name):
        require(flags.count(name) == 1 and flags.index(name)+1 < len(flags),
                'native manifest missing/duplicate '+name, path='h3_ar.assets')
        return flags[flags.index(name)+1]
    libraries = {name: ref(flag(argument)) for name, argument in
        (('pinned', '--pinned-audit-library'), ('rms', '--rms-library'), ('cutlass', '--cutlass-lora-library'))}
    require({name: r['sha256'] for name, r in libraries.items()} == LIBRARY_PINS,
            'accepted library bytes differ', path='h3_ar.assets')
    source_files = {str(p.relative_to(root)): file_sha256(p) for p in root.rglob('*.py')}
    source_receipt = ref(root/'source_receipt.json')
    validate_source_tree(root, source_receipt, source_files)
    prefix = flag('--startup-accelerate-gpu-info-prefix')
    checkpoint = Path(manifest['checkpoint']).resolve()
    checkpoint_files = admit_checkpoint_once(checkpoint, checkpoint_audit_path)
    result = H3ARAssets(source_root=str(root),
        source_files=source_files, source_receipt=source_receipt, helper_files=helpers,
        checkpoint_root=str(checkpoint), checkpoint_files=checkpoint_files,
        checkpoint_audit=ref(checkpoint_audit_path),
        cache_manifest=ref(manifest['cache_manifest']), libraries=libraries,
        startup_gpu_info={str(node): ref(prefix+str(node)+'_r2.json') for node in (0, 1)})
    return result
