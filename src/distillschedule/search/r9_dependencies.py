"""Explicit deployment dependency provenance; no inherited experiment imports."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

from distillschedule.search.contracts import require
from distillschedule.workloads.best_known.registry import H3_SYSTEM_TOOLCHAIN_PATH, file_sha256


def metadata(cfg):
    from .r9_contract import bound
    rows = json.loads(bound(cfg['dependency_provenance']).read_text())
    require([r['node'] for r in rows] == [0, 1], 'BOTH_REVIEWED_DEPENDENCY_NODES')
    return rows


def expected_overlays(cfg, node):
    return [{k: o[k] for k in ('path', 'identity_file', 'sha256')} for o in metadata(cfg)[node]['overlays']]


def validate_plan_dependencies(cfg, plan):
    require([n.get('python_path_overlays') for n in plan['asset_config']['nodes']] ==
            [expected_overlays(cfg, n) for n in (0, 1)], 'ONLY_TWO_REVIEWED_DEPENDENCY_OVERLAYS')


def verify_node_dependencies(cfg, node):
    rows = metadata(cfg)[node]['overlays']
    require(len(rows) == 2 and [r['identity_file'] for r in rows] ==
            ['torch/__init__.py', 'transformers/__init__.py'], 'DEPENDENCY_PACKAGE_SCOPE')
    for row, package in zip(rows, ('torch', 'transformers')):
        path = Path(row['path'])
        # No sitecustomize, observer, trainer, or arbitrary module may hitch a
        # ride on these import prefixes. The qualified donor had one entry each.
        require(path.is_dir() and sorted(p.name for p in path.iterdir()) == [package],
                'DEPENDENCY_NAMESPACE_SHADOWING')
        entry = path / package
        observed = dict(name=package, symlink=entry.is_symlink(), target=str(entry.resolve()))
        require(row['files'] == [observed], 'DEPENDENCY_PACKAGE_TARGET_CHANGED')
        require(file_sha256(path / row['identity_file']) == row['sha256'], 'DEPENDENCY_IDENTITY_BYTES')
    return rows


def import_provenance(cfg, node, *, torch, transformers, diffusers, trainer, entrypoint):
    rows = verify_node_dependencies(cfg, node)
    source = Path(cfg['source_root']) / 'src/distillschedule/backends/training/minimax_h3'
    expected = {'torch': Path(rows[0]['files'][0]['target']),
                'transformers': Path(rows[1]['files'][0]['target']),
                'diffusers': Path(cfg['environment_by_node'][node]['H3_DIFFUSERS_SOURCE']) / 'src/diffusers',
                'trainer': source, 'entrypoint': source}
    result = {}
    for name, module in dict(torch=torch, transformers=transformers, diffusers=diffusers,
                             trainer=trainer, entrypoint=entrypoint).items():
        path = Path(module.__file__).resolve()
        require(path.is_relative_to(expected[name].resolve()), 'UNDECLARED_IMPORT_ORIGIN: ' + name)
        result[name] = dict(path=str(path), sha256=file_sha256(path))
    extension = Path(torch._C.__file__).resolve()
    require(extension.is_relative_to(expected['torch'].resolve()), 'TORCH_EXTENSION_ORIGIN')
    result['torch_extension'] = dict(path=str(extension), sha256=file_sha256(extension))
    return dict(packages=result, scope='SELECTED_IMPORT_FILES_NOT_FULL_PACKAGE_HASH')


def qualify_toolchain(environment):
    # Deliberately no os.environ merge, shell fallback, CUDA or Torch import.
    env = dict(environment)
    require(env.get('PATH') == H3_SYSTEM_TOOLCHAIN_PATH, 'EXPLICIT_SYSTEM_TOOLCHAIN_PATH_REQUIRED')
    require(env.get('CUDA_VISIBLE_DEVICES') == '', 'CPU_TOOLCHAIN_REQUIRES_HIDDEN_CUDA')
    root = Path(env['TMPDIR'])
    require(root.is_absolute() and root.is_dir(), 'EXPLICIT_TOOLCHAIN_TMPDIR')
    tools = {}
    for name in ('gcc', 'ld'):
        path = shutil.which(name, path=env['PATH'])
        require(path is not None, 'MISSING_SYSTEM_TOOLCHAIN: ' + name)
        version = subprocess.run([path, '--version'], env=env, capture_output=True,
                                 text=True, check=True, timeout=10)
        tools[name] = dict(path=path, resolved_path=str(Path(path).resolve()),
                           sha256=file_sha256(Path(path)), version=version.stdout.splitlines()[0])
    with tempfile.TemporaryDirectory(prefix='cpu-linker-', dir=root) as directory:
        source, binary = Path(directory) / 'probe.c', Path(directory) / 'probe'
        source.write_text('int main(void) { return 0; }\n')
        compiled = subprocess.run([tools['gcc']['path'], str(source), '-o', str(binary)],
                                 env=env, capture_output=True, text=True, check=True, timeout=30)
        executed = subprocess.run([str(binary)], env=env, capture_output=True,
                                 text=True, check=True, timeout=10)
        return dict(status='PASS_CPU_COMPILE_LINK_EXECUTE_NO_CUDA', path=env['PATH'], tools=tools,
            source_sha256=file_sha256(source), binary_sha256=file_sha256(binary),
            compile_returncode=compiled.returncode, execute_returncode=executed.returncode,
            compile_stdout=compiled.stdout, compile_stderr=compiled.stderr,
            scope='SYSTEM_CPU_TOOLCHAIN_ONLY_NOT_TRITON_OR_GPU_VALIDATION')


def validate_toolchain_receipt(row):
    require(row['status'] == 'PASS_CPU_COMPILE_LINK_EXECUTE_NO_CUDA'
            and row['path'] == H3_SYSTEM_TOOLCHAIN_PATH
            and row['compile_returncode'] == row['execute_returncode'] == 0
            and set(row['tools']) == {'gcc', 'ld'}, 'CPU_TOOLCHAIN_RECEIPT')
