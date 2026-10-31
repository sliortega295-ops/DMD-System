"""C51 v2 only: identity of the existing local HF FA2/FA3 providers.

No distribution impersonation, download, replacement kernel or math policy.
Metadata admission is Torch-free. Explicit runtime admission invokes the SAME
native loaders used by C51, and attests their real public/autograd functions
and mapped extensions before original model construction.
"""
from copy import deepcopy
import importlib.machinery
import os
from pathlib import Path
import platform
import sys

from distillschedule.workloads.best_known.registry import file_sha256, SHA256_RE
from distillschedule.training.livewan_reference_assets import require

SCHEMA = "distillschedule.livewan_c51_hf_provider.v1"
ABI = dict(torch="2.10.0+cu128", torch_cuda="12.8", cxx11_abi=True, system="Linux", machine="x86_64")
EXTENSIONS = {"fa2": "_flash_attn2_cuda_a511a4c", "fa3": "_flash_attn3_cuda_373a45c"}
ROLE_BINDINGS = dict(generator="fa2_original_dense_readonly_splitKV", score="fa3_original_dense_split1_exact_SAC")


def provider_files():
    from . import livewan_attention_backend as fa2, livewan_score_fa3 as fa3
    from .livewan_attention_checkpoint import INTERFACE_SHA256
    return {"fa2": {"metadata.json": fa2.EXPECTED_METADATA_SHA256,
                    "__init__.py": "a5b677ae2101dc64c8ca6753bc2ef4db7c9d60359a967c44cbe2ead0319e6684",
                    "_ops.py": "71e83ed2be1343db1216c4d436b76c18d72b71ec98a75bb0dcde42ddb4bf43f6",
                    "bert_padding.py": "805d449ac749f87a50f3a3114385710f0f926ff45521275000b74d9e80721e5c",
                    "flash_attn_interface.py": INTERFACE_SHA256,
                    "flash_attn2/__init__.py": fa2.EXPECTED_PACKAGE_SHA256,
                    EXTENSIONS["fa2"] + ".abi3.so": fa2.EXPECTED_EXTENSION_SHA256},
            "fa3": {**fa3.HASHES,
                    "flash_attn3/__init__.py": fa2.EXPECTED_PACKAGE_SHA256,
                    "flash_attn_config.py": "bb1a3e7b231c0e2b7fffc603692e11b64e414cc2b34d22bdb4e2938dd5957bfc"}}


def provider_config(value):
    require(type(value) is dict and set(value) == {"schema_version", "abi"} and value["schema_version"] == SCHEMA,
            "explicit versioned C51 HF provider required")
    abi = value["abi"]
    require(type(abi) is dict and set(abi) == set(ABI) | {"torch_extension_sha256"}
            and all(type(abi[k]) is type(expected) and abi[k] == expected for k, expected in ABI.items())
            and type(abi["torch_extension_sha256"]) is str and SHA256_RE.fullmatch(abi["torch_extension_sha256"]),
            "C51 HF Torch2.10/cu128/cxx11 ABI identity changed")
    return deepcopy(value)


def source_origins(kernels):
    """Inspect preloaded provider namespaces before either native loader runs."""
    expected = {kind: {(Path(row["root"])/name).resolve(): sha for name, sha in row["files"].items()}
                for kind, row in kernels.items()}
    result = {}
    for name, module in tuple(sys.modules.items()):
        kind = ("fa2" if name == "flash_attn" or name.startswith("flash_attn.") or
                name == "flash_attn2" or name.startswith("flash_attn2.") or EXTENSIONS["fa2"] in name
                else "fa3" if name == "flash_attn3" or name.startswith("flash_attn3.") or EXTENSIONS["fa3"] in name else None)
        require(name != "flash_attn_interface", "C51 HF foreign generic FA3 interface preload")
        if kind is None:
            continue
        filename = getattr(module, "__file__", None)
        require(type(filename) is str, "C51 HF provider module origin missing: " + name)
        path = Path(filename).resolve()
        require(path in expected[kind] and path.is_file() and file_sha256(path) == expected[kind][path],
                "C51 HF provider import origin/hash changed: " + name)
        result[name] = dict(path=str(path), sha256=expected[kind][path])
    return result


def _extension_identity(kind, root):
    """Real extension loader plus bounded Linux mapping/inode verification."""
    from .helios_gan_manual.backend_library import read_maps
    stem = EXTENSIONS[kind]
    path = (root/(stem + ".abi3.so")).resolve()
    expected = provider_files()[kind][stem + ".abi3.so"]
    require(file_sha256(path) == expected, "C51 HF resolved extension changed")
    modules = [(name, module) for name, module in tuple(sys.modules.items()) if stem in name]
    require(bool(modules), "C51 HF actual native extension not imported")
    for name, module in modules:
        spec = getattr(module, "__spec__", None)
        require(Path(getattr(module, "__file__", "")).resolve() == path
                and spec is not None and isinstance(spec.loader, importlib.machinery.ExtensionFileLoader)
                and Path(spec.loader.get_filename(name)).resolve() == path,
                "C51 HF actual extension loader/origin changed")
    before = path.stat()
    matches = []
    for line in read_maps().splitlines():
        fields = line.split(None, 5)
        if len(fields) != 6:
            require(str(path) not in line and stem not in line, "C51 HF malformed extension mapping")
            continue
        raw = fields[5]
        if raw != str(path) and stem not in raw and raw.removesuffix(" (deleted)").split("/")[-1] != expected:
            continue
        require(raw == str(path), "C51 HF foreign/deleted native extension mapping")
        try:
            major, minor = (int(part, 16) for part in fields[3].split(":"))
            inode = int(fields[4])
        except ValueError:
            raise ValueError("LiveWan reference: C51 HF invalid extension device/inode") from None
        require((major, minor, inode) == (os.major(before.st_dev), os.minor(before.st_dev), before.st_ino),
                "C51 HF actual extension mapping inode changed")
        matches.append(raw)
    after = path.stat()
    require(matches and (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), "C51 HF extension unmapped/changed during admission")
    # Mapping counts/inodes are deliberately not persistent checkpoint identity.
    return dict(path=str(path), sha256=expected, native_loader_and_mapping_verified=True)


def _component_identity(kind, module, root, torch):
    from .helios_gan_manual.backend_library import attest_function
    pins = provider_files()[kind]
    package = Path(module.__file__).resolve()
    require(package in {(root/name).resolve() for name in pins if name.endswith("__init__.py")},
            "C51 HF actual provider package changed")
    source = root/"flash_attn_interface.py"
    functions = {}
    for name, cls_name in (("flash_attn_func", "FlashAttnFunc"), *(
            (("flash_attn_varlen_func", "FlashAttnVarlenFunc"),) if kind == "fa2" else ())):
        fn = getattr(module, name, None)
        attest_function(fn, source, name)
        interface = sys.modules.get(fn.__module__)
        require(interface is not None and Path(interface.__file__).resolve() == source.resolve()
                and fn.__globals__ is interface.__dict__ and interface.__dict__.get(name) is fn,
                "C51 HF actual public function/module binding changed")
        cls = interface.__dict__.get(cls_name)
        require(isinstance(cls, type) and issubclass(cls, torch.autograd.Function) and cls.__module__ == fn.__module__,
                "C51 HF original autograd class changed")
        for method in ("forward", "backward"):
            original = getattr(cls, method)
            require(original.__globals__ is interface.__dict__, "C51 HF autograd globals changed")
            attest_function(original, source, cls_name + "." + method)
        # The pinned wrapper names its parent from Python's process-salted
        # hash(path). Attest that actual module binding above; do not persist
        # the unstable namespace in a cross-rank/restart identity.
        functions[name] = dict(source=str(source.resolve()), source_sha256=pins["flash_attn_interface.py"],
                               original_public_and_autograd_bytecode_attested=True)
    return dict(package=str(package), package_sha256=file_sha256(package), functions=functions,
                dispatchers=_dispatcher_identity(kind, module, torch), extension=_extension_identity(kind, root))


def _dispatcher_identity(kind, module, torch):
    """Follow original public autograd to actual registered native dispatchers."""
    from torch._library.custom_ops import CustomOpDef
    interface = module.flash_attn_func.__globals__
    forward, backward = (interface.get(name) for name in ("_flash_attn_forward", "_flash_attn_backward"))
    require(isinstance(forward, CustomOpDef) and isinstance(backward, CustomOpDef),
            "C51 HF actual registered native CustomOpDefs required")
    stem = EXTENSIONS[kind]
    for name, value in (("_flash_attn_forward", forward), ("_flash_attn_backward", backward)):
        require(str(value._opoverload) == stem + "." + name + ".default"
                and value._opoverload is getattr(getattr(torch.ops, stem), name).default,
                "C51 HF actual native dispatcher binding changed")
    if kind == "fa3":
        from .helios_gan_manual.backend_abi import validate_actual_registered_FA3
        return validate_actual_registered_FA3(forward, backward, CustomOpDef)
    from .livewan_attention_checkpoint import discover_flash_forward
    _, result = discover_flash_forward(module)
    return dict(result, backward=str(backward._opoverload), backward_schema=str(backward._opoverload._schema))


def runtime_identity(provider, kernels, *, torch=None):
    """Explicit CPU-capable native imports, no CUDA context/device/model work.

    The caller owns all existing environment/precision/dependency gates. Only
    the exact FA2 module is aliased for upstream's ``import flash_attn``; this
    is an observed source binding, NOT fabricated flash-attn distribution data.
    """
    provider_config(provider)
    if torch is None:
        import torch
    abi = dict(torch=str(torch.__version__), torch_cuda=torch.version.cuda,
               cxx11_abi=torch._C._GLIBCXX_USE_CXX11_ABI, system=platform.system(), machine=platform.machine(),
               torch_extension_sha256=file_sha256(Path(torch._C.__file__)))
    require(abi == provider["abi"] and all(type(abi[k]) is type(provider["abi"][k]) for k in abi)
            and sys.implementation.name == "cpython", "C51 HF actual Torch/native ABI changed")
    pins = provider_files()
    require(type(kernels) is dict and set(kernels) == set(pins), "both C51 HF provider components required")
    for kind, expected in pins.items():
        row = kernels[kind]
        require(type(row) is dict and set(row) == {"root", "files"} and row["files"] == expected
                and type(row["root"]) is str and Path(row["root"]).is_absolute()
                and str(Path(row["root"]).resolve()) == row["root"], "C51 HF explicit pinned kernel binding changed")
        require(all((Path(row["root"])/name).is_file() and file_sha256(Path(row["root"])/name) == sha
                    for name, sha in expected.items()), "C51 HF provider files changed before import")
    source_origins(kernels)
    from .livewan_attention_backend import _load_flash_attn2
    from .livewan_score_fa3 import load_flash_attn3
    components = {}
    modules = {}
    for kind, load in (("fa2", _load_flash_attn2), ("fa3", load_flash_attn3)):
        module, root = load()
        require(str(root) == kernels[kind]["root"], "C51 HF actual native loader root changed")
        components[kind] = _component_identity(kind, module, root, torch)
        modules[kind] = module
    require("flash_attn" not in sys.modules or sys.modules["flash_attn"] is modules["fa2"],
            "C51 HF native flash_attn alias is a different provider")
    sys.modules["flash_attn"] = modules["fa2"]
    origins = source_origins(kernels)
    imported_files = [dict(path=path, sha256=sha) for path, sha in
                      sorted({(row["path"], row["sha256"]) for row in origins.values()})]
    return dict(schema_version=SCHEMA, provider=deepcopy(provider), kernels=deepcopy(kernels), abi=abi,
                role_bindings=deepcopy(ROLE_BINDINGS), native_flash_attn_alias="flash_attn2",
                torch_extension=dict(path=str(Path(torch._C.__file__).resolve()), sha256=abi["torch_extension_sha256"]),
                components=components, imported_files=imported_files,
                namespace_identity="EXACT_SOURCE_BINDING_NOT_PROCESS_SALTED_WRAPPER_NAME")
