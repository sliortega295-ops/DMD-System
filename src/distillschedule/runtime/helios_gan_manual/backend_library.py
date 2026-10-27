"""Existing pinned FA3 live identity, not a backend reimplementation/owner.
Stdlib import/default plan only. GPU-capable imports require external permit.
"""
from __future__ import annotations
import ast,hashlib,inspect,json,os,pathlib,sys,types
from . import backend_abi as backend
FA3_ROOT=pathlib.Path('/share/project/shuchenweng/yongyan_liu/DMD-System_assets/lightx2v_minimax_dmd_20260817/hf_home/hub/models--kernels-community--flash-attn3/snapshots/acb0f42e6901d3e9effa8a83bd32bc2bf0cfdfc2/build/torch210-cxx11-cu128-x86_64-linux')
FA3_PINS={'flash_attn_interface.py':'cb46c3d89606305817ea20eb7b7cc9376b013eeabfd7637a328e7967e1d008d4','_ops.py':'7f392b365cc945a3b3a420e24802ef776a79316c068e6e99e4da682e8927a1f6','_flash_attn3_cuda_373a45c.abi3.so':'8aa9bb49e174872be4e2cb8c2608f2d66de29b513f704fb8d91771f9cd39df56'}
MAX_MAP_BYTES=2<<20;MAX_MAP_ROWS=16384

def require(c,why):
 if not c:raise ValueError(why)

def sha(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb')as f:
  for data in iter(lambda:f.read(8<<20),b''):h.update(data)
 return h.hexdigest()

def verify_files(root=FA3_ROOT):
 require(pathlib.Path(root)==FA3_ROOT,'PINNED_EXISTING_FA3_PATH')
 return verify_explicit_files(root)

def verify_explicit_files(root):
 """Versioned stream only: same exact bytes, explicit already-local package."""
 root=pathlib.Path(root)
 require(root.is_absolute() and root.is_dir(),'EXPLICIT_LOCAL_PINNED_FA3_DIRECTORY')
 for name,pin in FA3_PINS.items():require(sha(pathlib.Path(root)/name)==pin,'PINNED_EXISTING_FA3_FILE:'+name)
 return dict(FA3_source_root=str(root),FA3_files_sha256=dict(FA3_PINS))

def _code_signature(code):
 return(code.co_code,tuple(_code_signature(x)if isinstance(x,types.CodeType)else x for x in code.co_consts),code.co_names,code.co_varnames,code.co_freevars,code.co_cellvars,code.co_argcount,code.co_posonlyargcount,code.co_kwonlyargcount,code.co_flags)

def _extract_code(source,name):
 code=compile(pathlib.Path(source).read_bytes(),str(source),'exec',dont_inherit=True,optimize=sys.flags.optimize)
 for part in name.split('.'):
  matches=[x for x in code.co_consts if isinstance(x,types.CodeType)and x.co_name==part]
  require(len(matches)==1,'EXACT_FA3_COMPILED_SOURCE_FUNCTION:'+name);code=matches[0]
 return code

def expected_function_defaults(source,name):
 node=ast.parse(pathlib.Path(source).read_bytes())
 for part in name.split('.'):
  matches=[x for x in node.body if isinstance(x,(ast.FunctionDef,ast.ClassDef))and x.name==part]
  require(len(matches)==1,'EXACT_FA3_AST_DEFAULTS:'+name);node=matches[0]
 require(isinstance(node,ast.FunctionDef),'EXACT_FA3_AST_FUNCTION:'+name)
 positional=tuple(ast.literal_eval(x)for x in node.args.defaults)or None
 keywords={arg.arg:ast.literal_eval(value)for arg,value in zip(node.args.kwonlyargs,node.args.kw_defaults)if value is not None}or None
 return positional,keywords

def _literal_signature(value):
 if type(value)in(tuple,list):return(type(value).__name__,tuple(_literal_signature(x)for x in value))
 if type(value)is dict:return('dict',tuple((k,_literal_signature(v))for k,v in sorted(value.items())))
 require(value is None or type(value)in(bool,int,float,str),'EXACT_LITERAL_FUNCTION_DEFAULT')
 return(type(value).__name__,value)

def attest_function(fn,source,name):
 require(type(fn)is types.FunctionType,'ACTUAL_FA3_ORIGINAL_FUNCTION:'+name)
 require(pathlib.Path(inspect.getsourcefile(fn)).samefile(source)and pathlib.Path(fn.__code__.co_filename).samefile(source),'ACTUAL_FA3_CODE_SOURCE_PATH:'+name)
 require(_code_signature(fn.__code__)==_code_signature(_extract_code(source,name)),'ACTUAL_FA3_LOADED_SOURCE_CODE:'+name)
 require(_literal_signature((fn.__defaults__,fn.__kwdefaults__))==_literal_signature(expected_function_defaults(source,name)),'ACTUAL_FA3_ORIGINAL_DEFAULT_VALUES:'+name)


def validate_maps(text,expected,stat):
 require(type(text)is str and len(text.encode())<=MAX_MAP_BYTES and len(text.splitlines())<=MAX_MAP_ROWS,'BOUNDED_SELF_MAPS')
 matches=[];expected=str(expected);basename='_flash_attn3_cuda_373a45c.abi3.so'
 for line in text.splitlines():
  fields=line.split(None,5)
  if len(fields)!=6:
   require(expected not in line and basename not in line,'PINNED_FA3_MAP_MALFORMED');continue
  raw=fields[5];candidate=raw==expected or basename in raw or raw.removesuffix(' (deleted)').split('/')[-1]==FA3_PINS[basename]
  if not candidate:continue
  require(not raw.endswith(' (deleted)'),'PINNED_FA3_MAPPING_DELETED')
  require(raw==expected,'UNEXPECTED_LOADED_FA3_LIBRARY_PATH')
  try:major,minor=(int(x,16)for x in fields[3].split(':'));inode=int(fields[4],10)
  except ValueError:raise ValueError('PINNED_FA3_MAP_DEVICE_OR_INODE')from None
  require(major==os.major(stat.st_dev)and minor==os.minor(stat.st_dev)and inode==stat.st_ino,'PINNED_FA3_MAP_DEVICE_INODE')
  matches.append(raw)
 require(matches,'PINNED_FA3_REAL_LIBRARY_NOT_MAPPED');return len(matches)


def read_maps():
 with pathlib.Path('/proc/self/maps').open('r')as f:text=f.read(MAX_MAP_BYTES+1)
 require(len(text.encode())<=MAX_MAP_BYTES,'BOUNDED_MAP_READ');return text


def load_actual(torch):
 """Caller already validated root permit/GPU UUID and original pinned FA2.

 No hub download, guessed namespace, fake dispatcher, unwrapped kernel or
 CUDA/math change. This module returns the original public autograd callable.
 """
 return _load_verified(torch,FA3_ROOT,verify_files())

def load_explicit(torch,root):
 """Same original callable/ABI/map attestation, no personal-path requirement."""
 root=pathlib.Path(root).resolve()
 return _load_verified(torch,root,verify_explicit_files(root))

def _load_verified(torch,root,pins):
 from kernels.utils import get_local_kernel
 module=get_local_kernel(root,'flash_attn3');public=module.flash_attn_func
 interface=sys.modules.get(public.__module__);require(interface is not None,'ACTUAL_PUBLIC_INTERFACE_MODULE')
 source=root/'flash_attn_interface.py'
 require(pathlib.Path(interface.__file__).samefile(source)and interface.__dict__.get('flash_attn_func')is public,'ACTUAL_PUBLIC_FA3_MODULE_FUNCTION_IDENTITY')
 attest_function(public,source,'flash_attn_func')
 cls=interface.__dict__.get('FlashAttnFunc');require(cls is not None,'ORIGINAL_FA3_AUTOGRAD_CLASS')
 require(issubclass(cls,torch.autograd.Function)and cls.__module__ == public.__module__ and public.__globals__ is interface.__dict__,'ORIGINAL_FA3_AUTOGRAD_PUBLIC_BINDING')
 for name in('forward','backward'):
  require(getattr(cls,name).__globals__ is interface.__dict__,'ORIGINAL_FA3_AUTOGRAD_FUNCTION_GLOBALS')
  attest_function(getattr(cls,name),source,'FlashAttnFunc.'+name)
 rawfw=interface.__dict__.get('_flash_attn_forward');rawbw=interface.__dict__.get('_flash_attn_backward')
 from torch._library.custom_ops import CustomOpDef
 schema=backend.validate_actual_registered_FA3(rawfw,rawbw,CustomOpDef)
 opmod=inspect.getmodule(interface.add_op_namespace_prefix)
 require(opmod is not None and pathlib.Path(opmod.__file__).samefile(root/'_ops.py'),'ACTUAL_PINNED_FA3_OPS_NAMESPACE_SOURCE')
 require(sha(opmod.__file__)==FA3_PINS['_ops.py'],'ACTUAL_OPS_SOURCE_SHA')
 basename='_flash_attn3_cuda_373a45c.abi3.so';resolved=(root/basename).resolve();stat=resolved.stat()
 require(sha(resolved)==FA3_PINS[basename],'ACTUAL_RESOLVED_FA3_LIBRARY_BYTES')
 compiled=sorted({str(pathlib.Path(m.__file__).resolve())for name,m in list(sys.modules.items())if'_flash_attn3_cuda_373a45c'in name and hasattr(m,'__file__')})
 require(compiled==[str(resolved)],'EXACT_ACTUAL_LOADED_FA3_EXTENSION')
 count=validate_maps(read_maps(),resolved,stat);after=resolved.stat();require((after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)==(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns),'FA3_LIBRARY_CHANGED_DURING_MAP_READ')
 return public,schema,dict(**pins,public_module=public.__module__,public_source=str(source),public_source_sha256=sha(source),original_loaded_public_and_autograd_bytecode_attested=True,actual_raw_dispatchers=schema,ops_module_file=str(opmod.__file__),compiled_paths=compiled,actual_mapped_FA3_library=dict(path=str(resolved),mapping_count=count,device_major_minor=[os.major(stat.st_dev),os.minor(stat.st_dev)],inode=stat.st_ino,sha256=FA3_PINS[basename],maps_is_library_identity_not_GPU_ownership=True))
