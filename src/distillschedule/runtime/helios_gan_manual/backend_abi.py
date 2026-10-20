"""Pinned real FA3 ABI and scoped dispatcher ownership, not a kernel implementation."""
from __future__ import annotations
import ast, contextlib, hashlib
from pathlib import Path

FA2_FW='_flash_attn2_cuda_a511a4c._flash_attn_forward.default'
FA3_FW='_flash_attn3_cuda_373a45c._flash_attn_forward.default'
FA3_BW='_flash_attn3_cuda_373a45c._flash_attn_backward.default'
INTERFACE_SHA='cb46c3d89606305817ea20eb7b7cc9376b013eeabfd7637a328e7967e1d008d4'
SOURCE=Path(__file__).with_name('fa3_pinned_interface.raw')

def require(condition,message):
    if not condition:raise ValueError(message)


def expected_forward_arguments():
    require(hashlib.sha256(SOURCE.read_bytes()).hexdigest()==INTERFACE_SHA,'EXACT_RETAINED_FA3_SOURCE')
    tree=ast.parse(SOURCE.read_text());f=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='_flash_attn_forward')
    return tuple(x.arg for x in f.args.args)


def validate_actual_registered_FA3(raw_forward,raw_backward,custom_op_type):
    """Caller imports pinned real module under root-owned lock/permit first.

    Validate actual live CustomOpDef and registered OpOverload, never a fake,
    unwrapped backend function, namespace string alone, or serialized claim.
    """
    require(isinstance(raw_forward,custom_op_type)and isinstance(raw_backward,custom_op_type),'ACTUAL_REGISTERED_CUSTOM_OPDEFS')
    require(str(raw_forward._opoverload)==FA3_FW and str(raw_backward._opoverload)==FA3_BW,'EXACT_EXISTING_FA3_DISPATCHERS')
    fs=raw_forward._opoverload._schema;bs=raw_backward._opoverload._schema
    require(tuple(a.name for a in fs.arguments)==expected_forward_arguments(),'ACTUAL_FA3_FORWARD_SCHEMA_SOURCE_ARGUMENTS')
    require(len(fs.returns)==4,'FA3_WHOLE_FOUR_TENSOR_RESULT_SAC')
    require(all(a.alias_info is None for a in fs.arguments),'FA3_FORWARD_PUBLIC_INPUTS_NONMUTATING')
    writes={a.name for a in bs.arguments if a.alias_info is not None and a.alias_info.is_write}
    require(writes=={'dq','dk','dv'},'ORIGINAL_FA3_BACKWARD_WRITES_ONLY_PRIVATE_DQDKDV')
    return dict(forward=FA3_FW,backward=FA3_BW,forward_schema=str(fs),backward_schema=str(bs),
        forward_return_pytree='4_TENSORS_MUST_SAVE_WHOLE',source_arguments=list(expected_forward_arguments()))


@contextlib.contextmanager
def actual_FA3_backend_scope(dispatcher,fsdp_static,fa3_public,*,actual_schema_receipt):
    """Only diagnostic process globals; all underlying source math stays pinned.

    Not a production installer. Caller owns original dispatch module and has
    verified actual source, public module file, pinned .so/maps plus schema.
    Every original/replay forward calls the SAME public FA3 function and the
    pinned original SAC policy gets its exact raw forward identity. Restore
    under failure. No tensor cast/cache/copy/barrier or changed default args.
    """
    require(actual_schema_receipt['forward']==FA3_FW and actual_schema_receipt['backward']==FA3_BW,'ACTUAL_SCHEMA_REQUIRED')
    require(fsdp_static.SELECTIVE_FLASH_ATTN_OP==FA2_FW,'ORIGINAL_FA2_SAC_POLICY_REQUIRED')
    old_fn=dispatcher.flash_attn_func;old_op=fsdp_static.SELECTIVE_FLASH_ATTN_OP
    require(old_fn is not fa3_public,'CANDIDATE_CANNOT_BE_ALREADY_INSTALLED')
    dispatcher.flash_attn_func=fa3_public;fsdp_static.SELECTIVE_FLASH_ATTN_OP=FA3_FW
    try:yield dict(backend='PINNED_FA3_PUBLIC',SAC_raw_forward_target=FA3_FW,source_files_written=False)
    finally:
        owned=dispatcher.flash_attn_func is fa3_public and fsdp_static.SELECTIVE_FLASH_ATTN_OP==FA3_FW
        dispatcher.flash_attn_func=old_fn;fsdp_static.SELECTIVE_FLASH_ATTN_OP=old_op
        require(owned,'EXCLUSIVE_DIAGNOSTIC_GLOBAL_OWNERSHIP')

