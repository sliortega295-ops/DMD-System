"""Delegate the frozen builder; fsync only our output, never global os.sync."""
import os
from pathlib import Path
import runpy
import sys

from h3_a100.model import _configure_local_flash_attn3

_configure_local_flash_attn3()
source=Path(sys.argv[1]);sys.argv=sys.argv[1:]
output=Path(sys.argv[sys.argv.index('--output-dir')+1])
previous=os.sync
def scoped_sync():
    fd=os.open(output/'adaln_grid1000.bf16.bin',os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
os.sync=scoped_sync
try:runpy.run_path(str(source),run_name='__main__')
finally:os.sync=previous
