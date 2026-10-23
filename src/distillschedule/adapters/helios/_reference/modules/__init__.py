"""Ordinary package search path, not a replacement for any native helios import."""
from distillschedule.workloads.best_known.registry import repository_root

# Our explicit dense backend is found here first; the transformer is imported
# whole from its byte-verified original file. No source is copied or rewritten.
__path__.append(str(repository_root() / "third_party/helios_compat/source/helios/modules"))
