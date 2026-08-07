"""Validated mechanism compatibility layer for the DistillSchedule refactor.

The stable public API lives under :mod:`distillschedule`.  Modules under this
package are intentionally imported explicitly (for example,
``from runtime.host_admission import ...``) while the strangler migration is in
progress.

Do not eagerly import mechanism modules here: several validated mechanisms have
optional PyTorch/Helios/DeepSpeed dependencies, while lightweight plan and host
admission tooling must remain importable in CPU-only development/CI environments.
"""
