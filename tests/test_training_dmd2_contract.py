"""Public DMD2 integration contracts; numerical parity lives in CPU tests."""


def test_official_sd_and_dmd2_are_real_public_factories():
    from distillschedule.training.sd import build_sd15, build_sdxl
    from distillschedule.training.dmd2 import build_dmd2
    from distillschedule.data.sd_inputs import prepare_inputs
    from distillschedule.training.upstream import PinnedSource

    assert all(callable(value) for value in
               (build_sd15, build_sdxl, build_dmd2, prepare_inputs, PinnedSource))
