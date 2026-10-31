"""Source-locked official optimization helpers, loaded only by admitted owners.

Importing this package does not load Torch, Triton, upstream model packages or
GPU kernels. Source discovery and verification remain standard-library only.
"""


def describe():
    from ..official_profiles import describe as profiles
    return profiles()


def resolve(profile_id, *, model=None):
    from ..official_profiles import resolve as profile
    return profile(profile_id, model=model)


def verify_sources(profile_id):
    from ..official_profiles import verify_sources as verify
    return verify(profile_id)


__all__ = ["describe", "resolve", "verify_sources"]
