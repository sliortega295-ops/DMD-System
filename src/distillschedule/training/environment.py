"""Nonsecret runtime evidence, shared by the original-reference owners."""
import re


NONSECRET_HUB_CONTROLS = frozenset({
    "HF_HOME", "HF_HUB_CACHE", "HF_ASSETS_CACHE", "HF_XET_CACHE", "HF_ENDPOINT", "HF_INFERENCE_ENDPOINT",
    "HF_HUB_OFFLINE", "HF_DEBUG", "HF_HUB_VERBOSITY", "HF_HUB_DISABLE_TELEMETRY",
    "HF_HUB_DISABLE_PROGRESS_BARS", "HF_HUB_DISABLE_SYMLINKS_WARNING", "HF_HUB_DISABLE_EXPERIMENTAL_WARNING",
    "HF_HUB_DISABLE_IMPLICIT_TOKEN", "HF_HUB_ENABLE_HF_TRANSFER", "HF_HUB_DISABLE_XET",
    "HF_HUB_LOCAL_DIR_AUTO_SYMLINK_THRESHOLD", "HF_HUB_ETAG_TIMEOUT", "HF_HUB_DOWNLOAD_TIMEOUT",
    "HF_HUB_USER_AGENT_ORIGIN", "TRANSFORMERS_CACHE", "TRANSFORMERS_OFFLINE", "TRANSFORMERS_VERBOSITY",
    "TRANSFORMERS_NO_ADVISORY_WARNINGS",
})
_CREDENTIAL_KEY = re.compile(
    r"(?:^|_)(?:TOKENS?|SECRETS?|PASSWORD|PASSWD|CREDENTIALS?|AUTH(?:ORIZATION)?|API_?KEY|ACCESS_?KEY|PRIVATE_?KEY)(?:_|$)")


def credential_key(name):
    # DISABLE_IMPLICIT_TOKEN is a boolean control, not a token or token path.
    return name not in NONSECRET_HUB_CONTROLS and bool(_CREDENTIAL_KEY.search(name.upper()))


def nonsecret_environment_snapshot(environment, *, prefixes, names, exclude=()):
    """Keep owner-specific execution knobs visible; never read credential values."""
    return {name: environment[name] for name in sorted(environment)
            if name not in exclude and not credential_key(name)
            and (name.startswith(prefixes) or name in names or name in NONSECRET_HUB_CONTROLS)}
