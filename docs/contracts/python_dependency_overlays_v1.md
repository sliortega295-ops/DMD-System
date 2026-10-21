# Explicit node-local Python dependency overlays v1

`assets.nodes[i].python_path_overlays` is an optional ordered list of objects
with exactly `path`, `identity_file`, and `sha256`. The directory path must be
absolute, single-entry (no path separator), and traversal-free. Its identity
file must be a nonempty relative path without `..`; SHA256 is mandatory.
Duplicate normalized directory paths and unknown fields reject. There are no
site paths, inferred environment variables, package downloads, or dependency
installation in this interface.

Only H3 currently admits nonempty overlays: its native launcher preserves the
resolved `PYTHONPATH`. Other native launchers reset that variable and therefore
reject overlays rather than silently ignore them. Empty/absent overlays leave
all five canonical profiles, default paths, and asset fingerprints unchanged.

For opted-in plans, the explicit per-node ordered overlay records enter the
asset fingerprint used by calibration, costs, correctness, search, saved plans,
and replay. Changing their path, order, or identity hash invalidates old costs.
Serialized `PYTHONPATH` is re-derived and checked before execution. Local
preflight requires an explicit node index and verifies only that node's files;
another node's `/tmp` need not be mounted. The metadata runtime probe verifies
them before importing Torch as well.

An identity-file hash is **not** a full package/environment hash. Dependency
directories may intentionally contain package symlinks; the identity's actual
resolved path is reported. An operator must independently qualify package
versions/import origins, prevent model/experiment code shadowing, and preserve
the selected package bytes. The sealed deployment packet can impose a narrower
allowlist than this portable library schema. No observer substitution or
workload-math override is authorized by this interface.

CPU tests cover schema/default compatibility, ordered lowering, serialized
tampering, local identity-byte changes, old-cost rejection, and local-node
verification. Real dependency import/GPU execution remains a separate gate.
