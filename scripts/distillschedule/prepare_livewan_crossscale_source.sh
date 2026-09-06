#!/usr/bin/env bash
# Build an immutable, user-owned Wan2.1 source overlay for the pinned LiveWan
# SF-DMD workload.  This script never installs packages or edits either
# submodule; callers add the emitted source directory to PYTHONPATH.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LIVEWAN_COMMIT="1eabf43c777bc5b2816926ca998cd4e4c2dfa3d5"
WAN21_COMMIT="9737cba9c1c3c4d04b33fcad41c111989865d315"
OUTPUT="${LIVEWAN_WAN21_SOURCE:-/home/lyy/Helios/runtime_sources/livewan-wan21-${WAN21_COMMIT}}"
RECEIPT="${LIVEWAN_SOURCE_RECEIPT:-${OUTPUT}.source_receipt.json}"

usage() {
  cat <<'EOF'
usage: prepare_livewan_crossscale_source.sh [--output DIR] [--receipt JSON]

Creates a patched Wan2.1 source tree without modifying Python/site-packages.
The destination is immutable: a non-matching existing tree fails closed.
EOF
}

while (($#)); do
  case "$1" in
    --output) OUTPUT="$2"; shift 2 ;;
    --receipt) RECEIPT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

LIVEWAN="$ROOT/third_party/livewan"
WAN21="$ROOT/third_party/wan21"
test -d "$LIVEWAN" || { echo "missing LiveWan submodule: $LIVEWAN" >&2; exit 1; }
test -d "$WAN21" || { echo "missing Wan2.1 submodule: $WAN21" >&2; exit 1; }
test "$(git -C "$LIVEWAN" rev-parse HEAD)" = "$LIVEWAN_COMMIT" || {
  echo "LiveWan revision mismatch" >&2; exit 1;
}
test "$(git -C "$WAN21" rev-parse HEAD)" = "$WAN21_COMMIT" || {
  echo "Wan2.1 revision mismatch" >&2; exit 1;
}

PATCH_ATTN="$LIVEWAN/wan21_patches/modules/attention.py"
PATCH_CONFIG="$LIVEWAN/wan21_patches/configs/__init__.py"
test -f "$PATCH_ATTN" && test -f "$PATCH_CONFIG"

MARKER="$OUTPUT/.distillschedule-livewan-source.json"
if [[ -e "$OUTPUT" ]]; then
  test -f "$MARKER" || {
    echo "refusing to reuse unmarked source directory: $OUTPUT" >&2; exit 1;
  }
  python3 - "$MARKER" "$LIVEWAN_COMMIT" "$WAN21_COMMIT" <<'PY'
import json, sys
p, livewan, wan = sys.argv[1:]
x = json.load(open(p, encoding="utf-8"))
if x.get("livewan_commit") != livewan or x.get("wan21_commit") != wan:
    raise SystemExit("existing source marker has a different revision")
PY
else
  mkdir -p "$(dirname "$OUTPUT")"
  TMP="$(mktemp -d "$(dirname "$OUTPUT")/.livewan-source.XXXXXX")"
  trap 'rm -rf "$TMP"' EXIT
  git -C "$WAN21" archive --format=tar HEAD | tar -xf - -C "$TMP"
  install -m 0644 "$PATCH_ATTN" "$TMP/wan/modules/attention.py"
  install -m 0644 "$PATCH_CONFIG" "$TMP/wan/configs/__init__.py"
  python3 - "$TMP/.distillschedule-livewan-source.json" \
    "$LIVEWAN_COMMIT" "$WAN21_COMMIT" <<'PY'
import json, sys
path, livewan, wan = sys.argv[1:]
with open(path, "w", encoding="utf-8") as f:
    json.dump({
        "schema_version": 1,
        "livewan_commit": livewan,
        "wan21_commit": wan,
        "patch_policy": "LiveWan pinned attention/config overlay",
    }, f, indent=2, sort_keys=True)
    f.write("\n")
PY
  mv "$TMP" "$OUTPUT"
  trap - EXIT
fi

cmp -s "$PATCH_ATTN" "$OUTPUT/wan/modules/attention.py" || {
  echo "prepared attention.py does not match the pinned LiveWan overlay" >&2; exit 1;
}
cmp -s "$PATCH_CONFIG" "$OUTPUT/wan/configs/__init__.py" || {
  echo "prepared configs/__init__.py does not match the pinned LiveWan overlay" >&2; exit 1;
}

mkdir -p "$(dirname "$RECEIPT")"
python3 - "$ROOT" "$OUTPUT" "$RECEIPT" "$LIVEWAN_COMMIT" "$WAN21_COMMIT" <<'PY'
import hashlib, json, os, pathlib, sys, tempfile
root = pathlib.Path(sys.argv[1]).resolve()
output = pathlib.Path(sys.argv[2]).resolve()
receipt = pathlib.Path(sys.argv[3]).resolve()
livewan_commit, wan_commit = sys.argv[4:6]

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

files = {
    "wan_attention": output / "wan/modules/attention.py",
    "wan_configs": output / "wan/configs/__init__.py",
    "wan_model": output / "wan/modules/model.py",
    "livewan_train_dmd": root / "third_party/livewan/scripts/train_dmd.py",
    "livewan_blockcausal": root / "third_party/livewan/wanstreamer/blockcausal.py",
    "livewan_dmd": root / "third_party/livewan/wanstreamer/dmd.py",
    "livewan_stream": root / "third_party/livewan/wanstreamer/stream.py",
}
missing = [str(path) for path in files.values() if not path.is_file()]
if missing:
    raise SystemExit("source receipt missing files: " + ", ".join(missing))
rows = {
    name: {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
    for name, path in files.items()
}
identity = {
    "schema_version": 1,
    "status": "PASS",
    "livewan_commit": livewan_commit,
    "wan21_commit": wan_commit,
    "source_root": str(output),
    "pythonpath": [str(root / "third_party/livewan"), str(output)],
    "files": rows,
}
canonical = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
identity["source_receipt_sha256"] = hashlib.sha256(canonical).hexdigest()
receipt.parent.mkdir(parents=True, exist_ok=True)
fd, tmp = tempfile.mkstemp(prefix=receipt.name + ".", dir=receipt.parent)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(identity, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, receipt)
finally:
    if os.path.exists(tmp):
        os.unlink(tmp)
print(json.dumps({"status": "PASS", "source_root": str(output),
                  "receipt": str(receipt),
                  "source_receipt_sha256": identity["source_receipt_sha256"]},
                 sort_keys=True))
PY
