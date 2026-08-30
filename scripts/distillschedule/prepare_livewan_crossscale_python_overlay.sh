#!/usr/bin/env bash
# Add only LiveWan's missing pure-Python dependency to a user-owned overlay.
# The validated Torch environment is never modified.
set -Eeuo pipefail

OUTPUT="${LIVEWAN_PYTHON_OVERLAY:-}"
PYTHON="${DISTILLSCHEDULE_PYTHON:-/tmp/distillschedule-helios-official-torch210/bin/python}"
INDEX_URL="${PIP_INDEX_URL:-http://10.1.1.16/repository/pypi-group/simple}"

while (($#)); do
  case "$1" in
    --output) OUTPUT="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    --index-url) INDEX_URL="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "${OUTPUT}" && -x "${PYTHON}" ]] || exit 2

RECEIPT="${OUTPUT}/.complete.json"
if [[ ! -f "${RECEIPT}" ]]; then
  [[ ! -e "${OUTPUT}" ]] || {
    echo "refusing to reuse incomplete overlay: ${OUTPUT}" >&2; exit 82;
  }
  mkdir -p "${OUTPUT}"
  "$(dirname "${PYTHON}")/pip" install --no-deps --target "${OUTPUT}" \
    --index-url "${INDEX_URL}" --trusted-host "$(python3 - "${INDEX_URL}" <<'PY'
from urllib.parse import urlparse
import sys
print(urlparse(sys.argv[1]).hostname or '')
PY
)" easydict==1.13
  PYTHONPATH="${OUTPUT}" "${PYTHON}" - "${OUTPUT}" <<'PY' >"${RECEIPT}.tmp"
import hashlib, json, pathlib, sys
import easydict

root=pathlib.Path(sys.argv[1]).resolve()
h=hashlib.sha256()
for path in sorted(p for p in root.rglob('*') if p.is_file() and p.name != '.complete.json.tmp'):
    h.update(str(path.relative_to(root)).encode())
    h.update(path.read_bytes())
print(json.dumps({
    'schema_version': 1,
    'status': 'PASS',
    'packages': {'easydict': '1.13'},
    'root': str(root),
    'tree_sha256': h.hexdigest(),
}, sort_keys=True))
PY
  mv "${RECEIPT}.tmp" "${RECEIPT}"
fi

PYTHONPATH="${OUTPUT}" "${PYTHON}" - "${RECEIPT}" <<'PY'
import json, pathlib, sys
import easydict
value=json.loads(pathlib.Path(sys.argv[1]).read_text())
assert value['status']=='PASS' and value['packages']=={'easydict':'1.13'}
print(json.dumps(value, sort_keys=True))
PY
