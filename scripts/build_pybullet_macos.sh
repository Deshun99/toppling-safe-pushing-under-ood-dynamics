#!/usr/bin/env bash
# Build pybullet from source on macOS (Apple Silicon) with the zutil.h fix.
#
# Why this exists:
#   - PyPI ships NO macOS wheel for pybullet 3.2.6/3.2.7 (only Linux/Windows
#     wheels + sdist), and the sdist fails to build on modern Apple SDKs.
#   - examples/ThirdPartyLibs/zlib/zutil.h contains
#         #define fdopen(fd, mode) NULL
#     which corrupts the SDK's own fdopen declaration in <stdio.h>.
#   This script disables that macro on __APPLE__ and installs into the active
#   virtualenv.
#
# Usage (from the repo root, no activation needed):
#   bash scripts/build_pybullet_macos.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$REPO_ROOT/.venv/bin/python"
if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This build script is for macOS only." >&2
  exit 1
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "Create .venv first; see README.md." >&2
  exit 1
fi
PYBULLET_VERSION=$("$PYTHON" -c 'import pathlib, re, sys; print(re.search(r"^pybullet==([^;\s]+)", pathlib.Path(sys.argv[1]).read_text(), re.MULTILINE)[1])' "$REPO_ROOT/requirements.txt")
WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

echo "Downloading pybullet ${PYBULLET_VERSION} sdist..."
URL=$(curl -fsS "https://pypi.org/pypi/pybullet/${PYBULLET_VERSION}/json" \
  | "$PYTHON" -c "import json,sys; print([u['url'] for u in json.load(sys.stdin)['urls'] if u['filename'].endswith('.tar.gz')][0])")
curl -fsSL -o "$WORKDIR/pybullet.tar.gz" "$URL"
tar -xzf "$WORKDIR/pybullet.tar.gz" -C "$WORKDIR"

cd "$WORKDIR/pybullet-${PYBULLET_VERSION}"
sed -i '' \
  's/#if defined(MACOS) || defined(TARGET_OS_MAC)/#if (defined(MACOS) || defined(TARGET_OS_MAC)) \&\& !defined(__APPLE__)/' \
  examples/ThirdPartyLibs/zlib/zutil.h

echo "Building (takes several minutes on Apple Silicon)..."
"$PYTHON" -m pip install .
echo "pybullet ${PYBULLET_VERSION} installed from patched source."
