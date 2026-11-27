#!/bin/sh
# Build a release tarball: prebuilt free-only tts_host plus the
# full source tree (so the binary can be rebuilt anywhere).
#
#   tools/make_release.sh [output-dir]
#
# The binary is portable across x86-64 AVX2 machines by design:
# only the AVX2 kernel TU is compiled with -mavx2/-mfma (the
# product baseline); AVX-512 paths are runtime-dispatched, and
# everything else is generic. No model weights are included --
# the host needs a converted root directory at run time (see
# README.md).
set -eu
cd "$(dirname "$0")/.."
ROOT=$(pwd)
OUT=${1:-"$ROOT/dist"}
VER=$(git describe --tags --always 2>/dev/null || git rev-parse --short HEAD)
NAME="gptsovits-v2pp-cpu-$VER-linux-x86_64"
BUILD=$(mktemp -d)
trap 'rm -rf "$BUILD"' EXIT

cmake -S engine -B "$BUILD/b" -DPOTATO_FREE_ONLY=ON -DCMAKE_BUILD_TYPE=Release > /dev/null
cmake --build "$BUILD/b" -j 4 --target tts_host 2>&1 | tail -1

mkdir -p "$BUILD/$NAME/bin"
cp "$BUILD/b/tts_host" "$BUILD/$NAME/bin/"
# Source tree: everything tracked, minus nothing -- a release the
# user can rebuild with the README commands.
mkdir -p "$BUILD/$NAME/src"
git archive HEAD | tar -x -C "$BUILD/$NAME/src"
cat > "$BUILD/$NAME/RELEASE.txt" <<EOF
gptsovits-v2pp-cpu $VER (linux x86-64, AVX2 baseline)
Built from git HEAD $(git rev-parse HEAD) on $(date -u +%Y-%m-%dT%H:%M:%SZ).
bin/tts_host is the free-engine-only build (no ONNX Runtime).
Rebuild: cmake -S src/engine -B build -DPOTATO_FREE_ONLY=ON && cmake --build build -j 4
No model weights are included. See src/README.md for what a
runtime root directory must contain.
EOF
cd "$BUILD/$NAME"
find . -type f -exec sha256sum {} + | sort -k2 > SHA256SUMS
cd "$BUILD"
mkdir -p "$OUT"
tar -czf "$OUT/$NAME.tar.gz" "$NAME"
echo "$OUT/$NAME.tar.gz"
