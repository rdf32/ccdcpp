#!/bin/bash
# compare.sh <repo> <out_dir> <corpus_dir> <image> <label>=<ref> ... : the Linux dump comparison.
#
# Archives each git ref of <repo> into <out_dir>/src-<label>, then runs in_container.sh in one
# throwaway linux/amd64 container of <image> (an image ID or name@digest, already pulled: the
# run never pulls). The container builds every tree with conda-forge GCC and compares each
# label's dumps and fixtures with the last label's. Everything it writes is under
# <out_dir>/linux-out/, its console under <out_dir>/linux-run.log, the refs in <out_dir>/refs.txt.
#
# Docker comes from the environment (DOCKER_HOST, DOCKER_CONFIG). CERTS=<dir> mounts a
# directory of *.crt files for a TLS-intercepting proxy. See README.md.
set -euo pipefail
if [ $# -lt 6 ]; then
  echo "usage: compare.sh <repo> <out_dir> <corpus_dir> <image> <label>=<ref> <label>=<ref> ..." >&2
  exit 2
fi
REPO=$1; OUT=$2; CORPUS=$3; IMAGE=$4; shift 4
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT/linux-out"
OUT=$(cd "$OUT" && pwd); CORPUS=$(cd "$CORPUS" && pwd)

mounts=(); labels=()
: > "$OUT/refs.txt"
for pair in "$@"; do
  label=${pair%%=*}; ref=${pair#*=}
  labels+=("$label")
  rm -rf "$OUT/src-$label"; mkdir -p "$OUT/src-$label"
  git -C "$REPO" archive "$ref" | tar -x -C "$OUT/src-$label"
  echo "$label $ref $(git -C "$REPO" rev-parse "$ref^{commit}")" >> "$OUT/refs.txt"
  mounts+=(-v "$OUT/src-$label:/src/$label:ro")
done
[ -n "${CERTS:-}" ] && mounts+=(-v "$CERTS:/certs:ro")

docker image inspect --format '{{.Id}} {{.Os}}/{{.Architecture}} {{json .RepoDigests}}' "$IMAGE" \
  > "$OUT/image.txt"
docker run --rm --pull never --platform linux/amd64 \
  "${mounts[@]}" -v "$CORPUS:/corpus:ro" -v "$HERE:/tool:ro" -v "$OUT/linux-out:/out" \
  "$IMAGE" bash /tool/in_container.sh "${labels[@]}" > "$OUT/linux-run.log" 2>&1
tail -n "$(( ${#labels[@]} - 1 ))" "$OUT/linux-run.log"
