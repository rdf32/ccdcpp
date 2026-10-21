#!/bin/bash
# bench.sh <python> <module_dir> <corpus.bin> [config ...]
#
# The synthetic bench: bench_synthetic.py once per config, each in its own process, which prints
# its time and peak memory. A config is "dtype threads reps"; the default set is f64 t1 (best of
# 2) and f64 t8 (best of 3). See README.md for the method.
set -eo pipefail
if [ $# -lt 3 ]; then
  echo "usage: bench.sh <python> <module_dir> <corpus.bin> [\"dtype threads reps\" ...]" >&2
  exit 2
fi
PY=$1; MODULE=$2; CORPUS=$3; shift 3
HERE=$(cd "$(dirname "$0")" && pwd)
CFGS=("$@")
[ ${#CFGS[@]} -eq 0 ] && CFGS=("f64 1 2" "f64 8 3")
for cfg in "${CFGS[@]}"; do
  read -r dt th reps <<< "$cfg"
  "$PY" -I "$HERE/bench_synthetic.py" "$MODULE" "$CORPUS" "$dt" "$th" "$reps" \
    || { echo "bench.sh: config \"$cfg\" failed" >&2; exit 1; }
done
