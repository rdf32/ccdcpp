#!/bin/bash
# bench.sh <python> <module_dir> <corpus.bin> [config ...]
#
# The synthetic bench: bench_synthetic.py once per config, each under /usr/bin/time for the
# process's peak RSS. A config is "dtype entry threads reps"; the default set is the one the
# state-machine refactor reports: cols f64 t1 (best of 2), cols f64 t8 and cube f64 t8 (best
# of 3 each). See README.md for the method.
set -e
if [ $# -lt 3 ]; then
  echo "usage: bench.sh <python> <module_dir> <corpus.bin> [\"dtype entry threads reps\" ...]" >&2
  exit 2
fi
PY=$1; MODULE=$2; CORPUS=$3; shift 3
HERE=$(cd "$(dirname "$0")" && pwd)
CFGS=("$@")
[ ${#CFGS[@]} -eq 0 ] && CFGS=("f64 cols 1 2" "f64 cols 8 3" "f64 cube 8 3")
TIMES=$(mktemp)
trap 'rm -f "$TIMES"' EXIT
for cfg in "${CFGS[@]}"; do
  read -r dt entry th reps <<< "$cfg"
  /usr/bin/time -l "$PY" -I "$HERE/bench_synthetic.py" "$MODULE" "$CORPUS" "$dt" "$entry" "$th" "$reps" 2> "$TIMES"
  grep -E "maximum resident|real" "$TIMES" | sed 's/^ */    /'
done
