#!/bin/bash
# bench.sh <python> <module_dir> <corpus.bin> [config ...]
#
# The synthetic bench: bench_synthetic.py once per config, each under /usr/bin/time for the
# process's peak RSS. A config is "dtype entry threads reps"; the default set is the one the
# state-machine refactor reports: cols f64 t1 (best of 2), cols f64 t8 and cube f64 t8 (best
# of 3 each). See README.md for the method.
set -eo pipefail
if [ $# -lt 3 ]; then
  echo "usage: bench.sh <python> <module_dir> <corpus.bin> [\"dtype entry threads reps\" ...]" >&2
  exit 2
fi
PY=$1; MODULE=$2; CORPUS=$3; shift 3
HERE=$(cd "$(dirname "$0")" && pwd)
CFGS=("$@")
[ ${#CFGS[@]} -eq 0 ] && CFGS=("f64 cols 1 2" "f64 cols 8 3" "f64 cube 8 3")
for cfg in "${CFGS[@]}"; do
  read -r dt entry th reps <<< "$cfg"
  # The bench's stdout goes straight out (fd 3); its stderr, where /usr/bin/time writes, is
  # piped to the filter, so no file is written.
  { /usr/bin/time -l "$PY" -I "$HERE/bench_synthetic.py" "$MODULE" "$CORPUS" "$dt" "$entry" "$th" "$reps" 2>&1 1>&3 \
      | grep -E "maximum resident|real" | sed 's/^ */    /'; } 3>&1 \
    || { echo "bench.sh: config \"$cfg\" failed; run bench_synthetic.py with it to see why" >&2; exit 1; }
done
