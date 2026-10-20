#!/bin/bash
# bench.sh <python> <module_dir> <corpus.bin> [config ...]
#
# The synthetic bench: bench_synthetic.py once per config, each under /usr/bin/time for the
# process's peak RSS. A config is "dtype threads reps"; the default set is f64 t1 (best of 2)
# and f64 t8 (best of 3). See README.md for the method.
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
  # The bench's stdout goes straight out (fd 3); its stderr, where /usr/bin/time writes, is
  # piped to the filter, so no file is written. The filter keeps time's wall and peak RSS lines
  # and every line of the bench's own: its import check and any traceback.
  { /usr/bin/time -l "$PY" -I "$HERE/bench_synthetic.py" "$MODULE" "$CORPUS" "$dt" "$th" "$reps" 2>&1 1>&3 \
      | grep -E "maximum resident| real |bench_synthetic\.py|Traceback|Error|^  File " \
      | sed 's/^ */    /'; } 3>&1 \
    || { echo "bench.sh: config \"$cfg\" failed; run bench_synthetic.py with it to see why" >&2; exit 1; }
done
