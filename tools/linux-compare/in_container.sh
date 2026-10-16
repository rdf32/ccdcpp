#!/bin/bash
# in_container.sh <label> ... : run by compare.sh inside a linux/amd64 container.
#
# Installs conda-forge GCC, builds each tree /src/<label> the same way (CMake Release, no
# -march, GCC's default -ffp-contract=fast), and writes to /out/<label>/ its golden-driver
# dumps on /corpus and the CSV pixels, and the Golden, Bands and Transitions fixtures its own
# tests write. Then compares every label's files with the last label's, byte for byte.
#
# Mounts: /src/<label> (read only), /corpus (read only), /tool (this directory, read only),
# /out (written), and optionally /certs (*.crt added to the trust store, for a proxy).
set -euo pipefail
[ $# -ge 2 ] || { echo "usage: in_container.sh <label> <label> ... (the last is compared with each)" >&2; exit 2; }
labels=("$@")
subject=${labels[$#-1]}

if compgen -G "/certs/*.crt" > /dev/null; then
  cp /certs/*.crt /usr/local/share/ca-certificates/ && update-ca-certificates > /dev/null 2>&1
fi
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
{ uname -m; grep PRETTY /etc/os-release; nproc; } | tee /out/container.txt

mkdir -p /opt/mm
curl -fsSL --cacert "$SSL_CERT_FILE" https://micro.mamba.pm/api/micromamba/linux-64/latest \
  | tar -xj -C /opt/mm bin/micromamba
export MAMBA_ROOT_PREFIX=/opt/mmroot
/opt/mm/bin/micromamba --version > /out/micromamba.txt
/opt/mm/bin/micromamba create -y -p /opt/env -c conda-forge --ssl-verify "$SSL_CERT_FILE" \
  gxx_linux-64 cmake make libboost-devel eigen > /out/mamba.log 2>&1 \
  || { tail -30 /out/mamba.log; exit 1; }
/opt/mm/bin/micromamba list -p /opt/env > /out/packages.txt
export PATH=/opt/env/bin:$PATH
CXX=$(ls /opt/env/bin/x86_64-conda-linux-gnu-g++)
"$CXX" --version | head -1 | tee /out/compiler.txt
N=$(nproc)

for t in "${labels[@]}"; do
  echo "== build and dump $t"
  O=/out/$t; mkdir -p "$O/golden"
  cp -r "/src/$t" "/work-$t"
  G=/work-$t/tests/ccd/golden
  cmake -S "/work-$t" -B "/b-$t" -DCMAKE_BUILD_TYPE=Release -DCCD_BUILD_PYTHON=OFF \
    -DCCD_ENABLE_OPENMP=ON -DCMAKE_CXX_COMPILER="$CXX" -DCMAKE_PREFIX_PATH=/opt/env \
    > "$O/configure.log" 2>&1
  targets="ccd_core test_golden test_bands"
  [ -f "$G/test_transitions.cpp" ] && targets="$targets test_transitions"
  cmake --build "/b-$t" -j"$N" --target $targets > "$O/build.log" 2>&1 \
    || { tail -30 "$O/build.log"; exit 1; }
  "$CXX" -std=c++17 -O3 -DNDEBUG -I"/work-$t/include" -I"$G" -I/opt/env/include \
    -I/opt/env/include/eigen3 -fopenmp /tool/golden_driver.cpp "/b-$t/libccd_core.a" \
    -o "/b-$t/golden_driver"
  "/b-$t/golden_driver" run /corpus "/work-$t/tests/ccd/procedure" "$O/golden" 2> "$O/driver.log"
  # Each fixture test in write mode rewrites its fixture in the tree copy, then skips.
  (cd "/b-$t" && CCD_GOLDEN_WRITE=1 ./test_golden --gtest_filter='Golden.BitIdenticalToFixture') \
    > "$O/write-golden.log" 2>&1 || true
  cp "$G/golden_fixture.txt" "$O/golden_fixture.txt"
  (cd "/b-$t" && CCD_BANDS_WRITE=1 ./test_bands --gtest_filter='Bands.BitIdenticalToFixture') \
    > "$O/write-bands.log" 2>&1 || true
  cp "$G/bands_fixture.txt" "$O/bands_fixture.txt"
  if [ -x "/b-$t/test_transitions" ]; then
    (cd "/b-$t" && CCD_TRANSITIONS_WRITE=1 ./test_transitions --gtest_filter='Transitions.BitIdenticalToFixture') \
      > "$O/write-transitions.log" 2>&1 || true
    cp "$G/transitions_fixture.txt" "$O/transitions_fixture.txt"
  fi
done

echo "== compare with $subject"
: > /out/compare.txt
for a in "${labels[@]}"; do
  [ "$a" = "$subject" ] && continue
  fail=0; n=0
  for f in /out/"$a"/golden/*.txt /out/"$a"/*_fixture.txt; do
    rel=${f#/out/$a/}
    n=$((n+1))
    if cmp -s "$f" "/out/$subject/$rel"; then
      echo "same $a $subject $rel" >> /out/compare.txt
    else
      echo "DIFF $a $subject $rel" | tee -a /out/compare.txt; fail=1
    fi
  done
  echo "$a vs $subject: compared $n files; fail=$fail" | tee -a /out/compare.txt
done
