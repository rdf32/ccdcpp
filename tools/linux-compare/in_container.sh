#!/bin/bash
# in_container.sh <label> ... : run by compare.sh inside a linux/amd64 container.
#
# Installs conda-forge GCC, builds each tree /src/<label> the same way (CMake Release, no
# -march, GCC's default -ffp-contract=fast), and writes to /out/<label>/ its golden-driver
# dumps on /corpus and the CSV pixels, and the Golden, Bands and Transitions fixtures its own
# tests write (each checked to carry this container's platform header). Then compares every
# label's files with the last label's, byte for byte.
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

# The platform header a fixture written in this container carries (tests/ccd/golden/golden.hpp
# platform()): this GCC, x86-64, Linux. The trees' committed fixtures are the Mac's, so a
# write that did not happen leaves a header that fails this.
platform_re="^# platform: gcc $(sed -n 's/.*) \([0-9][0-9.]*\).*/\1/p' /out/compiler.txt) / x86_64 / linux / "

# write_fixture <test> <ENV> <Suite> <fixture>: test_<test> of tree $t writes <fixture> in its
# tree copy ($G) in write mode, which must exit 0 and leave this platform's header; then the
# fixture is kept as $O/<fixture>.
write_fixture() {
  local test=$1 env=$2 suite=$3 fixture=$4 header
  (cd "/b-$t" && env "CCD_${env}_WRITE=1" "./test_$test" --gtest_filter="$suite.BitIdenticalToFixture") \
    > "$O/write-$test.log" 2>&1 || { echo "$t: test_$test in write mode failed"; tail -20 "$O/write-$test.log"; exit 1; }
  header=$(head -1 "$G/$fixture")
  [[ $header =~ $platform_re ]] || { echo "$t: $fixture not written here: $header"; exit 1; }
  cp "$G/$fixture" "$O/$fixture"
}

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
  "/b-$t/golden_driver" run /corpus "/work-$t/tests/ccd/machine" "$O/golden" 2> "$O/driver.log"
  # Each fixture test in write mode rewrites its fixture in the tree copy, then skips.
  write_fixture golden GOLDEN Golden golden_fixture.txt
  write_fixture bands BANDS Bands bands_fixture.txt
  if [ -x "/b-$t/test_transitions" ]; then
    write_fixture transitions TRANSITIONS Transitions transitions_fixture.txt
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
