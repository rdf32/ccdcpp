# Linux dump comparison

Builds several ccdcpp trees on Linux x86-64 with conda-forge GCC, the same way, and compares
every output byte for byte. The golden fixtures belong to one platform (clang / arm64 / macOS);
this is the check that a change which keeps them on the Mac also keeps another platform's
arithmetic: GCC contracts across statements by default (`-ffp-contract=fast`), so moving code
can change which multiply-adds fuse.

| File | What it does |
|---|---|
| `compare.sh` | Host side: archives each git ref, runs `in_container.sh` in one throwaway container, keeps everything under the output directory. |
| `in_container.sh` | Container side: installs conda-forge GCC with micromamba, builds each tree (CMake Release, no `-march`, so the conda-forge x86-64 baseline without FMA), writes each tree's golden-driver dumps and its Golden, Bands and Transitions fixtures, and compares each tree's files with the last tree's. |
| `golden_driver.cpp` | The driver that dumps every output of `detect()` and `detect_cube()` (threads 1 and 8) on the synthetic corpus and the CSV pixels as hexfloat text. |

## Image

The devbox base image, `mcr.microsoft.com/devcontainers/typescript-node:24`, as `linux/amd64`
(Rosetta on an arm64 Mac). Use it by digest so that runs compare like with like and the local
`:24` tag is left alone:

```
mcr.microsoft.com/devcontainers/typescript-node@sha256:91dc841966e9b47195a1a6290a8ef9bba5ba43c02aa22f87f8669c16bb6f0f6e
```

(Debian 13 trixie; the multi-platform index the `:24` tag pointed to on 2026-10-16.)

## Commands

Docker Desktop's credential helper can block on a keychain prompt, and the image is public, so
pull anonymously through a config without `credsStore`:

```
mkdir -p <scratch>/dockercfg && echo '{}' > <scratch>/dockercfg/config.json
export DOCKER_CONFIG=<scratch>/dockercfg
export DOCKER_HOST=unix://$HOME/.docker/run/docker.sock
docker pull --platform linux/amd64 \
  mcr.microsoft.com/devcontainers/typescript-node@sha256:91dc841966e9b47195a1a6290a8ef9bba5ba43c02aa22f87f8669c16bb6f0f6e
```

The synthetic corpus (`c16`, `c8`, `dup`, `medium`, `short` `.bin`) is the golden check's.
To make one, build the driver against any tree and run `golden_driver gen <corpus_dir>`.

Then, with the last label the tree under test:

```
CERTS=<dir of *.crt, behind a TLS-intercepting proxy> \
bash tools/linux-compare/compare.sh <repo> <out_dir> <corpus_dir> \
  mcr.microsoft.com/devcontainers/typescript-node@sha256:91dc841966e9b47195a1a6290a8ef9bba5ba43c02aa22f87f8669c16bb6f0f6e \
  v061=v0.6.1 main=main branch=<branch>
```

It prints one line per earlier label, `<label> vs <last>: compared <n> files; fail=<0|1>`.
Everything is kept under `<out_dir>`: `refs.txt` (label, ref, commit), `image.txt`,
`linux-run.log`, `src-<label>/`, and `linux-out/` with `container.txt`, `compiler.txt`,
`packages.txt` (the conda environment), `compare.txt` (every file, same or DIFF) and, per
label, the dumps, the three fixtures and the build, driver and fixture-write logs. The
container runs with `--rm` and `--pull never`.

A tree without `tests/ccd/golden/test_transitions.cpp` (v0.6.1 and earlier) has no
Transitions fixture, so its comparison has one file fewer: the 34 dumps and the Golden and
Bands fixtures.
