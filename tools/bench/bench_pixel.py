"""The per-pixel timing of one build.

The first `pixels` pixels of the corpus of `golden_driver gen` (c16.bin), one ccdcpp.detect call
each, single-threaded, best of `reps` passes, in three forms:

  spec       detect(d, s, q, spec=h): a DataSpec, the "ccdc" configuration by default
  profile    detect(d, s, q, spec=landsat_c1_dn()): a profile, the "ccdc" configuration
  config     detect(d, s, q, spec=h, config=cfg), cfg = ccdcpp.config("ccdc") taken once

The first two are the default paths, what a caller writes without thinking about the
configuration; the third names the configuration. Each form runs in its own process (`--only
<form>` runs one): prints, per form, the best pass's time per pixel and the process's peak
memory, also above the peak with the pixels loaded, and the file `import ccdcpp` loaded.

`package_dir` is a directory holding the ccdcpp package (profiles need it, not the bare
extension). Input conversion happens outside the timed calls.
"""
import argparse
import os
import sys
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common  # noqa: E402

FORMS = ("spec", "profile", "config")


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("package_dir")
    p.add_argument("corpus", help="c16.bin")
    p.add_argument("pixels", type=int)
    p.add_argument("reps", type=int)
    p.add_argument("--only", choices=FORMS, help="run one form")
    args = p.parse_args(argv)
    if args.only is None:
        results = _common.run_sides(__file__, argv, FORMS)
        print(f"ccdcpp {results[FORMS[0]]['ccdcpp']}")
        return 0
    _common.import_build(args.package_dir)
    import ccdcpp
    from ccdcpp import presets

    corpus = _common.read_corpus(args.corpus)
    H, W, _, T = corpus.spectral.shape
    dates = corpus.dates
    spec = corpus.spectral.reshape(H * W, 7, T)
    qas = corpus.qas.reshape(H * W, T)
    px = [(np.ascontiguousarray(spec[i]), np.ascontiguousarray(qas[i]))
          for i in range(min(args.pixels, H * W))]
    del corpus, spec, qas

    h = ccdcpp.DataSpec()
    profile = presets.landsat_c1_dn()
    cfg = ccdcpp.config("ccdc")
    forms: dict[str, Callable[[NDArray[np.float64], NDArray[np.uint8]], object]] = {
        "spec": lambda s, q: ccdcpp.detect(dates, s, q, spec=h),
        "profile": lambda s, q: ccdcpp.detect(dates, s, q, spec=profile),
        "config": lambda s, q: ccdcpp.detect(dates, s, q, spec=h, config=cfg),
    }

    call = forms[args.only]
    base = _common.peak_mb()
    call(*px[0])

    def one_pass() -> None:
        for s, q in px:
            call(s, q)

    best, _ = _common.best_of(args.reps, one_pass)
    print(_common.report(f"pixel {args.only}",
                         f"{best / len(px) * 1e3:.4f} ms/px over {len(px)} px",
                         _common.peak_mb(), base, "the pixels"))
    _common.emit({"ms_per_px": best / len(px) * 1e3, "peak_mb": _common.peak_mb(),
                  "ccdcpp": _common.build_of()})
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
