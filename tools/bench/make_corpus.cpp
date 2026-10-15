// make_corpus <out.bin>
//
// Writes the synthetic bench corpus: golden::make_cube(1, Cadence16, 40, 50)
// of tests/ccd/golden/synthetic.hpp, 2000 Landsat-like pixels at a 16-day
// cadence, 1984-2024. The same generator builds the golden corpus's c16
// cube, so the file is a deterministic function of synthetic.hpp and the
// platform's libm.
//
// Format, little-endian: uint64 H, W, T; int64 dates[T]; float64
// spectral[H][W][7][T]; uint8 qas[H][W][T]. bench_synthetic.py reads it.
//
// Build (header-only, no ccdcpp library needed):
//
//   c++ -std=c++17 -O2 -I <repo>/tests/ccd/golden <repo>/tools/bench/make_corpus.cpp -o make_corpus

#include <cstdint>
#include <cstdio>

#include "synthetic.hpp"

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        std::fprintf(stderr, "usage: make_corpus <out.bin>\n");
        return 2;
    }

    const golden::Cube c = golden::make_cube(1, golden::Regime::Cadence16, 40, 50);

    std::FILE* f = std::fopen(argv[1], "wb");
    if (!f)
    {
        std::perror(argv[1]);
        return 1;
    }

    const std::uint64_t header[3] = {c.H, c.W, c.T};
    std::fwrite(header, sizeof(header), 1, f);
    std::fwrite(c.dates.data(), sizeof(std::int64_t), c.dates.size(), f);
    std::fwrite(c.spectral.data(), sizeof(double), c.spectral.size(), f);
    std::fwrite(c.qas.data(), 1, c.qas.size(), f);

    return std::fclose(f) == 0 ? 0 : 1;
}
