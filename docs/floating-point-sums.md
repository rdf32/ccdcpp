# Floating-point sums: why their order is part of the output, and how to make them fast

This note explains one optimisation in ccdcpp's Lasso solver, the four partial sums in
`dot()` (`src/ccd/regression/lasso_solver.cpp`), from the model it serves down to the CPU.
It also covers the general rule behind it, so you can spot the same opportunity in other
code.

## 1. What the solver computes

CCDC models each band of a pixel over a stretch of time with a harmonic curve
(`include/ccd/regression/harmonic_basis.hpp`):

    y(t) = c0 + c1 t + c2 cos(wt) + c3 sin(wt) + c4 cos(2wt) + c5 sin(2wt) + c6 cos(3wt) + c7 sin(3wt)

where `t` is the day and `w = 2 pi / 365.25`. A window holds n observations (dates t_1..t_n and
values y_1..y_n). Column j of the design matrix X holds one term evaluated at every date, for
example column 2 is cos(w t_i) for i = 1..n. Fitting the curve means choosing the coefficients
c.

ccdcpp fits them with the Lasso: least squares plus a penalty on the size of the coefficients,
which pushes unneeded harmonics to exactly zero:

    minimise   (1 / 2n) * sum_i (y_i - c0 - sum_j X_ij c_j)^2  +  alpha * sum_j |c_j|

The intercept c0 is handled by centring: subtract each column's mean from X (`X_center`) and
y's mean from y, then fit the rest without it.

### Coordinate descent

The solver (`LassoSolver::fit`) improves one coefficient at a time while holding the others
fixed, and sweeps over the coefficients until nothing changes much. It keeps the current
residual

    r_i = y_i - sum_j X_ij c_j        (what the curve has not yet explained at observation i)

To update coefficient j:

1. Add coefficient j's own contribution back into the residual: r_i += X_ij c_j.
2. Measure how much of that residual column j could explain:

       rho_j = sum_i X_ij * r_i                                   (a dot product)

3. Shrink it toward zero (the soft threshold: if |rho_j| <= alpha n the coefficient becomes 0;
   otherwise its size shrinks by alpha n) and divide by sum_i X_ij^2.
4. Take the new contribution back out of the residual: r_i -= X_ij c_j.

A sweep does this for every coefficient (up to 7). A fit repeats sweeps until the largest
change is small, then confirms convergence with a duality-gap test. That test needs
`sum_i X_ij r_i` again, for every j.

So the innermost operation of the whole fit is the dot product rho_j = sum_i X_ij r_i. It runs
for every coefficient, in every sweep, for every band, in every refit. On the synthetic bench
it is most of the run time.

## 2. A dot product is a chain of additions

Written plainly, the dot product is one loop with one running total:

    rho = 0
    rho = rho + X_1 r_1
    rho = rho + X_2 r_2
    ...
    rho = rho + X_n r_n

Each line needs the total from the line before. The multiplications are independent and can
all run at once. The additions cannot: they form a chain, ((((0 + p1) + p2) + p3) + ...).

### Why the compiler won't fix this for you

In ordinary arithmetic you could add the terms in any order, because addition is associative:
(a + b) + c = a + (b + c). Floating-point addition is not. Each addition rounds its result to
53 binary digits (about 16 decimal digits), so different groupings round at different places.
A small example in doubles:

    (1e16 + 1) - 1e16  =  0      (1e16 + 1 rounds back to 1e16)
    (1e16 - 1e16) + 1  =  1

C and C++ therefore require the compiler to add floating-point numbers in exactly the order the
source code writes them. Changing that order would change the result, and the compiler may
only make changes that keep the result the same. (`-ffast-math` lifts that rule for a whole
program. ccdcpp does not use it, because then the output would depend on compiler choices
nobody can see.)

### Why a chain is slow

An addition takes the CPU a few cycles to finish (its *latency*, about 3-4 cycles on current
CPUs), but it can *start* several new additions every cycle (its *throughput*). It does that
only when the additions don't depend on each other. In a chain, every addition waits for the
previous one, so the loop runs at one addition per ~3-4 cycles while most of the CPU sits idle.

## 3. Partial sums: the same sum, as independent chains

Split the terms into K groups by position, sum each group separately, and add the K group sums
at the end. With K = 4:

    s0 = X_1 r_1 + X_5 r_5 + X_9 r_9  + ...
    s1 = X_2 r_2 + X_6 r_6 + X_10 r_10 + ...
    s2 = X_3 r_3 + X_7 r_7 + X_11 r_11 + ...
    s3 = X_4 r_4 + X_8 r_8 + X_12 r_12 + ...
    rho = (s0 + s1) + (s2 + s3)

Every term appears exactly once, so in exact arithmetic rho is unchanged. But s0, s1, s2 and s3
don't depend on each other, so the CPU works on all four at once and the loop runs up to four
times faster. Leftover terms, when n is not a multiple of 4, go into s0. The code
(`dot()` in `lasso_solver.cpp`) is exactly this.

### Is the answer still right?

Yes, to the same accuracy. The textbook bound for the rounding error of a sum of n terms in any
order is about (n - 1) * 1.1e-16 times the sum of the terms' absolute values. That holds for the
old order and the new one alike. Partial sums are, if anything, slightly more accurate, because
each chain is a quarter as long. Neither order is the true value; they are two equally valid
roundings of it.

What does change is the last few bits of every coefficient. Those then pass through the rest of
the computation. Measured on the Atlanta cell (250,000 pixels, 455,231 segments) and the golden
corpus (26,021 segments, 182,147 band fits):

- every segment is identical: the same start, end and break days, counts and change
  probabilities;
- no fit stops at a different iteration;
- the numbers move by at most about 1e-10 relative (typically 1e-16).

The one way a bigger difference could arise: a fit whose convergence test lands within
rounding of its threshold could stop one sweep earlier or later. Its coefficients would then
differ by up to the solver's tolerance, about 1e-4 relative. That never happened in the data
above. The same sensitivity already exists between a Mac build and a Linux build, whose
arithmetic differs in the same way.

Because the order is part of the output, changing it is a results version (`RESULTS_VERSION`
in `include/ccd/record.hpp`). A stored resume record from the old order is refused, and the
first run after upgrading is a full one.

## 4. Why four

Two effects decide how many chains pay off.

1. **Enough chains to keep the adder busy.** The number of independent additions needed is
   about latency x additions started per cycle. On top of that, the compiler packs pairs of
   chains into 2-wide vector registers (NEON on Apple silicon; AVX on x86 is 4-wide). So more
   chains help, up to a point.
2. **Short windows.** Most refits are small. While a window holds 24 or fewer observations,
   Extend refits at every step. With K chains, up to K - 1 leftover terms run one at a time,
   and the K sums have to be combined at the end. So large K costs more on small n.

Measured on an Apple M5 Max. The dot product alone, in nanoseconds per call:

| n | K=1 | K=2 | K=4 | K=8 |
|---|---|---|---|---|
| 24 | 3.3 | 4.7 | 4.7 | 2.2 |
| 100 | 21.9 | 18.7 | 23.6 | 8.4 |
| 1000 | 422.6 | 261.3 | 232.6 | 84.7 |

The whole pipeline (synthetic bench, one thread):

| K | 1 (0.9.0) | 2 | 4 | 8 |
|---|---|---|---|---|
| time | 3.87 s | 3.53 s | 3.06 s | 3.08 s |

Alone, 8 chains wins. In the solver, where n is often small and the dot product shares the time
with the rest of the sweep, 4 and 8 tie. 4 is the smaller change and leaves less work in the
tails, so it is the one ccdcpp uses.

K is fixed, not a setting. It decides where the sums round, so two values give two slightly
different outputs. A per-machine choice would make results depend on where they were computed,
and would break resume's promise that a resumed run equals a full one.

## 5. Where else this applies

The pattern to look for: **a hot loop that accumulates a floating-point value across
iterations.** Examples:

- sums and dot products (`s += a[i] * b[i]`);
- norms and residual sums of squares (`s += r[i] * r[i]`);
- means and variances;
- matrix-vector products, row by row (each row is a dot product).

How to tell it is worth doing:

1. **Profile first.** It only matters when the loop is a large share of the time. Elsewhere in
   ccdcpp sums like the mean of y run once per fit, not once per coefficient per sweep.
2. **Look for one running total updated every iteration.** That is the chain.
3. **Decide whether a last-bit change is acceptable.** In ccdcpp it is an output change: it
   needs a decision, new fixtures and a results version. In code whose result is only compared
   within a tolerance, it is free.

What does **not** need it:

- **Integer sums.** Integer addition is exactly associative, so the compiler already reorders
  and vectorises them.
- **Element-wise loops with no running total** (`r[i] -= x[i] * w`). Every iteration is
  independent; the compiler already runs them in parallel.
- **Max and min of numbers that are never NaN.** The result doesn't depend on order, so
  splitting them into partial maxima changes no bits: safe whenever it is worth it.

Related rules worth knowing:

- **Fused multiply-add (FMA)** computes a * b + c with one rounding instead of two. Whether
  `s += a * b` becomes an FMA is up to the compiler (clang fuses within one expression by
  default; GCC also across statements). That is another reason the same source can give
  different last bits on different compilers.
- **What the compiler is allowed to change.** The compiler may rearrange code only when the
  result is provably the same. Speed-ups that change rounding must be written into the source
  on purpose, as here.
