# TSGD

Julia scripts for numerical experiments with truncated stochastic gradient descent (TSGD,
the abbreviation used throughout the scripts). In the basic form, the iteration replaces the
step `α g` of stochastic gradient descent, with `g` a stochastic gradient, by the truncated
step

```
x ← x − α / (1 + α‖g‖) · g
```

(see [`tsgd_gaussian_hist.jl`](tsgd_gaussian_hist.jl)). The scripts run it on test functions
(the quadratic `x²/2`, `x⁴`, `eˣ − x`, the Rosenbrock function), on a logistic-regression
maximum-likelihood problem, on an Asian-option problem with an importance distribution
obtained by shifting a standard normal by `θ`, and on volume computations; they compare it
with gradient descent with diminishing step sizes, and estimate the autocorrelation time and
the stationary distribution of the iterates. Some scripts refer to figure numbers in
their comments (for example [`noisy_rosenbrock.jl`](noisy_rosenbrock.jl), "Figure 1").

## Running

The scripts were written in 2015 for a Julia release before 1.0 and do not run on current
Julia: they use `tic()` and `int()`, which were removed in Julia 1.0, and the Convex.jl and
ECOS.jl interfaces of that time. The packages the scripts load are `PyPlot` (matplotlib
through PyCall), `Distributions`, `Convex` and `ECOS` (most scripts), `Regression`
(`logistic_MLE.jl`, `test_logisticreg.jl`) and `Optim` (`asian_option.jl`).

With a pre-1.0 Julia and these packages installed, each script is run on its own from the
repository directory, for example

```julia
include("tsgd_gaussian_hist.jl")
```

The scripts are independent of each other. Several read data files from, or write figures to,
the directories `data/` and `paper/figures/`.

## Repository layout

The repository contains only top-level scripts. By file name:

| Files | Contents |
|---|---|
| `gaussian_*.jl`, `quart_*.jl`, `quartic_*.jl`, `exp_*.jl`, `testgaussianngdnsgd.jl`, `testing_gaussian.jl` | TSGD on `x²/2`, `x⁴` and `eˣ − x`: autocorrelation time, accuracy, stationary distribution (`tsgd_gaussian_hist.jl`) |
| `noisy_exponential.jl`, `noisy_quartic.jl`, `noisy_rosenbrock.jl` | noisy test problems, with figure numbers in the comments |
| `asian_*.jl`, `1dasianoption_test.jl`, `comparing_asian_option_gradients.jl`, `newstepsizeAsian.jl`, `tsgd_asia*.jl`, `tsgd_asian_*.jl`, `finance_sample.jl`, `monteCarloFinance.jl` | the Asian-option problem |
| `logistic_MLE.jl`, `test_logisticreg.jl`, `dirichlet_MLE.jl` | maximum-likelihood problems |
| `volume_polytope.jl`, `volume_quad.jl` | volume computations |
| `gd_dimstepsize.jl`, `gd_stepsizetest.jl` | gradient descent with diminishing step size compared with truncated gradient descent, and when gradient descent fails |
| `CLT_verify.jl`, `disc_term.jl`, `linear_dynamic.jl`, `logm.jl`, `NGD_speedtest.jl`, `test.jl`, `test_SDE.jl`, `tgd_twodim.jl`, `TSG_distance_plot.jl`, `ttest.jl`, `two_dimension_test.jl` | smaller experiments, checks and helpers; see the comments at the top of each file |

## Authors and citation

The code is by **Ahmed Bou-Rabee**.

## License

The code is licensed under the Apache License 2.0; see [`LICENSE`](LICENSE).
