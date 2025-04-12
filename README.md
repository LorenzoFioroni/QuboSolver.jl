# QuboSolver.jl

| **Article**       | ![Static Badge](https://img.shields.io/badge/DOI-10.48550%2FarXiv.2501.09078-blue?style=flat&logo=arXiv&link=https%3A%2F%2Fdoi.org%2F10.48550%2FarXiv.2501.09078) |
|:-----------------:|:-------------|
| **Runtests**      | [![Runtests][runtests-img]][runtests-url] [![Coverage][codecov-img]][codecov-url] |
| **Code Quality**  | [![Code Quality][code-quality-img]][code-quality-url] [![Aqua QA][aqua-img]][aqua-url] [![JET][jet-img]][jet-url] |
| **Documentation** | [![Doc-Stable][docs-stable-img]][docs-stable-url] |


[runtests-img]: https://github.com/LorenzoFioroni/QuboSolver.jl/actions/workflows/CI.yml/badge.svg?branch=main
[runtests-url]: https://github.com/LorenzoFioroni/QuboSolver.jl/actions/workflows/CI.yml?query=branch%3Amain

[codecov-img]: https://codecov.io/gh/LorenzoFioroni/QuboSolver.jl/branch/main/graph/badge.svg
[codecov-url]: https://codecov.io/gh/LorenzoFioroni/QuboSolver.jl

[code-quality-img]: https://github.com/LorenzoFioroni/QuboSolver.jl/actions/workflows/CodeQuality.yml/badge.svg 
[code-quality-url]: https://github.com/LorenzoFioroni/QuboSolver.jl/actions/workflows/CodeQuality.yml

[aqua-img]: https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg
[aqua-url]: https://github.com/JuliaTesting/Aqua.jl

[jet-img]: https://img.shields.io/badge/%F0%9F%9B%A9%EF%B8%8F_tested_with-JET.jl-233f9a
[jet-url]: https://github.com/aviatesk/JET.jl

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://lorenzofioroni.github.io/QuboSolver.jl/dev/

## Introduction

[`QuboSolver.jl`](https://github.com/LorenzoFioroni/QuboSolver.jl) is a [`Julia`](https://julialang.org/) package that provides a suite of tools for solving _Quadratic Unconstrained Binary Optimization_ (QUBO) problems. Importantly, it implements the __GCS__ algorithm which we propose in our preprint "[Entanglement-assisted heuristic for variational solutions of discrete optimization problems](https://arxiv.org/abs/2501.09078)". Additionally, it includes a variety of other solvers and utilities for benchmarking and comparing different approaches to QUBO optimization. 

## Features

`QuboSolver.jl` boasts the following features:

- **Multiple solvers:** Implements several exact and heuristic solvers for QUBO problems, including:
    - GCS
    - LQA
    - simulated annealing
    - a brute-force solver
    - an interface to the Gurobi solver
    - an interface to the TAMC library for the PT-ICM heuristic
- **Utility functions:** Provides a variety of utility functions for generating random QUBO instances of different classes as well as functions to convert between different representations of QUBO problems.
- **Easy Extension:** Easily extend the package, adding new solvers or utility functions as needed.

## Installation
    
!!! note "Requirements"
    `QuboSolver.jl` requires `Julia 1.10+`.

To install `QuboSolver.jl`, run the following commands inside Julia's interactive session (REPL):
```julia
using Pkg
Pkg.add(url="https://github.com/LorenzoFioroni/QuboSolver.jl")
```
Alternatively, this can also be run in `Julia`'s [Pkg REPL](https://julialang.github.io/Pkg.jl/v1/getting-started/) by pressing the key `]` in the REPL and running:
```julia-repl
(1.10) pkg> add https://github.com/LorenzoFioroni/QuboSolver.jl
```
Finally, to start using the package execute the following line of code:
```julia
using QuboSolver
```

The package is now ready to be used. You can start by checking the [Getting Started](https://lorenzofioroni.github.io/QuboSolver.jl/dev/getting_started) page from the documentation for a quick introduction to the package and its features.
You can also check the [API documentation](https://lorenzofioroni.github.io/QuboSolver.jl/dev/resources/api) for a more detailed overview of the available functions and their usage.

