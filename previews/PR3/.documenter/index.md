---
# https://vitepress.dev/reference/default-theme-home-page
layout: home

hero:
  name: "QuboSolver.jl"
  tagline: "A Julia suite implementing the GCS algorithm and other heuristics for solving QUBO problems"
  actions:
  - theme: brand
    text: Read the Article
    link: https://www.nature.com/articles/s42005-025-02338-0
  - theme: alt
    text: Getting Started
    link: /getting_started
  - theme: alt
    text: API
    link: /resources/api
  - theme: alt
    text: View on Github
    link: https://github.com/LorenzoFioroni/QuboSolver.jl
---


# Introduction {#doc:Introduction}

[`QuboSolver.jl`](https://github.com/LorenzoFioroni/QuboSolver.jl) is a [`Julia`](https://julialang.org/) package that provides a suite of tools for solving _Quadratic Unconstrained Binary Optimization_ (QUBO) problems. Importantly, it implements the **GCS** algorithm which we propose in our article &quot;[Entanglement-assisted variational algorithm for discrete optimization problems](https://www.nature.com/articles/s42005-025-02338-0)&quot;. Additionally, it includes a variety of other solvers and utilities for benchmarking and comparing different approaches to QUBO optimization. 

# Installation {#doc:Installation}

::: tip Requirements

`QuboSolver.jl` requires `Julia 1.10+`.

:::

To install `QuboSolver.jl`, run the following commands inside Julia&#39;s interactive session (REPL):

```julia
using Pkg
Pkg.add(url="https://github.com/LorenzoFioroni/QuboSolver.jl")
```


Alternatively, this can also be run in `Julia`&#39;s [Pkg REPL](https://julialang.github.io/Pkg.jl/v1/getting-started/) by pressing the key `]` in the REPL and running:

```julia
(1.10) pkg> add https://github.com/LorenzoFioroni/QuboSolver.jl
```


Finally, to start using the package execute the following line of code:

```julia
using QuboSolver
```


The package is now ready to be used. You can start by checking the [Getting Started](getting_started.md) page for a quick introduction to the package and its features. You can also check the [API documentation](resources/api.md) for a more detailed overview of the available functions and their usage.
