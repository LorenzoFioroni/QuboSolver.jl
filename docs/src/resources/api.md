```@meta
CurrentModule = QuboSolver
```

# [API](@id doc-API)

**Table of contents**

[[toc]]

## [Base QuboSolver](@id doc-API:QuboSolverBase)

### QuboProblem
```@docs
QuboProblem
QuboProblem(::AbstractMatrix{T}, ::Union{AbstractVector{T}, Nothing}) where {T<:AbstractFloat}
get_energy
```

### AbstractSolver
```@docs
AbstractSolver
```

### Solution
```@docs
Solution
Solution(::AbstractFloat, ::Vector{Int8}, ::AbstractSolver; kwargs...)
add_solution(::QuboProblem, ::Solution)
add_solution(::QuboProblem, ::Vector{Int8}, ::AbstractSolver; kwargs...)
```

## [Solvers](@id doc-API:Solvers)

### [GCS solver](@id doc-API:GCS)

```@meta
CurrentModule = QuboSolver.Solvers.GCS
```

```@docs
GCS_solver
SignRounding
SequentialRounding
solve!
```

### [Brute-force solver](@id doc-API:BruteForce)

```@meta
CurrentModule = QuboSolver.Solvers.BruteForce
```

```@docs
BruteForce_solver
solve!(::QuboProblem, ::BruteForce_solver)
```

### [Gurobi solver](@id doc-API:Gurobi)

```@meta
CurrentModule = QuboSolver.Solvers.GurobiLib
```

```@docs
Gurobi_solver
solve!(::QuboProblem, ::Gurobi_solver)
```

### [TAMC solver](@id doc-API:TAMC)

```@meta
CurrentModule = QuboSolver.Solvers.PTICM
```

```@docs
PTICM_solver
solve!(::QuboProblem, ::PTICM_solver)
```

### [Simulated annealing solver](@id doc-API:SA)

```@meta
CurrentModule = QuboSolver.Solvers.SA
```

```@docs
SA_solver
solve!(::QuboProblem, ::SA_solver)
```

### [Local Quantum Annealing solver](@id doc-API:LQA)

```@meta
CurrentModule = QuboSolver.Solvers.LQA
```

```@docs
LQA_solver
SignRounding
SequentialRounding
solve!(::QuboProblem, ::LQA_solver)
```

## [Utilities](@id doc-API:Utilities)

### Random generation of QUBO problems

```@docs
SherringtonKirkpatrick
Chimera
EdwardsAnderson
Base.rand
```

### Utility functions

```@docs
dense_similar
similar_named_tuple
conf_int2spin
conf_int2spin!
binary_to_spin
spin_to_binary
nonzero_triu
drop_target_sparsity
```

