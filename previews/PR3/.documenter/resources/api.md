


# API {#doc-API}

**Table of contents**

[[toc]]

## Base QuboSolver {#doc-API:QuboSolverBase}

### QuboProblem {#QuboProblem}
<details class='jldocstring custom-block' open>
<summary><a id='QuboSolver.QuboProblem' href='#QuboSolver.QuboProblem'><span class="jlbinding">QuboSolver.QuboProblem</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
struct QuboProblem{
    T<:AbstractFloat,
    Tw<:AbstractMatrix{T},
    Tc<:Union{AbstractVector{T},Nothing}
}
```


A QUBO problem with matrix `W`, optional bias vector `c`, and value type `T`.

The matrix `W` must be square and symmetric, and the diagonal must be zero. The bias vector `c` is optional and must have the same length as the number of rows in `W`.

**Fields**
- `W::Tw`: The QUBO matrix.
  
- `has_bias::Bool`: Indicates if a bias vector is provided.
  
- `c::Tc`: The bias vector, or `nothing` if not provided.
  
- `N::Int`: The number of variables in the problem.
  
- `solutions::Vector{Solution}`: A vector to store solutions found.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/LorenzoFioroni/QuboSolver.jl/blob/e65c4543ed7b54db05ab960d3f4e066780747dc8/src/problem.jl#L3-L21" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuboSolver.QuboProblem-Union{Tuple{T}, Tuple{AbstractMatrix{T}, Union{Nothing, AbstractVector{T}}}} where T<:AbstractFloat' href='#QuboSolver.QuboProblem-Union{Tuple{T}, Tuple{AbstractMatrix{T}, Union{Nothing, AbstractVector{T}}}} where T<:AbstractFloat'><span class="jlbinding">QuboSolver.QuboProblem</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
function QuboProblem(
    W::AbstractMatrix{T}, 
    bias::Union{AbstractVector{T},Nothing} = nothing
) where {T<:AbstractFloat}
```


Create a new QuboProblem instance with the given matrix `W` and optional bias vector `bias`.

**Example**

```julia
W = [0.0 1.0; 1.0 0.0]
bias = [0.0, 1.0]
println(QuboProblem(W))
println(QuboProblem(W, bias))

# output

QuboProblem{Float64, Matrix{Float64}, Nothing}([0.0 1.0; 1.0 0.0], false, nothing, 2, Solution[])
QuboProblem{Float64, Matrix{Float64}, Vector{Float64}}([0.0 1.0; 1.0 0.0], true, [0.0, 1.0], 2, Solution[])
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/LorenzoFioroni/QuboSolver.jl/blob/e65c4543ed7b54db05ab960d3f4e066780747dc8/src/problem.jl#L33-L53" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuboSolver.get_energy' href='#QuboSolver.get_energy'><span class="jlbinding">QuboSolver.get_energy</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
function get_energy(
    problem::QuboProblem, 
    configuration::AbstractVector
)
```


Calculate the energy of a given configuration for the given QuboProblem `problem`.

The configuration must be a vector of $-1$ or $1$ values. The returned energy is  $E = -\vec{x}^T W \vec{x} - \vec{c}^T \vec{x}$, where $\vec{x}$ is the  configuration vector, $W$ is the QUBO matrix, and $\vec{c}$ is the bias vector.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/LorenzoFioroni/QuboSolver.jl/blob/e65c4543ed7b54db05ab960d3f4e066780747dc8/src/problem.jl#L76-L87" target="_blank" rel="noreferrer">source</a></Badge>

</details>


### AbstractSolver {#AbstractSolver}
<details class='jldocstring custom-block' open>
<summary><a id='QuboSolver.AbstractSolver' href='#QuboSolver.AbstractSolver'><span class="jlbinding">QuboSolver.AbstractSolver</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
abstract type AbstractSolver end
```


Abstract type for a QUBO problem solver.

Concrete solvers should inherit from this type and implement the QuboSolver.solve! method. This should take a QuboProblem, the solver instance, and any additional arguments and return a [`Solution`](/resources/api#QuboSolver.Solution) object. 

**Example**

```julia
struct MySolver <: AbstractSolver end
function QuboSolver.solve!(problem::QuboProblem, solver::MySolver)
    # Implement the solver logic here
    # ...
    dummy_configuration = ones(Int8, problem.N)
    dummy_energy = 0.1
    return Solution(dummy_energy, dummy_configuration, solver)
end

problem = QuboProblem([0.0 1.0; 1.0 0.0])
solver = MySolver()
solution = QuboSolver.solve!(problem, solver)

# output

🟦🟦 - Energy: 0.1 - Solver: MySolver - Metadata count: 0
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/LorenzoFioroni/QuboSolver.jl/blob/e65c4543ed7b54db05ab960d3f4e066780747dc8/src/solver.jl#L4-L32" target="_blank" rel="noreferrer">source</a></Badge>

</details>


### Solution {#Solution}
<details class='jldocstring custom-block' open>
<summary><a id='QuboSolver.Solution' href='#QuboSolver.Solution'><span class="jlbinding">QuboSolver.Solution</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
struct Solution{Te<:AbstractFloat,Ts<:AbstractSolver}
```


A solution to a QUBO problem. 

Stores a reference to the solver used to find the solution, its energy, the configuration, and  any additional metadata. Once metadata is populated, its values can also be accessed indexing the solution directly. If the solution is displayed, the configuration is shown as a string of  colored squares, where blue represents $1$ and yellow represents $-1$.

**Fields**
- `energy::Te`: The energy of the solution.
  
- `configuration::Vector`: Vector of $-1$ or $1$ values representing the configuration.
  
- `solver::Ts`: The solver used to find the solution.
  
- `metadata::Dict{Symbol,Any}`: Additional metadata about the solution.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/LorenzoFioroni/QuboSolver.jl/blob/e65c4543ed7b54db05ab960d3f4e066780747dc8/src/solution.jl#L4-L19" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuboSolver.Solution-Tuple{AbstractFloat, Vector{Int8}, AbstractSolver}' href='#QuboSolver.Solution-Tuple{AbstractFloat, Vector{Int8}, AbstractSolver}'><span class="jlbinding">QuboSolver.Solution</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
function Solution(
    energy::AbstractFloat, 
    configuration::Vector{Int8}, 
    solver::AbstractSolver; 
    kwargs...
)
```


Create a new `Solution` instance. The keyword arguments specify additional metadata.

**Example**

```julia
struct solver <: AbstractSolver end
sol = Solution(0.5, [1, -1, 1], solver(); time=0.1)
println(sol.energy) 
println(sol["time"]) 
println(sol)

# output

0.5
0.1
🟦🟨🟦 - Energy: 0.5 - Solver: solver - Metadata count: 1
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/LorenzoFioroni/QuboSolver.jl/blob/e65c4543ed7b54db05ab960d3f4e066780747dc8/src/solution.jl#L26-L50" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuboSolver.add_solution-Tuple{QuboProblem, Solution}' href='#QuboSolver.add_solution-Tuple{QuboProblem, Solution}'><span class="jlbinding">QuboSolver.add_solution</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
add_solution(problem::QuboProblem, sol::Solution)
```


Add a [`Solution`](/resources/api#QuboSolver.Solution) `sol` to the [`QuboProblem`](/resources/api#QuboSolver.QuboProblem) `problem`. 

**Example**

```julia
struct MySolver <: AbstractSolver end
problem = QuboProblem([0.0 1.0; 1.0 0.0])
solution = Solution(2.0, [1, -1], MySolver(); runtime=1.1)

println(problem.solutions)
add_solution(problem, solution)
println(problem.solutions)

# output

Solution[]
Solution[🟦🟨 - Energy: 2.0 - Solver: MySolver - Metadata count: 1]
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/LorenzoFioroni/QuboSolver.jl/blob/e65c4543ed7b54db05ab960d3f4e066780747dc8/src/problem.jl#L102-L122" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuboSolver.add_solution-Tuple{QuboProblem, Vector{Int8}, AbstractSolver}' href='#QuboSolver.add_solution-Tuple{QuboProblem, Vector{Int8}, AbstractSolver}'><span class="jlbinding">QuboSolver.add_solution</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
add_solution(
    problem::QuboProblem, 
    configuration::Vector, 
    solver::AbstractSolver;
    kwargs...
)    

Add a solution with the given `configuration` and `solver` to the [`QuboProblem`](@ref QuboSolver.QuboProblem) `problem`.
```


The configuration must be a vector of $-1$ or $1$ values. Additional metadata can be passed  as keyword arguments.

**Returns**

The added [`Solution`](/resources/api#QuboSolver.Solution) object.

**Example**

```julia
struct MySolver <: AbstractSolver end
problem = QuboProblem([0.0 1.0; 1.0 0.0])

println(problem.solutions)
add_solution(problem, Int8[1, -1], MySolver(); runtime=1.1)
println(problem.solutions)

# output

Solution[]
Solution[🟦🟨 - Energy: 2.0 - Solver: MySolver - Metadata count: 1]
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/LorenzoFioroni/QuboSolver.jl/blob/e65c4543ed7b54db05ab960d3f4e066780747dc8/src/problem.jl#L125-L155" target="_blank" rel="noreferrer">source</a></Badge>

</details>


## Solvers {#doc-API:Solvers}

### GCS solver {#doc-API:GCS}


<details class='jldocstring custom-block' open>
<summary><a id='QuboSolver.Solvers.GCS.GCS_solver' href='#QuboSolver.Solvers.GCS.GCS_solver'><span class="jlbinding">QuboSolver.Solvers.GCS.GCS_solver</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
struct GCS_solver <: AbstractSolver end
```


Variational solver for QUBO problems using GCS states.

Use the GCS algorithm [[1](/resources/bibliography#fioroni2025entanglement)] to solve the given [`QuboProblem`](/resources/api#QuboSolver.QuboProblem).  The analytical form of the GCS states used is

$$\ket{ψ} = \mathcal{U}(x) \mathcal{V}(M) \ket{θ,φ}.$$

The operator $\mathcal{U}(x)$ is a product of single-spin rotations on all particles. The vector $x$ determines the rotation axis and angle for each spin. For our implementation, we parametrize it in spherical coordinates as 

$$x_i = \begin{bmatrix} r_i \cos(δ_i) \cos(Γ_i) \\ r_i \cos(δ_i) \sin(Γ_i) \\ r_i \sin(δ_i) \end{bmatrix}.$$

The operator $\mathcal{V}(M)$ entangles the qubits via correlated $σ_z\,σ_z$ rotations. Specifically,

$$\mathcal{V}(M) = \exp(-i \sum_{i,j} M_{ij} σ_z^{(i)} σ_z^{(j)}),$$

where $M$ is a coupling matrix assumed to be symmetric and with zero diagonal. Finally, $\ket{θ,φ}$ is the product state 

$$\ket{θ,φ} = ⊗_i \left(\cos(θ_i/2) \ket{0} + \sin(θ_i/2) e^{i φ_i} \ket{1}\right).$$

GCS states are thus obtained by applying an entangling operator to a product state, and then applying an additional rotation operator to each qubit.

::: tip Tip

For more information on the GCS algorithm, see [https://doi.org/10.1038/s42005-025-02338-0](https://doi.org/10.1038/s42005-025-02338-0).

:::

::: warning Warning

To use this solver, you need to explicitly import the `GCS` module in your code:

```julia
using QuboSolver.Solvers.GCS
```


:::


<Badge type="info" class="source-link" text="source"><a href="https://github.com/LorenzoFioroni/QuboSolver.jl/blob/e65c4543ed7b54db05ab960d3f4e066780747dc8/src/solvers/GCS.jl#L10-L47" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuboSolver.Solvers.GCS.SignRounding' href='#QuboSolver.Solvers.GCS.SignRounding'><span class="jlbinding">QuboSolver.Solvers.GCS.SignRounding</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
struct SignRounding <: RoundingMethod end
```


Discretization method that discretizes the quantum state to a binary configuration setting $b_i = \text{sign}(σ_z)$.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/LorenzoFioroni/QuboSolver.jl/blob/e65c4543ed7b54db05ab960d3f4e066780747dc8/src/solvers/GCS.jl#L60-L64" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuboSolver.Solvers.GCS.SequentialRounding' href='#QuboSolver.Solvers.GCS.SequentialRounding'><span class="jlbinding">QuboSolver.Solvers.GCS.SequentialRounding</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
struct SequentialRounding <: RoundingMethod end
```


Discretization method that implements the sequential rounding algorithm presented in [https://doi.org/10.1137/20M132016X](https://doi.org/10.1137/20M132016X).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/LorenzoFioroni/QuboSolver.jl/blob/e65c4543ed7b54db05ab960d3f4e066780747dc8/src/solvers/GCS.jl#L67-L71" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuboSolver.solve!' href='#QuboSolver.solve!'><span class="jlbinding">QuboSolver.solve!</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
solve!(problem::QuboProblem, solver::AbstractSolver, args...; kwargs...)
```


Solve the QuboProblem `problem` using the provided `solver`.

The `solver` instance must implement the `solve!` method.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/LorenzoFioroni/QuboSolver.jl/blob/e65c4543ed7b54db05ab960d3f4e066780747dc8/src/problem.jl#L177-L183" target="_blank" rel="noreferrer">source</a></Badge>



```julia
function solve!(
    problem::QuboProblem,
    solver::SA_solver;
    MCS::Integer = 1000,
    rng::Random.AbstractRNG = Random.GLOBAL_RNG,
    initial_conf::Union{AbstractVector{Int8},Nothing} = nothing,
    initial_temp::Real = 1.0,
    progressbar::Bool = true,
)
```


Solve the QuboProblem `problem` using simulated annealing [[4](/resources/bibliography#kirkpatrickOptimizationSimulatedAnnealing1983)] with a linearly decreasing temperature  schedule. 

::: warning Warning

To use this solver, you need to explicitly import the `SA` module in your code:

```julia
using QuboSolver.Solvers.SA
```


:::

**Arguments**
- `problem::QuboProblem`: [`QuboProblem`](/resources/api#QuboSolver.QuboProblem) instance to be solved.
  
- `solver::SA_solver`: Instance of [`SA_solver`](/resources/api#QuboSolver.Solvers.SA.SA_solver).
  
- `MCS::Integer`: Number of Monte Carlo steps. The default value is `1000`.
  
- `rng::Random.AbstractRNG`: Random number generator. The default value is `Random.GLOBAL_RNG`.
  
- `initial_conf::Union{AbstractVector{Int8},Nothing}`: Initial configuration. If `nothing`, a random    configuration is generated. The default value is `nothing`.
  
- `initial_temp::Real`: Initial temperature. The default value is `1.0`.
  
- `progressbar::Bool`: Whether to show a progress bar. The default value is `true`.
  

**Returns**

The optimal solution found by the solver. Metadata include the runtime as `runtime`.

**Example**

```julia
using QuboSolver.Solvers.SA

problem = QuboProblem([0.0 1.0; 1.0 0.0], [1.0, 0.0])
solution = solve!(problem, SA_solver())

# output

🟦🟦 - Energy: -3.0 - Solver: SA_solver - Metadata count: 1
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/LorenzoFioroni/QuboSolver.jl/blob/e65c4543ed7b54db05ab960d3f4e066780747dc8/src/solvers/SA.jl#L24-L68" target="_blank" rel="noreferrer">source</a></Badge>



```julia
function solve!(
    problem::QuboProblem{T,TW,Tc},
    solver::GCS_solver;
    rng::AbstractRNG = Random.GLOBAL_RNG,
    initial_conf::Union{ParamType{T},Nothing} = nothing,
    iterations::Int = 1000,
    inner_iterations::Int = 1,
    tf::T = one(T),
    rounding::Union{RoundingMethod,Tuple{Vararg{RoundingMethod}}} = SignRounding(),
    save_params::Bool = false,
    save_energy::Bool = false,
    opt::Optimisers.AbstractRule = Adam(0.05),
    progressbar::Bool = true,
)
```


Solve the QUBO problem using the Variational GCS method [[1](/resources/bibliography#fioroni2025entanglement)].

::: tip Tip

For more information on the GCS algorithm, see [https://doi.org/10.1038/s42005-025-02338-0](https://doi.org/10.1038/s42005-025-02338-0).

:::

::: warning Warning

To use this solver, you need to explicitly import the `GCS` module in your code:

```julia
using QuboSolver.Solvers.GCS
```


:::

**Arguments**
- `problem`: [`QuboProblem`](/resources/api#QuboSolver.QuboProblem) object.
  
- `solver`: [`GCS_solver`](/resources/api#QuboSolver.Solvers.GCS.GCS_solver) object.
  
- `rng`: random number generator (default: `Random.GLOBAL_RNG`).
  
- `initial_conf`: initial parameter configuration (default: `nothing` for random initialization).
  
- `iterations`: number of time steps (default: `1000`).
  
- `inner_iterations`: number of gradient descent steps each time step (default: `1`).
  
- `tf`: transverse field strength (default: `1.0`).
  
- `rounding`: Rounding method used to obtain the classical configuration from the GCS   state (default: [`SignRounding`](/resources/api#QuboSolver.Solvers.GCS.SignRounding)). Accepts both a single method or a tuple of methods.
  
- `save_params`: whether to store the final parameters of the GCS state (default: `false`).
  
- `save_energy`: whether to store the variational energy during the optimization (default: `false`).
  
- `opt`: optimizer for the gradient descent (default: `Adam(0.05)`).
  
- `progressbar`: whether to show a progress bar (default: `true`).
  

**Returns**

The optimal solution found by the solver. Metadata include the runtime as `runtime`, the final  parameters of the GCS state as `params` (if `save_params` is `true`), and the variational energy  during the optimization as `energy` (if `save_energy` is `true`).

**Example**

```julia
using QuboSolver.Solvers.GCS

problem = QuboProblem([0.0 1.0; 1.0 0.0], [1.0, 0.0])
solution = solve!(problem, GCS_solver(); save_params=true, save_energy=true, progressbar=false)

# output

🟦🟦 - Energy: -3.0 - Solver: GCS_solver - Metadata count: 3
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/LorenzoFioroni/QuboSolver.jl/blob/e65c4543ed7b54db05ab960d3f4e066780747dc8/src/solvers/GCS.jl#L2336-L2395" target="_blank" rel="noreferrer">source</a></Badge>



```julia
function solve!(problem::QuboProblem, solver::BruteForce_solver)
```


Solve the QuboProblem `problem` using the [`BruteForce_solver`](/resources/api#QuboSolver.Solvers.BruteForce.BruteForce_solver).	

Exhaustively search through all possible configurations to find the optimal solution. It is not  recommended for large problems due to its exponential time and memory complexity. By default, it uses all available threads to parallelize the search.

::: warning Warning

To use this solver, you need to explicitly import the `BruteForce` module in your code:

```julia
using QuboSolver.Solvers.BruteForce
```


:::

**Returns**

The optimal solution found by the solver. Metadata include the runtime as `runtime` and the  number of threads used as `nthreads`.

**Example**

```julia
using QuboSolver.Solvers.BruteForce

problem = QuboProblem([0.0 1.0; 1.0 0.0], [1.0, 0.0])
solution = solve!(problem, BruteForce_solver())

# output

🟦🟦 - Energy: -3.0 - Solver: BruteForce_solver - Metadata count: 2
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/LorenzoFioroni/QuboSolver.jl/blob/e65c4543ed7b54db05ab960d3f4e066780747dc8/src/solvers/BruteForce.jl#L40-L70" target="_blank" rel="noreferrer">source</a></Badge>



```julia
function solve!(
    problem::QuboProblem,
    solver::Gurobi_solver;
    mipgap::Float64 = 0.0,
    timelimit::Float64 = Inf64,
    output::Bool = true,
    threads::Int = Threads.nthreads(),
)
```


Solve the QuboProblem `problem` using the Gurobi library [[5](/resources/bibliography#gurobioptimizationllcGurobiOptimizerReference2024)].

::: warning Warning

The Gurobi solver requires a valid Gurobi license.  See the [Gurobi website](https://www.gurobi.com) for more information.

:::

::: warning Warning

To use this solver, you need to explicitly import the `GurobiLib` module in your code:

```julia
using QuboSolver.Solvers.GurobiLib
```


:::

**Arguments**
- `problem::QuboProblem`: The QUBO problem to be solved.
  
- `solver::Gurobi_solver`: Instance of [`Gurobi_solver`](/resources/api#QuboSolver.Solvers.GurobiLib.Gurobi_solver).
  
- `mipgap::Float64`: The maximum allowed gap between the found lower bound and the upper bound.    The default value is `0.0`, which means that the solver will try to find the optimal solution.
  
- `timelimit::Float64`: The maximum time limit for the solver in seconds. The default value is `Inf64`,
  
- `output::Bool`: Whether to print the solver output. The default value is `true`.
  
- `threads::Int`: The number of threads to use for the solver. The default value is the number of    available threads on the system.
  

**Returns**

The optimal solution found by the solver. Metadata include the runtime as `runtime` and the  maximum gap as `mipgap`.

**Example**

```julia
using QuboSolver.Solvers.GurobiLib

problem = QuboProblem([0.0 1.0; 1.0 0.0], [1.0, 0.0])
solution = solve!(problem, Gurobi_solver(), output=false)

# output

🟦🟦 - Energy: -3.0 - Solver: Gurobi_solver - Metadata count: 2
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/LorenzoFioroni/QuboSolver.jl/blob/e65c4543ed7b54db05ab960d3f4e066780747dc8/src/solvers/GurobiLib.jl#L27-L74" target="_blank" rel="noreferrer">source</a></Badge>



```julia
function solve!(
    problem::QuboProblem{T,Tw,Tc},
    solver::LQA_solver;
    rng::Random.AbstractRNG = Random.GLOBAL_RNG,
    initial_conf::Union{AbstractVector{T},Nothing} = nothing,
    iterations::Int = 1000,
    inner_iterations::Int = 1,
    opt::Optimisers.AbstractRule = Adam(),
    tf::T = one(T),
    rounding::Union{RoundingMethod,Tuple{Vararg{RoundingMethod}}} = SignRounding(),
    save_params::Bool = false,
    save_energy::Bool = false,
    progressbar::Bool = true,
) where {T<:AbstractFloat,Tw<:AbstractMatrix{T},Tc<:Union{Nothing,AbstractVector{T}}}
```


Solve the QUBO problem using the Local Quantum Annealing solver [[2](/resources/bibliography#bowlesQuadraticUnconstrainedBinary2022)].

::: tip Tip

For more information on the LQA algorithm, see the original paper at [https://doi.org/10.1103/PhysRevApplied.18.034016](https://doi.org/10.1103/PhysRevApplied.18.034016).

:::

::: warning Warning

To use this solver, you need to explicitly import the `LQA` module in your code:

```julia
using QuboSolver.Solvers.LQA
```


:::

**Arguments**
- `problem`: [`QuboProblem`](/resources/api#QuboSolver.QuboProblem) object.
  
- `solver`: [`LQA_solver`](/resources/api#QuboSolver.Solvers.LQA.LQA_solver) object.
  
- `rng`: Random number generator to use (default: `Random.GLOBAL_RNG`).
  
- `initial_conf`: Initial configuration (default: `nothing` for random initialization).
  
- `iterations`: Number of time steps (default: 1000).
  
- `inner_iterations`: Number of gradient descent steps each time step (default: 1).
  
- `opt`: Optimizer to use (default: `Adam()`).
  
- `tf`: Transverse field strength (default: 1.0).
  
- `rounding`: Rounding method used to obtain the classical configuration from the GCS   state (default: [`SignRounding`](/resources/api#QuboSolver.Solvers.LQA.SignRounding)). Accepts both a single method or a tuple of methods.
  
- `save_params`: Save the parameters of the variational state (default: `false`).
  
- `save_energy`: Save the energy of the variational state (default: `false`).
  
- `progressbar`: Show a progress bar (default: `true`).
  

**Returns**

A [`Solution`](/resources/api#QuboSolver.Solution) object containing the optimal solution found by the solver. Metadata include the runtime as `runtime`, the parameters of the variational state as `params` (if  `save_params` is `true`), and the energy of the variational state as `energy` (if `save_energy`  is `true`).

**Example**

```julia
using QuboSolver.Solvers.LQA

problem = QuboProblem([0.0 1.0; 1.0 0.0], [1.0, 0.0])
solution = solve!(problem, LQA_solver(); save_params=true, save_energy=true, progressbar=false)

# output

🟦🟦 - Energy: -3.0 - Solver: LQA_solver - Metadata count: 3
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/LorenzoFioroni/QuboSolver.jl/blob/e65c4543ed7b54db05ab960d3f4e066780747dc8/src/solvers/LQA.jl#L253-L313" target="_blank" rel="noreferrer">source</a></Badge>



```julia
function solve!(
    problem::QuboProblem{T,TW,Tc},
    solver::PTICM_solver;
    MCS::Integer = 1000,
    warmup_fraction::Real = 0.5,
    betas::Vector{T} = T.(QuboSolver.LogRange(0.1, 5.0, 32)),
    lo_num_beta::Integer = 8,
    threads::Integer = Threads.nthreads(),
    num_replica_chains::Integer = 2,
) where {T<:AbstractFloat,TW<:AbstractMatrix{T},Tc<:Union{Nothing,AbstractVector{T}}}
```


Solve the QuboProblem `problem` using the TAMC solver [[3](/resources/bibliography#bauzaScalingAdvantageApproximate2024)].

::: warning Warning

The TAMC solver requires the TAMC binaries to be installed. See the [https://github.com/USCqserver/tamc](https://github.com/USCqserver/tamc) for more information.

:::

::: warning Warning

To use this solver, you need to explicitly import the `PTICM` module in your code:

```julia
using QuboSolver.Solvers.PTICM
```


:::

**Arguments**
- `problem::QuboProblem`: The QUBO problem to be solved.
  
- `solver::PTICM_solver`: Instance of [`PTICM_solver`](/resources/api#QuboSolver.Solvers.PTICM.PTICM_solver).
  
- `MCS::Integer`: The number of Monte Carlo steps. The default value is `1000`.
  
- `warmup_fraction::Real`: The fraction of Monte Carlo steps to be used for warmup. The default value is    `0.5`.
  
- `betas::Vector{T}`: The inverse temperatures to be used. The default value is a logarithmic range   between `0.1` and `5.0` with `32` elements.
  
- `lo_num_beta::Integer`: The number of inverse temperatures to be used for the isoenergetic cluster   moves. The default value is `8`.
  
- `threads::Integer`: The number of threads to be used. The default value is the number of available   threads.
  
- `num_replica_chains::Integer`: The number of replica chains to be used. The default value is `2`.
  

**Returns**

The optimal solution found by the solver. Metadata include the runtime as `runtime`.

**Example**

```julia
using QuboSolver.Solvers.PTICM

problem = QuboProblem([0.0 1.0; 1.0 0.0], [1.0, 0.0])
solution = solve!(problem, PTICM_solver())

# output

🟦🟦 - Energy: -3.0 - Solver: PTICM_solver - Metadata count: 1
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/LorenzoFioroni/QuboSolver.jl/blob/e65c4543ed7b54db05ab960d3f4e066780747dc8/src/solvers/PTICM.jl#L114-L166" target="_blank" rel="noreferrer">source</a></Badge>

</details>


### Brute-force solver {#doc-API:BruteForce}


<details class='jldocstring custom-block' open>
<summary><a id='QuboSolver.Solvers.BruteForce.BruteForce_solver' href='#QuboSolver.Solvers.BruteForce.BruteForce_solver'><span class="jlbinding">QuboSolver.Solvers.BruteForce.BruteForce_solver</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
struct BruteForce_solver <: AbstractSolver end
```


Brute-force solver for QUBO problems.

Exhaustively search through all possible configurations to find the optimal solution. It is not  recommended for large problems due to its exponential time and memory complexity. 

::: warning Warning

To use this solver, you need to explicitly import the `BruteForce` module in your code:

```julia
using QuboSolver.Solvers.BruteForce
```


:::


<Badge type="info" class="source-link" text="source"><a href="https://github.com/LorenzoFioroni/QuboSolver.jl/blob/e65c4543ed7b54db05ab960d3f4e066780747dc8/src/solvers/BruteForce.jl#L7-L21" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuboSolver.solve!-Tuple{QuboProblem, QuboSolver.Solvers.BruteForce.BruteForce_solver}' href='#QuboSolver.solve!-Tuple{QuboProblem, QuboSolver.Solvers.BruteForce.BruteForce_solver}'><span class="jlbinding">QuboSolver.solve!</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
function solve!(problem::QuboProblem, solver::BruteForce_solver)
```


Solve the QuboProblem `problem` using the [`BruteForce_solver`](/resources/api#QuboSolver.Solvers.BruteForce.BruteForce_solver).	

Exhaustively search through all possible configurations to find the optimal solution. It is not  recommended for large problems due to its exponential time and memory complexity. By default, it uses all available threads to parallelize the search.

::: warning Warning

To use this solver, you need to explicitly import the `BruteForce` module in your code:

```julia
using QuboSolver.Solvers.BruteForce
```


:::

**Returns**

The optimal solution found by the solver. Metadata include the runtime as `runtime` and the  number of threads used as `nthreads`.

**Example**

```julia
using QuboSolver.Solvers.BruteForce

problem = QuboProblem([0.0 1.0; 1.0 0.0], [1.0, 0.0])
solution = solve!(problem, BruteForce_solver())

# output

🟦🟦 - Energy: -3.0 - Solver: BruteForce_solver - Metadata count: 2
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/LorenzoFioroni/QuboSolver.jl/blob/e65c4543ed7b54db05ab960d3f4e066780747dc8/src/solvers/BruteForce.jl#L40-L70" target="_blank" rel="noreferrer">source</a></Badge>

</details>


### Gurobi solver {#doc-API:Gurobi}


<details class='jldocstring custom-block' open>
<summary><a id='QuboSolver.Solvers.GurobiLib.Gurobi_solver' href='#QuboSolver.Solvers.GurobiLib.Gurobi_solver'><span class="jlbinding">QuboSolver.Solvers.GurobiLib.Gurobi_solver</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
struct Gurobi_solver <: AbstractSolver end
```


Gurobi solver for QUBO problems.

Use the Gurobi optimization library [[5](/resources/bibliography#gurobioptimizationllcGurobiOptimizerReference2024)] to solve the QUBO problem.

::: warning Warning

The Gurobi solver requires a valid Gurobi license and the Gurobi binaries to be installed. See the [Gurobi website](https://www.gurobi.com) for more information.

:::

::: warning Warning

To use this solver, you need to explicitly import the `GurobiLib` module in your code:

```julia
using QuboSolver.Solvers.GurobiLib
```


:::


<Badge type="info" class="source-link" text="source"><a href="https://github.com/LorenzoFioroni/QuboSolver.jl/blob/e65c4543ed7b54db05ab960d3f4e066780747dc8/src/solvers/GurobiLib.jl#L8-L24" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuboSolver.solve!-Tuple{QuboProblem, QuboSolver.Solvers.GurobiLib.Gurobi_solver}' href='#QuboSolver.solve!-Tuple{QuboProblem, QuboSolver.Solvers.GurobiLib.Gurobi_solver}'><span class="jlbinding">QuboSolver.solve!</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
solve!(problem::QuboProblem, solver::AbstractSolver, args...; kwargs...)
```


Solve the QuboProblem `problem` using the provided `solver`.

The `solver` instance must implement the `solve!` method.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/LorenzoFioroni/QuboSolver.jl/blob/e65c4543ed7b54db05ab960d3f4e066780747dc8/src/problem.jl#L177-L183" target="_blank" rel="noreferrer">source</a></Badge>



```julia
function solve!(
    problem::QuboProblem,
    solver::Gurobi_solver;
    mipgap::Float64 = 0.0,
    timelimit::Float64 = Inf64,
    output::Bool = true,
    threads::Int = Threads.nthreads(),
)
```


Solve the QuboProblem `problem` using the Gurobi library [[5](/resources/bibliography#gurobioptimizationllcGurobiOptimizerReference2024)].

::: warning Warning

The Gurobi solver requires a valid Gurobi license.  See the [Gurobi website](https://www.gurobi.com) for more information.

:::

::: warning Warning

To use this solver, you need to explicitly import the `GurobiLib` module in your code:

```julia
using QuboSolver.Solvers.GurobiLib
```


:::

**Arguments**
- `problem::QuboProblem`: The QUBO problem to be solved.
  
- `solver::Gurobi_solver`: Instance of [`Gurobi_solver`](/resources/api#QuboSolver.Solvers.GurobiLib.Gurobi_solver).
  
- `mipgap::Float64`: The maximum allowed gap between the found lower bound and the upper bound.    The default value is `0.0`, which means that the solver will try to find the optimal solution.
  
- `timelimit::Float64`: The maximum time limit for the solver in seconds. The default value is `Inf64`,
  
- `output::Bool`: Whether to print the solver output. The default value is `true`.
  
- `threads::Int`: The number of threads to use for the solver. The default value is the number of    available threads on the system.
  

**Returns**

The optimal solution found by the solver. Metadata include the runtime as `runtime` and the  maximum gap as `mipgap`.

**Example**

```julia
using QuboSolver.Solvers.GurobiLib

problem = QuboProblem([0.0 1.0; 1.0 0.0], [1.0, 0.0])
solution = solve!(problem, Gurobi_solver(), output=false)

# output

🟦🟦 - Energy: -3.0 - Solver: Gurobi_solver - Metadata count: 2
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/LorenzoFioroni/QuboSolver.jl/blob/e65c4543ed7b54db05ab960d3f4e066780747dc8/src/solvers/GurobiLib.jl#L27-L74" target="_blank" rel="noreferrer">source</a></Badge>

</details>


### TAMC solver {#doc-API:TAMC}


<details class='jldocstring custom-block' open>
<summary><a id='QuboSolver.Solvers.PTICM.PTICM_solver' href='#QuboSolver.Solvers.PTICM.PTICM_solver'><span class="jlbinding">QuboSolver.Solvers.PTICM.PTICM_solver</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
struct PTICM_solver <: AbstractSolver end
```


Tempering and Annealing Monte Carlo solver for QUBO problems.

Use the TAMC library [[3](/resources/bibliography#bauzaScalingAdvantageApproximate2024)] to solve the QUBO problem.

::: warning Warning

The TAMC solver requires the TAMC binaries to be installed. See the [https://github.com/USCqserver/tamc](https://github.com/USCqserver/tamc) for more information.

:::

::: warning Warning

To use this solver, you need to explicitly import the `PTICM` module in your code:

```julia
using QuboSolver.Solvers.PTICM
```


:::


<Badge type="info" class="source-link" text="source"><a href="https://github.com/LorenzoFioroni/QuboSolver.jl/blob/e65c4543ed7b54db05ab960d3f4e066780747dc8/src/solvers/PTICM.jl#L7-L23" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuboSolver.solve!-Tuple{QuboProblem, QuboSolver.Solvers.PTICM.PTICM_solver}' href='#QuboSolver.solve!-Tuple{QuboProblem, QuboSolver.Solvers.PTICM.PTICM_solver}'><span class="jlbinding">QuboSolver.solve!</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
solve!(problem::QuboProblem, solver::AbstractSolver, args...; kwargs...)
```


Solve the QuboProblem `problem` using the provided `solver`.

The `solver` instance must implement the `solve!` method.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/LorenzoFioroni/QuboSolver.jl/blob/e65c4543ed7b54db05ab960d3f4e066780747dc8/src/problem.jl#L177-L183" target="_blank" rel="noreferrer">source</a></Badge>



```julia
function solve!(
    problem::QuboProblem{T,TW,Tc},
    solver::PTICM_solver;
    MCS::Integer = 1000,
    warmup_fraction::Real = 0.5,
    betas::Vector{T} = T.(QuboSolver.LogRange(0.1, 5.0, 32)),
    lo_num_beta::Integer = 8,
    threads::Integer = Threads.nthreads(),
    num_replica_chains::Integer = 2,
) where {T<:AbstractFloat,TW<:AbstractMatrix{T},Tc<:Union{Nothing,AbstractVector{T}}}
```


Solve the QuboProblem `problem` using the TAMC solver [[3](/resources/bibliography#bauzaScalingAdvantageApproximate2024)].

::: warning Warning

The TAMC solver requires the TAMC binaries to be installed. See the [https://github.com/USCqserver/tamc](https://github.com/USCqserver/tamc) for more information.

:::

::: warning Warning

To use this solver, you need to explicitly import the `PTICM` module in your code:

```julia
using QuboSolver.Solvers.PTICM
```


:::

**Arguments**
- `problem::QuboProblem`: The QUBO problem to be solved.
  
- `solver::PTICM_solver`: Instance of [`PTICM_solver`](/resources/api#QuboSolver.Solvers.PTICM.PTICM_solver).
  
- `MCS::Integer`: The number of Monte Carlo steps. The default value is `1000`.
  
- `warmup_fraction::Real`: The fraction of Monte Carlo steps to be used for warmup. The default value is    `0.5`.
  
- `betas::Vector{T}`: The inverse temperatures to be used. The default value is a logarithmic range   between `0.1` and `5.0` with `32` elements.
  
- `lo_num_beta::Integer`: The number of inverse temperatures to be used for the isoenergetic cluster   moves. The default value is `8`.
  
- `threads::Integer`: The number of threads to be used. The default value is the number of available   threads.
  
- `num_replica_chains::Integer`: The number of replica chains to be used. The default value is `2`.
  

**Returns**

The optimal solution found by the solver. Metadata include the runtime as `runtime`.

**Example**

```julia
using QuboSolver.Solvers.PTICM

problem = QuboProblem([0.0 1.0; 1.0 0.0], [1.0, 0.0])
solution = solve!(problem, PTICM_solver())

# output

🟦🟦 - Energy: -3.0 - Solver: PTICM_solver - Metadata count: 1
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/LorenzoFioroni/QuboSolver.jl/blob/e65c4543ed7b54db05ab960d3f4e066780747dc8/src/solvers/PTICM.jl#L114-L166" target="_blank" rel="noreferrer">source</a></Badge>

</details>


### Simulated annealing solver {#doc-API:SA}


<details class='jldocstring custom-block' open>
<summary><a id='QuboSolver.Solvers.SA.SA_solver' href='#QuboSolver.Solvers.SA.SA_solver'><span class="jlbinding">QuboSolver.Solvers.SA.SA_solver</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
struct SA_solver <: AbstractSolver end
```


Simulated Annealing solver for QUBO problems [[4](/resources/bibliography#kirkpatrickOptimizationSimulatedAnnealing1983)].

::: warning Warning

To use this solver, you need to explicitly import the `SA` module in your code:

```julia
using QuboSolver.Solvers.SA
```


:::


<Badge type="info" class="source-link" text="source"><a href="https://github.com/LorenzoFioroni/QuboSolver.jl/blob/e65c4543ed7b54db05ab960d3f4e066780747dc8/src/solvers/SA.jl#L9-L19" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuboSolver.solve!-Tuple{QuboProblem, QuboSolver.Solvers.SA.SA_solver}' href='#QuboSolver.solve!-Tuple{QuboProblem, QuboSolver.Solvers.SA.SA_solver}'><span class="jlbinding">QuboSolver.solve!</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
function solve!(
    problem::QuboProblem,
    solver::SA_solver;
    MCS::Integer = 1000,
    rng::Random.AbstractRNG = Random.GLOBAL_RNG,
    initial_conf::Union{AbstractVector{Int8},Nothing} = nothing,
    initial_temp::Real = 1.0,
    progressbar::Bool = true,
)
```


Solve the QuboProblem `problem` using simulated annealing [[4](/resources/bibliography#kirkpatrickOptimizationSimulatedAnnealing1983)] with a linearly decreasing temperature  schedule. 

::: warning Warning

To use this solver, you need to explicitly import the `SA` module in your code:

```julia
using QuboSolver.Solvers.SA
```


:::

**Arguments**
- `problem::QuboProblem`: [`QuboProblem`](/resources/api#QuboSolver.QuboProblem) instance to be solved.
  
- `solver::SA_solver`: Instance of [`SA_solver`](/resources/api#QuboSolver.Solvers.SA.SA_solver).
  
- `MCS::Integer`: Number of Monte Carlo steps. The default value is `1000`.
  
- `rng::Random.AbstractRNG`: Random number generator. The default value is `Random.GLOBAL_RNG`.
  
- `initial_conf::Union{AbstractVector{Int8},Nothing}`: Initial configuration. If `nothing`, a random    configuration is generated. The default value is `nothing`.
  
- `initial_temp::Real`: Initial temperature. The default value is `1.0`.
  
- `progressbar::Bool`: Whether to show a progress bar. The default value is `true`.
  

**Returns**

The optimal solution found by the solver. Metadata include the runtime as `runtime`.

**Example**

```julia
using QuboSolver.Solvers.SA

problem = QuboProblem([0.0 1.0; 1.0 0.0], [1.0, 0.0])
solution = solve!(problem, SA_solver())

# output

🟦🟦 - Energy: -3.0 - Solver: SA_solver - Metadata count: 1
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/LorenzoFioroni/QuboSolver.jl/blob/e65c4543ed7b54db05ab960d3f4e066780747dc8/src/solvers/SA.jl#L24-L68" target="_blank" rel="noreferrer">source</a></Badge>

</details>


### Local Quantum Annealing solver {#doc-API:LQA}


<details class='jldocstring custom-block' open>
<summary><a id='QuboSolver.Solvers.LQA.LQA_solver' href='#QuboSolver.Solvers.LQA.LQA_solver'><span class="jlbinding">QuboSolver.Solvers.LQA.LQA_solver</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
struct LQA_solver <: AbstractSolver end
```


Variational solver for QUBO problems using produce states.

Use the Local Quantum Annealing algorithm [[2](/resources/bibliography#bowlesQuadraticUnconstrainedBinary2022)] to solve the QUBO problem. The analytical form of the  states in LQA is 

$$\ket{ψ} = ⊗_{i=1}^N \left( \cos(θ/2) \ket{0} + \sin(θ/2)\ket{1} \right),$$

where the variational parameters $θ$ are further parameterized as $θ = π/2 * tanh(z)$.

The variational states in LQA are thus product states.

::: tip Tip

For more information on the LQA algorithm, see the original paper at [https://doi.org/10.1103/PhysRevApplied.18.034016](https://doi.org/10.1103/PhysRevApplied.18.034016).

:::

::: warning Warning

To use this solver, you need to explicitly import the `LQA` module in your code:

```julia
using QuboSolver.Solvers.LQA
```


:::


<Badge type="info" class="source-link" text="source"><a href="https://github.com/LorenzoFioroni/QuboSolver.jl/blob/e65c4543ed7b54db05ab960d3f4e066780747dc8/src/solvers/LQA.jl#L10-L33" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuboSolver.Solvers.LQA.SignRounding' href='#QuboSolver.Solvers.LQA.SignRounding'><span class="jlbinding">QuboSolver.Solvers.LQA.SignRounding</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
struct SignRounding <: RoundingMethod end
```


Discretization method that discretizes the quantum state to a binary configuration setting $b_i = \text{sign}(σ_z)$.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/LorenzoFioroni/QuboSolver.jl/blob/e65c4543ed7b54db05ab960d3f4e066780747dc8/src/solvers/LQA.jl#L46-L50" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuboSolver.Solvers.LQA.SequentialRounding' href='#QuboSolver.Solvers.LQA.SequentialRounding'><span class="jlbinding">QuboSolver.Solvers.LQA.SequentialRounding</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
struct SequentialRounding <: RoundingMethod end
```


Discretization method that implements the sequential rounding algorithm presented in [https://doi.org/10.1137/20M132016X](https://doi.org/10.1137/20M132016X).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/LorenzoFioroni/QuboSolver.jl/blob/e65c4543ed7b54db05ab960d3f4e066780747dc8/src/solvers/LQA.jl#L53-L57" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuboSolver.solve!-Tuple{QuboProblem, QuboSolver.Solvers.LQA.LQA_solver}' href='#QuboSolver.solve!-Tuple{QuboProblem, QuboSolver.Solvers.LQA.LQA_solver}'><span class="jlbinding">QuboSolver.solve!</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
solve!(problem::QuboProblem, solver::AbstractSolver, args...; kwargs...)
```


Solve the QuboProblem `problem` using the provided `solver`.

The `solver` instance must implement the `solve!` method.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/LorenzoFioroni/QuboSolver.jl/blob/e65c4543ed7b54db05ab960d3f4e066780747dc8/src/problem.jl#L177-L183" target="_blank" rel="noreferrer">source</a></Badge>



```julia
function solve!(
    problem::QuboProblem{T,Tw,Tc},
    solver::LQA_solver;
    rng::Random.AbstractRNG = Random.GLOBAL_RNG,
    initial_conf::Union{AbstractVector{T},Nothing} = nothing,
    iterations::Int = 1000,
    inner_iterations::Int = 1,
    opt::Optimisers.AbstractRule = Adam(),
    tf::T = one(T),
    rounding::Union{RoundingMethod,Tuple{Vararg{RoundingMethod}}} = SignRounding(),
    save_params::Bool = false,
    save_energy::Bool = false,
    progressbar::Bool = true,
) where {T<:AbstractFloat,Tw<:AbstractMatrix{T},Tc<:Union{Nothing,AbstractVector{T}}}
```


Solve the QUBO problem using the Local Quantum Annealing solver [[2](/resources/bibliography#bowlesQuadraticUnconstrainedBinary2022)].

::: tip Tip

For more information on the LQA algorithm, see the original paper at [https://doi.org/10.1103/PhysRevApplied.18.034016](https://doi.org/10.1103/PhysRevApplied.18.034016).

:::

::: warning Warning

To use this solver, you need to explicitly import the `LQA` module in your code:

```julia
using QuboSolver.Solvers.LQA
```


:::

**Arguments**
- `problem`: [`QuboProblem`](/resources/api#QuboSolver.QuboProblem) object.
  
- `solver`: [`LQA_solver`](/resources/api#QuboSolver.Solvers.LQA.LQA_solver) object.
  
- `rng`: Random number generator to use (default: `Random.GLOBAL_RNG`).
  
- `initial_conf`: Initial configuration (default: `nothing` for random initialization).
  
- `iterations`: Number of time steps (default: 1000).
  
- `inner_iterations`: Number of gradient descent steps each time step (default: 1).
  
- `opt`: Optimizer to use (default: `Adam()`).
  
- `tf`: Transverse field strength (default: 1.0).
  
- `rounding`: Rounding method used to obtain the classical configuration from the GCS   state (default: [`SignRounding`](/resources/api#QuboSolver.Solvers.LQA.SignRounding)). Accepts both a single method or a tuple of methods.
  
- `save_params`: Save the parameters of the variational state (default: `false`).
  
- `save_energy`: Save the energy of the variational state (default: `false`).
  
- `progressbar`: Show a progress bar (default: `true`).
  

**Returns**

A [`Solution`](/resources/api#QuboSolver.Solution) object containing the optimal solution found by the solver. Metadata include the runtime as `runtime`, the parameters of the variational state as `params` (if  `save_params` is `true`), and the energy of the variational state as `energy` (if `save_energy`  is `true`).

**Example**

```julia
using QuboSolver.Solvers.LQA

problem = QuboProblem([0.0 1.0; 1.0 0.0], [1.0, 0.0])
solution = solve!(problem, LQA_solver(); save_params=true, save_energy=true, progressbar=false)

# output

🟦🟦 - Energy: -3.0 - Solver: LQA_solver - Metadata count: 3
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/LorenzoFioroni/QuboSolver.jl/blob/e65c4543ed7b54db05ab960d3f4e066780747dc8/src/solvers/LQA.jl#L253-L313" target="_blank" rel="noreferrer">source</a></Badge>

</details>


## Utilities {#doc-API:Utilities}

### Random generation of QUBO problems {#Random-generation-of-QUBO-problems}
<details class='jldocstring custom-block' open>
<summary><a id='QuboSolver.SherringtonKirkpatrick' href='#QuboSolver.SherringtonKirkpatrick'><span class="jlbinding">QuboSolver.SherringtonKirkpatrick</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
struct SherringtonKirkpatrick <: QuboProblemClass end
```


An instance of `QuboProblemClass` representing the Sherrington-Kirkpatrick model.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/LorenzoFioroni/QuboSolver.jl/blob/e65c4543ed7b54db05ab960d3f4e066780747dc8/src/random.jl#L10-L14" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuboSolver.Chimera' href='#QuboSolver.Chimera'><span class="jlbinding">QuboSolver.Chimera</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
struct Chimera <: QuboProblemClass end
```


An instance of `QuboProblemClass` representing the Chimera model.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/LorenzoFioroni/QuboSolver.jl/blob/e65c4543ed7b54db05ab960d3f4e066780747dc8/src/random.jl#L24-L28" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuboSolver.EdwardsAnderson' href='#QuboSolver.EdwardsAnderson'><span class="jlbinding">QuboSolver.EdwardsAnderson</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
struct EdwardsAnderson <: QuboProblemClass end
```


An instance of `QuboProblemClass` representing the Edwards-Anderson model.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/LorenzoFioroni/QuboSolver.jl/blob/e65c4543ed7b54db05ab960d3f4e066780747dc8/src/random.jl#L17-L21" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='Base.rand' href='#Base.rand'><span class="jlbinding">Base.rand</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
function rand(
    ::SherringtonKirkpatrick, 
    N::Int; 
    rng::AbstractRNG = Random.GLOBAL_RNG,
    eltype::Type = Float64
)
```


Generate a random QUBO matrix for the Sherrington-Kirkpatrick model.

**Arguments**
- `N::Int`: Number of variables.
  
- `rng::AbstractRNG`: Random number generator (default: `Random.GLOBAL_RNG`).
  
- `eltype::Type`: Element type of the matrix elements (default: `Float64`).
  

**Example**

```julia
W = rand(SherringtonKirkpatrick(), 4)
println(size(W))
println(all(diag(W) .== 0))
println(W == transpose(W))

# output

(4, 4)
true
true
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/LorenzoFioroni/QuboSolver.jl/blob/e65c4543ed7b54db05ab960d3f4e066780747dc8/src/random.jl#L31-L61" target="_blank" rel="noreferrer">source</a></Badge>



```julia
function rand(
    ::EdwardsAnderson, 
    N_side::Int; 
    rng::AbstractRNG = Random.GLOBAL_RNG,
    sparse::Bool = false,
    eltype::Type = Float64
)
```


Generate a random QUBO matrix for the 3D Edwards-Anderson model with open boundary conditions.

**Arguments**
- `N_side::Int`: Side length of the cubic lattice.
  
- `rng::AbstractRNG`: Random number generator (default: `Random.GLOBAL_RNG`).
  
- `sparse::Bool`: If true, generate a sparse matrix (default: `false`).
  
- `eltype::Type`: Element type of the matrix elements (default: `Float64`).
  

**Example**

```julia
W = rand(EdwardsAnderson(), 4; sparse = true) # 4x4x4 lattice
println(size(W))
println(typeof(W))
println(all(diag(W) .== 0))
println(W == transpose(W))

# output

(64, 64)
SparseMatrixCSC{Float64, Int64}
true
true
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/LorenzoFioroni/QuboSolver.jl/blob/e65c4543ed7b54db05ab960d3f4e066780747dc8/src/random.jl#L74-L108" target="_blank" rel="noreferrer">source</a></Badge>



```julia
function rand(
    ::Chimera, 
    N_rows::Int, 
    N_cols::Int, 
    N_spin_layer::Int = 4; 
    rng::AbstractRNG = Random.GLOBAL_RNG,
    sparse::Bool = false,
    eltype::Type = Float64
)
```


Generate a random QUBO matrix for the Chimera model.

**Arguments**
- `N_rows::Int`: Number of rows in the Chimera lattice.
  
- `N_cols::Int`: Number of columns in the Chimera lattice.
  
- `N_spin_layer::Int`: Number of spins per unit cell (default: 4).
  
- `rng::AbstractRNG`: Random number generator (default: `Random.GLOBAL_RNG`).
  
- `sparse::Bool`: If true, generate a sparse matrix (default: `false`).
  
- `eltype::Type`: Element type of the matrix elements (default: `Float64`).
  

**Example**

```julia
W = rand(Chimera(), 4, 4, 4; sparse = true) # 4x4x(4x2) lattice
println(size(W))
println(typeof(W))
println(all(diag(W) .== 0))
println(W == transpose(W))

# output

(128, 128)
SparseMatrixCSC{Float64, Int64}
true
true
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/LorenzoFioroni/QuboSolver.jl/blob/e65c4543ed7b54db05ab960d3f4e066780747dc8/src/random.jl#L136-L174" target="_blank" rel="noreferrer">source</a></Badge>

</details>


### Utility functions {#Utility-functions}
<details class='jldocstring custom-block' open>
<summary><a id='QuboSolver.dense_similar' href='#QuboSolver.dense_similar'><span class="jlbinding">QuboSolver.dense_similar</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
function dense_similar(A::AbstractArray, args...)
function dense_similar(A::AbstractSparseMatrix, args...)
```


Create a dense array similar to the input `A`.

Eventual additional arguments are passed to the `similar` function.

**Example**

```julia
A = sprand(Float32, 10, 10, 0.1)
B = dense_similar(A)
println(size(B)) 
println(eltype(B))
println(typeof(B))

# output

(10, 10)
Float32
Matrix{Float32}
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/LorenzoFioroni/QuboSolver.jl/blob/e65c4543ed7b54db05ab960d3f4e066780747dc8/src/utilities.jl#L10-L32" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuboSolver.similar_named_tuple' href='#QuboSolver.similar_named_tuple'><span class="jlbinding">QuboSolver.similar_named_tuple</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
function similar_named_tuple(a::NamedTuple)
```


Create a new NamedTuple by recursively applying `similar` to each element of the input `a`.

**Example**

```julia
a = (x = rand(3), y = rand(Float32, 2, 2))
b = similar_named_tuple(a)
println(typeof(b) == typeof(a))
println(size(b.x) == size(a.x))
println(size(b.y) == size(a.y))

# output

true
true
true
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/LorenzoFioroni/QuboSolver.jl/blob/e65c4543ed7b54db05ab960d3f4e066780747dc8/src/utilities.jl#L36-L55" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuboSolver.conf_int2spin' href='#QuboSolver.conf_int2spin'><span class="jlbinding">QuboSolver.conf_int2spin</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
function conf_int2spin(n::Integer, pad::Integer)
```


Create an n-long vector containing the binary representation of the integer `n` as $1$ and $-1$.

See also [`conf_int2spin!`](/resources/api#QuboSolver.conf_int2spin!)

**Example**

```julia
a = conf_int2spin(5, 7)
println(a) 

# output

Int8[1, -1, 1, -1, -1, -1, -1]
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/LorenzoFioroni/QuboSolver.jl/blob/e65c4543ed7b54db05ab960d3f4e066780747dc8/src/utilities.jl#L88-L104" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuboSolver.conf_int2spin!' href='#QuboSolver.conf_int2spin!'><span class="jlbinding">QuboSolver.conf_int2spin!</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
function conf_int2spin!(a::AbstractVector{<:Integer}, n::Integer)
```


Fill the vector `a` with the binary representation of the integer `n` as $1$ and $-1$.

See also [`conf_int2spin`](/resources/api#QuboSolver.conf_int2spin)

**Example**

```julia
a = Vector{Int8}(undef, 4)
conf_int2spin!(a, 5)
println(a) 

# output

Int8[1, -1, 1, -1]
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/LorenzoFioroni/QuboSolver.jl/blob/e65c4543ed7b54db05ab960d3f4e066780747dc8/src/utilities.jl#L58-L75" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuboSolver.binary_to_spin' href='#QuboSolver.binary_to_spin'><span class="jlbinding">QuboSolver.binary_to_spin</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
function binary_to_spin(
    W::AbstractMatrix{T}, 
    bias::Union{Nothing,<:AbstractVector{T}} = nothing
) where {T<:AbstractFloat}
```


Convert a QUBO matrix `W` and an optional bias vector `bias` from binary to spin representation.

**Returns**
- `J::AbstractMatrix{T}`: The spin coupling matrix.
  
- `c::Union{Nothing,<:AbstractVector{T}}`: The bias vector.
  

**Example**

```julia
W = [0.0 1.0; 1.0 0.0]
bias = [1.0, 0.0]
J, c = binary_to_spin(W, bias)
println(J)
println(c)

# output

[0.0 0.25; 0.25 0.0]
[1.0, 0.5]
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/LorenzoFioroni/QuboSolver.jl/blob/e65c4543ed7b54db05ab960d3f4e066780747dc8/src/utilities.jl#L111-L136" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuboSolver.spin_to_binary' href='#QuboSolver.spin_to_binary'><span class="jlbinding">QuboSolver.spin_to_binary</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
function spin_to_binary(
    J::AbstractMatrix{T}, 
    c::Union{Nothing,<:AbstractVector{T}} = nothing
) where {T<:AbstractFloat}
```


Convert a QUBO matrix `J` and an optional bias vector `c` from spin to binary representation.

**Returns**
- `W::AbstractMatrix{T}`: The binary coupling matrix.
  
- `bias::Union{Nothing,<:AbstractVector{T}}`: The bias vector.
  

**Example**

```julia
J = [0.0 0.25; 0.25 0.0]
c = [1.0, 0.5]
W, bias = spin_to_binary(J, c)
println(W)
println(bias)

# output

[0.0 1.0; 1.0 0.0]
[1.0, 0.0]
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/LorenzoFioroni/QuboSolver.jl/blob/e65c4543ed7b54db05ab960d3f4e066780747dc8/src/utilities.jl#L149-L174" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuboSolver.nonzero_triu' href='#QuboSolver.nonzero_triu'><span class="jlbinding">QuboSolver.nonzero_triu</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
function nonzero_triu(A::AbstractMatrix; skip_zeros = true)
```


Extract the coordinates and values of the non-zero elements in the upper triangular part of a matrix `A`.

**Arguments**
- `A::AbstractMatrix`: The input matrix.
  
- `skip_zeros::Bool`: If true, skip zero values (default: true).
  

**Returns**
- `coo::Vector{Tuple{Int,Int}}`: A vector of tuples representing the coordinates of the non-zero elements.
  
- `compact::Vector{eltype(A)}`: A vector of the non-zero values.
  

**Example**

```julia
W = [0.0 1.0 0.0; 1.0 0.0 2.0; 0.0 2.0 0.0]
coo, compact = nonzero_triu(W)
println(coo)
println(compact)

# output

[(1, 2), (2, 3)]
[1.0, 2.0]
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/LorenzoFioroni/QuboSolver.jl/blob/e65c4543ed7b54db05ab960d3f4e066780747dc8/src/utilities.jl#L194-L219" target="_blank" rel="noreferrer">source</a></Badge>



```julia
function nonzero_triu(A::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
```


Extract the coordinates and values of the non-zero elements in the upper triangular part of a sparse matrix `A`.

The diagonal elements are not included.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/LorenzoFioroni/QuboSolver.jl/blob/e65c4543ed7b54db05ab960d3f4e066780747dc8/src/utilities.jl#L242-L248" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='QuboSolver.drop_target_sparsity' href='#QuboSolver.drop_target_sparsity'><span class="jlbinding">QuboSolver.drop_target_sparsity</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
function drop_target_sparsity(
    W::AbstractMatrix, 
    target_sparsity::Real; 
    max_depth = 30
)
```


Drop the smallest elements of the matrix `W` until the target sparsity is reached.

**Arguments**
- `W::AbstractMatrix`: The input matrix.
  
- `target_sparsity::Real`: The target sparsity level (between 0 and 1).
  
- `max_depth::Int`: The maximum recursion depth (default: 30).
  

**Returns**

A copy of `W` with the smallest elements dropped to reach the target sparsity.

**Example**

```julia
A = randn(1000, 1000)
target_sparsity = 0.34
B = sparse(drop_target_sparsity(A, target_sparsity; max_depth = 50))
println(round(nnz(B)/length(B), digits = 2))

# output

0.34
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/LorenzoFioroni/QuboSolver.jl/blob/e65c4543ed7b54db05ab960d3f4e066780747dc8/src/utilities.jl#L312-L340" target="_blank" rel="noreferrer">source</a></Badge>

</details>

