module LQA

using ..QuboSolver
using Random
using Optimisers
using ProgressMeter

export LQA_solver, solve!, SignRounding, SequentialRounding

@doc raw"""
    struct LQA_solver <: AbstractSolver end

Variational solver for QUBO problems using produce states.

Use the Local Quantum Annealing algorithm [bowlesQuadraticUnconstrainedBinary2022](@cite) to solve the QUBO problem. The analytical form of the 
states in LQA is 
```math
\ket{ψ} = ⊗_{i=1}^N \left( \cos(θ/2) \ket{0} + \sin(θ/2)\ket{1} \right),
```
where the variational parameters ``θ`` are further parameterized as ``θ = π/2 * tanh(z)``.

The variational states in LQA are thus product states.

!!! tip 
    For more information on the LQA algorithm, see the original paper at
    [https://doi.org/10.1103/PhysRevApplied.18.034016](https://doi.org/10.1103/PhysRevApplied.18.034016).

!!! warning
    To use this solver, you need to explicitely import the `LQA` module in your code:
    ```julia
    using QuboSolver.Solvers.LQA
    ```
"""
struct LQA_solver <: AbstractSolver end

@doc raw"""
    abstract type RoundingMethod end

Abstract type for a discretization method.

Concrete types should be defined for each discretization method implemented, and they should 
define the `round_configuration` function.
"""
abstract type RoundingMethod end

@doc raw"""
    struct SignRounding <: RoundingMethod end

Discretization method that discretizes the quantum state to a binary configuration setting ``b_i = \text{sign}(σ_z)``.
"""
struct SignRounding <: RoundingMethod end

@doc raw"""
    struct SequentialRounding <: RoundingMethod end

Discretization method that implements the sequential rounding algorithm presented in [https://doi.org/10.1137/20M132016X](https://doi.org/10.1137/20M132016X).
"""
struct SequentialRounding <: RoundingMethod end

@doc raw"""
    function preprocess_input!(z::Vector{T}, initial_conf::Vector{T}) where {T<:AbstractFloat}

Preprocess the input configuration to the variational solver.

Process the input configuration `initial_conf` (containing the θ parameters of the state) and 
transform it to the `z` variable used in the variational solver. The variables are related by 
the relation ``θ = π/2 \tanh(z)``. 
"""
preprocess_input!(z::Vector{T}, initial_conf::Vector{T}) where {T<:AbstractFloat} =
    z .= atanh.(2 ./ pi .* initial_conf)

@doc raw"""
    function compute_trig!(sines::Vector{T}, cosines::Vector{T}, z::Vector{T}) where {T<:AbstractFloat}

Compute the sine and cosine of the variational parameters and store them in the `sines` and 
`cosines` vectors.
"""
function compute_trig!(
    sines::Vector{T},
    cosines::Vector{T},
    z::Vector{T},
) where {T<:AbstractFloat}
    sines .= sin.(pi / 2 * tanh.(z))
    cosines .= cos.(pi / 2 * tanh.(z))
    return nothing
end

@doc raw"""
    function compute_trig!(sines::Vector{T}, cosines::Vector{T}, z::Vector{T}) where {T<:AbstractFloat}

Compute the sine and cosine of the variational parameters and return them
"""
function compute_trig(z::AbstractVector{<:AbstractFloat})
    sines = similar(z)
    cosines = similar(z)
    compute_trig!(sines, cosines, z)
    return sines, cosines
end

@doc raw"""
    ease(x::AbstractFloat) = acos(1 - 2x) / π

Easing function for the time variable `x ∈ [0, 1]`.
"""
ease(x::AbstractFloat) = acos(1 - 2x) / π

@doc raw"""
    a(t::AbstractFloat) = 1 - ease(t)
    
Time-dependent coefficient for the transverse field term in the loss function.
"""
a(t::AbstractFloat) = 1 - ease(t)

@doc raw"""
    b(t::AbstractFloat) = ease(t)

Time-dependent coefficient for the Ising term in the loss function.
"""
b(t::AbstractFloat) = ease(t)

@doc raw"""
    function loss(
        problem::QuboProblem,
        sines::Vector{T},
        cosines::Vector{T},
        time::T,
        tf::T,
    ) where {T<:AbstractFloat}

Compute the loss function for the QUBO problem.

## Arguments
- `problem`: [`QuboProblem`](@ref QuboSolver.QuboProblem) object.
- `sines`: Vector of sine values of the variational parameters.
- `cosines`: Vector of cosine values of the variational parameters.
- `time`: time identifying the current Hamiltonian
- `tf`: transverse field strength.
"""
function loss(
    problem::QuboProblem,
    sines::Vector{T},
    cosines::Vector{T},
    time::T,
    tf::T,
) where {T<:AbstractFloat}
    E = -tf * a(time) * sum(cosines) - b(time) * dot(sines, problem.W * sines)
    problem.has_bias && (E -= b(time) * dot(problem.c, sines))
    return E
end

@doc raw"""
    function grad!(
        dz::Vector{T},
        problem::QuboProblem,
        z::Vector{T},
        sines::Vector{T},
        cosines::Vector{T},
        time::Real,
        tf::Real,
    ) where {T<:AbstractFloat}

Compute the gradient of the loss function with respect to the variational parameters.

## Arguments
- `dz`: Vector to store the gradient.
- `problem`: [`QuboProblem`](@ref QuboSolver.QuboProblem) object.
- `z`: Vector of variational parameters.
- `sines`: Vector of sine values of the variational parameters.
- `cosines`: Vector of cosine values of the variational parameters.
- `time`: time identifying the current Hamiltonian
- `tf`: transverse field strength.
"""
function grad!(
    dz::Vector{T},
    problem::QuboProblem,
    z::Vector{T},
    sines::Vector{T},
    cosines::Vector{T},
    time::Real,
    tf::Real,
) where {T<:AbstractFloat}
    @. dz =
        pi / 2 * (
            (tf * a(time) * sines - b(time) * (2 * ($*)(problem.W, sines) * cosines)) *
            (1 - tanh(z)^2)
        )

    if problem.has_bias
        @. dz -= pi / 2 * b(time) * problem.c * cosines * (1 - tanh(z)^2)
    end
end

@doc raw"""
    function round_configuration(
        problem::QuboProblem,
        z::Vector{T},
        ::SignRounding
    ) where {T<:AbstractFloat}

Get a classical configuration from the variational parameters using the SignRounding method.

## Arguments
- `problem`: [`QuboProblem`](@ref QuboSolver.QuboProblem) object.
- `z`: Vector of variational parameters.
- `::SignRounding`: Rounding method to use.
"""
function round_configuration(
    ::QuboProblem,
    z::AbstractVector{<:AbstractFloat},
    ::SignRounding,
)
    conf = similar(z, Int8)
    conf .= Int8.(sign.(sin.(pi / 2 .* tanh.(z))))

    conf[conf.==0] .= 1
    return conf
end

@doc raw"""
    function round_configuration(
        problem::QuboProblem,
        z::Vector{T},
        ::SequentialRounding
    ) where {T<:AbstractFloat}

Get a classical configuration from the variational parameters using the SequentialRounding method.

## Arguments
- `problem`: [`QuboProblem`](@ref QuboSolver.QuboProblem) object.
- `z`: Vector of variational parameters.
- `::SequentialRounding`: Rounding method to use.
"""
function round_configuration(
    problem::QuboProblem,
    z::AbstractVector{<:AbstractFloat},
    ::SequentialRounding,
)
    problem.has_bias && throw(
        ArgumentError("Sequential rounding is not supported for problems with bias terms"),
    )

    mean_z = sin.(pi / 2 .* tanh.(z))
    for i ∈ 1:problem.N
        mean_z[i] = sign(dot(problem.W[:, i], mean_z))
    end
    conf = Int8.(mean_z)

    conf[conf.==0] .= 1

    return conf
end

@doc raw"""
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

Solve the QUBO problem using the Local Quantum Annealing solver [bowlesQuadraticUnconstrainedBinary2022](@cite).

!!! tip 
    For more information on the LQA algorithm, see the original paper at
    [https://doi.org/10.1103/PhysRevApplied.18.034016](https://doi.org/10.1103/PhysRevApplied.18.034016).

!!! warning
    To use this solver, you need to explicitely import the `LQA` module in your code:
    ```julia
    using QuboSolver.Solvers.LQA
    ```

## Arguments
- `problem`: [`QuboProblem`](@ref QuboSolver.QuboProblem) object.
- `solver`: [`LQA_solver`](@ref QuboSolver.Solvers.LQA.LQA_solver) object.
- `rng`: Random number generator to use (default: `Random.GLOBAL_RNG`).
- `initial_conf`: Initial configuration (default: `nothing` for random initialization).
- `iterations`: Number of time steps (default: 1000).
- `inner_iterations`: Number of gradient descent steps each time step (default: 1).
- `opt`: Optimizer to use (default: `Adam()`).
- `tf`: Transverse field strength (default: 1.0).
- `rounding`: Rounding method used to obtain the classical configuration from the GCS
    state (default: [`SignRounding`](@ref QuboSolver.Solvers.LQA.SignRounding)). Accepts both a single method or a tuple of methods.
- `save_params`: Save the parameters of the variational state (default: `false`).
- `save_energy`: Save the energy of the variational state (default: `false`).
- `progressbar`: Show a progress bar (default: `true`).

## Returns
A [`Solution`](@ref QuboSolver.Solution) object containing the optimal solution found by the solver. Metadata
include the runtime as `runtime`, the parameters of the variational state as `params` (if 
`save_params` is `true`), and the energy of the variational state as `energy` (if `save_energy` 
is `true`).

## Example
```jldoctest; setup = :(using Random; Random.seed!(11))
using QuboSolver.Solvers.LQA

problem = QuboProblem([0.0 1.0; 1.0 0.0], [1.0, 0.0])
solution = solve!(problem, LQA_solver(); save_params=true, save_energy=true, progressbar=false)

# output

🟦🟦 - Energy: -3.0 - Solver: LQA_solver - Metadata count: 3
```
"""
function QuboSolver.solve!(
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
    iterations < 2 && throw(ArgumentError("Number of iterations must be at least 2"))

    if isnothing(initial_conf)
        initial_conf = dense_similar(problem.W, problem.N)
        randn!(rng, initial_conf)
        initial_conf .*= 0.001
    end

    z = similar(initial_conf)
    preprocess_input!(z, initial_conf)

    dz = similar(z)
    sines, cosines = compute_trig(z)

    state_tree = Optimisers.setup(opt, z)

    save_energy && (energy = Vector{T}(undef, iterations))

    initial_time = time()

    pbar = Progress(iterations; showspeed = true, enabled = progressbar)
    for it ∈ 1:iterations
        time = (it - 1) / (iterations - 1)
        for _ ∈ 1:inner_iterations
            grad!(dz, problem, z, sines, cosines, time, tf)
            state_tree, z = Optimisers.update(state_tree, z, dz)
            compute_trig!(sines, cosines, z)
        end

        save_energy && (energy[it] = loss(problem, sines, cosines, time, tf))
        next!(pbar)
    end

    delta_time = time() - initial_time

    metadata = (runtime = delta_time,)
    save_params && (metadata = merge(metadata, (params = z,)))
    save_energy && (metadata = merge(metadata, (energy = energy,)))

    if isa(rounding, RoundingMethod)
        return add_solution(
            problem,
            round_configuration(problem, z, rounding),
            solver;
            metadata...,
        )
    else
        return map(
            round_method -> add_solution(
                problem,
                round_configuration(problem, z, round_method),
                solver;
                metadata...,
            ),
            rounding,
        )
    end
end

end
