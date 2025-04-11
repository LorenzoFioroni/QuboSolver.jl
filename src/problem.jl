export QuboProblem, get_energy, add_solution, solve!

@doc raw"""
    struct QuboProblem{
        T<:AbstractFloat,
        Tw<:AbstractMatrix{T},
        Tc<:Union{AbstractVector{T},Nothing}
    }

A QUBO problem with matrix `W`, optional bias vector `c`, and value type `T`.

The matrix `W` must be square and symmetric, and the diagonal must be zero.
The bias vector `c` is optional and must have the same length as the number of rows in `W`.

# Fields
- `W::Tw`: The QUBO matrix.
- `has_bias::Bool`: Indicates if a bias vector is provided.
- `c::Tc`: The bias vector, or `nothing` if not provided.
- `N::Int`: The number of variables in the problem.
- `solutions::Vector{Solution}`: A vector to store solutions found.
"""
struct QuboProblem{
    T<:AbstractFloat,
    Tw<:AbstractMatrix{T},
    Tc<:Union{AbstractVector{T},Nothing},
}
    W::Tw
    has_bias::Bool
    c::Tc
    N::Int
    solutions::Vector{Solution}

    @doc raw"""
        function QuboProblem(
            W::AbstractMatrix{T}, 
            bias::Union{AbstractVector{T},Nothing} = nothing
        ) where {T<:AbstractFloat}
        
    Create a new QuboProblem instance with the given matrix `W` and optional bias vector `bias`.

    # Example
    ```jldoctest
    W = [0.0 1.0; 1.0 0.0]
    bias = [0.0, 1.0]
    println(QuboProblem(W))
    println(QuboProblem(W, bias))

    # output

    QuboProblem{Float64, Matrix{Float64}, Nothing}([0.0 1.0; 1.0 0.0], false, nothing, 2, Solution[])
    QuboProblem{Float64, Matrix{Float64}, Vector{Float64}}([0.0 1.0; 1.0 0.0], true, [0.0, 1.0], 2, Solution[])
    ```
    """
    function QuboProblem(
        W::AbstractMatrix{T},
        bias::Union{AbstractVector{T},Nothing} = nothing,
    ) where {T<:AbstractFloat}
        N, M = size(W)

        has_bias = !isnothing(bias)

        Tw = typeof(W)
        Tc = typeof(bias)

        N == M || throw(ArgumentError("Matrix W must be square"))
        all(iszero, W[diagind(W)]) ||
            throw(ArgumentError("Matrix W must have zero diagonal"))
        (!has_bias || length(bias) == N) ||
            throw(ArgumentError("Bias vector must have length $N"))
        triu(W, 1) == tril(W, -1)' || throw(ArgumentError("Matrix W must be symmetric"))

        return new{T,Tw,Tc}(W, has_bias, bias, N, Solution[])
    end
end

@doc raw"""
    function get_energy(
        problem::QuboProblem, 
        configuration::AbstractVector
    )

Calculate the energy of a given configuration for the given QuboProblem `problem`.

The configuration must be a vector of ``-1`` or ``1`` values. The returned energy is 
``E = -\vec{x}^T W \vec{x} - \vec{c}^T \vec{x}``, where ``\vec{x}`` is the
 configuration vector, ``W`` is the QUBO matrix, and ``\vec{c}`` is the bias vector.
"""
function get_energy end

function get_energy(problem::QuboProblem, configuration::AbstractVector{Int8})
    any(x -> abs(x) != 1, configuration) &&
        throw(ArgumentError("Configuration must be a +-1 vector"))
    E = configuration' * problem.W * configuration
    problem.has_bias && (E += problem.c' * configuration)
    return -E
end

function get_energy(problem::QuboProblem, configuration::AbstractVector{<:Real})
    return get_energy(problem, Int8.(configuration))
end

@doc raw"""
    add_solution(problem::QuboProblem, sol::Solution)

Add a [`Solution`](@ref QuboSolver.Solution) `sol` to the [`QuboProblem`](@ref QuboSolver.QuboProblem) `problem`. 

# Example
```jldoctest
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
"""
add_solution(problem::QuboProblem, sol::Solution) = push!(problem.solutions, sol)

@doc raw"""
    add_solution(
        problem::QuboProblem, 
        configuration::Vector, 
        solver::AbstractSolver;
        kwargs...
    )    

    Add a solution with the given `configuration` and `solver` to the [`QuboProblem`](@ref QuboSolver.QuboProblem) `problem`.

The configuration must be a vector of ``-1`` or ``1`` values. Additional metadata can be passed 
as keyword arguments.

# Returns
The added [`Solution`](@ref QuboSolver.Solution) object.

# Example
```jldoctest
struct MySolver <: AbstractSolver end
problem = QuboProblem([0.0 1.0; 1.0 0.0])

println(problem.solutions)
add_solution(problem, Int8[1, -1], MySolver(); runtime=1.1)
println(problem.solutions)

# output

Solution[]
Solution[🟦🟨 - Energy: 2.0 - Solver: MySolver - Metadata count: 1]
```
"""
function add_solution(
    problem::QuboProblem,
    configuration::Vector{Int8},
    solver::AbstractSolver;
    kwargs...,
)
    E = get_energy(problem, configuration)
    sol = Solution(E, configuration, solver; kwargs...)
    add_solution(problem, sol)
    return sol
end

function add_solution(
    problem::QuboProblem,
    configuration::Vector{<:Real},
    solver::AbstractSolver;
    kwargs...,
)
    return add_solution(problem, Int8.(configuration), solver, kwargs...)
end

@doc raw"""
    solve!(problem::QuboProblem, solver::AbstractSolver, args...; kwargs...)

Solve the QuboProblem `problem` using the provided `solver`.

The `solver` instance must implement the `solve!` method.
"""
function solve!(problem::QuboProblem, solver::AbstractSolver, args...; kwargs...)
    throw(ErrorException("$solver does not implement the solve! method"))
end
