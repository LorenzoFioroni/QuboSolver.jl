export QuboProblem, get_energy, add_solution, solve!

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
add_solution(problem::QuboProblem, sol::Solution) = push!(problem.solutions, sol)

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

function solve!(problem::QuboProblem, solver::AbstractSolver, args...; kwargs...)
    throw(ErrorException("$solver does not implement the solve! method"))
end
