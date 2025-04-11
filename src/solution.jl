
export Solution

@doc raw"""
    struct Solution{Te<:AbstractFloat,Ts<:AbstractSolver}

A solution to a QUBO problem. 

Stores a reference to the solver used to find the solution, its energy, the configuration, and 
any additional metadata. Once metadata is populated, its values can also be accessed indexing
the solution directly. If the solution is displayed, the configuration is shown as a string of 
colored squares, where blue represents ``1`` and yellow represents ``-1``.

# Fields
- `energy::Te`: The energy of the solution.
- `configuration::Vector`: Vector of ``-1`` or ``1`` values representing the configuration.
- `solver::Ts`: The solver used to find the solution.
- `metadata::Dict{Symbol,Any}`: Additional metadata about the solution.
"""
struct Solution{Te<:AbstractFloat,Ts<:AbstractSolver}
    energy::Te
    configuration::Vector{Int8}
    solver::Ts
    metadata::Dict{Symbol,Any}

    @doc raw"""
        function Solution(
            energy::AbstractFloat, 
            configuration::Vector{Int8}, 
            solver::AbstractSolver; 
            kwargs...
        )

    Create a new `Solution` instance. The keyword arguments specify additional metadata.

    # Example
    ```jldoctest
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
    """
    function Solution(
        energy::AbstractFloat,
        configuration::Vector{Int8},
        solver::AbstractSolver;
        kwargs...,
    )
        any(x -> abs(x) != 1, configuration) &&
            throw(ArgumentError("Configuration must be a +-1 vector"))
        return new{typeof(energy),typeof(solver)}(
            energy,
            configuration,
            solver,
            Dict(kwargs),
        )
    end
end

function Solution(
    energy::Real,
    configuration::Vector{<:Real},
    solver::AbstractSolver;
    kwargs...,
)
    return Solution(Float64.(energy), Int8.(configuration), solver; kwargs...)
end

function Base.show(io::IO, sol::Solution)
    conf = join(c == 1 ? "🟦" : "🟨" for c ∈ sol.configuration)
    return print(
        io,
        "$conf - Energy: $(round(sol.energy, digits=4)) - Solver: $(Base.typename(typeof(sol.solver)).name) - Metadata count: $(length(sol.metadata))",
    )
end

Base.getindex(sol::Solution, key::Symbol) = sol.metadata[key]
Base.getindex(sol::Solution, key::String) = sol[Symbol(key)]

function Base.isequal(sol1::Solution, sol2::Solution)
    return all(getfield(sol1, f) == getfield(sol2, f) for f ∈ fieldnames(Solution))
end
