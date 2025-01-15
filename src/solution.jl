
export Solution

struct Solution{Te<:AbstractFloat,Ts<:AbstractSolver}
    energy::Te
    configuration::Vector{Int8}
    solver::Ts
    metadata::Dict{Symbol,Any}

    function Solution(energy::AbstractFloat, configuration::Vector{Int8}, solver::AbstractSolver; kwargs...)
        any(x -> abs(x) != 1, configuration) && throw(ArgumentError("Configuration must be a +-1 vector"))
        return new{typeof(energy),typeof(solver)}(energy, configuration, solver, Dict(kwargs))
    end
end

Solution(energy::Real, configuration::Vector{<:Real}, solver::AbstractSolver; kwargs...) =
    Solution(Float64.(energy), Int8.(configuration), solver; kwargs...)

function Base.show(io::IO, sol::Solution)
    conf = join(c == true ? "🟦" : "⬜" for c in sol.configuration)
    return print(
        io,
        "$conf - Energy: $(round(sol.energy, digits=4)) - Solver: $(Base.typename(typeof(sol.solver)).wrapper) - Metadata count: $(length(sol.metadata))",
    )
end

Base.getindex(sol::Solution, key::Symbol) = sol.metadata[key]
Base.getindex(sol::Solution, key::String) = sol[Symbol(key)]

Base.isequal(sol1::Solution, sol2::Solution) = all(getfield(sol1, f) == getfield(sol2, f) for f in fieldnames(Solution))
