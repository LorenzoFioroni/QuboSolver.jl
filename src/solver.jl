
export AbstractSolver

abstract type AbstractSolver end

function Base.show(io::IO, solver::AbstractSolver)
    return print(io, "Solver: ", Base.typename(typeof(solver)).name)
end
