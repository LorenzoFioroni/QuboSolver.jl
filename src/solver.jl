
export AbstractSolver

abstract type AbstractSolver end

Base.show(io::IO, solver::AbstractSolver) = print(io, "Solver: ", Base.typename(typeof(solver)).wrapper)
