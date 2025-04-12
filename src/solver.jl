
export AbstractSolver

@doc raw"""
    abstract type AbstractSolver end

Abstract type for a QUBO problem solver.

Concrete solvers should inherit from this type and implement the QuboSolver.solve! method. This
should take a QuboProblem, the solver instance, and any additional arguments and return a [`Solution`](@ref QuboSolver.Solution)
object. 

## Example
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
"""
abstract type AbstractSolver end

function Base.show(io::IO, solver::AbstractSolver)
    return print(io, "Solver: ", Base.typename(typeof(solver)).name)
end
