module BruteForce

using ..QuboSolver

export BruteForce_solver, solve!

@doc raw"""
    struct BruteForce_solver <: AbstractSolver end

Brute-force solver for QUBO problems.

Exhaustively search through all possible configurations to find the optimal solution. It is not 
recommended for large problems due to its exponential time and memory complexity. 

!!! warning
    To use this solver, you need to explicitly import the `BruteForce` module in your code:
    ```julia
    using QuboSolver.Solvers.BruteForce
    ```

"""
struct BruteForce_solver <: AbstractSolver end

function _test_chunk(problem::QuboProblem, chunk::UnitRange{Int})
    best_energy = Inf
    best_configuration = dense_similar(problem.W, Int8, problem.N)
    binary_state = dense_similar(problem.W, Int8, problem.N)

    for n ∈ chunk
        conf_int2spin!(binary_state, n)
        energy = get_energy(problem, binary_state)
        if energy < best_energy
            best_energy = energy
            best_configuration .= binary_state
        end
    end
    return best_configuration, best_energy
end

@doc raw"""
    function solve!(problem::QuboProblem, solver::BruteForce_solver)

Solve the QuboProblem `problem` using the [`BruteForce_solver`](@ref QuboSolver.Solvers.BruteForce.BruteForce_solver).	

Exhaustively search through all possible configurations to find the optimal solution. It is not 
recommended for large problems due to its exponential time and memory complexity. By default, it
uses all available threads to parallelize the search.

!!! warning
    To use this solver, you need to explicitly import the `BruteForce` module in your code:
    ```julia
    using QuboSolver.Solvers.BruteForce
    ```

# Returns
The optimal solution found by the solver. Metadata include the runtime as `runtime` and the 
number of threads used as `nthreads`.

# Example
```jldoctest
using QuboSolver.Solvers.BruteForce

problem = QuboProblem([0.0 1.0; 1.0 0.0], [1.0, 0.0])
solution = solve!(problem, BruteForce_solver())

# output

🟦🟦 - Energy: -3.0 - Solver: BruteForce_solver - Metadata count: 2
```
"""
function QuboSolver.solve!(problem::QuboProblem, solver::BruteForce_solver)
    initial_time = time()

    n = problem.N
    problem.has_bias || (n -= 1)
    range = 0:(2^n-1)

    n_batches = max(1, length(range) ÷ Threads.nthreads())
    chunks = Iterators.partition(range, n_batches)
    tasks = map(chunks) do chunk
        Threads.@spawn _test_chunk(problem, chunk)
    end

    configuration, _ = mapreduce(fetch, (x, y) -> x[2] < y[2] ? x : y, tasks)
    delta_time = time() - initial_time

    sol = add_solution(
        problem,
        configuration,
        solver;
        runtime = delta_time,
        nthreads = Threads.nthreads(),
    )
    return sol
end

end
