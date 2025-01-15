module BruteForce

using ..QuboSolver

export BruteForceSolver, solve!

struct BruteForceSolver <: AbstractSolver end

function _test_chunk(problem::QuboProblem, chunk::UnitRange{Int})
    best_energy = Inf
    best_configuration = dense_similar(problem.W, Int8, problem.N)
    binary_state = dense_similar(problem.W, Int8, problem.N)

    for n in chunk
        spindigits!(binary_state, n)
        energy = get_energy(problem, binary_state)
        if energy < best_energy
            best_energy = energy
            best_configuration .= binary_state
        end
    end
    return best_configuration, best_energy
end

function QuboSolver.solve!(problem::QuboProblem, solver::BruteForceSolver)
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

    sol = add_solution(problem, configuration, solver, runtime = delta_time, nthreads = Threads.nthreads())
    return sol
end

end
