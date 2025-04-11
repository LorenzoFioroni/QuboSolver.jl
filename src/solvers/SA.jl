module SA

using ..QuboSolver
using Random
using ProgressMeter

export SA_solver, solve!

@doc raw"""
    struct SA_solver <: AbstractSolver end

Simulated Annealing solver for QUBO problems [kirkpatrickOptimizationSimulatedAnnealing1983](@cite).

!!! warning
    To use this solver, you need to explicitly import the `SA` module in your code:
    ```julia
    using QuboSolver.Solvers.SA
    ```
"""
struct SA_solver <: AbstractSolver end

schedule(it::Integer, N_it::Integer) = 1 - (it - 1) / (N_it - 1)

@doc raw"""
    function solve!(
        problem::QuboProblem,
        solver::SA_solver;
        MCS::Integer = 1000,
        rng::Random.AbstractRNG = Random.GLOBAL_RNG,
        initial_conf::Union{AbstractVector{Int8},Nothing} = nothing,
        initial_temp::Real = 1.0,
        progressbar::Bool = true,
    )

Solve the QuboProblem `problem` using simulated annealing [kirkpatrickOptimizationSimulatedAnnealing1983](@cite) with a linearly decreasing temperature 
schedule. 

!!! warning
    To use this solver, you need to explicitly import the `SA` module in your code:
    ```julia
    using QuboSolver.Solvers.SA
    ```

## Arguments
- `problem::QuboProblem`: [`QuboProblem`](@ref QuboSolver.QuboProblem) instance to be solved.
- `solver::SA_solver`: Instance of [`SA_solver`](@ref QuboSolver.Solvers.SA.SA_solver).
- `MCS::Integer`: Number of Monte Carlo steps. The default value is `1000`.
- `rng::Random.AbstractRNG`: Random number generator. The default value is `Random.GLOBAL_RNG`.
- `initial_conf::Union{AbstractVector{Int8},Nothing}`: Initial configuration. If `nothing`, a random 
    configuration is generated. The default value is `nothing`.
- `initial_temp::Real`: Initial temperature. The default value is `1.0`.
- `progressbar::Bool`: Whether to show a progress bar. The default value is `true`.

## Returns
The optimal solution found by the solver. Metadata include the runtime as `runtime`.

## Example
```jldoctest
using QuboSolver.Solvers.SA

problem = QuboProblem([0.0 1.0; 1.0 0.0], [1.0, 0.0])
solution = solve!(problem, SA_solver())

# output

🟦🟦 - Energy: -3.0 - Solver: SA_solver - Metadata count: 1
```
"""
function QuboSolver.solve!(
    problem::QuboProblem,
    solver::SA_solver;
    MCS::Integer = 1000,
    rng::Random.AbstractRNG = Random.GLOBAL_RNG,
    initial_conf::Union{AbstractVector{Int8},Nothing} = nothing,
    initial_temp::Real = 1.0,
    progressbar::Bool = true,
)
    if isnothing(initial_conf)
        z = dense_similar(problem.W, Int8, problem.N)
        rand!(rng, z, Int8.((-1, 1)))
    else
        z = copy(initial_conf)
    end

    MCS >= 2 || throw(ArgumentError("The number of Monte Carlo steps must be at least 2"))

    E = get_energy(problem, z)

    z_min = copy(z)
    E_min = E

    initial_time = time()

    pbar = Progress(MCS; showspeed = true, enabled = progressbar)
    for mcs ∈ 1:MCS
        T = initial_temp * schedule(mcs, MCS)

        for _ ∈ 1:problem.N
            flip_idx = rand(rng, 1:problem.N)

            delta_E = 4 * z[flip_idx] * dot(problem.W[:, flip_idx], z)
            problem.has_bias && (delta_E += 2 * problem.c[flip_idx] * z[flip_idx])

            if delta_E < 0 || rand(rng) < exp(-delta_E / T)
                z[flip_idx] *= -1
                E += delta_E

                if E < E_min
                    z_min .= z
                    E_min = E
                end
            end
        end
        next!(pbar)
    end

    delta_time = time() - initial_time

    sol = add_solution(problem, z_min, solver; runtime = delta_time)
    return sol
end

end
