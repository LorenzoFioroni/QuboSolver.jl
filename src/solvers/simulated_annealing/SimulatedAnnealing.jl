module SimulatedAnnealing

using ..QuboSolver
using Random
using ProgressMeter

export SimulatedAnnealingSolver, solve!

struct SimulatedAnnealingSolver <: AbstractSolver end

schedule(it::Integer, N_it::Integer) = 1 - (it - 1) / (N_it - 1)

function QuboSolver.solve!(
    problem::QuboProblem,
    solver::SimulatedAnnealingSolver;
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

    @showprogress showspeed = true enabled = progressbar for mcs in 1:MCS
        T = initial_temp * schedule(mcs, MCS)

        for _ in 1:problem.N
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
    end

    delta_time = time() - initial_time

    sol = add_solution(problem, z_min, solver, runtime = delta_time)
    return sol
end

end
