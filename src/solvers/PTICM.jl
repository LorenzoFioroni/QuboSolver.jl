module PTICM

using ..QuboSolver

export PTICM_solver, solve!

@doc raw"""
    struct PTICM_solver <: AbstractSolver end

Tempering and Annealing Monte Carlo solver for QUBO problems.

Use the TAMC library [bauzaScalingAdvantageApproximate2024](@cite) to solve the QUBO problem.

!!! warning 
    The TAMC solver requires the TAMC binaries to be installed.
    See the [https://github.com/USCqserver/tamc](https://github.com/USCqserver/tamc) for more information.

!!! warning
    To use this solver, you need to explicitly import the `PTICM` module in your code:
    ```julia
    using QuboSolver.Solvers.PTICM
    ```
"""
struct PTICM_solver <: AbstractSolver end

function write_method_file(
    MCS::Integer,
    warmup_fraction::Real,
    betas::Vector{T},
    lo_num_beta::Integer,
    threads::Integer,
    num_replica_chains::Integer,
) where {T}
    method_file = tempname()
    open(method_file, "w") do io
        println(io, "---")
        println(io, "PT:")
        println(io, "  num_sweeps: $MCS")
        println(io, "  warmup_fraction: $warmup_fraction")
        println(io, "  beta:")
        println(io, "    Arr:")
        for beta ∈ betas
            println(io, "    - $beta")
        end
        println(io, "  icm: true")
        println(io, "  lo_num_beta: $lo_num_beta")
        println(io, "  threads: $threads")
        return println(io, "  num_replica_chains: $num_replica_chains")
    end
    return method_file
end

function write_instance_file(
    problem::QuboProblem{T,TW,Tc},
) where {T<:AbstractFloat,TW<:AbstractMatrix{T},Tc<:Union{Nothing,AbstractVector{T}}}
    compact_idx, compact_W = nonzero_triu(problem.W)

    instance_file = tempname()
    open(instance_file, "w") do io
        for ((i, j), w) ∈ zip(compact_idx, compact_W)
            println(io, "$(i-1) $(j-1) $(-w)")
        end
        if problem.has_bias
            for (i, c) ∈ enumerate(problem.c)
                println(io, "$(i-1) $(i-1) $(c)")
            end
        end
    end

    return instance_file
end

function parse_solution_file(problem, output_file::String)
    energies = Float64[]
    states = Vector{BigInt}[]
    runtime = nothing
    open(output_file, "r") do io
        while !eof(io)
            line = readline(io)
            startswith(line, "timing:") && (runtime = 1e-6 * parse(Float64, line[8:end]))
            startswith(line, "gs_energies:") && break
        end
        while !eof(io)
            line = readline(io)
            startswith(line, "gs_states:") && break
            push!(energies, parse(eltype(problem.W), line[5:end]))
        end
        for _ ∈ energies
            push!(states, BigInt[])
        end
        s_idx = 0
        while !eof(io)
            line = readline(io)
            startswith(line, "num_measurements") && break
            if startswith(line, "  - - ")
                s_idx += 1
            end
            push!(states[s_idx], parse(BigInt, line[7:end]))
        end
    end

    min_idx = argmin(energies)
    min_state = Int8[]
    n = problem.N
    for j ∈ eachindex(states[min_idx])
        pad = min(64, n)
        append!(min_state, conf_int2spin(states[min_idx][j], pad))
        n -= pad
    end

    return runtime, min_state
end

@doc raw"""
    function solve!(
        problem::QuboProblem{T,TW,Tc},
        solver::PTICM_solver;
        MCS::Integer = 1000,
        warmup_fraction::Real = 0.5,
        betas::Vector{T} = T.(QuboSolver.LogRange(0.1, 5.0, 32)),
        lo_num_beta::Integer = 8,
        threads::Integer = Threads.nthreads(),
        num_replica_chains::Integer = 2,
    ) where {T<:AbstractFloat,TW<:AbstractMatrix{T},Tc<:Union{Nothing,AbstractVector{T}}}

Solve the QuboProblem `problem` using the TAMC solver [bauzaScalingAdvantageApproximate2024](@cite).

!!! warning 
    The TAMC solver requires the TAMC binaries to be installed.
    See the [https://github.com/USCqserver/tamc](https://github.com/USCqserver/tamc) for more information.

!!! warning
    To use this solver, you need to explicitly import the `PTICM` module in your code:
    ```julia
    using QuboSolver.Solvers.PTICM
    ```

## Arguments
- `problem::QuboProblem`: The QUBO problem to be solved.
- `solver::PTICM_solver`: Instance of [`PTICM_solver`](@ref QuboSolver.Solvers.PTICM.PTICM_solver).
- `MCS::Integer`: The number of Monte Carlo steps. The default value is `1000`.
- `warmup_fraction::Real`: The fraction of Monte Carlo steps to be used for warmup. The default value is 
    `0.5`.
- `betas::Vector{T}`: The inverse temperatures to be used. The default value is a logarithmic range
    between `0.1` and `5.0` with `32` elements.
- `lo_num_beta::Integer`: The number of inverse temperatures to be used for the isoenergetic cluster
    moves. The default value is `8`.
- `threads::Integer`: The number of threads to be used. The default value is the number of available
    threads.
- `num_replica_chains::Integer`: The number of replica chains to be used. The default value is `2`.

## Returns
The optimal solution found by the solver. Metadata include the runtime as `runtime`.

## Example
```julia
using QuboSolver.Solvers.PTICM

problem = QuboProblem([0.0 1.0; 1.0 0.0], [1.0, 0.0])
solution = solve!(problem, PTICM_solver())

# output

🟦🟦 - Energy: -3.0 - Solver: PTICM_solver - Metadata count: 1
```
"""
function QuboSolver.solve!(
    problem::QuboProblem{T,TW,Tc},
    solver::PTICM_solver;
    MCS::Integer = 1000,
    warmup_fraction::Real = 0.5,
    betas::Vector{T} = T.(QuboSolver.LogRange(0.1, 5.0, 32)),
    lo_num_beta::Integer = 8,
    threads::Integer = Threads.nthreads(),
    num_replica_chains::Integer = 2,
) where {T<:AbstractFloat,TW<:AbstractMatrix{T},Tc<:Union{Nothing,AbstractVector{T}}}
    0 < warmup_fraction < 1 ||
        throw(ArgumentError("The warmup fraction must be between 0 and 1"))
    0 < MCS || throw(ArgumentError("The number of Monte Carlo steps must be positive"))
    length(betas) > 0 || throw(ArgumentError("The number of betas must be positive"))
    0 < lo_num_beta <= length(betas) || throw(
        ArgumentError(
            "The number of betas for the isoenergetic cluster moves must be positive and less than the total number of betas",
        ),
    )
    threads > 0 || throw(ArgumentError("The number of threads must be positive"))

    sort!(betas)

    method_file = write_method_file(
        MCS,
        warmup_fraction,
        betas,
        lo_num_beta,
        threads,
        num_replica_chains,
    )
    instance_file = write_instance_file(problem)
    output_file = tempname()

    try
        run(
            pipeline(
                `tamc $method_file $instance_file $output_file`;
                stdout = devnull,
                stderr = devnull,
            ),
        )
    catch
        throw(ArgumentError("Error while running TAMC"))
    end

    runtime, conf = parse_solution_file(problem, output_file)

    sol = Solution(get_energy(problem, conf), conf, solver; runtime = runtime)
    add_solution(problem, sol)

    rm(method_file)
    rm(instance_file)
    rm(output_file)

    return sol
end

end
