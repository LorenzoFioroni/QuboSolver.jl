module GurobiLib

using ..QuboSolver
using Gurobi

export Gurobi_solver, solve!

@doc raw"""
    struct Gurobi_solver <: AbstractSolver end

Gurobi solver for QUBO problems.

Use the Gurobi optimization library [gurobioptimizationllcGurobiOptimizerReference2024](@cite) to solve the QUBO problem.

!!! warning 
    The Gurobi solver requires a valid Gurobi license and the Gurobi binaries to be installed.
    See the [Gurobi website](https://www.gurobi.com) for more information.

!!! warning
    To use this solver, you need to explicitly import the `GurobiLib` module in your code:
    ```julia
    using QuboSolver.Solvers.GurobiLib
    ```
"""
struct Gurobi_solver <: AbstractSolver end

@doc raw"""
    function solve!(
        problem::QuboProblem,
        solver::Gurobi_solver;
        mipgap::Float64 = 0.0,
        timelimit::Float64 = Inf64,
        output::Bool = true,
        threads::Int = Threads.nthreads(),
    )

Solve the QuboProblem `problem` using the Gurobi library [gurobioptimizationllcGurobiOptimizerReference2024](@cite).

!!! warning 
    The Gurobi solver requires a valid Gurobi license. 
    See the [Gurobi website](https://www.gurobi.com) for more information.

!!! warning
    To use this solver, you need to explicitly import the `GurobiLib` module in your code:
    ```julia
    using QuboSolver.Solvers.GurobiLib
    ```

## Arguments
- `problem::QuboProblem`: The QUBO problem to be solved.
- `solver::Gurobi_solver`: Instance of [`Gurobi_solver`](@ref QuboSolver.Solvers.GurobiLib.Gurobi_solver).
- `mipgap::Float64`: The maximum allowed gap between the found lower bound and the upper bound. 
    The default value is `0.0`, which means that the solver will try to find the optimal solution.
- `timelimit::Float64`: The maximum time limit for the solver in seconds. The default value is `Inf64`,
- `output::Bool`: Whether to print the solver output. The default value is `true`.
- `threads::Int`: The number of threads to use for the solver. The default value is the number of 
    available threads on the system.

## Returns
The optimal solution found by the solver. Metadata include the runtime as `runtime` and the 
maximum gap as `mipgap`.

## Example
```julia
using QuboSolver.Solvers.GurobiLib

problem = QuboProblem([0.0 1.0; 1.0 0.0], [1.0, 0.0])
solution = solve!(problem, Gurobi_solver(), output=false)

# output

🟦🟦 - Energy: -3.0 - Solver: Gurobi_solver - Metadata count: 2
```
"""
function QuboSolver.solve!(
    problem::QuboProblem{T,TW,Tc},
    solver::Gurobi_solver;
    mipgap::Float64 = 0.0,
    timelimit::Float64 = Inf64,
    output::Bool = true,
    threads::Int = Threads.nthreads(),
) where {T<:AbstractFloat,TW<:AbstractMatrix{T},Tc<:Union{Nothing,AbstractVector{T}}}
    initial_time = time()

    0.0 <= mipgap <= 1.0 || throw(ArgumentError("mipgap must be in [0, 1]"))

    W_bin, c_bin = spin_to_binary(problem.W, problem.c)
    N = problem.N

    env_p = Ref{Ptr{Cvoid}}()
    model_p = Ref{Ptr{Cvoid}}()
    solution = Vector{Float64}(undef, N)

    qrow = Cint[]
    qcol = Cint[]
    qval = Cdouble[]

    for i ∈ 1:N
        for j ∈ 1:N
            if W_bin[i, j] != 0.0
                push!(qrow, i - 1)
                push!(qcol, j - 1)
                push!(qval, W_bin[i, j])
            end
        end
    end

    error = GRBemptyenv(env_p)
    iszero(error) || throw(ErrorException(unsafe_string(GRBgeterrormsg(env_p[]))))
    env = env_p[]

    error = GRBsetintparam(env, "OutputFlag", output ? 1 : 0)
    iszero(error) || throw(ErrorException(unsafe_string(GRBgeterrormsg(env_p[]))))

    error = GRBsetintparam(env, "Threads", threads)
    iszero(error) || throw(ErrorException(unsafe_string(GRBgeterrormsg(env_p[]))))

    error = GRBstartenv(env)
    iszero(error) || throw(ErrorException(unsafe_string(GRBgeterrormsg(env_p[]))))

    error = GRBsetdblparam(env, "MIPGap", mipgap)
    iszero(error) || throw(ErrorException(unsafe_string(GRBgeterrormsg(env_p[]))))

    error = GRBsetdblparam(env, "TimeLimit", timelimit)
    iszero(error) || throw(ErrorException(unsafe_string(GRBgeterrormsg(env_p[]))))

    error = GRBnewmodel(
        env,
        model_p,
        "qp",
        N,
        c_bin,
        C_NULL,
        C_NULL,
        repeat(GRB_BINARY, N),
        C_NULL,
    )
    iszero(error) || throw(ErrorException(unsafe_string(GRBgeterrormsg(env_p[]))))
    model = model_p[]

    error = GRBaddqpterms(model, length(qval), qrow, qcol, qval)
    iszero(error) || throw(ErrorException(unsafe_string(GRBgeterrormsg(env_p[]))))

    error = GRBsetintattr(model, "ModelSense", GRB_MAXIMIZE)
    iszero(error) || throw(ErrorException(unsafe_string(GRBgeterrormsg(env_p[]))))

    error = GRBoptimize(model)
    iszero(error) || throw(ErrorException(unsafe_string(GRBgeterrormsg(env_p[]))))

    error = GRBgetdblattrarray(model, "X", 0, N, solution)
    iszero(error) || throw(ErrorException(unsafe_string(GRBgeterrormsg(env_p[]))))

    spinsol = round.(Int8, 2 .* solution .- 1)

    delta_time = time() - initial_time

    error = GRBfreemodel(model)
    iszero(error) || throw(ErrorException(unsafe_string(GRBgeterrormsg(env_p[]))))

    GRBfreeenv(env)

    return add_solution(problem, spinsol, solver; runtime = delta_time, mipgap = mipgap)
end

end
