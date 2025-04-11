@testitem "gurobi solver" begin
    using StableRNGs
    import Suppressor: @capture_out
    using QuboSolver.Solvers.GurobiLib

    @test Gurobi_solver <: AbstractSolver

    # Test exact solution
    N = 5
    W = (-1) .^ ((0:N-1)' .+ (0:N-1)) .* rand(N, N)
    W = triu(W, 1) + triu(W, 1)'
    bias = zeros(N)
    idx = rand(1:N)
    bias[idx] = 0.1
    problem = QuboProblem(W, bias)
    solver = Gurobi_solver()
    mipgap = 0.0
    sol = solve!(problem, solver; mipgap = 0.0, output = false)
    @test sol isa Solution{Float64,Gurobi_solver}
    @test sol.energy isa Float64
    @test sol.configuration isa Vector{Int8}
    @test sol.energy ≈ -sum(abs, W) - sum(bias)
    @test sol.configuration == [1, -1, 1, -1, 1] .* (-1)^(idx - 1) # spin idx is going to be +1
    @test sol.solver === solver
    @test sol.metadata[:runtime] isa Float64
    @test 0.0 < sol.metadata[:runtime] < 60
    @test sol.metadata[:mipgap] == 0.0

    # Test approximate solution
    L = 10
    rng = StableRNG(1234)
    W = rand(EdwardsAnderson(), L; rng = rng, sparse = true)
    problem = QuboProblem(W)
    exact_energy = -1625.8733846669998
    mipgap = 0.1
    sol = solve!(problem, Gurobi_solver(); mipgap = mipgap, output = false)
    @test sol.metadata[:mipgap] == mipgap
    bound = abs(sol.energy - exact_energy) / abs(sol.energy)
    @test bound < mipgap

    # Test timeout
    N = 100
    W = rand(SherringtonKirkpatrick(), N)
    problem = QuboProblem(W)
    sol = solve!(problem, Gurobi_solver(); timelimit = 1.0, output = false)
    @test 0.99 < sol.metadata[:runtime] < 1.5

    # Test output capturing
    N = 5
    W = (-1) .^ ((0:N-1)' .+ (0:N-1)) .* rand(N, N)
    W = triu(W, 1) + triu(W, 1)'
    problem = QuboProblem(W)
    output = @capture_out begin
        sol = solve!(problem, Gurobi_solver(); output = false)
    end
    expected_output = ""
    @test output == expected_output
end
