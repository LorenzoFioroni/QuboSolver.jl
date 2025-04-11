@testitem "simulated annealing solver" begin
    using StableRNGs
    import Suppressor: @capture_err
    using QuboSolver.Solvers.SA

    @test SA_solver <: AbstractSolver

    # Test schedule
    times = SA.schedule.(1:300, 300)
    @test all(x -> 0.0 <= x <= 1.0, times)
    @test times[1] == 1.0
    @test times[end] == 0.0

    # Test result
    N = 5
    W = (-1) .^ ((0:N-1)' .+ (0:N-1)) .* rand(N, N)
    W = triu(W, 1) + triu(W, 1)'
    bias = zeros(N)
    idx = rand(1:N)
    bias[idx] = 2.0
    problem = QuboProblem(W, bias)
    solver = SA_solver()
    rng = StableRNG(1234)
    sol = solve!(problem, solver; MCS = 100_000, rng = rng, progressbar = false)
    @test sol isa Solution{Float64,SA_solver}
    @test sol.energy isa Float64
    @test sol.configuration isa Vector{Int8}
    @test sol.energy ≈ -sum(abs, W) - sum(bias)
    @test sol.configuration == [1, -1, 1, -1, 1] .* (-1)^(idx - 1) # spin idx is going to be +1
    @test sol.solver === solver
    @test sol.metadata[:runtime] isa Float64

    # Test initial configuration
    N = 5
    W = (-1) .^ ((0:N-1)' .+ (0:N-1)) .* rand(N, N)
    W = triu(W, 1) + triu(W, 1)'
    bias = zeros(N)
    idx = rand(1:N)
    bias[idx] = 2.0
    problem = QuboProblem(W, bias)
    solver = SA_solver()
    rng = StableRNG(1234)
    initial_conf = Int8[-1, -1, -1, -1, -1]
    sol = solve!(
        problem,
        solver;
        MCS = 2,
        rng = rng,
        initial_conf = initial_conf,
        progressbar = false,
    )
    @test sol.configuration == [-1, 1, -1, 1, -1] # valid with this initial conf

    # Test progress bar suppression
    N = 5
    W = (-1) .^ ((0:N-1)' .+ (0:N-1)) .* rand(N, N)
    W = triu(W, 1) + triu(W, 1)'
    problem = QuboProblem(W)
    rng = StableRNG(1234)
    output = @capture_err begin
        sol = solve!(problem, solver; MCS = 10_000, rng = rng, progressbar = false)
    end
    expected_output = ""
    @test output == expected_output

    # Test progress bar 
    N = 5
    W = (-1) .^ ((0:N-1)' .+ (0:N-1)) .* rand(N, N)
    W = triu(W, 1) + triu(W, 1)'
    problem = QuboProblem(W)
    rng = StableRNG(1234)
    output = @capture_err begin
        sol = solve!(problem, solver; MCS = 10_000_000, rng = rng, progressbar = true)
    end
    @test output != ""

    # Test MCS bounds
    N = 5
    W = (-1) .^ ((0:N-1)' .+ (0:N-1)) .* rand(N, N)
    W = triu(W, 1) + triu(W, 1)'
    problem = QuboProblem(W)
    rng = StableRNG(1234)
    @test_throws ArgumentError solve!(
        problem,
        solver,
        MCS = 1,
        rng = rng,
        progressbar = false,
    )
end
